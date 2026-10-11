"""阶段一：单卡物理容量与逐步 Roofline 计时金标准。"""

from dataclasses import asdict

import pytest
from pydantic import ValidationError

from simcore.analytical import step_cost
from simcore.capacity import resolve_config
from simcore.engine import Engine
from simcore.perf_model import (
    DecodeWork,
    PrefillWork,
    StepWorkload,
    restore_time,
    restore_time_tokens,
)
from simcore.runner import simulate_analytical, simulate_resolved
from simcore.schemas import (
    AttentionGroup,
    EngineConfig,
    FixedLength,
    HardwareSpec,
    InferenceRequest,
    ModelSpec,
    PerfModelConfig,
    ResolvedRequest,
    SimConfig,
    StorageSpec,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)


def example_request() -> ResolvedRequest:
    return ResolvedRequest(
        config=SimConfig(
            num_instances=1,
            workload=WorkloadConfig(
                synthetic=SyntheticWorkloadConfig(
                    num_requests=1,
                    session_rate=1,
                    prompt_len=FixedLength(value=4),
                    turn_increment_len=FixedLength(value=1),
                    output_len=FixedLength(value=2),
                    max_new_tokens=2,
                    seed=0,
                )
            ),
            engine=EngineConfig(max_running_requests=1, max_total_tokens=4, page_size=4),
            perf_model=PerfModelConfig(prefill_c1=1, decode_d1=2, decode_d2=1),
        ),
        model=ModelSpec(
            name="toy",
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=8,
            ffn_intermediate_size=32,
            vocab_size=16,
        ),
        hardware=HardwareSpec(
            name="toy",
            memory_bytes_per_gpu=2_000_000,
            effective_fp16_tflops=1,
            effective_hbm_gb_s=1,
        ),
        storage=StorageSpec(
            l1_available_gb=0.001,
            l2_available_gb=0.000000512,
            l3_available_gb=0.000000256,
            runtime_reserve_bytes=1000,
            l2_to_l1_gb_s=0.001,
            l3_to_l2_gb_s=0.002,
            l2_latency_ms=3,
            l3_latency_ms=4,
            l3_lookup_latency_ms=1.25,
        ),
    )


def test_attention_groups_distinguish_mla_from_mla_dsa() -> None:
    request = example_request()
    mla = AttentionGroup(
        kind="mla", layers=1, num_attention_heads=2,
        q_rank=4, kv_rank=4, rope_dim=2, nope_dim=6, value_dim=8,
    )
    dsa = AttentionGroup(
        kind="mla_dsa", layers=1, num_attention_heads=2,
        q_rank=4, kv_rank=4, rope_dim=2, nope_dim=6, value_dim=8,
        topk=4, index_heads=2, index_dim=128, full_indexer_layers=1,
    )
    model = request.model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": [mla, dsa],
    })
    simple_storage = request.storage.model_copy(update={
        "l2_available_gb": 0, "l3_available_gb": 0,
    })
    resolved = resolve_config(request.model_copy(update={
        "model": model, "storage": simple_storage,
    }))
    assert resolved.capacity.kv_bytes_per_token == (4 + 2) * 2 * 2 + 132
    assert resolved.capacity.estimated_total_parameters > 0
    workload = StepWorkload("DECODE", (), (DecodeWork(1, 100),), 0, 1)
    cost = step_cost(workload, model, request.hardware)
    assert cost.flops > 0 and cost.duration_ms > 0
    gqa = AttentionGroup(kind="gqa", layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=8)
    grouped = request.model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": [gqa],
    })
    assert (
        resolve_config(request.model_copy(update={"model": grouped})).capacity
        == resolve_config(request).capacity
    )
    assert (
        step_cost(workload, grouped, request.hardware)
        == step_cost(workload, request.model, request.hardware)
    )


def test_dsa_recompute_layer_count_affects_selection_but_not_mla_attention() -> None:
    request = example_request()
    group = AttentionGroup(
        kind="mla_dsa", layers=2, num_attention_heads=2,
        q_rank=4, kv_rank=4, rope_dim=2, nope_dim=6, value_dim=8,
        topk=4, index_heads=2, index_dim=128, full_indexer_layers=1,
    )
    two_full = group.model_copy(update={"full_indexer_layers": 2})
    one = request.model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": [group],
    })
    two = request.model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": [two_full],
    })
    short = StepWorkload("DECODE", (), (DecodeWork(1, 10),), 0, 1)
    long = StepWorkload("DECODE", (), (DecodeWork(1, 100),), 0, 1)
    assert (
        step_cost(long, two, request.hardware).flops
        > step_cost(long, one, request.hardware).flops
    )
    storage = request.storage.model_copy(update={"l2_available_gb": 0, "l3_available_gb": 0})
    assert resolve_config(request.model_copy(update={
        "model": two, "storage": storage,
    })).capacity.kv_bytes_per_token == (
        resolve_config(request.model_copy(update={
            "model": one, "storage": storage,
        })).capacity.kv_bytes_per_token + 132
    )
    # 超过 top-k 后，随上下文增长的 FLOPs 来自 full 索引器，非共享层主注意力。
    delta = (
        step_cost(long, one, request.hardware).flops
        - step_cost(short, one, request.hardware).flops
    )
    assert delta == pytest.approx(2 * 2 * 128 * 90)


def test_mixed_attention_groups_tp2_count_replication_and_tiered_index_cache() -> None:
    request = example_request()
    groups = [
        AttentionGroup(kind="gqa", layers=1, num_attention_heads=2, num_kv_heads=1, head_dim=8),
        AttentionGroup(kind="mla_dsa", layers=1, num_attention_heads=2,
                       q_rank=4, kv_rank=4, rope_dim=2, nope_dim=6, value_dim=8,
                       topk=4, index_heads=2, index_dim=128, full_indexer_layers=1),
    ]
    model = request.model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": groups,
    })
    hardware = request.hardware.model_copy(update={
        "gpu_count": 2, "effective_interconnect_gb_s": 100,
        "collective_latency_ms": 0.01,
    })
    storage = request.storage.model_copy(update={
        "l2_available_gb": 0.001, "l3_available_gb": 0,
    })
    resolved = resolve_config(request.model_copy(update={
        "model": model, "hardware": hardware, "storage": storage, "tp_size": 2,
    }))
    # GQA 单 KV 头逐 rank 复制 32B；MLA 的 6 维每层复制 12B；索引 Key 132B。
    assert resolved.capacity.rank_reports[0].kv_bytes_per_token == 32 + 12 + 132
    assert resolved.capacity.kv_bytes_per_token == 2 * (32 + 12 + 132)
    assert resolved.config.perf_model.kv_bytes_per_token == resolved.capacity.kv_bytes_per_token
    assert resolved.capacity.rank_reports[0].sharded_weights_bytes > 0
    workload = StepWorkload("DECODE", (), (DecodeWork(1, 100),), 0, 1)
    assert step_cost(workload, model, hardware, 2).collectives == 4


def hy4_request() -> ResolvedRequest:
    request = example_request()
    model = ModelSpec(
        name="tencent/Hy4-preview", revision="705d81ee51566a186d645b74c974d642ef2828fe",
        num_layers=78, hidden_size=6144,
        num_attention_heads=64, num_kv_heads=8, head_dim=256,
        ffn_intermediate_size=18432, vocab_size=120832,
        moe_layers=77, num_experts=256, experts_per_token=8,
        expert_intermediate_size=2048, attention_layout="hy4_preview",
        num_shared_experts=1, shared_expert_intermediate_size=2048,
        attention_groups=[AttentionGroup(
            kind="mla_dsa", layers=78, num_attention_heads=64,
            q_rank=2048, kv_rank=512, rope_dim=64, nope_dim=192, value_dim=256,
            index_heads=32, index_dim=128, topk=2048, full_indexer_layers=21,
            output_gate=True,
        )],
    )
    hardware = request.hardware.model_copy(update={
        "gpu_count": 8, "memory_bytes_per_gpu": 288_000_000_000,
        "effective_interconnect_gb_s": 600, "collective_latency_ms": 0.01,
        "ep_communication_latency_ms": 0.02,
    })
    storage = request.storage.model_copy(update={
        "l1_available_gb": 260, "l2_available_gb": 0, "l3_available_gb": 0,
    })
    return request.model_copy(update={
        "model": model, "hardware": hardware, "storage": storage,
        "tp_size": 8, "ep_size": 8,
    })


def test_hy4_bf16_tp8_ep8_main_cache_and_weight_placement() -> None:
    resolved = resolve_config(hy4_request())
    capacity = resolved.capacity
    assert resolved.runnable
    assert capacity.kv_bytes_per_token == 8 * (78 * 576 * 2 + 21 * 132)
    assert capacity.estimated_total_parameters == 769_907_402_653  # 仅主干，不含 MTP
    assert capacity.estimated_active_parameters == 49_057_539_997
    assert len(capacity.rank_reports) == 8
    rank = capacity.rank_reports[0]
    assert rank.kv_bytes_per_token == 92_628
    assert rank.sharded_weights_bytes == 5_322_574_272
    assert rank.replicated_weights_bytes == 9_331_865_716
    assert rank.expert_weights_bytes == 186_025_771_008
    assert capacity.weights_resident_bytes == 8 * (
        rank.sharded_weights_bytes + rank.replicated_weights_bytes + rank.expert_weights_bytes
    )
    assert resolved.config.engine.max_total_tokens == rank.l1_capacity_pages * 4
    request = hy4_request()
    storage = request.storage.model_copy(update={"l2_available_gb": 10})
    tiered = resolve_config(request.model_copy(update={"storage": storage}))
    assert tiered.capacity.rank_reports[0].kv_bytes_per_token == 78 * (512 + 64) * 2 + 78 * 132
    assert tiered.config.perf_model.kv_bytes_per_token == tiered.capacity.kv_bytes_per_token


def test_hy4_dsa_index_reuse_and_sparse_attention_workload() -> None:
    request = hy4_request()
    workload = StepWorkload("DECODE", (), (DecodeWork(1, 100_000),), 0, 1)
    short = StepWorkload("DECODE", (), (DecodeWork(1, 1_000),), 0, 1)
    long_cost = step_cost(workload, request.model, request.hardware, 8, 8)
    short_cost = step_cost(short, request.model, request.hardware, 8, 8)
    near = StepWorkload("DECODE", (), (DecodeWork(1, 90_000),), 0, 1)
    near_cost = step_cost(near, request.model, request.hardware, 8, 8)
    # 主注意力已经达到 2048 上限，继续增长的只有 21 个 full 索引器的扫描。
    assert long_cost.flops - near_cost.flops == pytest.approx(21 * 2 * 32 * 128 * 10_000)
    assert long_cost.hbm_bytes - near_cost.hbm_bytes == pytest.approx(21 * 132 * 10_000)
    assert long_cost.flops > short_cost.flops
    assert long_cost.communication_ms == short_cost.communication_ms
    assert long_cost.ep_exchanges == 2 * 77
    assert long_cost.collectives == 78 + 1


@pytest.mark.parametrize(
    (
        "phase",
        "reused",
        "new",
        "context",
        "flops",
        "hbm",
        "duration",
        "communication_bytes",
        "tp_bytes",
        "ep_bytes",
        "communication_ms",
    ),
    [
        (
            "PREFILL",
            2046,
            5,
            None,
            106949074944,
            200870670696,
            200889.1739939991,
            13978799.462155823,
            8494080,
            2742359.731077911,
            3.893297999103593,
        ),
        (
            "PREFILL",
            0,
            4,
            None,
            78923415552,
            199942575312,
            199961.0739503993,
            11183039.569724657,
            6795264,
            2193887.7848623283,
            3.8886383992828746,
        ),
        (
            "DECODE",
            0,
            0,
            2047,
            21388959744,
            200131632840,
            200150.1184328367,
            3355702,
            1698816,
            828443,
            3.8755928366666668,
        ),
        (
            "DECODE",
            0,
            0,
            2048,
            21389770752,
            200131725780,
            200150.2113728367,
            3355702,
            1698816,
            828443,
            3.8755928366666668,
        ),
        (
            "DECODE",
            0,
            0,
            2049,
            21389942784,
            200131728552,
            200150.2141448367,
            3355702,
            1698816,
            828443,
            3.8755928366666668,
        ),
        (
            "DECODE",
            0,
            0,
            100000,
            38240649216,
            200403248724,
            200421.7343168367,
            3355702,
            1698816,
            828443,
            3.8755928366666668,
        ),
        (
            "MIXED",
            2046,
            5,
            100000,
            145189724160,
            201332627388,
            201351.13534559894,
            16774559.354586985,
            10192896,
            3290831.677293493,
            3.897957598924312,
        ),
    ],
)
def test_hy4_generic_cost_preserves_original_step_cost(
    phase,
    reused,
    new,
    context,
    flops,
    hbm,
    duration,
    communication_bytes,
    tp_bytes,
    ep_bytes,
    communication_ms,
) -> None:
    request = hy4_request()
    hardware = request.hardware.model_copy(update={"kernel_overhead_ms": 0.03})
    prefill = (PrefillWork(1, reused, new, reused + new),) if new else ()
    decode = (DecodeWork(2, context),) if context is not None else ()
    workload = StepWorkload(phase, prefill, decode, 0, len(prefill) + len(decode))
    cost = step_cost(workload, request.model, hardware, 8, 8)
    assert asdict(cost) == pytest.approx(
        {
            "duration_ms": duration,
            "flops": flops,
            "hbm_bytes": hbm,
            "compute_ms": flops / 1e9,
            "memory_ms": hbm / 1e6,
            "communication_ms": communication_ms,
            "communication_bytes": communication_bytes,
            "collectives": 79,
            "tp_communication_ms": 0.79 + tp_bytes / 600e6,
            "tp_communication_bytes": tp_bytes,
            "ep_dispatch_ms": 1.54 + ep_bytes / 600e6,
            "ep_combine_ms": 1.54 + ep_bytes / 600e6,
            "ep_dispatch_bytes": ep_bytes,
            "ep_combine_bytes": ep_bytes,
            "ep_exchanges": 154,
        }
    )
    custom = ModelSpec.model_validate(
        request.model.model_dump()
        | {
            "name": "custom",
            "revision": None,
            "attention_layout": "grouped",
        }
    )
    assert step_cost(workload, custom, hardware, 8, 8) == cost


def test_hy4_requires_complete_explicit_components_and_empty_steps_are_free() -> None:
    request = hy4_request()
    payload = request.model.model_dump()
    for update in (
        {"attention_groups": None},
        {"attention_groups": [payload["attention_groups"][0] | {"output_gate": False}]},
        {"num_shared_experts": 0, "shared_expert_intermediate_size": None},
    ):
        with pytest.raises(ValidationError, match="HY4-Preview"):
            ModelSpec.model_validate(payload | update)
    restore = StepWorkload("RESTORE", (), (), 0, 1, (("a", "L2"),), 4, (0,))
    cost = step_cost(restore, request.model, request.hardware, 8, 8)
    assert all(value == 0 for value in asdict(cost).values())


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("kind", ["mla", "mla_dsa"])
def test_output_gate_uses_group_dimensions_and_sharded_capacity(kind, tp_size) -> None:
    request = example_request()
    group = AttentionGroup(
        kind=kind,
        layers=2,
        num_attention_heads=4,
        q_rank=4,
        kv_rank=4,
        rope_dim=2,
        nope_dim=6,
        value_dim=6,
        **(
            {"index_heads": 2, "index_dim": 128, "topk": 4, "full_indexer_layers": 1}
            if kind == "mla_dsa"
            else {}
        ),
    )
    model = request.model.model_copy(
        update={
            "attention_layout": "grouped",
            "attention_groups": [group],
        }
    )
    gated = model.model_copy(
        update={"attention_groups": [group.model_copy(update={"output_gate": True})]}
    )
    hardware = request.hardware.model_copy(
        update={
            "gpu_count": tp_size,
            "effective_interconnect_gb_s": 100,
            "collective_latency_ms": 0.01,
            "kernel_overhead_ms": 0.03,
        }
    )
    workload = StepWorkload("DECODE", (), (DecodeWork(1, 100),), 0, 1)
    base = step_cost(workload, model, hardware, tp_size)
    cost = step_cost(workload, gated, hardware, tp_size)
    local_gate = 2 * 16 * (4 // tp_size) * 6
    assert cost.flops - base.flops == pytest.approx(2 * local_gate)
    assert cost.hbm_bytes - base.hbm_bytes == pytest.approx(2 * local_gate)
    assert cost.duration_ms - base.duration_ms == pytest.approx(2 * local_gate / 1e6)
    assert cost.communication_bytes == base.communication_bytes
    storage = request.storage.model_copy(update={"l2_available_gb": 0, "l3_available_gb": 0})
    base_capacity = resolve_config(
        request.model_copy(
            update={
                "model": model,
                "hardware": hardware,
                "storage": storage,
                "tp_size": tp_size,
            }
        )
    ).capacity
    capacity = resolve_config(
        request.model_copy(
            update={
                "model": gated,
                "hardware": hardware,
                "storage": storage,
                "tp_size": tp_size,
            }
        )
    ).capacity
    logical_gate = local_gate * tp_size
    assert (
        capacity.weights_resident_bytes - base_capacity.weights_resident_bytes == 2 * logical_gate
    )
    assert (
        capacity.estimated_total_parameters - base_capacity.estimated_total_parameters
        == logical_gate
    )
    assert (
        capacity.estimated_active_parameters - base_capacity.estimated_active_parameters
        == logical_gate
    )
    assert capacity.kv_bytes_per_token == base_capacity.kv_bytes_per_token
    if tp_size > 1:
        assert (
            capacity.rank_reports[0].sharded_weights_bytes
            - base_capacity.rank_reports[0].sharded_weights_bytes
        ) == 2 * local_gate
        assert (
            capacity.rank_reports[0].replicated_weights_bytes
            == base_capacity.rank_reports[0].replicated_weights_bytes
        )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("grouped", [False, True])
def test_shared_experts_are_replicated_and_do_not_change_routing(tp_size, grouped) -> None:
    request = example_request()
    model = request.model.model_copy(
        update={
            "moe_layers": 1,
            "num_experts": 4,
            "experts_per_token": 2,
            "expert_intermediate_size": 16,
        }
    )
    if grouped:
        model = model.model_copy(
            update={
                "attention_layout": "grouped",
                "attention_groups": [
                    AttentionGroup(
                        kind="gqa", layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=8
                    ),
                ],
            }
        )
    shared = model.model_copy(
        update={"num_shared_experts": 2, "shared_expert_intermediate_size": 8}
    )
    hardware = request.hardware.model_copy(
        update={
            "gpu_count": tp_size,
            "effective_interconnect_gb_s": 100,
            "collective_latency_ms": 0.01,
            "ep_communication_latency_ms": 0.02,
            "kernel_overhead_ms": 0.03,
        }
    )
    workload = StepWorkload("MIXED", (PrefillWork(1, 3, 2, 5),), (DecodeWork(2, 10),), 0, 2)
    base = step_cost(workload, model, hardware, tp_size, tp_size)
    cost = step_cost(workload, shared, hardware, tp_size, tp_size)
    shared_params = 1 * 2 * 3 * 16 * 8
    assert cost.flops - base.flops == pytest.approx(2 * 3 * shared_params)
    assert cost.hbm_bytes - base.hbm_bytes == pytest.approx(2 * shared_params)
    assert cost.duration_ms - base.duration_ms == pytest.approx(0.03 + 2 * shared_params / 1e6)
    for name, value in asdict(base).items():
        if name not in {"duration_ms", "flops", "hbm_bytes", "compute_ms", "memory_ms"}:
            assert getattr(cost, name) == value
    storage = request.storage.model_copy(update={"l2_available_gb": 0, "l3_available_gb": 0})
    base_capacity = resolve_config(
        request.model_copy(
            update={
                "model": model,
                "hardware": hardware,
                "storage": storage,
                "tp_size": tp_size,
                "ep_size": tp_size,
            }
        )
    ).capacity
    capacity = resolve_config(
        request.model_copy(
            update={
                "model": shared,
                "hardware": hardware,
                "storage": storage,
                "tp_size": tp_size,
                "ep_size": tp_size,
            }
        )
    ).capacity
    assert (
        capacity.weights_resident_bytes - base_capacity.weights_resident_bytes
        == 2 * shared_params * tp_size
    )
    assert (
        capacity.estimated_total_parameters - base_capacity.estimated_total_parameters
        == shared_params
    )
    assert (
        capacity.estimated_active_parameters - base_capacity.estimated_active_parameters
        == shared_params
    )
    assert capacity.estimated_expert_parameters == base_capacity.estimated_expert_parameters
    if tp_size > 1:
        assert (
            capacity.rank_reports[0].replicated_weights_bytes
            - base_capacity.rank_reports[0].replicated_weights_bytes
        ) == 2 * shared_params
        assert (
            capacity.rank_reports[0].expert_weights_bytes
            == base_capacity.rank_reports[0].expert_weights_bytes
        )


def test_dsa_charges_int32_candidate_reads_in_each_attention_layer() -> None:
    request = example_request()
    mla = AttentionGroup(
        kind="mla",
        layers=2,
        num_attention_heads=2,
        q_rank=4,
        kv_rank=4,
        rope_dim=2,
        nope_dim=6,
        value_dim=8,
    )
    dsa = AttentionGroup(
        kind="mla_dsa",
        layers=2,
        num_attention_heads=2,
        q_rank=4,
        kv_rank=4,
        rope_dim=2,
        nope_dim=6,
        value_dim=8,
        index_heads=2,
        index_dim=128,
        topk=10,
        full_indexer_layers=1,
    )
    model = request.model.model_copy(
        update={"attention_layout": "grouped", "attention_groups": [mla]}
    )
    sparse = model.model_copy(update={"attention_groups": [dsa]})
    workload = StepWorkload("MIXED", (PrefillWork(1, 1, 2, 3),), (DecodeWork(2, 4),), 0, 2)
    base = step_cost(workload, model, request.hardware)
    cost = step_cost(workload, sparse, request.hardware)
    pairs = 2 + 3 + 4
    index_weights = 2 * 128 * 4 + 16 * (128 + 2)
    index_hbm = 2 * index_weights + 132 * (3 + 4 + 3)
    assert cost.hbm_bytes - base.hbm_bytes == pytest.approx(index_hbm + 2 * 4 * pairs)
    assert cost.flops - base.flops == pytest.approx(2 * 3 * index_weights + 2 * 2 * 128 * pairs)


def test_resolve_three_tiers_and_bandwidth_rounds_only_whole_pages() -> None:
    resolved = resolve_config(example_request())
    report = resolved.capacity
    assert report.kv_bytes_per_token == 64
    assert report.kv_bytes_per_page == 256
    assert report.estimated_total_parameters == report.estimated_active_parameters == 5232
    assert report.estimated_expert_parameters == 0
    assert report.weights_resident_bytes == 10464
    assert report.l1_kv_budget_bytes == 988_536
    assert (report.l1_capacity_pages, report.l2_capacity_pages, report.l3_capacity_pages) == (
        3861,
        2,
        1,
    )
    assert (
        resolved.config.engine.max_total_tokens,
        resolved.config.engine.l2_capacity_tokens,
        resolved.config.engine.l3_capacity_tokens,
    ) == (3861 * 4, 8, 4)
    assert report.l2_to_l1_tokens_per_ms == pytest.approx(1000 / 64)
    assert report.l3_to_l2_tokens_per_ms == pytest.approx(2000 / 64)
    assert resolved.config.perf_model.kv_bytes_per_token == 64
    assert resolved.config.perf_model.l3_lookup_latency_ms == 1.25
    assert resolved.config.engine.l3_prefetch_min_tokens == 256


def test_qwen3_8b_single_b300_capacity_example() -> None:
    request = example_request()
    qwen = ModelSpec(
        name="Qwen/Qwen3-8B",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        ffn_intermediate_size=12288,
        vocab_size=151936,
    )
    resolved = resolve_config(
        request.model_copy(
            update={
                "model": qwen,
                "hardware": request.hardware.model_copy(
                    update={
                        "name": "NVIDIA B300",
                        "memory_bytes_per_gpu": 288_000_000_000,
                    }
                ),
                "storage": request.storage.model_copy(
                    update={
                        "l1_available_gb": 200,
                        "l2_available_gb": 0,
                        "l3_available_gb": 0,
                        "l2_to_l1_gb_s": None,
                        "l3_to_l2_gb_s": None,
                    }
                ),
            }
        )
    )
    assert resolved.capacity.kv_bytes_per_token == 144 * 1024
    assert resolved.capacity.estimated_total_parameters == 8_190_735_360
    assert resolved.capacity.estimated_active_parameters == 8_190_735_360
    assert resolved.capacity.weights_resident_bytes == 2 * 8_190_735_360
    assert resolved.capacity.l1_capacity_pages == (200_000_000_000 - 2 * 8_190_735_360 - 1000) // (
        4 * 144 * 1024
    )


def test_reject_unsupported_topologies_and_missing_bandwidth() -> None:
    original = example_request()
    with pytest.raises(ValueError, match="TP 数"):
        resolve_config(
            original.model_copy(
                update={"hardware": original.hardware.model_copy(update={"gpu_count": 8})}
            )
        )
    with pytest.raises(ValueError, match="MoE"):
        resolve_config(
            original.model_copy(
                update={"model": original.model.model_copy(update={"moe_layers": 1})}
            )
        )
    with pytest.raises(ValueError, match="L2"):
        resolve_config(
            original.model_copy(
                update={"storage": original.storage.model_copy(update={"l2_to_l1_gb_s": None})}
            )
        )
    with pytest.raises(ValueError, match="不足"):
        resolve_config(
            original.model_copy(
                update={
                    "storage": original.storage.model_copy(update={"l1_available_gb": 0.000001})
                }
            )
        )


def test_tp4_preview_uses_per_rank_l1_and_group_l2_l3_budgets() -> None:
    request = example_request()
    model = ModelSpec(
        name="Qwen3-8B",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        ffn_intermediate_size=12288,
        vocab_size=151936,
    )
    hardware = request.hardware.model_copy(
        update={
            "gpu_count": 4,
            "memory_bytes_per_gpu": 288_000_000_000,
            "effective_interconnect_gb_s": 100,
            "collective_latency_ms": 0.1,
        }
    )
    storage = request.storage.model_copy(
        update={
            "l1_available_gb": 20,
            "l2_available_gb": 1,
            "l3_available_gb": 0.5,
        }
    )
    result = resolve_config(
        request.model_copy(
            update={
                "model": model,
                "hardware": hardware,
                "storage": storage,
                "tp_size": 4,
            }
        )
    )
    report = result.capacity
    assert result.runnable and result.tp_size == 4
    assert not resolve_config(
        request.model_copy(
            update={
                "model": model,
                "hardware": hardware,
                "storage": storage,
                "tp_size": 4,
                "timing": "empirical",
            }
        )
    ).runnable
    assert len(report.rank_reports) == 4
    assert {rank.kv_bytes_per_token for rank in report.rank_reports} == {36 * 1024}
    assert report.kv_bytes_per_token == 144 * 1024
    assert report.estimated_total_parameters == 8_190_735_360
    for rank in report.rank_reports:
        assert (
            rank.sharded_weights_bytes
            + rank.replicated_weights_bytes
            + rank.l1_kv_budget_bytes
            + storage.runtime_reserve_bytes
        ) == 20_000_000_000
        assert rank.l1_capacity_pages == rank.l1_kv_budget_bytes // (rank.kv_bytes_per_token * 4)
    assert (
        result.config.engine.max_total_tokens
        == min(rank.l1_capacity_pages for rank in report.rank_reports) * 4
    )
    assert report.l2_capacity_pages == 1_000_000_000 // (4 * report.kv_bytes_per_token)
    assert report.l3_capacity_pages == 500_000_000 // (4 * report.kv_bytes_per_token)


def test_tp8_replicates_four_kv_heads_but_rejects_moe_for_now() -> None:
    request = example_request()
    model = ModelSpec(
        name="dense-kv4",
        num_layers=94,
        hidden_size=4096,
        num_attention_heads=64,
        num_kv_heads=4,
        head_dim=128,
        ffn_intermediate_size=12288,
        vocab_size=151936,
    )
    hardware = request.hardware.model_copy(
        update={
            "gpu_count": 8,
            "memory_bytes_per_gpu": 288_000_000_000,
            "effective_interconnect_gb_s": 100,
            "collective_latency_ms": 0.1,
        }
    )
    storage = request.storage.model_copy(
        update={
            "l1_available_gb": 50,
            "l2_available_gb": 1,
            "l3_available_gb": 0,
        }
    )
    tp = request.model_copy(
        update={
            "tp_size": 8,
            "model": model,
            "hardware": hardware,
            "storage": storage,
        }
    )
    report = resolve_config(tp).capacity
    assert {rank.kv_bytes_per_token for rank in report.rank_reports} == {48_128}
    assert report.kv_bytes_per_token == 8 * 48_128
    result, analysis = simulate_resolved(resolve_config(tp))
    assert result.meta.num_completed == 1 and analysis is not None
    assert analysis.collectives == 2 * 94 * analysis.steps
    assert analysis.communication_ms > 0
    with pytest.raises(ValueError, match="MoE"):
        resolve_config(tp.model_copy(update={"model": model.model_copy(update={"moe_layers": 94})}))
    with pytest.raises(ValueError, match="MoE"):
        resolve_config(
            tp.model_copy(
                update={
                    "model": model.model_copy(
                        update={
                            "num_experts": 128,
                        }
                    )
                }
            )
        )


def test_tp_preview_rejects_incomplete_placement_and_gpu_mismatch() -> None:
    request = example_request()
    with pytest.raises(ValueError, match="TP 数"):
        resolve_config(request.model_copy(update={"tp_size": 4}))
    hardware = request.hardware.model_copy(update={"gpu_count": 4})
    valid_model = request.model.model_copy(
        update={
            "vocab_size": 16,
            "num_attention_heads": 4,
        }
    )
    storage = request.storage.model_copy(update={"l2_available_gb": 0, "l3_available_gb": 0})
    preview = resolve_config(
        request.model_copy(
            update={
                "tp_size": 4,
                "hardware": hardware,
                "model": valid_model,
                "storage": storage,
            }
        )
    )
    assert not preview.runnable and len(preview.capacity.rank_reports) == 4
    with pytest.raises(ValueError, match="集合通信延迟"):
        simulate_resolved(preview)
    hardware = hardware.model_copy(
        update={"effective_interconnect_gb_s": 100, "collective_latency_ms": 0.1}
    )
    assert resolve_config(
        request.model_copy(
            update={
                "tp_size": 4,
                "hardware": hardware,
                "model": valid_model,
                "storage": storage,
            }
        )
    ).runnable
    with pytest.raises(ValueError, match="vocab_size"):
        resolve_config(
            request.model_copy(
                update={
                    "tp_size": 4,
                    "hardware": hardware,
                    "model": valid_model.model_copy(update={"vocab_size": None}),
                }
            )
        )


def test_moe_capacity_distributes_all_experts_and_replaces_dense_ffn() -> None:
    base = example_request()
    model = ModelSpec(
        name="toy-moe",
        num_layers=2,
        moe_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        ffn_intermediate_size=32,
        expert_intermediate_size=8,
        num_experts=4,
        experts_per_token=2,
        vocab_size=16,
    )
    hardware = base.hardware.model_copy(update={"gpu_count": 2})
    storage = base.storage.model_copy(
        update={
            "l1_available_gb": 0.00001,
            "l2_available_gb": 0,
            "l3_available_gb": 0,
            "runtime_reserve_bytes": 0,
        }
    )
    request = base.model_copy(
        update={
            "tp_size": 2,
            "ep_size": 2,
            "model": model,
            "hardware": hardware,
            "storage": storage,
        }
    )
    preview = resolve_config(request)
    assert preview.ep_size == preview.tp_size == 2
    assert not preview.runnable
    assert preview.capacity.kv_bytes_per_token == 128
    assert (
        preview.capacity.estimated_total_parameters,
        preview.capacity.estimated_active_parameters,
        preview.capacity.estimated_expert_parameters,
    ) == (5296, 4528, 1536)
    assert len(preview.capacity.rank_reports) == 2
    for rank in preview.capacity.rank_reports:
        assert rank.sharded_weights_bytes == 4096
        assert rank.replicated_weights_bytes == 352
        assert rank.expert_weights_bytes == 1536
        assert rank.l1_kv_budget_bytes == 10000 - 4096 - 352 - 1536
        assert rank.l1_capacity_pages == rank.l1_kv_budget_bytes // (64 * 4)
    assert preview.capacity.weights_resident_bytes == 2 * (4096 + 352 + 1536)
    assert preview.config.engine.max_total_tokens == 4 * preview.capacity.l1_capacity_pages
    with pytest.raises(ValueError, match="EP 单次通信开销"):
        simulate_resolved(preview)


@pytest.mark.parametrize(
    "updates, reason",
    [
        ({"ep_size": 1}, "EP 数"),
        ({"model": {"experts_per_token": 5}}, "激活专家数"),
        ({"model": {"num_experts": 3}}, "专家数"),
        ({"model": {"expert_intermediate_size": None}}, "专家中间维度"),
    ],
)
def test_moe_preview_rejects_inconsistent_layout(updates: dict, reason: str) -> None:
    base = example_request()
    model = base.model.model_copy(
        update={
            "vocab_size": 16,
            "moe_layers": 1,
            "num_experts": 4,
            "experts_per_token": 2,
            "expert_intermediate_size": 8,
            **updates.get("model", {}),
        }
    )
    request = base.model_copy(
        update={
            "model": model,
            "ep_size": updates.get("ep_size", 2),
            "tp_size": 2,
            "hardware": base.hardware.model_copy(update={"gpu_count": 2}),
        }
    )
    with pytest.raises(ValueError, match=reason):
        resolve_config(request)


def test_moe_step_replaces_ffn_and_separates_tp_ep_modes() -> None:
    model = ModelSpec(
        name="toy-moe",
        num_layers=2,
        moe_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        ffn_intermediate_size=32,
        expert_intermediate_size=8,
        num_experts=4,
        experts_per_token=2,
        vocab_size=16,
    )
    hardware = HardwareSpec(
        name="two-gpu",
        gpu_count=2,
        memory_bytes_per_gpu=1_000_000,
        effective_fp16_tflops=1,
        effective_hbm_gb_s=1,
        effective_interconnect_gb_s=1,
        collective_latency_ms=0.3,
        ep_communication_latency_ms=0.7,
    )
    prefill = StepWorkload("PREFILL", (PrefillWork(0, 0, 2, 2),), (), 0, 1)
    decode = StepWorkload("DECODE", (), (DecodeWork(0, 10), DecodeWork(1, 11)), 0, 2)
    mixed = StepWorkload(
        "MIXED",
        (PrefillWork(2, 4, 1, 5),),
        (DecodeWork(0, 10), DecodeWork(1, 11)),
        0,
        3,
    )
    normal = step_cost(prefill, model, hardware, tp_size=2, ep_size=2)
    low_latency = step_cost(decode, model, hardware, tp_size=2, ep_size=2)
    combined = step_cost(mixed, model, hardware, tp_size=2, ep_size=2)
    assert normal.collectives == low_latency.collectives == combined.collectives == 3
    assert normal.ep_exchanges == low_latency.ep_exchanges == combined.ep_exchanges == 2
    assert normal.ep_dispatch_bytes == pytest.approx(2 * 0.5 * (5 / 6) * 32 + 2 * 0.5 * 8)
    assert low_latency.ep_dispatch_bytes == pytest.approx(2 * 0.5 * (32 + 8))
    assert combined.ep_dispatch_bytes == pytest.approx(3 * 0.5 * (5 / 6) * 32 + 3 * 0.5 * 8)
    for cost in (normal, low_latency, combined):
        assert cost.ep_combine_bytes == cost.ep_dispatch_bytes
        assert cost.ep_dispatch_ms == pytest.approx(0.7 + cost.ep_dispatch_bytes / 1e6)
        assert cost.ep_combine_ms == cost.ep_dispatch_ms
        assert cost.communication_ms == pytest.approx(
            cost.tp_communication_ms + cost.ep_dispatch_ms + cost.ep_combine_ms
        )
        assert cost.duration_ms >= cost.communication_ms
    dense = model.model_copy(update={"moe_layers": 0})
    assert step_cost(prefill, dense, hardware, tp_size=2).collectives == 4
    assert normal.flops != step_cost(prefill, dense, hardware, tp_size=2).flops
    slower_ep = step_cost(
        prefill,
        model,
        hardware.model_copy(update={"ep_communication_latency_ms": 1.7}),
        tp_size=2,
        ep_size=2,
    )
    assert slower_ep.tp_communication_ms == normal.tp_communication_ms
    assert slower_ep.duration_ms - normal.duration_ms == pytest.approx(2 * model.moe_layers)
    restore = StepWorkload("RESTORE", (), (), 0, 1, (("a", "L2"),), 4, (0,))
    assert step_cost(restore, model, hardware, tp_size=2, ep_size=2).duration_ms == 0
    assert step_cost(restore, model, hardware, tp_size=2, ep_size=2).ep_exchanges == 0


def test_moe_analytical_run_uses_ep_cost_and_keeps_empirical_unavailable() -> None:
    base = example_request()
    model = base.model.model_copy(
        update={
            "vocab_size": 16,
            "moe_layers": 1,
            "num_experts": 4,
            "experts_per_token": 2,
            "expert_intermediate_size": 8,
        }
    )
    hardware = base.hardware.model_copy(
        update={
            "gpu_count": 2,
            "effective_interconnect_gb_s": 1,
            "collective_latency_ms": 0.3,
            "ep_communication_latency_ms": 0.7,
        }
    )
    storage = base.storage.model_copy(
        update={
            "l2_available_gb": 0,
            "l3_available_gb": 0,
        }
    )
    request = base.model_copy(
        update={
            "model": model,
            "hardware": hardware,
            "storage": storage,
            "tp_size": 2,
            "ep_size": 2,
        }
    )
    resolution = resolve_config(request)
    assert resolution.runnable
    result, summary = simulate_resolved(resolution)
    assert summary is not None and result.meta.num_completed == 1
    assert summary.steps == result.meta.num_engine_steps
    assert summary.collectives == 3 * summary.steps
    assert summary.ep_exchanges == 2 * summary.steps
    assert summary.communication_ms == pytest.approx(
        summary.tp_communication_ms + summary.ep_dispatch_ms + summary.ep_combine_ms
    )
    assert result.requests[0].finish - result.requests[0].prefill_start == pytest.approx(
        summary.device_ms + summary.restore_ms
    )
    with pytest.raises(NotImplementedError, match="四系数"):
        simulate_resolved(resolve_config(request.model_copy(update={"timing": "empirical"})))


def test_moe_mixed_engine_step_charges_ep_once_and_advances_clock() -> None:
    model = ModelSpec(
        name="mixed-moe",
        num_layers=2,
        moe_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        ffn_intermediate_size=32,
        num_experts=4,
        experts_per_token=2,
        expert_intermediate_size=8,
        vocab_size=16,
    )
    hardware = HardwareSpec(
        name="two-gpu",
        gpu_count=2,
        memory_bytes_per_gpu=1_000_000,
        effective_fp16_tflops=1,
        effective_hbm_gb_s=1,
        effective_interconnect_gb_s=1,
        collective_latency_ms=0.3,
        ep_communication_latency_ms=0.7,
    )
    engine = Engine(
        EngineConfig(
            max_running_requests=2, max_total_tokens=128, page_size=4, pd_mutually_exclusive=False
        ),
        PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=1),
        output_targets={0: 2, 1: 2},
        analytical=(model, hardware, 2, 2),
    )
    request0 = InferenceRequest(0, 0, None, 4, 2, (), 0)
    request1 = InferenceRequest(1, 1, None, 4, 2, (), 0)
    engine.submit(request0, 0)
    first_duration, first_finished = engine.step(0, 0)
    assert first_finished == ()
    engine.submit(request1, first_duration)
    mixed_duration, mixed_finished = engine.step(first_duration, 1)
    assert list(engine._log.steps())[-1].phase == "MIXED"
    assert mixed_finished == (0,)
    expected = step_cost(
        StepWorkload("MIXED", (PrefillWork(1, 0, 4, 4),), (DecodeWork(0, 5),), 2, 2),
        model,
        hardware,
        tp_size=2,
        ep_size=2,
    )
    assert mixed_duration == pytest.approx(expected.duration_ms)
    assert expected.ep_exchanges == 2 * model.moe_layers
    engine.complete(first_duration + mixed_duration, 0)
    last_duration, last_finished = engine.step(first_duration + mixed_duration, 2)
    assert last_finished == (1,)
    engine.complete(first_duration + mixed_duration + last_duration, 1)
    run = engine.finalize()
    assert [item.phase for item in run.trace.steps()] == ["PREFILL", "MIXED", "DECODE"]
    assert run.analysis is not None
    assert run.analysis.steps == 3
    assert run.analysis.ep_exchanges == 3 * expected.ep_exchanges
    assert run.analysis.device_ms == pytest.approx(first_duration + mixed_duration + last_duration)
    assert run.records[-1].finish == pytest.approx(run.analysis.device_ms)
    assert len(run.records) == 2 and not run.aborted_requests and not run.skipped_requests


def test_token_bandwidth_matches_physical_two_hop_transfer() -> None:
    resolved = resolve_config(example_request())
    workload = StepWorkload(
        "RESTORE",
        (),
        (),
        0,
        1,
        (("a", "L2"), ("b", "L3")),
        4,
        (0,),
    )
    expected = restore_time(workload, resolved.config.perf_model)
    actual = restore_time_tokens(
        workload,
        l2_tokens_per_ms=resolved.capacity.l2_to_l1_tokens_per_ms,
        l3_tokens_per_ms=resolved.capacity.l3_to_l2_tokens_per_ms,
        l2_latency_ms=resolved.config.perf_model.l2_latency_ms,
        l3_latency_ms=resolved.config.perf_model.l3_latency_ms,
    )
    assert actual == pytest.approx(expected)
    assert actual == pytest.approx(3 + 8 * 64 / 1000 + 4 + 4 * 64 / 2000)


def test_single_layer_roofline_takes_stage_max_then_sums_dependencies() -> None:
    model = ModelSpec(
        name="toy",
        num_layers=1,
        hidden_size=2,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=1,
        ffn_intermediate_size=4,
        vocab_size=16,
    )
    hardware = HardwareSpec(
        name="toy",
        memory_bytes_per_gpu=10000,
        effective_fp16_tflops=1e-9,
        effective_hbm_gb_s=1e-6,
        kernel_overhead_ms=0.2,
    )
    prefill = StepWorkload("PREFILL", (PrefillWork(0, 0, 1, 1),), (), 1, 1)
    cost = step_cost(prefill, model, hardware)
    assert cost.flops == 80
    assert cost.hbm_bytes == 80
    assert cost.compute_ms == pytest.approx(80)
    assert cost.memory_ms == pytest.approx(80)
    assert cost.duration_ms == pytest.approx(80.6)
    restore = StepWorkload("RESTORE", (), (), 0, 1, (("a", "L2"),), 4, (0,))
    assert step_cost(restore, model, hardware).duration_ms == 0
    decode = StepWorkload("DECODE", (), (DecodeWork(0, 12),), 0, 1)
    assert step_cost(decode, model, hardware).duration_ms > cost.duration_ms


def test_tp2_roofline_counts_two_collectives_per_dense_layer() -> None:
    model = ModelSpec(
        name="toy",
        num_layers=1,
        hidden_size=2,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=1,
        ffn_intermediate_size=4,
        vocab_size=16,
    )
    hardware = HardwareSpec(
        name="toy",
        gpu_count=2,
        memory_bytes_per_gpu=10000,
        effective_fp16_tflops=1e-9,
        effective_hbm_gb_s=1e-6,
        effective_interconnect_gb_s=1e-6,
        collective_latency_ms=0.3,
        kernel_overhead_ms=0.2,
    )
    prefill = StepWorkload("PREFILL", (PrefillWork(0, 0, 1, 1),), (), 1, 1)
    cost = step_cost(prefill, model, hardware, tp_size=2)
    assert cost.flops == 44
    assert cost.hbm_bytes == 48
    assert cost.compute_ms == pytest.approx(44)
    assert cost.memory_ms == pytest.approx(48)
    assert cost.collectives == 2
    assert cost.communication_bytes == 8
    assert cost.communication_ms == pytest.approx(2 * (0.3 + 4))
    assert cost.duration_ms == pytest.approx(48.8 + cost.communication_ms)
    restore = StepWorkload("RESTORE", (), (), 0, 1, (("a", "L2"),), 4, (0,))
    assert step_cost(restore, model, hardware, tp_size=2).collectives == 0


def test_tp2_runner_uses_communication_cost_and_preserves_request_results() -> None:
    base = example_request()
    model = base.model.model_copy(
        update={
            "vocab_size": 16,
        }
    )
    hardware = base.hardware.model_copy(
        update={
            "gpu_count": 2,
            "effective_interconnect_gb_s": 1e-6,
            "collective_latency_ms": 0.1,
        }
    )
    storage = base.storage.model_copy(
        update={
            "l1_available_gb": 0.0002,
            "l2_available_gb": 0,
            "l3_available_gb": 0,
        }
    )
    request = base.model_copy(
        update={
            "model": model,
            "hardware": hardware,
            "storage": storage,
            "tp_size": 2,
        }
    )
    resolved = resolve_config(request)
    result, analysis = simulate_resolved(resolved)
    assert resolved.runnable and analysis is not None
    assert result.meta.num_completed == 1
    assert result.meta.num_aborted == result.meta.num_skipped == 0
    assert analysis.steps == result.meta.num_engine_steps
    assert analysis.collectives == 2 * model.num_layers * analysis.steps
    assert analysis.communication_ms > 0
    assert result.requests[0].finish - result.requests[0].prefill_start == pytest.approx(
        analysis.device_ms + analysis.restore_ms
    )
    with pytest.raises(NotImplementedError, match="四系数"):
        simulate_resolved(resolve_config(request.model_copy(update={"timing": "empirical"})))


def test_tp_mixed_step_charges_collectives_once_per_layer() -> None:
    model = ModelSpec(
        name="toy",
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        ffn_intermediate_size=32,
        vocab_size=16,
    )
    hardware = HardwareSpec(
        name="toy",
        gpu_count=2,
        memory_bytes_per_gpu=1_000_000,
        effective_fp16_tflops=1,
        effective_hbm_gb_s=1,
        effective_interconnect_gb_s=1,
        collective_latency_ms=0.25,
    )
    workload = StepWorkload(
        "MIXED",
        (PrefillWork(1, 4, 2, 6),),
        (DecodeWork(0, 12),),
        1,
        2,
    )
    cost = step_cost(workload, model, hardware, tp_size=2)
    assert cost.collectives == 4
    assert cost.communication_bytes == 4 * (2 + 1) * 16 * 2
    assert cost.communication_ms == pytest.approx(4 * 0.25 + cost.communication_bytes / 1e6)


def test_prefill_kv_hbm_reads_scale_with_context_not_attention_pairs() -> None:
    model = ModelSpec(
        name="toy",
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        ffn_intermediate_size=32,
        vocab_size=16,
    )
    hardware = HardwareSpec(
        name="toy",
        memory_bytes_per_gpu=1_000_000,
        effective_fp16_tflops=1,
        effective_hbm_gb_s=1,
    )

    def cost(length: int) -> float:
        workload = StepWorkload(
            "PREFILL",
            (PrefillWork(0, 0, length, length),),
            (),
            1,
            1,
        )
        return step_cost(workload, model, hardware).hbm_bytes

    assert cost(1024) - cost(1) == (1024 - 1) * 2 * 8 * 2 * 2


def test_analytical_clock_uses_workload_not_empirical_coefficients() -> None:
    resolution = resolve_config(example_request())
    result, summary = simulate_analytical(resolution.config, resolution.model, resolution.hardware)
    assert result.meta.num_completed == 1
    assert summary.steps == result.meta.num_engine_steps
    assert summary.device_ms > 0
    assert result.requests[0].finish - result.requests[0].prefill_start == pytest.approx(
        summary.device_ms + summary.restore_ms
    )
