"""阶段一：单卡物理容量与逐步 Roofline 计时金标准。"""

from dataclasses import asdict, replace
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from simcore import simulate
from simcore.analytical import step_cost
from simcore.capacity import prepare_runtime, resolve_config
from simcore.engine import Engine
from simcore.events import EventKind
from simcore.perf_model import DecodeWork, PrefillWork, StepWorkload
from simcore.prefix_hash import encode
from simcore.schemas import (
    AnalyticalTimingConfig,
    AttentionGroup,
    AuxWeightDtype,
    EmpiricalTimingConfig,
    FixedLength,
    HardwareSpec,
    IHCSpec,
    InferenceRequest,
    ModelSpec,
    NormSpec,
    PhysicalInferenceConfig,
    RouterSpec,
    SchedulerConfig,
    SimConfig,
    StorageSpec,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)


def physical(config: SimConfig) -> PhysicalInferenceConfig:
    return cast(PhysicalInferenceConfig, config.inference)


def update_inference(config: SimConfig, *, update: dict[str, object]) -> SimConfig:
    return config.model_copy(update={"inference": physical(config).model_copy(update=update)})


def example_request() -> SimConfig:
    return SimConfig(
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
        scheduler=SchedulerConfig(max_running_requests=1),
        inference=PhysicalInferenceConfig(
            page_size=4,
            timing=AnalyticalTimingConfig(),
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
    model = physical(request).model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": [mla, dsa],
    })
    simple_storage = physical(request).storage.model_copy(update={
        "l2_available_gb": 0, "l3_available_gb": 0,
    })
    resolved = resolve_config(update_inference(request, update={
        "model": model, "storage": simple_storage,
    }))
    assert resolved.capacity.kv_bytes_per_token == (4 + 2) * 2 * 2 + 132
    assert resolved.capacity.estimated_total_parameters > 0
    workload = StepWorkload("DECODE", (), (DecodeWork(1, 100),), 0, 1)
    cost = step_cost(workload, model, physical(request).hardware)
    assert cost.flops > 0 and cost.duration_ms > 0
    gqa = AttentionGroup(kind="gqa", layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=8)
    grouped = physical(request).model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": [gqa],
    })
    assert (
        resolve_config(update_inference(request, update={"model": grouped})).capacity
        == resolve_config(request).capacity
    )
    assert (
        step_cost(workload, grouped, physical(request).hardware)
        == step_cost(workload, physical(request).model, physical(request).hardware)
    )


def test_dsa_recompute_layer_count_affects_selection_but_not_mla_attention() -> None:
    request = example_request()
    group = AttentionGroup(
        kind="mla_dsa", layers=2, num_attention_heads=2,
        q_rank=4, kv_rank=4, rope_dim=2, nope_dim=6, value_dim=8,
        topk=4, index_heads=2, index_dim=128, full_indexer_layers=1,
    )
    two_full = group.model_copy(update={"full_indexer_layers": 2})
    one = physical(request).model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": [group],
    })
    two = physical(request).model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": [two_full],
    })
    short = StepWorkload("DECODE", (), (DecodeWork(1, 10),), 0, 1)
    long = StepWorkload("DECODE", (), (DecodeWork(1, 100),), 0, 1)
    assert (
        step_cost(long, two, physical(request).hardware).flops
        > step_cost(long, one, physical(request).hardware).flops
    )
    storage = physical(request).storage.model_copy(update={
        "l2_available_gb": 0, "l3_available_gb": 0,
    })
    assert resolve_config(update_inference(request, update={
        "model": two, "storage": storage,
    })).capacity.kv_bytes_per_token == (
        resolve_config(update_inference(request, update={
            "model": one, "storage": storage,
        })).capacity.kv_bytes_per_token + 132
    )
    # 超过 top-k 后，随上下文增长的 FLOPs 来自 full 索引器，非共享层主注意力。
    delta = (
        step_cost(long, one, physical(request).hardware).flops
        - step_cost(short, one, physical(request).hardware).flops
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
    model = physical(request).model.model_copy(update={
        "attention_layout": "grouped", "attention_groups": groups,
    })
    hardware = physical(request).hardware.model_copy(update={
        "gpu_count": 2, "effective_interconnect_gb_s": 100,
        "collective_latency_ms": 0.01,
    })
    storage = physical(request).storage.model_copy(update={
        "l2_available_gb": 0.001, "l3_available_gb": 0,
    })
    request = update_inference(request, update={
        "model": model, "hardware": hardware, "storage": storage, "tp_size": 2,
    })
    resolved = resolve_config(request)
    # GQA 单 KV 头逐 rank 复制 32B；MLA 的 6 维每层复制 12B；索引 Key 132B。
    assert resolved.capacity.rank_reports[0].kv_bytes_per_token == 32 + 12 + 132
    assert resolved.capacity.kv_bytes_per_token == 2 * (32 + 12 + 132)
    assert prepare_runtime(request).perf_model.kv_bytes_per_token == (
        resolved.capacity.kv_bytes_per_token
    )
    assert resolved.capacity.rank_reports[0].sharded_weights_bytes > 0
    workload = StepWorkload("DECODE", (), (DecodeWork(1, 100),), 0, 1)
    assert step_cost(workload, model, hardware, 2).collectives == 4


def hy4_request() -> SimConfig:
    request = example_request()
    model = ModelSpec(
        name="tencent/Hy4-preview", revision="705d81ee51566a186d645b74c974d642ef2828fe",
        num_layers=78, hidden_size=6144,
        num_attention_heads=64, num_kv_heads=8, head_dim=256,
        ffn_intermediate_size=18432, vocab_size=120832,
        moe_layers=77, num_experts=256, experts_per_token=8,
        expert_intermediate_size=2048, attention_layout="hy4_preview",
        num_shared_experts=1, shared_expert_intermediate_size=2048,
        router=RouterSpec(dtype="fp32", correction_bias=True), ihc=IHCSpec(mult=4),
        attention_groups=[AttentionGroup(
            kind="mla_dsa", layers=78, num_attention_heads=64,
            q_rank=2048, kv_rank=512, rope_dim=64, nope_dim=192, value_dim=256,
            index_heads=32, index_dim=128, topk=2048, full_indexer_layers=21,
            output_gate=True, index_norm_dtype="fp32", sink_dtype="fp32",
        )],
    )
    hardware = physical(request).hardware.model_copy(update={
        "gpu_count": 8, "memory_bytes_per_gpu": 288_000_000_000,
        "effective_interconnect_gb_s": 600, "collective_latency_ms": 0.01,
        "ep_communication_latency_ms": 0.02,
    })
    storage = physical(request).storage.model_copy(update={
        "l1_available_gb": 260, "l2_available_gb": 0, "l3_available_gb": 0,
    })
    return update_inference(request, update={
        "model": model, "hardware": hardware, "storage": storage,
        "tp_size": 8, "ep_size": 8,
    })


def test_hy4_bf16_tp8_ep8_main_cache_and_weight_placement() -> None:
    resolved = resolve_config(hy4_request())
    capacity = resolved.capacity
    assert resolved.runnable
    assert capacity.kv_bytes_per_token == 8 * (78 * 576 * 2 + 21 * 132)
    assert capacity.estimated_total_parameters == 769_907_408_797  # 仅主干，不含 MTP
    assert capacity.estimated_active_parameters == 49_057_546_141
    assert len(capacity.rank_reports) == 8
    rank = capacity.rank_reports[0]
    assert rank.kv_bytes_per_token == 92_628
    assert rank.sharded_weights_bytes == 5_322_574_272
    assert rank.replicated_weights_bytes == 9_331_878_004
    assert rank.expert_weights_bytes == 186_025_771_008
    assert capacity.weights_resident_bytes == 8 * (
        rank.sharded_weights_bytes + rank.replicated_weights_bytes + rank.expert_weights_bytes
    )
    assert resolved.max_total_tokens == rank.l1_capacity_pages * 4
    request = hy4_request()
    storage = physical(request).storage.model_copy(update={"l2_available_gb": 10})
    tiered = resolve_config(update_inference(request, update={"storage": storage}))
    assert tiered.capacity.rank_reports[0].kv_bytes_per_token == 78 * (512 + 64) * 2 + 78 * 132
    runtime = prepare_runtime(update_inference(request, update={"storage": storage}))
    assert runtime.perf_model.kv_bytes_per_token == tiered.capacity.kv_bytes_per_token
    assert tiered.capacity.weights_resident_bytes == capacity.weights_resident_bytes
    assert tiered.capacity.estimated_total_parameters == capacity.estimated_total_parameters
    for main_rank, tiered_rank in zip(
        capacity.rank_reports, tiered.capacity.rank_reports, strict=True,
    ):
        assert (
            tiered_rank.sharded_weights_bytes,
            tiered_rank.replicated_weights_bytes,
            tiered_rank.expert_weights_bytes,
        ) == (
            main_rank.sharded_weights_bytes,
            main_rank.replicated_weights_bytes,
            main_rank.expert_weights_bytes,
        )


@pytest.mark.parametrize("tp_size", [1, 8])
@pytest.mark.parametrize("hierarchical", [False, True])
def test_hy4_kv_uses_group_budget_for_all_cache_tiers(
    tp_size: int, hierarchical: bool,
) -> None:
    request = hy4_request()
    hardware = physical(request).hardware.model_copy(update={
        "gpu_count": tp_size, "memory_bytes_per_gpu": 3_000_000_000_000,
    })
    storage = physical(request).storage.model_copy(update={
        "l1_available_gb": 2000,
        "l2_available_gb": 10 if hierarchical else 0,
        "l3_available_gb": 10 if hierarchical else 0,
    })
    request = update_inference(request, update={
        "hardware": hardware, "storage": storage, "tp_size": tp_size, "ep_size": tp_size,
    })
    resolved = resolve_config(request)
    custom = ModelSpec.model_validate(physical(request).model.model_dump() | {
        "name": "custom", "revision": None, "attention_layout": "grouped",
    })
    grouped = resolve_config(update_inference(request, update={"model": custom}))
    local_kv = 78 * 576 * 2 + (78 if hierarchical else 21) * 132
    group_kv = local_kv * tp_size
    capacity = resolved.capacity
    assert capacity == grouped.capacity
    assert capacity.kv_bytes_per_token == grouped.capacity.kv_bytes_per_token == group_kv
    assert capacity.kv_bytes_per_page == grouped.capacity.kv_bytes_per_page == group_kv * 4
    assert all(rank.kv_bytes_per_token == local_kv for rank in capacity.rank_reports)
    assert prepare_runtime(request).perf_model.kv_bytes_per_token == group_kv
    assert resolved.max_total_tokens == capacity.l1_capacity_pages * 4
    if hierarchical:
        expected_pages = 10_000_000_000 // (group_kv * 4)
        assert capacity.l2_capacity_pages == capacity.l3_capacity_pages == expected_pages
        assert resolved.l2_capacity_tokens == capacity.l2_capacity_pages * 4
        assert resolved.l3_capacity_tokens == capacity.l3_capacity_pages * 4
    else:
        assert capacity.l2_capacity_pages == capacity.l3_capacity_pages == 0
    groups = physical(request).model.attention_groups
    assert groups is not None and groups[0].full_indexer_layers == 21


def test_hy4_dsa_index_reuse_and_sparse_attention_workload() -> None:
    request = hy4_request()
    workload = StepWorkload("DECODE", (), (DecodeWork(1, 100_000),), 0, 1)
    short = StepWorkload("DECODE", (), (DecodeWork(1, 1_000),), 0, 1)
    long_cost = step_cost(workload, physical(request).model, physical(request).hardware, 8, 8)
    short_cost = step_cost(short, physical(request).model, physical(request).hardware, 8, 8)
    near = StepWorkload("DECODE", (), (DecodeWork(1, 90_000),), 0, 1)
    near_cost = step_cost(near, physical(request).model, physical(request).hardware, 8, 8)
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
    phase: Literal["PREFILL", "DECODE", "MIXED"],
    reused: int,
    new: int,
    context: int | None,
    flops: float,
    hbm: float,
    duration: float,
    communication_bytes: float,
    tp_bytes: float,
    ep_bytes: float,
    communication_ms: float,
) -> None:
    request = hy4_request()
    hardware = physical(request).hardware.model_copy(update={"kernel_overhead_ms": 0.03})
    prefill = (PrefillWork(1, reused, new, reused + new),) if new else ()
    decode = (DecodeWork(2, context),) if context is not None else ()
    workload = StepWorkload(phase, prefill, decode, 0, len(prefill) + len(decode))
    cost = step_cost(workload, physical(request).model, hardware, 8, 8)
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
        physical(request).model.model_dump()
        | {
            "name": "custom",
            "revision": None,
            "attention_layout": "grouped",
        }
    )
    assert step_cost(workload, custom, hardware, 8, 8) == cost


def test_hy4_parameters_are_editable_and_empty_steps_are_rejected() -> None:
    request = hy4_request()
    payload = physical(request).model.model_dump()
    with pytest.raises(ValidationError, match="必须提供注意力组"):
        ModelSpec.model_validate(payload | {"attention_groups": None})
    updates: tuple[dict[str, object], ...] = (
        {"name": "modified", "revision": "user-revision"},
        {"hidden_size": 4096},
        {"ffn_intermediate_size": 16384},
        {"weight_dtype": "fp16", "kv_dtype": "fp16"},
        {"attention_groups": [payload["attention_groups"][0] | {"output_gate": False}]},
        {"num_shared_experts": 0, "shared_expert_intermediate_size": None},
        {"norms": payload["norms"] | {"kind": "layer"}},
        {"norms": payload["norms"] | {"block_per_layer": 1}},
        {"norms": payload["norms"] | {"final_norm": False}},
        {"norms": payload["norms"] | {"dtype": "fp32"}},
        {"router": payload["router"] | {"dtype": "bf16"}},
        {"router": payload["router"] | {"correction_bias": False}},
        {"router": payload["router"] | {"correction_dtype": "fp16"}},
        {"ihc": None},
        {"ihc": payload["ihc"] | {"mult": 2}},
        {"ihc": payload["ihc"] | {"dtype": "bf16"}},
    )
    for update in updates:
        modified = ModelSpec.model_validate(payload | update)
        assert ModelSpec.model_validate_json(modified.model_dump_json()) == modified
        assert resolve_config(update_inference(request, update={"model": modified})).runnable
    for field, value in (
        ("qk_norm", False), ("norm_dtype", "fp32"),
        ("index_norm", "rms"), ("index_norm_dtype", None), ("sink_dtype", None),
        ("topk", 1024), ("q_rank", 1024), ("kv_rank", 256),
    ):
        modified = ModelSpec.model_validate(payload | {
            "attention_groups": [payload["attention_groups"][0] | {field: value}],
        })
        assert resolve_config(update_inference(request, update={"model": modified})).runnable
    with pytest.raises(ValueError):
        StepWorkload("DECODE", (), (), 0, 1, restore_only=(0,))


def auxiliary_weight_request(tp_size: int) -> SimConfig:
    request = example_request()
    model = ModelSpec.model_validate(physical(request).model.model_dump() | {
        "num_layers": 4, "num_attention_heads": 4, "num_kv_heads": 2,
        "moe_layers": 2, "num_experts": 4, "experts_per_token": 2,
        "expert_intermediate_size": 8, "attention_layout": "grouped",
        "attention_groups": [
            AttentionGroup(kind="gqa", layers=1, num_attention_heads=4,
                           num_kv_heads=2, head_dim=8),
            AttentionGroup(kind="mla", layers=1, num_attention_heads=4,
                           q_rank=4, kv_rank=6, rope_dim=2, nope_dim=6, value_dim=8),
            AttentionGroup(kind="mla_dsa", layers=2, num_attention_heads=4,
                           q_rank=4, kv_rank=6, rope_dim=2, nope_dim=6, value_dim=8,
                           index_heads=2, index_dim=128, topk=4, full_indexer_layers=1),
        ],
    })
    return update_inference(request, update={
        "model": model, "tp_size": tp_size, "ep_size": tp_size,
        "hardware": physical(request).hardware.model_copy(update={
            "gpu_count": tp_size, "effective_interconnect_gb_s": 100,
            "collective_latency_ms": 0.01, "ep_communication_latency_ms": 0.02,
        }),
        "storage": physical(request).storage.model_copy(update={
            "l2_available_gb": 0.001, "l3_available_gb": 0.001,
        }),
    })


def assert_auxiliary_weight_delta(
    request: SimConfig, model: ModelSpec, parameters: int,
    replicated_bytes: int, sharded_bytes: int = 0,
) -> None:
    base = resolve_config(request)
    resolved = resolve_config(update_inference(request, update={"model": model}))
    capacity, original = resolved.capacity, base.capacity
    assert capacity.estimated_total_parameters - original.estimated_total_parameters == parameters
    assert capacity.estimated_active_parameters - original.estimated_active_parameters == parameters
    assert capacity.estimated_expert_parameters == original.estimated_expert_parameters
    assert capacity.weights_resident_bytes - original.weights_resident_bytes == (
        replicated_bytes + sharded_bytes
    ) * physical(request).tp_size
    for rank, base_rank in zip(capacity.rank_reports, original.rank_reports, strict=True):
        assert (
            rank.replicated_weights_bytes - base_rank.replicated_weights_bytes == replicated_bytes
        )
        assert rank.sharded_weights_bytes - base_rank.sharded_weights_bytes == sharded_bytes
        assert rank.expert_weights_bytes == base_rank.expert_weights_bytes
        assert rank.l1_kv_budget_bytes - base_rank.l1_kv_budget_bytes == -(
            replicated_bytes + sharded_bytes
        )
        assert rank.kv_bytes_per_token == base_rank.kv_bytes_per_token
    assert capacity.kv_bytes_per_token == original.kv_bytes_per_token
    assert capacity.kv_bytes_per_page == original.kv_bytes_per_page
    assert capacity.l2_capacity_pages == original.l2_capacity_pages
    assert capacity.l3_capacity_pages == original.l3_capacity_pages
    assert prepare_runtime(update_inference(request, update={"model": model})).perf_model == (
        prepare_runtime(request).perf_model
    )
    workload = StepWorkload("MIXED", (PrefillWork(1, 3, 2, 5),), (DecodeWork(2, 10),), 0, 2)
    inference = physical(request)
    assert step_cost(workload, model, inference.hardware, inference.tp_size, inference.ep_size) == (
        step_cost(
            workload, inference.model, inference.hardware, inference.tp_size, inference.ep_size,
        )
    )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("kind,blocks,final,dtype,parameters,bytes_per_parameter", [
    ("none", 3, True, "fp32", 0, 4),
    ("rms", 0, False, None, 0, 2),
    ("rms", 0, True, None, 16, 2),
    ("rms", 3, False, "fp16", 192, 2),
    ("rms", 2, True, "bf16", 144, 2),
    ("layer", 1, True, "fp32", 160, 4),
])
def test_block_and_final_norm_capacity_follows_explicit_spec(
    tp_size: int, kind: Literal["none", "rms", "layer"], blocks: int, final: bool,
    dtype: AuxWeightDtype | None, parameters: int, bytes_per_parameter: int,
) -> None:
    request = auxiliary_weight_request(tp_size)
    request = update_inference(request, update={
        "model": physical(request).model.model_copy(update={"norms": NormSpec(kind="none")}),
    })
    model = physical(request).model.model_copy(update={
        "norms": NormSpec(kind=kind, block_per_layer=blocks, final_norm=final, dtype=dtype),
    })
    assert_auxiliary_weight_delta(request, model, parameters, parameters * bytes_per_parameter)


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("group_index,parameters", [(0, 16), (1, 10), (2, 20)])
@pytest.mark.parametrize("dtype,bytes_per_parameter", [(None, 2), ("fp16", 2), ("fp32", 4)])
def test_qk_norm_uses_head_or_low_rank_width_and_is_replicated(
    tp_size: int, group_index: int, parameters: int,
    dtype: AuxWeightDtype | None, bytes_per_parameter: int,
) -> None:
    request = auxiliary_weight_request(tp_size)
    original_groups = physical(request).model.attention_groups
    assert original_groups is not None
    groups = list(original_groups)
    groups[group_index] = groups[group_index].model_copy(update={"qk_norm": False})
    request = update_inference(request, update={
        "model": physical(request).model.model_copy(update={"attention_groups": groups}),
    })
    enabled = list(groups)
    enabled[group_index] = groups[group_index].model_copy(update={
        "qk_norm": True, "norm_dtype": dtype,
    })
    model = physical(request).model.model_copy(update={"attention_groups": enabled})
    assert_auxiliary_weight_delta(request, model, parameters, parameters * bytes_per_parameter)
    disabled = list(groups)
    disabled[group_index] = groups[group_index].model_copy(update={"norm_dtype": dtype})
    assert_auxiliary_weight_delta(
        request, physical(request).model.model_copy(update={"attention_groups": disabled}), 0, 0,
    )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("dtype,bytes_per_parameter", [("bf16", 2), ("fp16", 2), ("fp32", 4)])
def test_attention_sink_is_sharded_by_each_groups_heads(
    tp_size: int, dtype: AuxWeightDtype, bytes_per_parameter: int,
) -> None:
    request = auxiliary_weight_request(tp_size)
    groups = physical(request).model.attention_groups
    assert groups is not None
    model = physical(request).model.model_copy(update={
        "attention_groups": [
            group.model_copy(update={"sink_dtype": dtype})
            for group in groups
        ],
    })
    assert_auxiliary_weight_delta(request, model, 16, 0, 16 // tp_size * bytes_per_parameter)


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("full_layers", [1, 2])
@pytest.mark.parametrize("kind,dtype,parameters_per_layer,bytes_per_parameter", [
    ("none", "fp32", 0, 4), ("rms", None, 128, 2),
    ("rms", "fp16", 128, 2), ("layer", "bf16", 256, 2),
    ("layer", "fp32", 256, 4),
])
def test_index_norm_counts_only_full_indexer_layers(
    tp_size: int, full_layers: int, kind: Literal["none", "rms", "layer"],
    dtype: AuxWeightDtype | None, parameters_per_layer: int, bytes_per_parameter: int,
) -> None:
    request = auxiliary_weight_request(tp_size)
    original_groups = physical(request).model.attention_groups
    assert original_groups is not None
    groups = list(original_groups)
    groups[2] = groups[2].model_copy(update={
        "index_norm": "none", "full_indexer_layers": full_layers,
    })
    request = update_inference(request, update={
        "model": physical(request).model.model_copy(update={"attention_groups": groups}),
    })
    normalized = list(groups)
    normalized[2] = groups[2].model_copy(update={"index_norm": kind, "index_norm_dtype": dtype})
    parameters = full_layers * parameters_per_layer
    assert_auxiliary_weight_delta(
        request, physical(request).model.model_copy(update={"attention_groups": normalized}),
        parameters, parameters * bytes_per_parameter,
    )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("dtype,correction,correction_dtype,matrix_bytes,correction_bytes", [
    (None, False, "fp32", 2, 4), ("fp16", False, "fp32", 2, 4),
    ("fp32", False, "fp32", 4, 4), ("fp32", True, "fp32", 4, 4),
    ("bf16", True, "fp16", 2, 2), ("fp16", True, "bf16", 2, 2),
])
def test_router_dtype_replaces_matrix_bytes_and_adds_correction_once(
    tp_size: int, dtype: AuxWeightDtype | None, correction: bool,
    correction_dtype: AuxWeightDtype, matrix_bytes: int, correction_bytes: int,
) -> None:
    request = auxiliary_weight_request(tp_size)
    model = physical(request).model.model_copy(update={
        "router": RouterSpec(dtype=dtype, correction_bias=correction,
                             correction_dtype=correction_dtype),
    })
    correction_parameters = 8 if correction else 0
    assert_auxiliary_weight_delta(
        request, model, correction_parameters,
        128 * (matrix_bytes - 2) + correction_parameters * correction_bytes,
    )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("mult,parameters", [(1, 306), (2, 1139), (4, 4437)])
def test_ihc_counts_two_sites_per_layer_and_final_head_as_fp32_copies(
    tp_size: int, mult: int, parameters: int,
) -> None:
    request = auxiliary_weight_request(tp_size)
    model = physical(request).model.model_copy(update={"ihc": IHCSpec(mult=mult)})
    assert_auxiliary_weight_delta(request, model, parameters, parameters * 4)
    disabled = model.model_copy(update={"ihc": None})
    assert resolve_config(update_inference(request, update={"model": disabled})).capacity == (
        resolve_config(request).capacity
    )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("kind", ["mla", "mla_dsa"])
def test_output_gate_uses_group_dimensions_and_sharded_capacity(
    kind: Literal["mla", "mla_dsa"], tp_size: int,
) -> None:
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
        index_heads=2 if kind == "mla_dsa" else None,
        index_dim=128 if kind == "mla_dsa" else None,
        topk=4 if kind == "mla_dsa" else None,
        full_indexer_layers=1 if kind == "mla_dsa" else None,
    )
    model = physical(request).model.model_copy(
        update={
            "attention_layout": "grouped",
            "attention_groups": [group],
        }
    )
    gated = model.model_copy(
        update={"attention_groups": [group.model_copy(update={"output_gate": True})]}
    )
    hardware = physical(request).hardware.model_copy(
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
    storage = physical(request).storage.model_copy(update={
        "l2_available_gb": 0, "l3_available_gb": 0,
    })
    base_capacity = resolve_config(
        update_inference(request,
            update={
                "model": model,
                "hardware": hardware,
                "storage": storage,
                "tp_size": tp_size,
            }
        )
    ).capacity
    capacity = resolve_config(
        update_inference(request,
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
def test_shared_experts_are_replicated_and_do_not_change_routing(
    tp_size: int, grouped: bool,
) -> None:
    request = example_request()
    model = physical(request).model.model_copy(
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
    hardware = physical(request).hardware.model_copy(
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
    storage = physical(request).storage.model_copy(update={
        "l2_available_gb": 0, "l3_available_gb": 0,
    })
    base_capacity = resolve_config(
        update_inference(request,
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
        update_inference(request,
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
    model = physical(request).model.model_copy(
        update={"attention_layout": "grouped", "attention_groups": [mla]}
    )
    sparse = model.model_copy(update={"attention_groups": [dsa]})
    workload = StepWorkload("MIXED", (PrefillWork(1, 1, 2, 3),), (DecodeWork(2, 4),), 0, 2)
    base = step_cost(workload, model, physical(request).hardware)
    cost = step_cost(workload, sparse, physical(request).hardware)
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
        resolved.max_total_tokens,
        resolved.l2_capacity_tokens,
        resolved.l3_capacity_tokens,
    ) == (3861 * 4, 8, 4)
    assert report.l2_to_l1_tokens_per_ms == pytest.approx(1000 / 64)
    assert report.l3_to_l2_tokens_per_ms == pytest.approx(2000 / 64)
    runtime = prepare_runtime(example_request())
    assert runtime.resolution == resolved
    assert runtime.perf_model.kv_bytes_per_token == 64
    assert runtime.perf_model.l3_lookup_latency_ms == 1.25
    assert runtime.engine.l3_prefetch_min_tokens == 256


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
        update_inference(request,
            update={
                "model": qwen,
                "hardware": physical(request).hardware.model_copy(
                    update={
                        "name": "NVIDIA B300",
                        "memory_bytes_per_gpu": 288_000_000_000,
                    }
                ),
                "storage": physical(request).storage.model_copy(
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
            update_inference(original,
                update={"hardware": physical(original).hardware.model_copy(update={"gpu_count": 8})}
            )
        )
    with pytest.raises(ValueError, match="MoE"):
        resolve_config(
            update_inference(original,
                update={"model": physical(original).model.model_copy(update={"moe_layers": 1})}
            )
        )
    with pytest.raises(ValueError, match="L2"):
        resolve_config(
            update_inference(original,
                update={"storage": physical(original).storage.model_copy(update={
                    "l2_to_l1_gb_s": None,
                })}
            )
        )
    with pytest.raises(ValueError, match="不足"):
        resolve_config(
            update_inference(original,
                update={
                    "storage": physical(original).storage.model_copy(update={
                        "l1_available_gb": 0.000001,
                    })
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
    hardware = physical(request).hardware.model_copy(
        update={
            "gpu_count": 4,
            "memory_bytes_per_gpu": 288_000_000_000,
            "effective_interconnect_gb_s": 100,
            "collective_latency_ms": 0.1,
        }
    )
    storage = physical(request).storage.model_copy(
        update={
            "l1_available_gb": 20,
            "l2_available_gb": 1,
            "l3_available_gb": 0.5,
        }
    )
    result = resolve_config(
        update_inference(request,
            update={
                "model": model,
                "hardware": hardware,
                "storage": storage,
                "tp_size": 4,
            }
        )
    )
    report = result.capacity
    assert result.runnable and hardware.gpu_count == 4
    assert not resolve_config(
        update_inference(request,
            update={
                "model": model,
                "hardware": hardware,
                "storage": storage,
                "tp_size": 4,
                "timing": EmpiricalTimingConfig(prefill_c1=1, decode_d1=2, decode_d2=1),
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
        result.max_total_tokens
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
    hardware = physical(request).hardware.model_copy(
        update={
            "gpu_count": 8,
            "memory_bytes_per_gpu": 288_000_000_000,
            "effective_interconnect_gb_s": 100,
            "collective_latency_ms": 0.1,
        }
    )
    storage = physical(request).storage.model_copy(
        update={
            "l1_available_gb": 50,
            "l2_available_gb": 1,
            "l3_available_gb": 0,
        }
    )
    tp = update_inference(request,
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
    result = simulate(tp)
    analysis = result.analysis
    assert result.meta.num_completed == 1 and analysis is not None
    assert analysis.collectives == 2 * 94 * analysis.steps
    assert analysis.communication_ms > 0
    with pytest.raises(ValueError, match="MoE"):
        resolve_config(update_inference(tp, update={
            "model": model.model_copy(update={"moe_layers": 94}),
        }))
    with pytest.raises(ValueError, match="MoE"):
        resolve_config(
            update_inference(tp,
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
        resolve_config(update_inference(request, update={"tp_size": 4}))
    hardware = physical(request).hardware.model_copy(update={"gpu_count": 4})
    valid_model = physical(request).model.model_copy(
        update={
            "vocab_size": 16,
            "num_attention_heads": 4,
        }
    )
    storage = physical(request).storage.model_copy(update={
        "l2_available_gb": 0, "l3_available_gb": 0,
    })
    request = update_inference(request, update={
        "tp_size": 4,
        "hardware": hardware,
        "model": valid_model,
        "storage": storage,
    })
    preview = resolve_config(request)
    assert not preview.runnable and len(preview.capacity.rank_reports) == 4
    with pytest.raises(ValueError, match="集合通信延迟"):
        simulate(request)
    hardware = hardware.model_copy(
        update={"effective_interconnect_gb_s": 100, "collective_latency_ms": 0.1}
    )
    assert resolve_config(
        update_inference(request,
            update={
                "tp_size": 4,
                "hardware": hardware,
                "model": valid_model,
                "storage": storage,
            }
        )
    ).runnable
    with pytest.raises(ValidationError, match="vocab_size"):
        ModelSpec.model_validate(valid_model.model_dump() | {"vocab_size": None})


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
    hardware = physical(base).hardware.model_copy(update={"gpu_count": 2})
    storage = physical(base).storage.model_copy(
        update={
            "l1_available_gb": 0.00001,
            "l2_available_gb": 0,
            "l3_available_gb": 0,
            "runtime_reserve_bytes": 0,
        }
    )
    request = update_inference(base,
        update={
            "tp_size": 2,
            "ep_size": 2,
            "model": model,
            "hardware": hardware,
            "storage": storage,
        }
    )
    preview = resolve_config(request)
    assert physical(request).ep_size == physical(request).tp_size == 2
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
    assert preview.max_total_tokens == 4 * preview.capacity.l1_capacity_pages
    with pytest.raises(ValueError, match="EP 单次通信开销"):
        simulate(request)


@pytest.mark.parametrize(
    "updates, reason",
    [
        ({"ep_size": 1}, "EP 数"),
        ({"model": {"experts_per_token": 5}}, "激活专家数"),
        ({"model": {"num_experts": 3}}, "专家数"),
        ({"model": {"expert_intermediate_size": None}}, "专家中间维度"),
    ],
)
def test_moe_preview_rejects_inconsistent_layout(
    updates: dict[str, object], reason: str,
) -> None:
    base = example_request()
    model = physical(base).model.model_copy(
        update={
            "vocab_size": 16,
            "moe_layers": 1,
            "num_experts": 4,
            "experts_per_token": 2,
            "expert_intermediate_size": 8,
            **cast(dict[str, object], updates.get("model", {})),
        }
    )
    request = update_inference(base,
        update={
            "model": model,
            "ep_size": updates.get("ep_size", 2),
            "tp_size": 2,
            "hardware": physical(base).hardware.model_copy(update={"gpu_count": 2}),
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
    for workload, expected in ((prefill, normal), (decode, low_latency), (mixed, combined)):
        with_restore = StepWorkload(
            workload.phase,
            workload.prefill,
            workload.decode,
            workload.allocated_pages,
            workload.batch_requests + 1,
            restore_only=(3,),
        )
        assert step_cost(with_restore, model, hardware, tp_size=2, ep_size=2) == expected


def test_moe_analytical_run_uses_ep_cost_and_keeps_empirical_unavailable() -> None:
    base = example_request()
    model = physical(base).model.model_copy(
        update={
            "vocab_size": 16,
            "moe_layers": 1,
            "num_experts": 4,
            "experts_per_token": 2,
            "expert_intermediate_size": 8,
        }
    )
    hardware = physical(base).hardware.model_copy(
        update={
            "gpu_count": 2,
            "effective_interconnect_gb_s": 1,
            "collective_latency_ms": 0.3,
            "ep_communication_latency_ms": 0.7,
        }
    )
    storage = physical(base).storage.model_copy(
        update={
            "l2_available_gb": 0,
            "l3_available_gb": 0,
        }
    )
    request = update_inference(base,
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
    result = simulate(request)
    summary = result.analysis
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
        simulate(update_inference(request, update={
            "timing": EmpiricalTimingConfig(prefill_c1=1, decode_d1=2, decode_d2=1),
        }))


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
    config = update_inference(example_request(), update={
        "model": model, "hardware": hardware, "tp_size": 2, "ep_size": 2,
        "storage": StorageSpec(l1_available_gb=0.001),
    }).model_copy(update={
        "scheduler": SchedulerConfig(max_running_requests=2, pd_mutually_exclusive=False),
    })
    runtime = prepare_runtime(config)
    # 组件测试只缩小运行容量；物理解析报告与原始配置保持不变。
    runtime = replace(runtime, engine=runtime.engine.model_copy(update={"max_total_tokens": 128}))
    engine = Engine(runtime, output_targets={0: 2, 1: 2}, record_trace=True)
    request0 = InferenceRequest(0, 0, None, 4, 2, (), 0)
    request1 = InferenceRequest(1, 1, None, 4, 2, (), 0)
    engine.submit(request0, 0)
    first = engine.start_step(0, 0)
    assert first is not None
    first_duration, _ = first
    assert engine.finish_step(first_duration) == ()
    engine.submit(request1, first_duration)
    mixed = engine.start_step(first_duration, 1)
    assert mixed is not None
    mixed_duration, _ = mixed
    assert list(engine._log.steps())[-1].phase == "MIXED"
    assert engine.finish_step(first_duration + mixed_duration) == (0,)
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
    last = engine.start_step(first_duration + mixed_duration, 2)
    assert last is not None
    last_duration, _ = last
    assert engine.finish_step(first_duration + mixed_duration + last_duration) == (1,)
    engine.complete(first_duration + mixed_duration + last_duration, 1)
    run = engine.finalize()
    assert [item.phase for item in run.trace.steps()] == ["PREFILL", "MIXED", "DECODE"]
    assert run.analysis is not None
    assert run.analysis.steps == 3
    assert run.analysis.ep_exchanges == 3 * expected.ep_exchanges
    assert run.analysis.device_ms == pytest.approx(first_duration + mixed_duration + last_duration)
    assert run.records[-1].finish == pytest.approx(run.analysis.device_ms)
    assert len(run.records) == 2 and not run.aborted_requests and not run.skipped_requests


@pytest.mark.parametrize("tp_size", [1, 2])
def test_engine_config_derived_links_match_physical_two_hop_transfer(tp_size: int) -> None:
    request = example_request()
    model = physical(request).model.model_copy(update={
        "moe_layers": 1, "num_experts": 4, "experts_per_token": 2,
        "expert_intermediate_size": 8,
    })
    hardware = physical(request).hardware.model_copy(update={
        "gpu_count": tp_size, "effective_interconnect_gb_s": 100,
        "collective_latency_ms": 0.1, "ep_communication_latency_ms": 0.2,
    })
    storage = physical(request).storage.model_copy(update={
        "l2_available_gb": 0.001, "l3_available_gb": 0.001,
    })
    config = update_inference(request, update={
        "model": model, "hardware": hardware, "storage": storage,
        "tp_size": tp_size, "ep_size": tp_size,
    })
    runtime = prepare_runtime(config)
    resolved = runtime.resolution
    assert resolved is not None
    original_data = config.model_dump()
    # 组件场景只缩小运行容量，报告及配置仍代表完整物理预算。
    runtime = replace(runtime, engine=runtime.engine.model_copy(update={
        "max_total_tokens": 12, "l2_capacity_tokens": 4, "l3_capacity_tokens": 4,
    }))
    assert runtime.resolution == resolve_config(config)
    assert runtime.links == prepare_runtime(config).links
    page_bytes = runtime.engine.page_size * resolved.capacity.kv_bytes_per_token
    l2_service = physical(request).storage.l2_latency_ms + page_bytes / 1000
    l3_service = physical(request).storage.l3_latency_ms + page_bytes / 2000
    expected_ms = 2 * l2_service + l3_service
    l2_link, l3_link = runtime.links.l2_to_l1, runtime.links.l3_to_l2
    assert l2_link is not None and l3_link is not None
    assert l2_link.page_bytes == l3_link.page_bytes == page_bytes
    assert l2_link.latency_ms + l2_link.transfer_ms == pytest.approx(l2_service)
    assert l3_link.latency_ms + l3_link.transfer_ms == pytest.approx(l3_service)
    assert l2_service == pytest.approx(
        storage.l2_latency_ms
        + runtime.engine.page_size / cast(float, resolved.capacity.l2_to_l1_tokens_per_ms)
    )
    assert l3_service == pytest.approx(
        storage.l3_latency_ms
        + runtime.engine.page_size / cast(float, resolved.capacity.l3_to_l2_tokens_per_ms)
    )
    engine = Engine(runtime, output_targets={}, record_trace=True)
    cache = engine._prefix_cache
    for key in range(3):
        parent = InferenceRequest(key, key, None, 4, 1, (), 0)
        cache.pin(parent, 0)
        cache.admit(key)
        cache.extend(key, 4)
        cache.publish(key)
        cache.finish(key)
    cache.ensure_free(2)
    now = 10.0
    elapsed = 0.0
    for chunk_id, source, destination, kind, service in (
        (encode(1, 0), "L2", "L1", EventKind.L2_L1_DONE, l2_service),
        (encode(0, 0), "L3", "L2", EventKind.L3_L2_DONE, l3_service),
        (encode(0, 0), "L2", "L1", EventKind.L2_L1_DONE, l2_service),
    ):
        page = cache.page(chunk_id)
        assert page is not None and page.tier == source
        transfer_id = engine._start_transfer(chunk_id, destination, now, 99)
        [(end, event_kind, event_id)] = engine.drain_events()
        assert (event_kind, event_id) == (kind, transfer_id)
        assert end - now == pytest.approx(service)
        page = cache.page(chunk_id)
        assert page is not None and page.tier == source
        assert engine._restore_time_ms == pytest.approx(elapsed)
        engine.on_transfer_done(kind, transfer_id, end)
        page = cache.page(chunk_id)
        assert page is not None and page.tier == destination
        elapsed += service
        now = end
    run = engine.finalize()
    assert run.restore_time_ms == pytest.approx(expected_ms)
    assert now == pytest.approx(10 + expected_ms)
    assert run.restore_bytes == 3 * page_bytes
    assert run.analysis is not None
    assert run.analysis.restore_ms == pytest.approx(expected_ms)
    assert all(
        value == 0
        for name, value in run.analysis.model_dump().items()
        if name != "restore_ms"
    )
    assert run.num_engine_steps == 0 and list(run.trace.steps()) == []
    assert not engine._transfers and not cache._transfers
    assert config.model_dump() == original_data


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
    with_restore = StepWorkload("PREFILL", prefill.prefill, (), 1, 2, restore_only=(1,))
    assert step_cost(with_restore, model, hardware) == cost
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
    with_restore = StepWorkload("PREFILL", prefill.prefill, (), 1, 2, restore_only=(1,))
    assert step_cost(with_restore, model, hardware, tp_size=2) == cost


def test_tp2_runner_uses_communication_cost_and_preserves_request_results() -> None:
    base = example_request()
    model = physical(base).model.model_copy(
        update={
            "vocab_size": 16,
        }
    )
    hardware = physical(base).hardware.model_copy(
        update={
            "gpu_count": 2,
            "effective_interconnect_gb_s": 1e-6,
            "collective_latency_ms": 0.1,
        }
    )
    storage = physical(base).storage.model_copy(
        update={
            "l1_available_gb": 0.0002,
            "l2_available_gb": 0,
            "l3_available_gb": 0,
        }
    )
    request = update_inference(base,
        update={
            "model": model,
            "hardware": hardware,
            "storage": storage,
            "tp_size": 2,
        }
    )
    resolved = resolve_config(request)
    result = simulate(request)
    analysis = result.analysis
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
        simulate(update_inference(request, update={
            "timing": EmpiricalTimingConfig(prefill_c1=1, decode_d1=2, decode_d2=1),
        }))


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
    request = example_request()
    original_data = request.model_dump()
    result = simulate(request)
    assert result.meta.config.model_dump() == original_data
    assert request.model_dump() == original_data
    assert result.resolution == resolve_config(request)
    summary = result.analysis
    assert summary is not None
    assert result.meta.num_completed == 1
    assert summary.steps == result.meta.num_engine_steps
    assert summary.device_ms > 0
    assert result.requests[0].finish - result.requests[0].prefill_start == pytest.approx(
        summary.device_ms + summary.restore_ms
    )
