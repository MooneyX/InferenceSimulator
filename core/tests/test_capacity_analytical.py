"""阶段一：单卡物理容量与逐步 Roofline 计时金标准。"""

import pytest

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


def hy4_request() -> ResolvedRequest:
    request = example_request()
    model = ModelSpec(
        name="tencent/Hy4-preview", revision="705d81ee51566a186d645b74c974d642ef2828fe",
        num_layers=78, hidden_size=6144,
        num_attention_heads=64, num_kv_heads=8, head_dim=256,
        ffn_intermediate_size=18432, vocab_size=120832,
        moe_layers=77, num_experts=256, experts_per_token=8,
        expert_intermediate_size=2048, attention_layout="hy4_preview",
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
