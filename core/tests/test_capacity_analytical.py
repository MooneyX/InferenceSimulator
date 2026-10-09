"""阶段一：单卡物理容量与逐步 Roofline 计时金标准。"""

import pytest

from simcore.analytical import step_cost
from simcore.capacity import resolve_config
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
            workload=WorkloadConfig(synthetic=SyntheticWorkloadConfig(
                num_requests=1, arrival_rate=1, prompt_len=FixedLength(value=4),
                output_len=FixedLength(value=2), max_new_tokens=2, seed=0,
            )),
            engine=EngineConfig(max_running_requests=1, max_total_tokens=4, page_size=4),
            perf_model=PerfModelConfig(prefill_c1=1, decode_d1=2, decode_d2=1),
        ),
        model=ModelSpec(
            name="toy", num_layers=2, hidden_size=16, num_attention_heads=2,
            num_kv_heads=1, head_dim=8, ffn_intermediate_size=32,
            total_parameters=1000,
        ),
        hardware=HardwareSpec(
            name="toy", memory_bytes_per_gpu=2_000_000,
            effective_fp16_tflops=1, effective_hbm_gb_s=1,
        ),
        storage=StorageSpec(
            l1_available_gb=0.001, l2_available_gb=0.000000512,
            l3_available_gb=0.000000256, runtime_reserve_bytes=1000,
            l2_to_l1_gb_s=0.001, l3_to_l2_gb_s=0.002,
            l2_latency_ms=3, l3_latency_ms=4,
        ),
    )


def test_resolve_three_tiers_and_bandwidth_rounds_only_whole_pages() -> None:
    resolved = resolve_config(example_request())
    report = resolved.capacity
    assert report.kv_bytes_per_token == 64
    assert report.kv_bytes_per_page == 256
    assert report.weights_resident_bytes == 2000
    assert report.l1_kv_budget_bytes == 997_000
    assert (report.l1_capacity_pages, report.l2_capacity_pages, report.l3_capacity_pages) == (
        3894, 2, 1,
    )
    assert (resolved.config.engine.max_total_tokens, resolved.config.engine.l2_capacity_tokens,
            resolved.config.engine.l3_capacity_tokens) == (3894 * 4, 8, 4)
    assert report.l2_to_l1_tokens_per_ms == pytest.approx(1000 / 64)
    assert report.l3_to_l2_tokens_per_ms == pytest.approx(2000 / 64)
    assert resolved.config.perf_model.kv_bytes_per_token == 64


def test_qwen3_8b_single_b300_capacity_example() -> None:
    request = example_request()
    qwen = ModelSpec(
        name="Qwen/Qwen3-8B", num_layers=36, hidden_size=4096,
        num_attention_heads=32, num_kv_heads=8, head_dim=128,
        ffn_intermediate_size=12288, total_parameters=8_200_000_000,
    )
    resolved = resolve_config(request.model_copy(update={
        "model": qwen,
        "hardware": request.hardware.model_copy(update={
            "name": "NVIDIA B300", "memory_bytes_per_gpu": 288_000_000_000,
        }),
        "storage": request.storage.model_copy(update={
            "l1_available_gb": 200, "l2_available_gb": 0, "l3_available_gb": 0,
            "l2_to_l1_gb_s": None, "l3_to_l2_gb_s": None,
        }),
    }))
    assert resolved.capacity.kv_bytes_per_token == 144 * 1024
    assert resolved.capacity.weights_resident_bytes == 16_400_000_000
    assert resolved.capacity.l1_capacity_pages == (200_000_000_000 - 16_400_000_000 - 1000) // (
        4 * 144 * 1024
    )


def test_reject_unsupported_topologies_and_missing_bandwidth() -> None:
    original = example_request()
    with pytest.raises(ValueError, match="TP 数"):
        resolve_config(original.model_copy(update={
            "hardware": original.hardware.model_copy(update={"gpu_count": 8})
        }))
    with pytest.raises(ValueError, match="Dense"):
        resolve_config(original.model_copy(update={
            "model": original.model.model_copy(update={"moe_layers": 1})
        }))
    with pytest.raises(ValueError, match="L2"):
        resolve_config(original.model_copy(update={
            "storage": original.storage.model_copy(update={"l2_to_l1_gb_s": None})
        }))
    with pytest.raises(ValueError, match="不足"):
        resolve_config(original.model_copy(update={
            "storage": original.storage.model_copy(update={"l1_available_gb": 0.000001})
        }))


def test_tp4_preview_uses_per_rank_l1_and_group_l2_l3_budgets() -> None:
    request = example_request()
    model = ModelSpec(
        name="Qwen3-8B", num_layers=36, hidden_size=4096,
        num_attention_heads=32, num_kv_heads=8, head_dim=128,
        ffn_intermediate_size=12288, vocab_size=151936, total_parameters=8_200_000_000,
    )
    hardware = request.hardware.model_copy(update={
        "gpu_count": 4, "memory_bytes_per_gpu": 288_000_000_000,
        "effective_interconnect_gb_s": 100, "collective_latency_ms": 0.1,
    })
    storage = request.storage.model_copy(update={
        "l1_available_gb": 20, "l2_available_gb": 1, "l3_available_gb": 0.5,
    })
    result = resolve_config(request.model_copy(update={
        "model": model, "hardware": hardware, "storage": storage, "tp_size": 4,
    }))
    report = result.capacity
    assert result.runnable and result.tp_size == 4
    assert len(report.rank_reports) == 4
    assert {rank.kv_bytes_per_token for rank in report.rank_reports} == {36 * 1024}
    assert report.kv_bytes_per_token == 144 * 1024
    assert report.unassigned_parameters == 9_264_640
    for rank in report.rank_reports:
        assert (rank.sharded_weights_bytes + rank.replicated_weights_bytes
                + rank.l1_kv_budget_bytes + storage.runtime_reserve_bytes) == 20_000_000_000
        assert rank.l1_capacity_pages == rank.l1_kv_budget_bytes // (rank.kv_bytes_per_token * 4)
    assert result.config.engine.max_total_tokens == min(
        rank.l1_capacity_pages for rank in report.rank_reports
    ) * 4
    assert report.l2_capacity_pages == 1_000_000_000 // (4 * report.kv_bytes_per_token)
    assert report.l3_capacity_pages == 500_000_000 // (4 * report.kv_bytes_per_token)


def test_tp8_replicates_four_kv_heads_but_rejects_moe_for_now() -> None:
    request = example_request()
    model = ModelSpec(
        name="dense-kv4", num_layers=94, hidden_size=4096,
        num_attention_heads=64, num_kv_heads=4, head_dim=128,
        ffn_intermediate_size=12288, vocab_size=151936, total_parameters=30_000_000_000,
    )
    hardware = request.hardware.model_copy(update={
        "gpu_count": 8, "memory_bytes_per_gpu": 288_000_000_000,
        "effective_interconnect_gb_s": 100, "collective_latency_ms": 0.1,
    })
    storage = request.storage.model_copy(update={
        "l1_available_gb": 50, "l2_available_gb": 1, "l3_available_gb": 0,
    })
    tp = request.model_copy(update={
        "tp_size": 8, "model": model, "hardware": hardware, "storage": storage,
    })
    report = resolve_config(tp).capacity
    assert {rank.kv_bytes_per_token for rank in report.rank_reports} == {48_128}
    assert report.kv_bytes_per_token == 8 * 48_128
    result, analysis = simulate_resolved(resolve_config(tp))
    assert result.meta.num_completed == 1 and analysis is not None
    assert analysis.collectives == 2 * 94 * analysis.steps
    assert analysis.communication_ms > 0
    with pytest.raises(ValueError, match="MoE"):
        resolve_config(tp.model_copy(update={"model": model.model_copy(update={"moe_layers": 94})}))


def test_tp_preview_rejects_incomplete_placement_and_gpu_mismatch() -> None:
    request = example_request()
    with pytest.raises(ValueError, match="TP 数"):
        resolve_config(request.model_copy(update={"tp_size": 4}))
    hardware = request.hardware.model_copy(update={"gpu_count": 4})
    with pytest.raises(ValueError, match="互联带宽"):
        resolve_config(request.model_copy(update={"tp_size": 4, "hardware": hardware}))
    hardware = hardware.model_copy(update={"effective_interconnect_gb_s": 100,
                                   "collective_latency_ms": 0.1})
    with pytest.raises(ValueError, match="vocab_size"):
        resolve_config(request.model_copy(update={"tp_size": 4, "hardware": hardware}))


def test_token_bandwidth_matches_physical_two_hop_transfer() -> None:
    resolved = resolve_config(example_request())
    workload = StepWorkload(
        "RESTORE", (), (), 0, 1,
        (("a", "L2"), ("b", "L3")), 4, (0,),
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
        name="toy", num_layers=1, hidden_size=2, num_attention_heads=2,
        num_kv_heads=1, head_dim=1, ffn_intermediate_size=4,
        total_parameters=1000,
    )
    hardware = HardwareSpec(
        name="toy", memory_bytes_per_gpu=10000, effective_fp16_tflops=1e-9,
        effective_hbm_gb_s=1e-6, kernel_overhead_ms=0.2,
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
        name="toy", num_layers=1, hidden_size=2, num_attention_heads=2,
        num_kv_heads=1, head_dim=1, ffn_intermediate_size=4,
        total_parameters=1000,
    )
    hardware = HardwareSpec(
        name="toy", gpu_count=2, memory_bytes_per_gpu=10000,
        effective_fp16_tflops=1e-9, effective_hbm_gb_s=1e-6,
        effective_interconnect_gb_s=1e-6, collective_latency_ms=0.3,
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
    model = base.model.model_copy(update={
        "vocab_size": 16, "total_parameters": 10_000,
    })
    hardware = base.hardware.model_copy(update={
        "gpu_count": 2, "effective_interconnect_gb_s": 1e-6,
        "collective_latency_ms": 0.1,
    })
    storage = base.storage.model_copy(update={
        "l1_available_gb": 0.0002, "l2_available_gb": 0,
        "l3_available_gb": 0,
    })
    request = base.model_copy(update={
        "model": model, "hardware": hardware, "storage": storage, "tp_size": 2,
    })
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
        name="toy", num_layers=2, hidden_size=16, num_attention_heads=2,
        num_kv_heads=1, head_dim=8, ffn_intermediate_size=32, total_parameters=10_000,
    )
    hardware = HardwareSpec(
        name="toy", gpu_count=2, memory_bytes_per_gpu=1_000_000,
        effective_fp16_tflops=1, effective_hbm_gb_s=1,
        effective_interconnect_gb_s=1, collective_latency_ms=0.25,
    )
    workload = StepWorkload(
        "MIXED", (PrefillWork(1, 4, 2, 6),), (DecodeWork(0, 12),), 1, 2,
    )
    cost = step_cost(workload, model, hardware, tp_size=2)
    assert cost.collectives == 4
    assert cost.communication_bytes == 4 * (2 + 1) * 16 * 2
    assert cost.communication_ms == pytest.approx(4 * 0.25 + cost.communication_bytes / 1e6)


def test_prefill_kv_hbm_reads_scale_with_context_not_attention_pairs() -> None:
    model = ModelSpec(
        name="toy", num_layers=1, hidden_size=16, num_attention_heads=2,
        num_kv_heads=1, head_dim=8, ffn_intermediate_size=32,
        total_parameters=1000,
    )
    hardware = HardwareSpec(
        name="toy", memory_bytes_per_gpu=1_000_000,
        effective_fp16_tflops=1, effective_hbm_gb_s=1,
    )
    def cost(length: int) -> float:
        workload = StepWorkload(
            "PREFILL", (PrefillWork(0, 0, length, length),), (), 1, 1,
        )
        return step_cost(workload, model, hardware).hbm_bytes

    assert cost(1024) - cost(1) == (1024 - 1) * 2 * 8 * 2 * 2


def test_analytical_clock_uses_workload_not_empirical_coefficients() -> None:
    resolution = resolve_config(example_request())
    result, summary = simulate_analytical(
        resolution.config, resolution.model, resolution.hardware
    )
    assert result.meta.num_completed == 1
    assert summary.steps == result.meta.num_engine_steps
    assert summary.device_ms > 0
    assert result.requests[0].finish - result.requests[0].prefill_start == pytest.approx(
        summary.device_ms + summary.restore_ms
    )
