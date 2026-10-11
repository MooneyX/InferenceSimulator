"""请求计划定时、依赖、同会话块复用及非法输出的内核回归。"""

from dataclasses import FrozenInstanceError
from typing import Literal

import pytest

from simcore import SimConfig, TraceReplay, simulate
from simcore.analytical import step_cost
from simcore.capacity import resolve_config
from simcore.perf_model import PrefillWork, StepWorkload
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    HardwareSpec,
    ModelSpec,
    PerfModelConfig,
    ResolvedRequest,
    StorageSpec,
    SyntheticWorkloadConfig,
    TracePlan,
    TraceTurn,
    WorkloadConfig,
)


def config(*, capacity: int = 2048) -> SimConfig:
    return SimConfig(
        num_instances=2,
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=2,
                session_rate=1,
                prompt_len=FixedLength(value=129),
                turn_increment_len=FixedLength(value=1),
                output_len=FixedLength(value=1),
                seed=0,
            )
        ),
        engine=EngineConfig(
            max_running_requests=4, max_total_tokens=capacity, page_size=64
        ),
        perf_model=PerfModelConfig(
            prefill_c1=0.01, prefill_c2=0, decode_d1=0.01, decode_d2=0
        ),
    )


def plan(*, output: int = 1, second_offset: float = 1000) -> TracePlan:
    return TracePlan(
        sessions=("original-session",),
        block_size=64,
        requests=(
            TraceTurn(
                request_id=0, session_id=0, arrival_offset_ms=0,
                prompt_len=129, output_len=output, hash_ids=(0, 1),
            ),
            TraceTurn(
                request_id=1, session_id=0, parent_request_id=0,
                arrival_offset_ms=second_offset, think_time_ms=20,
                prompt_len=193, output_len=1, hash_ids=(0, 1, 2),
            ),
        ),
    )


def test_trace_open_loop_reuses_input_pages_on_same_instance() -> None:
    result, _ = simulate(
        config(), replay=TraceReplay(plan(), mode="open_loop", session_rate=1, seed=7)
    )
    assert result.meta.num_completed == 2
    first, second = result.requests
    assert first.instance_id == second.instance_id
    assert second.arrival == pytest.approx(first.arrival + 1000)
    assert second.actual_hit_tokens >= 128
    assert result.meta.config.workload.mode == "trace"


def test_open_loop_target_qps_scales_merged_arrivals_without_changing_shape() -> None:
    source = TracePlan(
        sessions=("first", "second"), block_size=64,
        requests=(
            TraceTurn(request_id=i, session_id=session, arrival_offset_ms=offset,
                      prompt_len=129, output_len=1, hash_ids=(session * 10, session * 10 + 1))
            for i, session, offset in ((0, 0, 0), (1, 0, 900), (2, 1, 0), (3, 1, 3500))
        ),
    )
    baseline, _ = simulate(
        config(), replay=TraceReplay(source, mode="open_loop", session_rate=2, seed=7)
    )
    scaled, _ = simulate(
        config(),
        replay=TraceReplay(source, mode="open_loop", session_rate=2, seed=7, target_qps=5),
    )
    original = {request.id: request.arrival for request in baseline.requests}
    arrivals = {request.id: request.arrival for request in scaled.requests}
    first, last = min(original.values()), max(original.values())
    factor = (len(original) - 1) * 1000 / (5 * (last - first))
    for request_id, arrival in original.items():
        assert arrivals[request_id] == pytest.approx(first + (arrival - first) * factor)
    assert scaled.metrics.arrival_qps == pytest.approx(5)
    assert baseline.meta.num_requests == scaled.meta.num_requests == 4
    assert scaled.metrics.num_sessions == 2


@pytest.mark.parametrize(
    ("mode", "target"), [("session", 5), ("open_loop", 0), ("open_loop", float("inf"))]
)
def test_target_qps_only_accepts_positive_finite_open_loop(mode: str, target: float) -> None:
    with pytest.raises(ValueError, match="目标 QPS"):
        simulate(
            config(),
            replay=TraceReplay(plan(), mode=mode, session_rate=1, seed=7, target_qps=target),
        )


def test_target_qps_requires_a_nonzero_arrival_window() -> None:
    one = TracePlan(sessions=("only",), block_size=64, requests=(
        TraceTurn(request_id=0, session_id=0, arrival_offset_ms=0,
                  prompt_len=129, output_len=1, hash_ids=(0, 1)),
    ))
    with pytest.raises(ValueError, match="两个到达时刻不同"):
        simulate(
            config(),
            replay=TraceReplay(one, mode="open_loop", session_rate=1, seed=7, target_qps=5),
        )
    simultaneous = TracePlan(sessions=("only",), block_size=64, requests=(
        TraceTurn(request_id=i, session_id=0, arrival_offset_ms=0,
                  prompt_len=129, output_len=1, hash_ids=(0, 1)) for i in range(2)
    ))
    with pytest.raises(ValueError, match="两个到达时刻不同"):
        simulate(
            config(),
            replay=TraceReplay(
                simultaneous, mode="open_loop", session_rate=1, seed=7, target_qps=5
            ),
        )


def test_trace_closed_loop_uses_simulated_parent_completion() -> None:
    result, _ = simulate(
        config(), replay=TraceReplay(plan(), mode="session", session_rate=1, seed=7)
    )
    first, second = result.requests
    assert second.arrival == pytest.approx(first.finish + 20)
    assert second.actual_hit_tokens >= 128


def test_trace_zero_output_aborts_and_skips_only_closed_loop_descendants() -> None:
    closed, _ = simulate(
        config(), replay=TraceReplay(plan(output=0), mode="session", session_rate=1, seed=7)
    )
    assert closed.meta.num_aborted == 1
    assert closed.aborted_requests[0].reason == "invalid_trace_output"
    assert [item.id for item in closed.skipped_requests] == [1]
    opened, _ = simulate(
        config(), replay=TraceReplay(plan(output=0), mode="open_loop", session_rate=1, seed=7)
    )
    assert opened.meta.num_aborted == 1
    assert opened.meta.num_completed == 1
    assert opened.meta.num_skipped == 0


def test_trace_never_truncates_output_for_capacity() -> None:
    result, _ = simulate(
        config(capacity=256),
        replay=TraceReplay(plan(output=200), mode="open_loop", session_rate=1, seed=7),
    )
    assert result.aborted_requests[0].reason == "output_exceeds_kv_capacity"


def test_equal_raw_block_ids_do_not_share_kv_across_sessions() -> None:
    source = TracePlan(
        sessions=("first", "second"), block_size=64,
        requests=(
            TraceTurn(request_id=0, session_id=0, arrival_offset_ms=0,
                      prompt_len=129, output_len=1, hash_ids=(0, 1)),
            TraceTurn(request_id=1, session_id=1, arrival_offset_ms=0,
                      prompt_len=129, output_len=1, hash_ids=(0, 1)),
        ),
    )
    single = config().model_copy(update={"num_instances": 1})
    result, _ = simulate(
        single, replay=TraceReplay(source, mode="open_loop", session_rate=1, seed=7)
    )
    assert result.meta.num_completed == 2
    assert result.requests[1].arrival > result.requests[0].arrival
    assert all(record.actual_hit_tokens == 0 for record in result.requests)


def test_timed_root_ratio_counts_all_branches_without_changing_session_count() -> None:
    source = TracePlan(
        sessions=("one",), block_size=64,
        requests=(
            TraceTurn(request_id=0, session_id=0, arrival_offset_ms=0,
                      prompt_len=129, output_len=1, hash_ids=(0, 1)),
            TraceTurn(request_id=1, session_id=0, arrival_offset_ms=100,
                      prompt_len=129, output_len=1, hash_ids=(0, 1)),
        ),
    )
    result, _ = simulate(
        config(), replay=TraceReplay(source, mode="session", session_rate=2, seed=3)
    )
    assert result.metrics.num_sessions == 1
    assert result.metrics.root_request_ratio == 1.0


@pytest.mark.parametrize("timing", ["empirical", "analytical"])
@pytest.mark.parametrize("mode", ["open_loop", "session"])
def test_trace_replay_uses_resolved_capacity_and_timing_without_mutating_inputs(
    timing: Literal["empirical", "analytical"], mode: str,
) -> None:
    original = config(capacity=64)
    original_data = original.model_dump()
    request = ResolvedRequest(
        config=original,
        model=ModelSpec(
            name="toy", num_layers=2, hidden_size=16, num_attention_heads=2,
            num_kv_heads=1, head_dim=8, ffn_intermediate_size=32, vocab_size=16,
        ),
        hardware=HardwareSpec(
            name="toy", memory_bytes_per_gpu=2_000_000,
            effective_fp16_tflops=1, effective_hbm_gb_s=1,
        ),
        storage=StorageSpec(
            l1_available_gb=0.0001, l2_available_gb=0.000008192,
            l3_available_gb=0.000004096, runtime_reserve_bytes=1000,
            l2_to_l1_gb_s=0.001, l3_to_l2_gb_s=0.002,
            l2_latency_ms=3, l3_latency_ms=4, l3_lookup_latency_ms=1.25,
        ),
        timing=timing,
    )
    resolution = resolve_config(request)
    resolution_data = resolution.model_dump()
    source = plan()
    result, analysis = simulate(
        resolution, replay=TraceReplay(source, mode=mode, session_rate=1, seed=7)
    )

    assert result.meta.num_completed == 2
    assert result.meta.config.workload.mode == "trace"
    assert result.meta.config.workload.synthetic is None
    assert result.meta.config.engine == resolution.config.engine
    assert result.meta.config.perf_model == resolution.config.perf_model
    assert result.meta.config.engine.max_total_tokens == (
        resolution.capacity.l1_capacity_pages * source.block_size
    )
    assert result.meta.config.engine.max_total_tokens > original.engine.max_total_tokens
    assert result.meta.config.engine.l2_capacity_tokens == 128
    assert result.meta.config.engine.l3_capacity_tokens == 64
    assert result.meta.config.perf_model.kv_bytes_per_token == 64
    assert result.meta.config.perf_model.l2_to_l1_bandwidth_bytes_per_ms == 1000
    assert result.meta.config.perf_model.l3_to_l2_bandwidth_bytes_per_ms == 2000

    first, second = result.requests
    if mode == "open_loop":
        assert second.arrival == pytest.approx(first.arrival + 1000)
    else:
        assert second.arrival == pytest.approx(first.finish + 20)
    if timing == "empirical":
        assert analysis is None
        assert first.prefill_end - first.prefill_start == pytest.approx(129 * 0.01)
    else:
        assert analysis is not None
        assert analysis.steps == result.meta.num_engine_steps == 2
        assert analysis.flops > 0
        expected = step_cost(
            StepWorkload(
                phase="PREFILL", prefill=(PrefillWork(0, 0, 129, 129),), decode=(),
                allocated_pages=3, batch_requests=1,
            ),
            request.model, request.hardware,
        )
        assert first.prefill_end - first.prefill_start == pytest.approx(expected.duration_ms)
        assert analysis.device_ms == pytest.approx(
            sum(record.finish - record.prefill_start for record in result.requests)
        )

    assert original.model_dump() == original_data
    assert resolution.model_dump() == resolution_data
    assert original.workload.mode == resolution.config.workload.mode == "synthetic"


def test_trace_replay_is_frozen_and_has_default_seed_and_target_qps() -> None:
    source = plan()
    replay = TraceReplay(source, mode="open_loop", session_rate=1)
    assert replay.plan is source
    assert replay.seed == 0
    assert replay.target_qps is None
    with pytest.raises(FrozenInstanceError):
        replay.__setattr__("mode", "session")
