"""请求计划定时、依赖、同会话块复用及非法输出的内核回归。"""

from typing import Literal

import pytest
from pydantic import ValidationError

from simcore import SimConfig, simulate
from simcore.analytical import step_cost
from simcore.capacity import prepare_runtime, resolve_config
from simcore.perf_model import PrefillWork, StepWorkload
from simcore.schemas import (
    AnalyticalTimingConfig,
    EngineConfig,
    HardwareSpec,
    InferenceConfig,
    ModelSpec,
    PerfModelConfig,
    SharedL3Config,
    StorageSpec,
    TracePlan,
    TraceTurn,
    TraceWorkloadConfig,
    WorkloadConfig,
)
from test_schemas import resolved_config


def config(
    *,
    capacity: int = 2048,
    mode: Literal["open_loop", "closed_loop"] = "open_loop",
    session_rate: float = 1,
    seed: int = 7,
    target_qps: float | None = None,
    max_output_tokens: int = 65_536,
) -> SimConfig:
    return resolved_config(
        num_instances=2,
        workload=WorkloadConfig(
            mode="trace",
            arrival_mode=mode,
            session_rate=session_rate,
            seed=seed,
            target_qps=target_qps,
            trace=TraceWorkloadConfig(plan_id="test-plan"),
        ),
        engine=EngineConfig(
            max_running_requests=4,
            max_total_tokens=capacity,
            max_output_tokens=max_output_tokens,
            page_size=64,
        ),
        perf_model=PerfModelConfig(prefill_c1=0.01, decode_d1=0.01, decode_d2=0),
    )


def plan(*, output: int = 1, second_offset: float = 1000) -> TracePlan:
    return TracePlan(
        sessions=("original-session",),
        block_size=64,
        requests=(
            TraceTurn(
                request_id=0,
                session_id=0,
                arrival_offset_ms=0,
                prompt_len=129,
                output_len=output,
                hash_ids=(0, 1),
            ),
            TraceTurn(
                request_id=1,
                session_id=0,
                parent_request_id=0,
                arrival_offset_ms=second_offset,
                think_time_ms=20,
                prompt_len=193,
                output_len=1,
                hash_ids=(0, 1, 2),
            ),
        ),
    )


@pytest.mark.parametrize("mode", ["open_loop", "closed_loop"])
@pytest.mark.parametrize("output", [0, 1])
def test_trace_progress_counts_turns_and_terminal_outcomes(mode, output) -> None:
    snapshots = []
    source = plan(output=output)
    result = simulate(config(mode=mode), trace_plan=source, on_progress=snapshots.append)
    assert all(item.total == len(source.requests) for item in snapshots)
    assert snapshots[0].completed == snapshots[0].aborted == snapshots[0].skipped == 0
    assert snapshots[-1].model_dump() == {
        "total": 2, "completed": result.meta.num_completed,
        "aborted": result.meta.num_aborted, "skipped": result.meta.num_skipped,
    }
    assert sum(snapshots[-1].model_dump().values()) == 4


def test_trace_open_loop_reuses_input_pages_on_same_instance() -> None:
    original = config().model_copy(update={"routing_policy": "cache_affinity"})
    original_data = original.model_dump()
    result = simulate(original, trace_plan=plan())
    assert result.meta.config.model_dump() == original_data
    assert original.model_dump() == original_data
    assert result.meta.num_completed == 2
    first, second = result.requests
    assert first.instance_id == second.instance_id
    assert second.arrival == pytest.approx(first.arrival + 1000)
    assert second.actual_hit_tokens >= 128
    assert result.meta.config.workload.mode == "trace"


def test_open_loop_target_qps_scales_merged_arrivals_without_changing_shape() -> None:
    source = TracePlan(
        sessions=("first", "second"),
        block_size=64,
        requests=tuple(
            TraceTurn(
                request_id=i,
                session_id=session,
                arrival_offset_ms=offset,
                prompt_len=129,
                output_len=1,
                hash_ids=(session * 10, session * 10 + 1),
            )
            for i, session, offset in ((0, 0, 0), (1, 0, 900), (2, 1, 0), (3, 1, 3500))
        ),
    )
    baseline = simulate(config(session_rate=2), trace_plan=source)
    scaled_config = config(session_rate=2, target_qps=5)
    scaled = simulate(scaled_config, trace_plan=source)
    assert scaled.meta.config == scaled_config
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
    ("mode", "target"), [("closed_loop", 5), ("open_loop", 0), ("open_loop", float("inf"))]
)
def test_target_qps_only_accepts_positive_finite_open_loop(
    mode: Literal["open_loop", "closed_loop"],
    target: float,
) -> None:
    with pytest.raises(ValueError, match=r"目标 QPS|target_qps"):
        simulate(config(mode=mode, target_qps=target), trace_plan=plan())


def test_target_qps_requires_a_nonzero_arrival_window() -> None:
    one = TracePlan(
        sessions=("only",),
        block_size=64,
        requests=(
            TraceTurn(
                request_id=0,
                session_id=0,
                arrival_offset_ms=0,
                prompt_len=129,
                output_len=1,
                hash_ids=(0, 1),
            ),
        ),
    )
    with pytest.raises(ValueError, match="两个到达时刻不同"):
        simulate(config(target_qps=5), trace_plan=one)
    simultaneous = TracePlan(
        sessions=("only",),
        block_size=64,
        requests=tuple(
            TraceTurn(
                request_id=i,
                session_id=0,
                arrival_offset_ms=0,
                prompt_len=129,
                output_len=1,
                hash_ids=(0, 1),
            )
            for i in range(2)
        ),
    )
    with pytest.raises(ValueError, match="两个到达时刻不同"):
        simulate(config(target_qps=5), trace_plan=simultaneous)


def test_trace_closed_loop_uses_simulated_parent_completion() -> None:
    settings = config(mode="closed_loop").model_copy(update={"routing_policy": "cache_affinity"})
    result = simulate(settings, trace_plan=plan())
    first, second = result.requests
    assert second.arrival == pytest.approx(first.finish + 20)
    assert second.actual_hit_tokens >= 128


def test_trace_zero_output_aborts_and_skips_only_closed_loop_descendants() -> None:
    closed = simulate(config(mode="closed_loop"), trace_plan=plan(output=0))
    assert closed.meta.num_aborted == 1
    assert closed.aborted_requests[0].reason == "invalid_trace_output"
    assert [item.id for item in closed.skipped_requests] == [1]
    opened = simulate(config(), trace_plan=plan(output=0))
    assert opened.meta.num_aborted == 1
    assert opened.meta.num_completed == 1
    assert opened.meta.num_skipped == 0


def test_trace_never_truncates_output_for_capacity() -> None:
    source = plan(output=200)
    result = simulate(config(capacity=256), trace_plan=source)
    assert source.requests[0].output_len == 200
    assert result.aborted_requests[0].reason == "decode_kv_exhausted"
    assert result.aborted_requests[0].generated_tokens == 128
    assert result.aborted_requests[1].reason == "input_exceeds_kv_capacity"
    assert result.meta.num_aborted == 2 and result.meta.num_skipped == 0


@pytest.mark.parametrize("mode", ["open_loop", "closed_loop"])
@pytest.mark.parametrize("cap", [64, 100])
def test_trace_output_exceeding_engine_limit_aborts_at_arrival(
    mode: Literal["open_loop", "closed_loop"],
    cap: int,
) -> None:
    source = plan(output=cap + 1)
    result = simulate(config(mode=mode, max_output_tokens=cap), trace_plan=source)
    failure = result.aborted_requests[0]
    assert failure.reason == "output_exceeds_engine_limit"
    assert failure.failed_at == failure.arrival and failure.generated_tokens == 0
    assert source.requests[0].output_len == cap + 1
    assert result.meta.num_completed == (1 if mode == "open_loop" else 0)
    assert [item.id for item in result.skipped_requests] == ([] if mode == "open_loop" else [1])


@pytest.mark.parametrize("cap", [64, 100])
def test_trace_output_at_engine_cap_is_preserved(cap: int) -> None:
    result = simulate(config(max_output_tokens=cap), trace_plan=plan(output=cap))
    assert result.meta.num_completed == 2
    assert result.requests[0].output_len == cap


def test_equal_raw_block_ids_do_not_share_kv_across_sessions() -> None:
    source = TracePlan(
        sessions=("first", "second"),
        block_size=64,
        requests=(
            TraceTurn(
                request_id=0,
                session_id=0,
                arrival_offset_ms=0,
                prompt_len=129,
                output_len=1,
                hash_ids=(0, 1),
            ),
            TraceTurn(
                request_id=1,
                session_id=1,
                arrival_offset_ms=0,
                prompt_len=129,
                output_len=1,
                hash_ids=(0, 1),
            ),
        ),
    )
    single = config().model_copy(update={"num_instances": 1})
    result = simulate(single, trace_plan=source)
    assert result.meta.num_completed == 2
    assert result.requests[1].arrival > result.requests[0].arrival
    assert all(record.actual_hit_tokens == 0 for record in result.requests)


def test_timed_root_ratio_counts_all_branches_without_changing_session_count() -> None:
    source = TracePlan(
        sessions=("one",),
        block_size=64,
        requests=(
            TraceTurn(
                request_id=0,
                session_id=0,
                arrival_offset_ms=0,
                prompt_len=129,
                output_len=1,
                hash_ids=(0, 1),
            ),
            TraceTurn(
                request_id=1,
                session_id=0,
                arrival_offset_ms=100,
                prompt_len=129,
                output_len=1,
                hash_ids=(0, 1),
            ),
        ),
    )
    result = simulate(config(mode="closed_loop", session_rate=2, seed=3), trace_plan=source)
    assert result.metrics.num_sessions == 1
    assert result.metrics.root_request_ratio == 1.0


@pytest.mark.parametrize("timing", ["empirical", "analytical"])
@pytest.mark.parametrize("mode", ["open_loop", "closed_loop"])
def test_trace_replay_uses_resolved_capacity_and_timing_without_mutating_inputs(
    timing: Literal["empirical", "analytical"],
    mode: Literal["open_loop", "closed_loop"],
) -> None:
    baseline = config(capacity=64, mode=mode)
    original = baseline.model_copy(
        update={
            "shared_l3": SharedL3Config(available_gb=0.000004096, lookup_latency_ms=1.25),
            "inference": InferenceConfig(
                page_size=64,
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
                    l1_available_gb=0.0001,
                    l2_available_gb=0.000008192,
                    runtime_reserve_bytes=1000,
                    l2_to_l1_gb_s=0.001,
                    l3_to_l2_gb_s=0.002,
                    l2_latency_ms=3,
                    l3_latency_ms=4,
                ),
                timing=(
                    baseline.inference.timing if timing == "empirical" else AnalyticalTimingConfig()
                ),
            ),
        }
    )
    original_data = original.model_dump()
    resolution = resolve_config(original)
    resolution_data = resolution.model_dump()
    runtime = prepare_runtime(original)
    source = plan()
    source_data = source.model_dump()
    result = simulate(original, trace_plan=source)
    analysis = result.analysis

    assert result.meta.num_completed == 2
    assert result.meta.config == original
    assert result.meta.config.model_dump() == original_data
    assert result.meta.config.workload.mode == "trace"
    assert result.meta.config.workload.synthetic is None
    trace = result.meta.config.workload.trace
    assert trace is not None and trace.plan_id == "test-plan"
    assert result.resolution == resolution
    assert (
        runtime.engine.max_total_tokens
        == resolution.max_total_tokens
        == (resolution.capacity.l1_capacity_pages * source.block_size)
    )
    assert runtime.engine.max_total_tokens > resolve_config(baseline).max_total_tokens
    assert runtime.engine.l2_capacity_tokens == resolution.l2_capacity_tokens == 128
    assert runtime.engine.l3_capacity_tokens == resolution.l3_capacity_tokens == 64
    assert runtime.perf_model.kv_bytes_per_token == 64
    assert runtime.perf_model.l2_to_l1_bandwidth_bytes_per_ms == 1000
    assert runtime.perf_model.l3_to_l2_bandwidth_bytes_per_ms == 2000

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
                phase="PREFILL",
                prefill=(PrefillWork(0, 0, 129, 129),),
                decode=(),
                allocated_pages=3,
                batch_requests=1,
            ),
            original.inference.model,
            original.inference.hardware,
        )
        assert first.prefill_end - first.prefill_start == pytest.approx(expected.duration_ms)
        assert analysis.device_ms == pytest.approx(
            sum(record.finish - record.prefill_start for record in result.requests)
        )

    assert original.model_dump() == original_data
    assert resolution.model_dump() == resolution_data
    assert source.model_dump() == source_data
    assert original.workload.mode == "trace"


def test_trace_workload_parent_is_frozen_and_has_default_seed_and_target_qps() -> None:
    workload = WorkloadConfig(mode="trace", session_rate=1, trace=TraceWorkloadConfig())
    assert workload.seed == 0 and workload.target_qps is None
    assert workload.arrival_mode == "closed_loop"
    assert workload.trace is not None and workload.trace.model_dump() == {"plan_id": None}
    with pytest.raises(ValidationError, match="frozen"):
        workload.__setattr__("arrival_mode", "open_loop")
    with pytest.raises(ValidationError, match="frozen"):
        workload.trace.__setattr__("plan_id", "changed")
