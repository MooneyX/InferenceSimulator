"""请求计划定时、依赖、同会话块复用及非法输出的内核回归。"""

import pytest

from simcore import SimConfig, simulate_plan
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    PerfModelConfig,
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
    result = simulate_plan(config(), plan(), mode="open_loop", session_rate=1, seed=7)
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
    baseline = simulate_plan(config(), source, mode="open_loop", session_rate=2, seed=7)
    scaled = simulate_plan(
        config(), source, mode="open_loop", session_rate=2, seed=7, target_qps=5
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
        simulate_plan(config(), plan(), mode=mode, session_rate=1, seed=7, target_qps=target)


def test_target_qps_requires_a_nonzero_arrival_window() -> None:
    one = TracePlan(sessions=("only",), block_size=64, requests=(
        TraceTurn(request_id=0, session_id=0, arrival_offset_ms=0,
                  prompt_len=129, output_len=1, hash_ids=(0, 1)),
    ))
    with pytest.raises(ValueError, match="两个到达时刻不同"):
        simulate_plan(config(), one, mode="open_loop", session_rate=1, seed=7, target_qps=5)
    simultaneous = TracePlan(sessions=("only",), block_size=64, requests=(
        TraceTurn(request_id=i, session_id=0, arrival_offset_ms=0,
                  prompt_len=129, output_len=1, hash_ids=(0, 1)) for i in range(2)
    ))
    with pytest.raises(ValueError, match="两个到达时刻不同"):
        simulate_plan(config(), simultaneous, mode="open_loop", session_rate=1,
                      seed=7, target_qps=5)


def test_trace_closed_loop_uses_simulated_parent_completion() -> None:
    result = simulate_plan(config(), plan(), mode="session", session_rate=1, seed=7)
    first, second = result.requests
    assert second.arrival == pytest.approx(first.finish + 20)
    assert second.actual_hit_tokens >= 128


def test_trace_zero_output_aborts_and_skips_only_closed_loop_descendants() -> None:
    closed = simulate_plan(config(), plan(output=0), mode="session", session_rate=1, seed=7)
    assert closed.meta.num_aborted == 1
    assert closed.aborted_requests[0].reason == "invalid_trace_output"
    assert [item.id for item in closed.skipped_requests] == [1]
    opened = simulate_plan(config(), plan(output=0), mode="open_loop", session_rate=1, seed=7)
    assert opened.meta.num_aborted == 1
    assert opened.meta.num_completed == 1
    assert opened.meta.num_skipped == 0


def test_trace_never_truncates_output_for_capacity() -> None:
    result = simulate_plan(
        config(capacity=256), plan(output=200), mode="open_loop", session_rate=1, seed=7
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
    result = simulate_plan(single, source, mode="open_loop", session_rate=1, seed=7)
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
    result = simulate_plan(config(), source, mode="session", session_rate=2, seed=3)
    assert result.metrics.num_sessions == 1
    assert result.metrics.root_request_ratio == 1.0
