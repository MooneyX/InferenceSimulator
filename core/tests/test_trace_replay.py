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
