"""F001 完成/失败/跳过结果、时延、首次命中和物理容量指标。"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from simcore.engine import Engine, run_engine
from simcore.kv_cache import L1Snapshot
from simcore.metrics import compute_metrics
from simcore.runner import simulate
from simcore.schemas import (
    AbortedRequest,
    EngineConfig,
    FixedLength,
    InferenceRequest,
    Metrics,
    PerfModelConfig,
    RequestRecord,
    SessionConfig,
    SimConfig,
    SimMeta,
    SimResult,
    SkippedRequest,
    SyntheticWorkloadConfig,
    TurnPlan,
    WorkloadConfig,
    WorkloadPlan,
)


def config() -> SimConfig:
    return SimConfig(
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=4,
                arrival_rate=10,
                prompt_len=FixedLength(value=5),
                output_len=FixedLength(value=1),
                seed=7,
                session=SessionConfig(num_sessions=2, target_ideal_prefix_hit_rate=4 / 26),
            )
        ),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=16, page_size=4),
        perf_model=PerfModelConfig(prefill_c1=1, decode_d1=0, decode_d2=0),
    )


def test_direct_engine_and_run_engine_reject_missing_transfer_config() -> None:
    engine = EngineConfig(
        max_running_requests=1, max_total_tokens=8, page_size=4, l2_capacity_tokens=4
    )
    perf = PerfModelConfig(prefill_c1=1, decode_d1=0, decode_d2=0)
    with pytest.raises(ValueError, match="kv_bytes_per_token"):
        Engine(engine, perf, output_targets={})
    with pytest.raises(ValueError, match="kv_bytes_per_token"):
        run_engine(engine, perf, [])


def mixed_plan() -> WorkloadPlan:
    turns = {
        0: TurnPlan(0, 0, None, 0, None, 4, 1, 2, 0),
        1: TurnPlan(1, 0, 0, None, 0, 5, 1, 2, 1),
        2: TurnPlan(2, 1, None, 0, None, 12, 0, 0, 0),
        3: TurnPlan(3, 1, 2, None, 0, 5, 0, 0, 0),
    }
    return WorkloadPlan(
        turns,
        (2, 0),
        {0: (1,), 2: (3,)},
        constructed_ideal_prefix_hit_rate=4 / 26,
    )


def test_mixed_result_lists_status_counts_and_only_success_metrics() -> None:
    settings = config()
    with patch("simcore.runner.build_plan", return_value=mixed_plan()):
        result = simulate(settings)
        second = simulate(settings)
    assert result.model_dump_json() == second.model_dump_json()
    assert (result.meta.num_requests, result.meta.num_completed) == (4, 2)
    assert (result.meta.num_aborted, result.meta.num_skipped, result.meta.num_engine_steps) == (
        1,
        1,
        2,
    )
    assert [request.id for request in result.requests] == [0, 1]
    assert [request.id for request in result.aborted_requests] == [2]
    assert [request.id for request in result.skipped_requests] == [3]
    aborted = result.aborted_requests[0]
    assert (aborted.arrival, aborted.failed_at, aborted.reason, aborted.generated_tokens) == (
        0,
        0,
        "input_exceeds_kv_capacity",
        0,
    )
    assert result.skipped_requests[0].parent_request_id == 2
    assert "arrival" not in result.skipped_requests[0].model_dump()
    assert (result.requests[0].prefill_end, result.requests[0].finish) == (4, 4)
    assert (result.requests[1].arrival, result.requests[1].prefill_end) == (4, 5)
    assert result.metrics.ttft_mean == 2.5
    assert result.metrics.ttft_p50 == 2.5
    assert result.metrics.ttft_p99 == pytest.approx(3.97)
    assert result.metrics.tpot_mean is None
    assert (result.metrics.makespan, result.metrics.throughput) == (5, 400)
    assert result.metrics.total_input_tokens == 9
    assert result.metrics.ideal_hit_tokens == result.metrics.actual_hit_tokens == 4
    assert result.metrics.actual_prefill_tokens == 5
    assert result.metrics.replayed_prefill_tokens == 0
    assert result.metrics.l1_capacity_blocks == 4
    assert (
        result.metrics.l1_free_blocks,
        result.metrics.l1_resident_blocks,
        result.metrics.l1_active_blocks,
        result.metrics.l1_evictable_blocks,
    ) == (3, 1, 0, 1)
    assert result.metrics.l1_peak_resident_blocks == 2
    assert result.metrics.l1_reserved_future_blocks == 0
    assert SimResult.model_validate_json(result.model_dump_json()) == result


def test_zero_duration_success_has_no_throughput_but_defined_tpot() -> None:
    settings = SimConfig(
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=1,
                arrival_rate=10,
                prompt_len=FixedLength(value=4),
                output_len=FixedLength(value=2),
                seed=7,
            )
        ),
        engine=EngineConfig(max_running_requests=1, max_total_tokens=16, page_size=4),
        perf_model=PerfModelConfig(prefill_c1=0, decode_d1=0, decode_d2=0),
    )
    result = simulate(settings)
    assert (result.meta.num_completed, result.meta.num_engine_steps) == (1, 2)
    assert result.metrics.makespan == 0 and result.metrics.throughput is None
    assert result.metrics.tpot_mean == 0
    assert result.metrics.ttft_mean == 0


def test_failed_before_success_is_counted_in_makespan_not_ttft_denominator() -> None:
    successful = RequestRecord(
        id=3,
        arrival=10,
        prefill_start=12,
        prefill_end=14,
        finish=20,
        prompt_len=4,
        output_len=3,
        actual_prefill_tokens=4,
    )
    metrics = compute_metrics([successful], earliest_sent_arrival=1, replayed_prefill_tokens=7)
    assert (metrics.ttft_mean, metrics.ttft_p50, metrics.ttft_p99) == (4, 4, 4)
    assert metrics.tpot_mean == 3  # 只计算成功请求的后续两个输出 token
    assert metrics.makespan == 19
    assert metrics.throughput == pytest.approx(1000 / 19)
    assert metrics.total_input_tokens == metrics.actual_prefill_tokens == 4
    assert metrics.replayed_prefill_tokens == 7


def test_no_success_metrics_are_undefined_but_physical_pages_remain_counted() -> None:
    snapshot = L1Snapshot(4, 1, 0, 1, 3, 0)
    metrics = compute_metrics([], cache_snapshot=snapshot, replayed_prefill_tokens=11)
    assert (metrics.ttft_mean, metrics.ttft_p50, metrics.ttft_p99) == (None, None, None)
    assert metrics.tpot_mean is None and metrics.throughput is None
    assert metrics.makespan == 0 and metrics.total_input_tokens == 0
    assert metrics.replayed_prefill_tokens == 11
    assert (metrics.l1_free_blocks, metrics.l1_resident_blocks) == (3, 1)
    assert metrics.l1_reserved_future_blocks == 0


def test_replay_tokens_from_request_that_later_aborts_remain_in_total() -> None:
    cfg = EngineConfig(max_running_requests=1, max_total_tokens=12, page_size=4)
    perf = PerfModelConfig(prefill_c1=1, decode_d1=0, decode_d2=0)
    engine = Engine(cfg, perf, output_targets={0: 3})
    req = InferenceRequest(0, 0, None, 3, 3, (), 0)
    engine.submit(req, 0)
    engine.step(0, 0)
    engine.step(3, 1)
    assert engine._prefix_cache.state(0).k == 4 and engine._running[0] == 2
    engine._prefix_cache.withdraw(0)
    del engine._running[0]
    engine._retry_counts[0] = 1
    engine._prefix_scheduler.requeue(req, 3, generated_tokens=2)
    assert engine.step(3, 2) == (4, ())
    assert engine._replayed_tokens[0] == 4
    holder = InferenceRequest(99, 0, None, 8, 1, (), 0)
    engine._prefix_cache.pin(holder, 0)
    engine._prefix_cache.admit(99)
    engine._prefix_cache.extend(99, 8)
    assert engine.step(7, 3) == (0, ())
    engine._prefix_cache.withdraw(99)
    run = engine.finalize()
    assert run.records == () and run.aborted_requests[0].retry_count == 1
    assert run.aborted_requests[0].generated_tokens == 2
    assert run.replayed_prefill_tokens == 4
    metrics = compute_metrics(
        run.records,
        cache_snapshot=run.cache_snapshot,
        l1_peak_resident_blocks=run.l1_peak_resident_blocks,
        replayed_prefill_tokens=run.replayed_prefill_tokens,
    )
    assert metrics.replayed_prefill_tokens == 4
    assert metrics.total_input_tokens == metrics.actual_prefill_tokens == 0


def test_status_result_rejects_duplicates_and_successful_parent_skip() -> None:
    settings = config()
    record = RequestRecord(
        id=0,
        arrival=0,
        prefill_start=0,
        prefill_end=1,
        finish=1,
        prompt_len=4,
        output_len=1,
        actual_prefill_tokens=4,
    )
    aborted = AbortedRequest(
        id=0,
        arrival=0,
        failed_at=0,
        reason="input_exceeds_kv_capacity",
        generated_tokens=0,
        retry_count=0,
    )
    metrics = Metrics(ttft_mean=1, ttft_p50=1, ttft_p99=1, makespan=1)
    meta = SimMeta(
        config=settings,
        num_requests=2,
        num_completed=1,
        num_aborted=1,
        num_skipped=0,
        num_engine_steps=1,
        sim_start=0,
        sim_end=1,
    )
    with pytest.raises(ValidationError, match="不可同时"):
        SimResult(meta=meta, metrics=metrics, requests=[record], aborted_requests=[aborted])
    meta = meta.model_copy(update={"num_aborted": 0, "num_skipped": 1})
    with pytest.raises(ValidationError, match="父轮"):
        SimResult(
            meta=meta,
            metrics=metrics,
            requests=[record],
            skipped_requests=[SkippedRequest(id=1, parent_request_id=0)],
        )
    meta = meta.model_copy(update={"num_requests": 3, "num_completed": 2})
    with pytest.raises(ValidationError, match="严格升序"):
        SimResult(
            meta=meta,
            metrics=metrics,
            requests=[record, record],
            skipped_requests=[SkippedRequest(id=1, parent_request_id=0)],
        )


def test_result_rejects_metrics_inconsistent_with_success_count() -> None:
    settings = config()
    failed = AbortedRequest(
        id=0,
        arrival=0,
        failed_at=0,
        reason="input_exceeds_kv_capacity",
        generated_tokens=0,
        retry_count=0,
    )
    no_success = SimMeta(
        config=settings,
        num_requests=1,
        num_completed=0,
        num_aborted=1,
        num_skipped=0,
        num_engine_steps=0,
        sim_start=0,
        sim_end=0,
    )
    with pytest.raises(ValidationError, match="无成功"):
        SimResult(
            meta=no_success,
            metrics=Metrics(ttft_mean=0, ttft_p50=0, ttft_p99=0, makespan=0),
            aborted_requests=[failed],
        )
    succeeded = RequestRecord(
        id=0,
        arrival=0,
        prefill_start=0,
        prefill_end=1,
        finish=1,
        prompt_len=4,
        output_len=1,
        actual_prefill_tokens=4,
    )
    one_success = no_success.model_copy(
        update={
            "num_completed": 1,
            "num_aborted": 0,
            "num_engine_steps": 1,
            "sim_end": 1,
        }
    )
    with pytest.raises(ValidationError, match="有成功"):
        SimResult(
            meta=one_success,
            metrics=Metrics(ttft_mean=None, ttft_p50=None, ttft_p99=None, makespan=1),
            requests=[succeeded],
        )
    with pytest.raises(ValidationError, match="吞吐"):
        SimResult(
            meta=one_success,
            metrics=Metrics(ttft_mean=1, ttft_p50=1, ttft_p99=1, makespan=1),
            requests=[succeeded],
        )
    with pytest.raises(ValidationError, match="TPOT"):
        SimResult(
            meta=one_success,
            metrics=Metrics(
                ttft_mean=1,
                ttft_p50=1,
                ttft_p99=1,
                tpot_mean=1,
                throughput=1000,
                makespan=1,
            ),
            requests=[succeeded],
        )


@pytest.mark.parametrize(
    "override",
    [
        {"l1_reserved_future_blocks": 1},
        {"l1_evictable_blocks": 0},
        {"l1_free_blocks": 0},
        {"l1_peak_resident_blocks": 0},
    ],
)
def test_physical_capacity_rejects_future_reservation_or_inconsistent_counts(
    override: dict[str, int],
) -> None:
    valid = {
        "ttft_mean": None,
        "ttft_p50": None,
        "ttft_p99": None,
        "makespan": 0,
        "l1_capacity_blocks": 4,
        "l1_free_blocks": 3,
        "l1_resident_blocks": 1,
        "l1_active_blocks": 0,
        "l1_evictable_blocks": 1,
        "l1_peak_resident_blocks": 1,
    }
    assert Metrics.model_validate(valid).l1_capacity_blocks == 4
    with pytest.raises(ValidationError, match="守恒"):
        Metrics.model_validate({**valid, **override})


@pytest.mark.parametrize("tier", ["l2", "l3"])
@pytest.mark.parametrize("field", ["free", "active", "peak_resident"])
def test_lower_tier_capacity_rejects_inconsistent_counts(tier: str, field: str) -> None:
    valid = {
        "ttft_mean": None,
        "ttft_p50": None,
        "ttft_p99": None,
        "makespan": 0,
        f"{tier}_capacity_blocks": 3,
        f"{tier}_resident_blocks": 1,
        f"{tier}_active_blocks": 0,
        f"{tier}_evictable_blocks": 1,
        f"{tier}_free_blocks": 2,
        f"{tier}_peak_resident_blocks": 2,
    }
    assert Metrics.model_validate(valid).model_dump()[f"{tier}_resident_blocks"] == 1
    with pytest.raises(ValidationError, match="容量不守恒"):
        Metrics.model_validate({**valid, f"{tier}_{field}_blocks": 1 if field == "active" else 0})
