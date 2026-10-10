"""F001 Decode 缺页、回撤重算与请求级失败的闭环验收。"""

from simcore.engine import Engine, run_engine
from simcore.kv_cache import L1KVCache, TieredKVCache
from simcore.metrics import compute_metrics
from simcore.prefix_hash import encode
from simcore.runner import simulate
from simcore.scheduler import F001Scheduler
from simcore.schemas import (
    Arrival,
    EngineConfig,
    FixedLength,
    InferenceRequest,
    PerfModelConfig,
    SessionConfig,
    SimConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)

PERF = PerfModelConfig(prefill_c1=1, decode_d1=5, decode_d2=1)


def request(key: int, *, prompt: int = 4, maximum: int = 3) -> InferenceRequest:
    return InferenceRequest(key, key, None, prompt, maximum, (), 0)


def occupy(cache: TieredKVCache, key: int, length: int) -> None:
    holder = request(key, prompt=length)
    cache.pin(holder, 0)
    cache.admit(key)
    cache.extend(key, length)
    cache.publish(key)


def test_decode_evicts_cached_pages_before_considering_withdrawal() -> None:
    cfg = EngineConfig(max_running_requests=2, max_total_tokens=12, page_size=4)
    engine = Engine(cfg, PERF, output_targets={0: 2})
    item = request(0, maximum=2)
    engine.submit(item, 0)
    assert engine.step(0, 0) == (4, ())
    occupy(engine._prefix_cache, 99, 4)
    engine._prefix_cache.finish(99)
    occupy(engine._prefix_cache, 98, 4)
    engine._prefix_cache.finish(98)
    assert engine._prefix_cache.free_pages == 0
    assert engine.step(4, 1) == (10, (0,))
    trace = list(engine._log.steps())[-1]
    assert trace.evicted_blocks == 1 and trace.withdrawn == ()
    assert engine._prefix_cache.evictions == 1
    engine.complete(14, 0)
    assert engine.finalize().aborted_requests == ()


def test_decode_withdraws_when_lower_tier_is_protected_by_inflight_load() -> None:
    cfg = EngineConfig(
        max_running_requests=2, max_total_tokens=12, page_size=4, l2_capacity_tokens=4
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=5,
        decode_d2=1,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
    )
    engine = Engine(cfg, perf, output_targets={0: 2})
    engine.submit(request(0, maximum=2), 0)
    assert engine.step(0, 0) == (4, ())
    cache = engine._prefix_cache
    for key in (99, 98):
        occupy(cache, key, 4)
        cache.finish(key)
    cache.ensure_free(1)
    source = encode(99, 0)
    cached = cache.page(source)
    assert cached is not None and cached.tier == "L2"
    cache.reserve_transfer(source, "L1")
    assert engine.step(4, 1) == (0, ())
    assert engine._aborted[0].reason == "decode_kv_exhausted"
    cache.cancel_transfer(source)


def test_withdraws_deterministically_then_replays_without_sampling_again() -> None:
    cfg = EngineConfig(max_running_requests=3, max_total_tokens=32, page_size=4)
    engine = Engine(cfg, PERF, output_targets={0: 3, 1: 3})
    engine.submit(request(0, prompt=4), 0)
    engine.submit(request(1, prompt=8), 0)
    assert engine.step(0, 0) == (12, ())
    occupy(engine._prefix_cache, 99, 20)
    assert engine._prefix_cache.free_pages == 0
    assert engine.step(12, 1) == (10, ())
    step = list(engine._log.steps())[-1]
    assert step.withdrawn == (1,) and step.evicted_blocks == 1
    assert engine._prefix_scheduler.ratio == 1.0
    assert engine._prefix_scheduler.num_waiting == 1
    assert engine._prefix_cache.free_pages + engine._prefix_cache.resident_pages == 8
    engine._prefix_cache.withdraw(99)
    assert engine.step(22, 2) == (8, ())
    assert list(engine._log.steps())[-1].phase == "PREFILL"
    assert engine._running[1] == 1 and engine._prefix_cache.state(1).k == 8
    assert engine._prefill_end[1] == 12
    assert engine._replayed_tokens[1] == 8
    duration, finished = engine.step(30, 3)
    assert duration == 20 and finished == (0,)
    engine.complete(50, 0)
    assert engine.step(50, 4) == (15, (1,))
    engine.complete(65, 1)
    run = engine.finalize()
    assert run.records[1].retry_count == 1
    assert run.records[1].replayed_prefill_tokens == 8
    assert run.records[1].actual_prefill_tokens == 8
    assert run.records[1].prefill_start == 0 and run.records[1].prefill_end == 12
    assert run.replayed_prefill_tokens == 8


def test_evictable_pages_are_removed_before_withdrawing_active_requests() -> None:
    cfg = EngineConfig(max_running_requests=3, max_total_tokens=24, page_size=4)
    engine = Engine(cfg, PERF, output_targets={0: 3, 1: 3})
    engine.submit(request(0), 0)
    engine.submit(request(1), 0)
    engine.step(0, 0)
    occupy(engine._prefix_cache, 98, 4)
    engine._prefix_cache.finish(98)
    occupy(engine._prefix_cache, 99, 12)
    assert engine._prefix_cache.free_pages == 0
    assert engine.step(8, 1) == (10, ())
    trace = list(engine._log.steps())[-1]
    assert trace.evicted_blocks == 1 and trace.withdrawn == (0,)
    assert engine._prefix_cache.page("chunk:98:0") is None
    assert engine._prefix_cache.page("chunk:0:0") is not None


def test_withdrawal_tie_breaks_by_smaller_request_id() -> None:
    cfg = EngineConfig(max_running_requests=3, max_total_tokens=24, page_size=4)
    engine = Engine(cfg, PERF, output_targets={0: 3, 1: 3})
    engine.submit(request(0), 0)
    engine.submit(request(1), 0)
    assert engine.step(0, 0) == (8, ())
    occupy(engine._prefix_cache, 99, 16)
    assert engine.step(8, 1) == (10, ())
    assert list(engine._log.steps())[-1].withdrawn == (0,)
    assert engine._running == {1: 2}
    assert engine._retry_counts[0] == 1


def test_fused_mode_fixes_decode_set_before_prefill_admission() -> None:
    cfg = EngineConfig(
        max_running_requests=3,
        max_total_tokens=32,
        page_size=4,
        pd_mutually_exclusive=False,
    )
    engine = Engine(cfg, PERF, output_targets={0: 3, 1: 3, 2: 1})
    engine.submit(request(0, prompt=4), 0)
    engine.submit(request(1, prompt=8), 0)
    engine.step(0, 0)
    occupy(engine._prefix_cache, 99, 20)
    engine.submit(request(2, maximum=1), 12)
    assert engine.step(12, 1) == (10, ())
    step = list(engine._log.steps())[-1]
    assert step.phase == "DECODE"
    assert step.num_decoding == 1 and step.batch_total_tokens == 5
    assert step.admitted == ()
    assert list(engine._log.steps())[-1].withdrawn == (1,)
    assert engine._prefix_scheduler.num_waiting == 2
    assert 1 not in engine._running  # 本步回撤者不能重新被 Prefill 准入


def test_restore_with_two_sampled_tokens_recomputes_only_p_plus_one() -> None:
    cfg = EngineConfig(max_running_requests=1, max_total_tokens=128, page_size=4)
    engine = Engine(cfg, PERF, output_targets={0: 3})
    item = request(0, maximum=10)
    engine.submit(item, 0)
    engine.step(0, 0)
    engine.step(4, 1)
    assert engine._running[0] == 2 and engine._prefix_cache.state(0).k == 5
    engine._prefix_cache.withdraw(0)
    del engine._running[0]
    engine._retry_counts[0] = 1
    engine._prefix_scheduler.requeue(item, 14, generated_tokens=2)
    assert engine.step(14, 2) == (1, ())
    step = list(engine._log.steps())[-1]
    assert step.phase == "PREFILL" and step.admitted == (0,)
    assert step.batch_total_tokens == 5 and step.matched_cache_tokens == 4
    assert engine._prefix_cache.state(0).k == 5 and engine._running[0] == 2
    assert engine._prefill_end[0] == 4
    assert engine.step(15, 3) == (11, (0,))
    engine.complete(26, 0)
    record = engine.finalize().records[0]
    assert (record.output_len, record.retry_count, record.replayed_prefill_tokens) == (3, 1, 1)
    assert record.actual_hit_tokens == 0 and record.actual_prefill_tokens == 4


def test_full_hit_restores_at_boundary_without_empty_engine_step() -> None:
    cfg = EngineConfig(max_running_requests=1, max_total_tokens=128, page_size=4)
    engine = Engine(cfg, PERF, output_targets={0: 6})
    item = request(0, maximum=6)
    engine.submit(item, 0)
    engine._prefix_scheduler.select_prefill({})
    engine._prefix_cache.admit(0)
    engine._prefix_cache.extend(0, 8)
    engine._prefix_cache.publish(0)
    engine._prefix_cache.withdraw(0)
    engine._prefix_scheduler.requeue(item, 0, generated_tokens=5)
    duration, finished = engine.step(0, 0)
    assert duration == 0 and finished == ()
    assert engine.num_steps == 0
    assert engine._running[0] == 5 and engine._prefix_cache.state(0).k == 8
    assert engine.step(0, 0) == (14, (0,))


def test_single_request_that_cannot_decode_is_aborted_instead_of_looping() -> None:
    cfg = EngineConfig(max_running_requests=1, max_total_tokens=12, page_size=4)
    engine = Engine(cfg, PERF, output_targets={0: 2})
    engine.submit(request(0, maximum=2), 0)
    engine.step(0, 0)
    occupy(engine._prefix_cache, 99, 8)
    assert engine.step(4, 1) == (0, ())
    assert engine.num_steps == 1
    assert engine.drain_aborts() == (0,)
    engine._prefix_cache.withdraw(99)
    run = engine.finalize()
    assert run.records == () and len(run.aborted_requests) == 1
    failure = run.aborted_requests[0]
    assert (failure.id, failure.reason, failure.generated_tokens, failure.retry_count) == (
        0,
        "decode_kv_exhausted",
        1,
        0,
    )
    assert failure.failed_at == 4


def test_input_limit_aborts_at_actual_arrival_without_scheduling_step() -> None:
    cfg = EngineConfig(max_running_requests=1, max_total_tokens=128, page_size=64)
    arrival = Arrival(request_id=4, sim_time=5, prompt_len=64, output_len=2)
    run = run_engine(cfg, PERF, (arrival,))
    assert run.num_engine_steps == 0 and run.records == ()
    assert run.aborted_requests[0].reason == "input_exceeds_kv_capacity"
    assert run.aborted_requests[0].arrival == run.aborted_requests[0].failed_at == 5
    assert run.cache_snapshot.free_blocks == run.cache_snapshot.capacity_blocks == 2


def test_aborted_root_does_not_block_later_success_or_shift_makespan_start() -> None:
    cfg = EngineConfig(max_running_requests=1, max_total_tokens=192, page_size=64)
    run = run_engine(
        cfg,
        PERF,
        (
            Arrival(request_id=0, sim_time=0, prompt_len=128, output_len=1),
            Arrival(request_id=1, sim_time=1, prompt_len=1, output_len=1),
        ),
    )
    assert [item.id for item in run.aborted_requests] == [0]
    assert [item.id for item in run.records] == [1]
    assert (run.sim_start, run.sim_end, run.num_engine_steps) == (0, 2, 1)
    metrics = compute_metrics(run.records, earliest_sent_arrival=run.sim_start)
    assert metrics.makespan == 2 and metrics.ttft_mean == 1


def test_scheduler_failure_at_empty_pool_continues_later_fcfs_candidate() -> None:
    cfg = EngineConfig(max_running_requests=2, max_total_tokens=192, page_size=64)
    engine = Engine(cfg, PERF, output_targets={0: 1, 1: 1})
    engine.submit(request(0, prompt=128, maximum=1), 0)
    engine.submit(request(1, prompt=1, maximum=1), 0)
    assert engine.step(0, 0) == (1, (1,))
    assert engine.drain_aborts() == (0,)
    engine.complete(1, 1)
    run = engine.finalize()
    assert (len(run.records), len(run.aborted_requests), run.num_engine_steps) == (1, 1, 1)
    assert next(run.trace.steps()).aborted == (0,)


def test_failed_parent_skips_all_unsent_descendants_in_success_response() -> None:
    config = SimConfig(
        num_instances=1,
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=3,
                session_rate=10,
                prompt_len=FixedLength(value=64),
                turn_increment_len=FixedLength(value=1),
                output_len=FixedLength(value=2),
                seed=7,
                session=SessionConfig(num_sessions=1),
            )
        ),
        engine=EngineConfig(max_running_requests=1, max_total_tokens=128, page_size=64),
        perf_model=PERF,
    )
    result = simulate(config)
    assert (result.meta.num_requests, result.meta.num_completed) == (3, 0)
    assert (result.meta.num_aborted, result.meta.num_skipped, result.meta.num_engine_steps) == (
        1,
        2,
        0,
    )
    assert result.requests == [] and result.aborted_requests[0].id == 0
    assert [item.id for item in result.skipped_requests] == [1, 2]
    assert result.skipped_requests[1].parent_request_id == 1
    assert result.metrics.ttft_mean is None and result.metrics.throughput is None
    assert result.metrics.makespan == 0
    exclude = {"meta": {"wall_clock_ms"}}
    assert result.model_dump_json(exclude=exclude) == simulate(config).model_dump_json(
        exclude=exclude
    )


def test_requeue_after_future_step_end_preserves_fcfs_of_midstep_arrival() -> None:
    cfg = EngineConfig(max_running_requests=1, max_total_tokens=128, page_size=4)
    cache = L1KVCache(cfg.max_total_tokens, cfg.page_size)
    scheduler = F001Scheduler(cfg, cache)
    first = request(0, maximum=2)
    scheduler.enqueue(first, 0)
    batch = scheduler.select_prefill({})
    assert len(batch.admitted) == 1
    cache.admit(0)
    cache.extend(0, 4)
    cache.withdraw(0)
    scheduler.requeue(first, 20, generated_tokens=1)
    scheduler.enqueue(request(1), 10)
    later = scheduler.select_prefill({})
    assert later.admitted[0].waiting.request.request_id == 1
    assert scheduler.num_waiting == 1
