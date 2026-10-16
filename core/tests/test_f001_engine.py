"""F001 单步引擎：P/D 互斥、融合、KV 增长、发布和确定性。

直接注入缓存种子或 withdraw/requeue 的用例是组件状态测试，不是端到端恢复。
"""


import pytest

from simcore import simulate
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.events import EventKind
from simcore.prefix_hash import encode
from simcore.router import Router
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    InferenceRequest,
    PerfModelConfig,
    SessionConfig,
    SyntheticWorkloadConfig,
    TurnPlan,
    WorkloadConfig,
    WorkloadPlan,
)
from test_client import planned_client
from test_schemas import physical_config, runtime_config

PERF = PerfModelConfig(prefill_c1=1, prefill_c2=0, decode_d1=5, decode_d2=1)


def make_engine(
    *, exclusive: bool = True, page_size: int = 4, capacity: int = 128
) -> Engine:
    config = EngineConfig(
        max_running_requests=4,
        max_total_tokens=capacity,
        page_size=page_size,
        pd_mutually_exclusive=exclusive,
    )
    return Engine(runtime_config(config, PERF))


def request(
    key: int, *, prompt: int = 4, maximum: int = 2, output: int = 2,
    inherited: tuple[str, ...] = (),
) -> InferenceRequest:
    return InferenceRequest(key, 0, None, prompt, maximum, output, inherited, len(inherited))


def test_prefill_samples_first_token_but_does_not_create_its_kv() -> None:
    engine = make_engine(page_size=64, capacity=512)
    engine.submit(request(0, prompt=65, maximum=100, output=3), 0)
    started = engine.start_step(0, 0)
    assert started == (65, ())
    assert engine._prefix_cache.page(encode(0, 0)) is None
    assert engine.finish_step(65) == ()
    assert engine._running == {0: 1}
    assert engine._prefix_cache.resident_pages == 2
    assert engine._prefix_cache.state(0).k == 65
    assert engine._prefix_cache.state(0).f == 64
    assert engine._prefix_cache.private_page_count(0) == 1
    assert engine._prefix_cache.snapshot().internal_fragmentation_tokens == 63
    assert engine._prefix_cache.snapshot().reserved_future_blocks == 0
    assert engine.start_step(65, 1) == (71, ())
    assert engine.finish_step(136) == ()
    assert engine._running == {0: 2}
    assert engine._prefix_cache.state(0).k == 66
    assert engine.start_step(136, 2) == (72, (0,))
    assert engine.finish_step(208) == (0,)
    assert engine._prefix_cache.state(0).k == 67  # 最后采样输出不建立 KV
    engine.complete(208, 0)
    assert engine.finalize().cache_snapshot.active_blocks == 0
    run = engine.finalize()
    assert (run.records[0].prefill_end, run.records[0].finish) == (65, 208)
    assert run.cache_snapshot.resident_blocks == run.cache_snapshot.evictable_blocks == 1
    assert run.cache_snapshot.free_blocks == 7


def test_output_len_one_finishes_at_prefill_end_and_reclaims_partial_page() -> None:
    engine = make_engine(page_size=64, capacity=256)
    engine.submit(request(0, prompt=63, output=1), 0)
    assert engine.start_step(0, 0) == (63, (0,))
    assert engine.finish_step(63) == (0,)
    assert engine._prefix_cache.state(0).k == 63
    engine.complete(63, 0)
    run = engine.finalize()
    assert run.cache_snapshot.resident_blocks == 0
    assert run.records[0].prefill_end == run.records[0].finish == 63


@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("output", [1, 2])
def test_short_output_with_large_request_limit_is_not_permanently_rejected(
    exclusive: bool, output: int,
) -> None:
    engine = make_engine(exclusive=exclusive, page_size=64, capacity=192)
    engine.submit(request(0, prompt=65, maximum=65_536, output=output), 0)
    now = 0.0
    for step in range(output):
        started = engine.start_step(now, step)
        assert started is not None
        now += started[0]
        completed = engine.finish_step(now)
    assert completed == (0,)
    engine.complete(now, 0)
    run = engine.finalize()
    assert run.records[0].output_len == output and not run.aborted_requests
    assert run.num_engine_steps == output and run.cache_snapshot.reserved_future_blocks == 0


def test_decode_full_page_remains_private_until_request_completion() -> None:
    engine = make_engine(page_size=64, capacity=256)
    engine.submit(request(0, prompt=63), 0)
    assert engine.start_step(0, 0) == (63, ())
    assert engine.finish_step(63) == ()
    assert engine._prefix_cache.page(encode(0, 0)) is None
    assert engine.start_step(63, 1) == (69, (0,))
    assert engine.finish_step(132) == (0,)
    assert engine._prefix_cache.state(0).k == 64
    assert engine._prefix_cache.page(encode(0, 0)) is None
    engine.complete(132, 0)
    page = engine._prefix_cache.page(encode(0, 0))
    assert page is not None and page.ref_count == 0
    assert engine.finalize().cache_snapshot.resident_blocks == 1


def test_exclusive_mode_prefills_next_arrival_without_decoding_existing_request() -> None:
    engine = make_engine()
    engine.submit(request(0), 0)
    assert engine.start_step(0, 0) == (4, ())
    assert engine.finish_step(4) == ()
    engine.submit(request(1), 4)
    assert engine.start_step(4, 1) == (4, ())
    assert engine.finish_step(8) == ()
    assert engine._running == {0: 1, 1: 1}
    assert engine._prefix_cache.state(0).k == 4
    assert engine.start_step(8, 2) == (15, (0, 1))
    assert engine.finish_step(23) == (0, 1)
    assert engine._prefix_cache.state(0).k == engine._prefix_cache.state(1).k == 5
    engine.complete(23, 0)
    engine.complete(23, 1)
    assert engine.finalize().num_engine_steps == 3


def test_fused_mode_only_decodes_request_running_at_step_start() -> None:
    engine = make_engine(exclusive=False)
    engine.submit(request(0), 0)
    assert engine.start_step(0, 0) == (4, ())
    assert engine.finish_step(4) == ()
    engine.submit(request(1), 4)
    assert engine.start_step(4, 1) == (10, (0,))
    assert engine.finish_step(14) == (0,)
    assert engine._prefix_cache.state(0).k == 5
    assert engine._prefix_cache.state(1).k == 4
    assert engine._running == {1: 1}
    engine.complete(14, 0)
    assert engine.start_step(14, 2) == (10, (1,))
    assert engine.finish_step(24) == (1,)
    engine.complete(24, 1)
    assert engine.finalize().num_engine_steps == 3


def test_same_batch_duplicate_private_pages_deduplicate_after_execution() -> None:
    engine = make_engine(page_size=64, capacity=512)
    shared = (encode(0, 0),)
    engine.submit(request(1, prompt=65, maximum=1, output=1, inherited=shared), 0)
    engine.submit(request(2, prompt=65, maximum=1, output=1, inherited=shared), 0)
    assert engine.start_step(0, 0) == (130, (1, 2))
    assert engine._prefix_cache.page(shared[0]) is None
    assert engine._prefix_cache.resident_pages == 4
    assert engine.finish_step(130) == (1, 2)
    page = engine._prefix_cache.page(shared[0])
    assert page is not None and page.ref_count == 2
    assert engine._prefix_cache.resident_pages == 3
    assert engine._prefix_cache.peak_resident_pages == 4
    engine.complete(130, 1)
    engine.complete(130, 2)
    snapshot = engine.finalize().cache_snapshot
    assert snapshot.resident_blocks == 1 and snapshot.evictable_blocks == 1
    assert snapshot.free_blocks + snapshot.resident_blocks == snapshot.capacity_blocks


def test_reused_prefill_cost_uses_reused_context_not_full_prompt() -> None:
    cfg = EngineConfig(max_running_requests=2, max_total_tokens=512, page_size=64)
    perf = PerfModelConfig(prefill_c1=0, prefill_c2=1, decode_d1=0, decode_d2=0)
    engine = Engine(runtime_config(cfg, perf))
    engine.submit(request(0, prompt=64, maximum=1, output=1), 0)
    assert engine.start_step(0, 0) == (4096, (0,))
    assert engine.finish_step(4096) == (0,)
    engine.complete(4096, 0)
    engine.submit(request(1, prompt=65, maximum=1, output=1, inherited=(encode(0, 0),)), 4096)
    assert engine.start_step(4096, 1) == (129, (1,))
    assert engine.finish_step(4225) == (1,)
    assert engine._admission_stats[1] == (64, 64)
    engine.complete(4225, 1)
    assert engine.finalize().records[1].actual_prefill_tokens == 1


def test_zero_cost_engine_progresses_and_is_deterministic() -> None:
    cfg = EngineConfig(max_running_requests=2, max_total_tokens=128, page_size=4)
    zero = PerfModelConfig(prefill_c1=0, prefill_c2=0, decode_d1=0, decode_d2=0)
    plan = WorkloadPlan({0: TurnPlan(0, 0, None, 1, None, 5, 3, 3, 0)}, (0,), {})
    first = Coordinator(
        (Engine(runtime_config(cfg, zero)),),
        planned_client(plan, cfg), Router(1),
    ).run()[0]
    second = Coordinator(
        (Engine(runtime_config(cfg, zero)),),
        planned_client(plan, cfg), Router(1),
    ).run()[0]
    assert first.num_engine_steps == 3
    assert first.records[0].finish == first.records[0].arrival == 1
    assert first == second
    assert first.cache_snapshot.reserved_future_blocks == 0


def test_multi_turn_simulation_uses_published_parent_prefix_and_is_reproducible() -> None:
    config = physical_config(
        num_instances=1,
        workload=WorkloadConfig(
            session_rate=10,
            seed=7,
            synthetic=SyntheticWorkloadConfig(
                num_requests=3,
                prompt_len=FixedLength(value=65),
                turn_increment_len=FixedLength(value=1),
                output_len=FixedLength(value=2),
                session=SessionConfig(num_sessions=1),
            )
        ),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=512, page_size=64),
        perf_model=PERF,
    )
    first = simulate(config)
    second = simulate(config)
    exclude = {"meta": {"wall_clock_ms"}}
    assert first.model_dump_json(exclude=exclude) == second.model_dump_json(exclude=exclude)
    assert first.meta.num_completed == first.meta.num_requests == 3
    assert first.metrics.actual_hit_tokens == 128
    assert [record.prompt_len for record in first.requests] == [65, 68, 71]
    assert first.metrics.l1_reserved_future_blocks == 0
    assert all(request.output_len == 2 for request in first.requests)


def test_empty_step_is_rejected_without_advancing_clock() -> None:
    engine = make_engine()
    with pytest.raises(RuntimeError, match="没有可执行"):
        engine.start_step(0, 0)
    assert engine.num_steps == 0


def test_shared_l2_promotion_is_charged_once_but_both_requests_get_l2_hits() -> None:
    config = EngineConfig(
        max_running_requests=3,
        max_total_tokens=28,
        page_size=4,
        l2_capacity_tokens=8,
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=5,
        decode_d2=1,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l2_latency_ms=3,
    )
    engine = Engine(runtime_config(config, perf))
    cache = engine._prefix_cache
    parent = request(0, prompt=4)
    cache.pin(parent, 0)
    cache.admit(0)
    cache.extend(0, 4)
    cache.publish(0)
    cache.finish(0)
    filler = request(99, prompt=28)
    cache.pin(filler, 0)
    cache.admit(99)
    cache.extend(99, 28)
    cache.publish(99)
    cache.finish(99)
    shared = encode(0, 0)
    demoted = cache.page(shared)
    assert demoted is not None and demoted.tier == "L2"
    for key in (1, 2):
        engine.submit(request(key, prompt=5, maximum=1, output=1, inherited=(shared,)), 0)
    assert engine.start_step(0, 0) is None
    assert engine.num_steps == 0
    events = engine.drain_events()
    assert len(events) == 1
    now, kind, transfer_id = events[0]
    assert (now, kind) == (5, EventKind.L2_L1_DONE)
    engine.on_transfer_done(kind, transfer_id, now)
    assert engine.start_step(now, 0) == (2, (1, 2))
    assert engine.finish_step(7) == (1, 2)
    assert engine.num_steps == 1
    promoted_page = cache.page(shared)
    assert promoted_page is not None and promoted_page.tier == "L1"
    assert promoted_page.ref_count == 2
    engine.complete(7, 1)
    engine.complete(7, 2)
    run = engine.finalize()
    assert run.promotions == 1 and run.restore_bytes == 8 and run.restore_time_ms == 5
    assert [
        (record.l1_hit_tokens, record.l2_hit_tokens, record.l3_hit_tokens) for record in run.records
    ] == [(0, 4, 0), (0, 4, 0)]
    assert run.l2_snapshot is not None
    assert run.l2_snapshot.free_blocks + run.l2_snapshot.resident_blocks == 2


def test_full_l2_hit_restore_is_transfer_only_and_does_not_resample_output() -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=16,
        page_size=4,
        l2_capacity_tokens=8,
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=5,
        decode_d2=1,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l2_latency_ms=3,
    )
    engine = Engine(runtime_config(config, perf))
    item = request(0, prompt=4, maximum=2)
    engine.submit(item, 0)
    assert engine.start_step(0, 0) == (4, ())
    assert engine.finish_step(4) == ()
    cache = engine._prefix_cache
    cache.withdraw(0)
    del engine._running[0]
    engine._prefix_scheduler.requeue(item, 4, generated_tokens=1)
    filler = request(99, prompt=16)
    cache.pin(filler, 0)
    cache.admit(99)
    cache.extend(99, 16)
    cache.publish(99)
    cache.finish(99)
    demoted_page = cache.page(encode(0, 0))
    assert demoted_page is not None and demoted_page.tier == "L2"
    assert engine.start_step(4, 1) is None
    events = engine.drain_events()
    assert len(events) == 1
    now, kind, transfer_id = events[0]
    assert (now, kind) == (9, EventKind.L2_L1_DONE)
    engine.on_transfer_done(kind, transfer_id, now)
    assert engine.start_step(now, 1) == (0, ())
    assert engine.finish_step(now) == ()
    assert engine.num_steps == 1
    assert engine._running[0] == 1 and engine._prefill_end[0] == 4
    assert engine.start_step(9, 1) == (10, (0,))
    assert engine.finish_step(19) == (0,)
    assert engine.num_steps == 2
    engine.complete(19, 0)
    run = engine.finalize()
    assert run.records[0].finish == 19
    assert run.records[0].replayed_prefill_tokens == 0
    assert run.restore_time_ms == 5 and run.restore_bytes == 8


def test_fused_decode_overlaps_transfer_without_decoding_restored_request() -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=28,
        page_size=4,
        l2_capacity_tokens=8,
        pd_mutually_exclusive=False,
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=5,
        decode_d2=1,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l2_latency_ms=3,
    )
    engine = Engine(runtime_config(config, perf))
    item = request(0, prompt=4, maximum=2)
    active = request(1, prompt=4, maximum=2)
    engine.submit(item, 0)
    engine.submit(active, 0)
    assert engine.start_step(0, 0) == (8, ())
    assert engine.finish_step(8) == ()
    cache = engine._prefix_cache
    cache.withdraw(0)
    del engine._running[0]
    engine._prefix_scheduler.requeue(item, 8, generated_tokens=1)
    filler = request(99, prompt=24)
    cache.pin(filler, 0)
    cache.admit(99)
    cache.extend(99, 24)
    cache.publish(99)
    cache.finish(99)
    demoted_page = cache.page(encode(0, 0))
    assert demoted_page is not None and demoted_page.tier == "L2"
    assert engine.start_step(8, 1) is None
    events = engine.drain_events()
    assert len(events) == 1
    now, kind, transfer_id = events[0]
    assert (now, kind) == (13, EventKind.L2_L1_DONE)
    assert engine.start_step(8, 1) == (10, (1,))
    assert 0 not in engine._running
    engine.on_transfer_done(kind, transfer_id, now)
    assert engine.finish_step(18) == (1,)
    assert 0 not in engine._running
    assert engine._pending_selection is not None
    engine.complete(18, 1)
    assert engine.start_step(18, 2) == (0, ())
    assert engine.finish_step(18) == ()
    assert engine._running[0] == 1 and engine._prefill_end[0] == 8
    assert engine.start_step(18, 2) == (10, (0,))
    assert engine.finish_step(28) == (0,)
    engine.complete(28, 0)
    run = engine.finalize()
    assert run.records[0].replayed_prefill_tokens == 0
    assert run.restore_time_ms == 5 and run.restore_bytes == 8


def test_l3_prefill_hit_traverses_two_links_and_reports_original_source() -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=16,
        page_size=4,
        l2_capacity_tokens=4,
        l3_capacity_tokens=8,
        l3_prefetch_min_tokens=0,
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=5,
        decode_d2=1,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l3_to_l2_bandwidth_bytes_per_ms=2,
        l2_latency_ms=3,
        l3_latency_ms=5,
    )
    engine = Engine(runtime_config(config, perf))
    cache = engine._prefix_cache
    for key, size in ((0, 4), (99, 16), (98, 4)):
        seed = request(key, prompt=size)
        cache.pin(seed, 0)
        cache.admit(key)
        cache.extend(key, size)
        cache.publish(key)
        cache.finish(key)
    cache.ensure_free(1)
    shared = encode(0, 0)
    demoted_page = cache.page(shared)
    assert demoted_page is not None and demoted_page.tier == "L3"
    assert cache.free_pages == 1
    engine.submit(request(1, prompt=5, maximum=1, output=1, inherited=(shared,)), 0)
    lookup_events = engine.drain_events()
    assert lookup_events == ((0, EventKind.L3_LOOKUP_DONE, 1),)
    engine.on_l3_lookup(1, 0)
    events = engine.drain_events()
    assert len(events) == 1
    now, kind, transfer_id = events[0]
    assert (now, kind) == (9, EventKind.L3_L2_DONE)
    assert not engine.can_step
    engine.on_transfer_done(kind, transfer_id, now)
    assert engine.start_step(now, 0) is None
    events = engine.drain_events()
    assert len(events) == 1
    now, kind, transfer_id = events[0]
    assert (now, kind) == (14, EventKind.L2_L1_DONE)
    engine.on_transfer_done(kind, transfer_id, now)
    assert engine.start_step(now, 0) == (1, (1,))
    assert engine.finish_step(15) == (1,)
    assert engine.num_steps == 1
    engine.complete(15, 1)
    run = engine.finalize()
    assert run.records[0].l3_hit_tokens == 4
    assert run.restore_bytes == 16 and run.restore_time_ms == 14
    assert run.promotions == 2
    assert run.l3_snapshot is not None
    assert run.l3_snapshot.free_blocks + run.l3_snapshot.resident_blocks == 2


def test_prefill_recomputes_same_id_as_other_candidates_pinned_lower_page() -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=36,
        page_size=4,
        l2_capacity_tokens=8,
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=0,
        decode_d2=0,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
    )
    engine = Engine(runtime_config(config, perf))
    cache = engine._prefix_cache
    for key, size in ((0, 4), (99, 36)):
        seed = request(key, prompt=size)
        cache.pin(seed, 0)
        cache.admit(key)
        cache.extend(key, size)
        cache.publish(key)
        cache.finish(key)
    shared = encode(0, 0)
    demoted_page = cache.page(shared)
    assert demoted_page is not None and demoted_page.tier == "L2"
    engine.submit(request(1, prompt=5, maximum=1, output=1, inherited=(shared,)), 0)
    engine.submit(
        request(
            2,
            prompt=9,
            maximum=1,
            output=1,
            inherited=(encode(77, 0), shared),
        ),
        0,
    )
    assert engine.start_step(0, 0) is None
    events = engine.drain_events()
    assert len(events) == 1
    now, kind, transfer_id = events[0]
    assert (now, kind) == (2, EventKind.L2_L1_DONE)
    engine.on_transfer_done(kind, transfer_id, now)
    assert engine.start_step(now, 0) == (10, (1, 2))
    assert engine.finish_step(12) == (1, 2)
    published = cache.page(shared)
    assert published is not None and published.tier == "L1"
    assert published.ref_count == 2
    assert cache.promotions == 1
    engine.complete(12, 1)
    engine.complete(12, 2)
    assert cache.page(shared) is not None
    assert [(r.l2_hit_tokens, r.replayed_prefill_tokens) for r in engine.finalize().records] == [
        (4, 0),
        (0, 0),
    ]


@pytest.mark.parametrize("exclusive", [True, False])
def test_zero_lower_tiers_match_legacy_results_field_by_field(exclusive: bool) -> None:
    legacy = EngineConfig(
        max_running_requests=2,
        max_total_tokens=32,
        page_size=4,
        pd_mutually_exclusive=exclusive,
    )
    explicitly_disabled = EngineConfig(
        max_running_requests=2,
        max_total_tokens=32,
        page_size=4,
        l2_capacity_tokens=0,
        l3_capacity_tokens=0,
        pd_mutually_exclusive=exclusive,
    )
    extra_perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=5,
        decode_d2=1,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l3_to_l2_bandwidth_bytes_per_ms=2,
        l2_latency_ms=3,
        l3_latency_ms=5,
    )
    plan = WorkloadPlan({
        0: TurnPlan(0, 0, None, 0, None, 5, 3, 3, 0),
        1: TurnPlan(1, 1, None, 5, None, 4, 2, 2, 0),
    }, (0, 1), {})
    old = Coordinator(
        (Engine(runtime_config(legacy, PERF)),),
        planned_client(plan, legacy), Router(1),
    ).run()[0]
    new = Coordinator(
        (Engine(runtime_config(explicitly_disabled, extra_perf)),),
        planned_client(plan, explicitly_disabled), Router(1),
    ).run()[0]
    assert old == new
    assert new.promotions == new.demotions == new.restore_bytes == new.restore_time_ms == 0
