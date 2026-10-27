"""F001 同刻完成、客户端回调、子轮发送和单步调度全序验收。"""

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from simcore import simulate
from simcore.analytical import StepCost, step_cost
from simcore.client import Client
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.events import EventKind
from simcore.kv_cache import CacheCapacityError, CacheEntry
from simcore.prefix_hash import encode
from simcore.router import Router
from simcore.schemas import (
    EngineConfig,
    InferenceRequest,
    PerfModelConfig,
    TracePlan,
    TraceTurn,
    TraceWorkloadConfig,
    TurnPlan,
    WorkloadConfig,
    WorkloadPlan,
)
from test_client import planned_client
from test_f001_scheduler import TrackedWaitingQueue
from test_schemas import resolved_config, runtime_config


def make_plan(*, ordinary_at: float = 8, think: float = 0) -> WorkloadPlan:
    turns = {
        0: TurnPlan(0, 0, None, 0, None, 4, 1, 1, 0),
        1: TurnPlan(1, 1, None, 0, None, 4, 1, 1, 0),
        2: TurnPlan(2, 2, None, ordinary_at, None, 4, 1, 1, 0),
        3: TurnPlan(3, 0, 0, None, think, 5, 1, 1, 1),
        4: TurnPlan(4, 1, 1, None, think, 5, 1, 1, 1),
    }
    return WorkloadPlan(turns, (2, 1, 0), {0: (3,), 1: (4,)})


def make_run(plan: WorkloadPlan) -> tuple[Coordinator, Engine]:
    config = EngineConfig(max_running_requests=5, max_total_tokens=128, page_size=4)
    engine = Engine(runtime_config(config, PerfModelConfig()))
    return Coordinator((engine,), planned_client(plan, config), Router(1)), engine


def test_parent_done_callback_precedes_sends_including_immediate_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("simcore.engine.step_cost", lambda *args: StepCost(8, 0, 0, 0, 0))
    coordinator, engine = make_run(make_plan())
    order = Mock()
    with (
        patch.object(engine, "complete", wraps=engine.complete) as completions,
        patch.object(
            coordinator.client,
            "on_request_completed",
            wraps=coordinator.client.on_request_completed,
        ) as callbacks,
        patch.object(engine, "submit", wraps=engine.submit) as submissions,
    ):
        order.attach_mock(completions, "complete")
        order.attach_mock(callbacks, "callback")
        order.attach_mock(submissions, "send")
        run = coordinator.run()[0]
    assert [
        (
            call[0],
            call.args[1]
            if call[0] == "complete"
            else call.args[0]
            if call[0] == "callback"
            else call.args[0].request_id,
        )
        for call in order.mock_calls[:9]
    ] == [
        ("send", 0),
        ("send", 1),
        ("complete", 0),
        ("callback", 0),
        ("complete", 1),
        ("callback", 1),
        ("send", 2),
        ("send", 3),
        ("send", 4),
    ]
    assert [
        call.args[0].request_id for call in submissions.call_args_list if call.args[1] == 8
    ] == [2, 3, 4]
    assert [call.args[0] for call in callbacks.call_args_list if call.args[1] == 8] == [0, 1]
    assert [run.records[key].prefill_start for key in (2, 3, 4)] == [8, 8, 8]
    assert run.records[3].actual_hit_tokens == run.records[4].actual_hit_tokens == 4
    assert run.records[3].arrival == run.records[0].finish == 8
    assert run.num_engine_steps == 2


def test_future_think_time_child_is_ordered_like_other_same_time_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("simcore.engine.step_cost", lambda *args: StepCost(8, 0, 0, 0, 0))
    coordinator, engine = make_run(make_plan(ordinary_at=12, think=4))
    with patch.object(engine, "submit", wraps=engine.submit) as submissions:
        run = coordinator.run()[0]
    assert [
        call.args[0].request_id for call in submissions.call_args_list if call.args[1] == 12
    ] == [2, 3, 4]
    assert [run.records[key].arrival for key in (2, 3, 4)] == [12, 12, 12]
    assert [run.records[key].prefill_start for key in (2, 3, 4)] == [12, 12, 12]
    assert run.num_engine_steps == 2
    assert run.records[3].arrival == run.records[0].finish + 4


def test_same_time_roots_are_sent_in_id_order_even_if_created_in_reverse() -> None:
    coordinator, _ = make_run(make_plan(ordinary_at=0))
    run = coordinator.run()[0]
    assert [run.records[key].arrival for key in (0, 1, 2)] == [0, 0, 0]
    assert [run.records[key].prefill_start for key in (0, 1, 2)] == [0, 0, 0]


@pytest.mark.parametrize("lookup_ms", [0.0, 3.0])
@pytest.mark.parametrize("l2_pages", [1, 2])
def test_immediate_child_and_normal_send_share_l3_prefetch_order(
    lookup_ms: float, l2_pages: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "simcore.engine.step_cost",
        lambda workload, *args: (
            StepCost(1, 0, 0, 0, 0)
            if workload.prefill and workload.prefill[0].request_id == 0
            else step_cost(workload, *args)
        ),
    )
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=64,
        page_size=4,
        l2_capacity_tokens=l2_pages * 4,
        l3_capacity_tokens=256,
        l3_prefetch_min_tokens=0,
    )
    perf = PerfModelConfig(
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l3_to_l2_bandwidth_bytes_per_ms=2,
        l2_latency_ms=3,
        l3_latency_ms=5,
        l3_lookup_latency_ms=lookup_ms,
    )
    send_ms = 1
    source = TracePlan(
        sessions=("competition",),
        block_size=4,
        requests=(
            TraceTurn(
                request_id=0,
                session_id=0,
                arrival_offset_ms=0,
                prompt_len=1,
                output_len=1,
                hash_ids=(),
            ),
            TraceTurn(
                request_id=1,
                session_id=0,
                arrival_offset_ms=send_ms,
                prompt_len=5,
                output_len=1,
                hash_ids=(20,),
            ),
            TraceTurn(
                request_id=2,
                session_id=0,
                parent_request_id=0,
                arrival_offset_ms=send_ms,
                think_time_ms=0,
                prompt_len=5,
                output_len=1,
                hash_ids=(10,),
            ),
            TraceTurn(
                request_id=3,
                session_id=0,
                arrival_offset_ms=1000,
                prompt_len=5,
                output_len=1,
                hash_ids=(20,),
            ),
            TraceTurn(
                request_id=4,
                session_id=0,
                arrival_offset_ms=1000,
                prompt_len=5,
                output_len=1,
                hash_ids=(10,),
            ),
        ),
    )
    workload = WorkloadConfig(mode="trace", session_rate=10, trace=TraceWorkloadConfig())
    with patch("simcore.client.random.Random.expovariate", return_value=0.0):
        client = Client(workload, config, trace_plan=source)
    engine = Engine(runtime_config(config, perf))
    cache = engine._prefix_cache
    seeds = (
        InferenceRequest(90, 0, None, 4, 1, 1, (), 0, trace_hash_ids=(20,), is_trace=True),
        InferenceRequest(91, 0, None, 4, 1, 1, (), 0, trace_hash_ids=(10,), is_trace=True),
        InferenceRequest(92, 92, None, 64, 1, 1, (), 0),
        InferenceRequest(93, 93, None, l2_pages * 4, 1, 1, (), 0),
    )
    for seed in seeds:
        cache.pin(seed, ())
        cache.admit(seed.request_id)
        cache.extend(seed.request_id, seed.prompt_len)
        cache.publish(seed.request_id)
        cache.finish(seed.request_id)
    cache.ensure_free(4)
    for chunk_id in ("trace:0:20", "trace:0:10"):
        page = cache.page(chunk_id)
        assert page is not None and page.tier == "L3"
    order = Mock()
    with (
        patch.object(engine, "submit", wraps=engine.submit) as submissions,
        patch.object(engine, "on_l3_lookup", wraps=engine.on_l3_lookup) as lookups,
        patch.object(cache, "reserve_transfer", wraps=cache.reserve_transfer) as reserves,
    ):
        order.attach_mock(submissions, "send")
        order.attach_mock(lookups, "lookup")
        run = Coordinator((engine,), client, Router(1)).run()[0]
    assert [call.args[0].request_id for call in submissions.call_args_list] == [0, 1, 2, 3, 4]
    assert [(call.args[0], call.args[1]) for call in lookups.call_args_list[:2]] == [
        (1, send_ms + lookup_ms),
        (2, send_ms + lookup_ms),
    ]
    assert [call[0] for call in order.mock_calls[:5]] == (
        ["send", "send", "lookup", "send", "lookup"]
        if lookup_ms == 0
        else ["send", "send", "send", "lookup", "lookup"]
    )
    assert [call.args for call in reserves.call_args_list if call.args[1] == "L2"] == [
        ("trace:0:20", "L2"),
        ("trace:0:10", "L2"),
    ]
    records = {item.id: item for item in run.records}
    assert records[0].finish == records[1].arrival == records[2].arrival == send_ms
    assert records[1].l3_hit_tokens == 4
    assert records[2].l3_hit_tokens == (0 if l2_pages == 1 else 4)
    assert records[1].prefill_start == send_ms + lookup_ms + 14
    assert run.restore_bytes == (16 if l2_pages == 1 else 32)
    assert not run.aborted_requests and not run.skipped_requests
    assert not engine._transfers and not engine._prefetch_waiting
    assert run.cache_snapshot.active_blocks == 0


@pytest.mark.parametrize("with_ready_root", [False, True])
@pytest.mark.parametrize("prefetch_threshold", [0, 8])
def test_queue_time_local_hit_loss_requeries_l3_and_skips_waiting_candidate(
    with_ready_root: bool,
    prefetch_threshold: int,
) -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=16,
        page_size=4,
        l2_capacity_tokens=8,
        l3_capacity_tokens=64,
        l3_prefetch_min_tokens=prefetch_threshold,
    )
    perf = PerfModelConfig(
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l3_to_l2_bandwidth_bytes_per_ms=2,
        l3_lookup_latency_ms=2,
    )
    engine = Engine(runtime_config(config, perf))
    cache = engine._prefix_cache
    parent = InferenceRequest(90, 90, None, 8, 1, 1, (), 0)
    cache.pin(parent, ())
    cache.admit(90)
    cache.extend(90, 8)
    cache.publish(90)
    cache.finish(90)

    child = InferenceRequest(
        1,
        90,
        90,
        9,
        1,
        1,
        (encode(90, 0), encode(90, 1)),
        2,
    )
    assert engine.submit(child, 0)
    if with_ready_root:
        root = InferenceRequest(2, 2, None, 1, 1, 1, (), 0)
        assert engine.submit(root, 0)
    assert not engine._lookup_waiting
    filler = InferenceRequest(91, 91, None, 16, 1, 1, (), 0)
    cache.pin(filler, ())
    cache.admit(91)
    cache.ensure_free(4)
    cache.extend(91, 16)
    cache.publish(91)
    cache.finish(91)
    cache.ensure_free(4)
    assert all(cache.page(chunk).tier == "L3" for chunk in child.inherited_prefix_ids)

    assert engine.can_step
    assert engine._prefix_scheduler.ready_count == 1 + int(with_ready_root)
    with (
        patch.object(cache, "probe", wraps=cache.probe) as probes,
        patch.object(cache, "pin", wraps=cache.pin) as pins,
    ):
        started = engine.start_step(0, 0)
    assert engine._prefix_scheduler.ready_count == 0
    assert sum(call.args[0].request_id == 1 for call in probes.call_args_list) == 1
    assert all(call.args[0].request_id != 1 for call in pins.call_args_list)
    assert engine._lookup_waiting == {1}
    assert engine._prefix_scheduler.head_request_id == 1
    if with_ready_root:
        assert started is not None and started[1] == (2,)
        end = started[0]
        assert 0 < end < 2
        assert engine.finish_step(end) == (2,)
        engine.complete(end, 2)
    else:
        assert started is None and not engine.can_step
        assert engine.num_steps == 0
    [(lookup_at, kind, key)] = engine.drain_events()
    assert (lookup_at, kind, key) == (2, EventKind.L3_LOOKUP_DONE, 1)
    engine.on_l3_lookup(key, lookup_at)
    assert engine._prefix_scheduler.ready_count == int(bool(prefetch_threshold))
    prefetch = sorted(engine.drain_events())
    if prefetch_threshold:
        assert not prefetch and engine.can_step
        started = engine.start_step(lookup_at, engine.num_steps)
        assert started is not None
        duration, _ = started
        end = lookup_at + duration
        assert engine.finish_step(end) == (1,)
        engine.complete(end, 1)
        record = next(item for item in engine.finalize().records if item.id == 1)
        assert record.actual_hit_tokens == 0
        assert not engine.drain_events()
        return
    assert len(prefetch) == 2
    for index, (time, kind, transfer_id) in enumerate(prefetch):
        assert kind is EventKind.L3_L2_DONE
        engine.on_transfer_done(kind, transfer_id, time)
        assert engine._prefix_scheduler.ready_count == int(index == 1)
    assert engine.start_step(prefetch[-1][0], engine.num_steps) is None
    assert engine._prefix_scheduler.ready_count == 0
    loads = sorted(engine.drain_events())
    assert len(loads) == 2
    for time, kind, transfer_id in loads:
        assert kind is EventKind.L2_L1_DONE
        engine.on_transfer_done(kind, transfer_id, time)
    started = engine.start_step(loads[-1][0], engine.num_steps)
    assert started is not None
    duration, _ = started
    end = loads[-1][0] + duration
    assert engine.finish_step(end) == (1,)
    engine.complete(end, 1)
    child_record = next(item for item in engine.finalize().records if item.id == 1)
    assert child_record.actual_hit_tokens == child_record.l3_hit_tokens == 8


def test_lookup_burst_readiness_checks_do_not_scan_waiting_queue() -> None:
    config = EngineConfig(
        max_running_requests=4,
        max_total_tokens=64,
        page_size=4,
        l2_capacity_tokens=16,
        l3_capacity_tokens=16,
    )
    perf = PerfModelConfig(l3_lookup_latency_ms=10)
    engine = Engine(runtime_config(config, perf))
    scheduler = engine._prefix_scheduler
    waiting = TrackedWaitingQueue()
    scheduler._waiting = waiting
    for key in range(128):
        assert engine.submit(InferenceRequest(key, key, None, 5, 1, 1, (), 0), 0)
        assert scheduler.ready_count == 0 and not engine.can_step
    for count, (time, kind, key) in enumerate(engine.drain_events(), 1):
        assert kind is EventKind.L3_LOOKUP_DONE
        engine.on_l3_lookup(key, time)
        assert scheduler.ready_count == count and engine.can_step
    assert waiting.iterations == waiting.pops == waiting.removals == 0


def test_zero_ready_waiters_do_not_stop_decode_or_trigger_queue_scan() -> None:
    config = EngineConfig(
        max_running_requests=4,
        max_total_tokens=64,
        page_size=4,
        l2_capacity_tokens=16,
        l3_capacity_tokens=16,
    )
    perf = PerfModelConfig(l3_lookup_latency_ms=100)
    engine = Engine(runtime_config(config, perf))
    assert engine.submit(InferenceRequest(0, 0, None, 1, 4, 4, (), 0), 0)
    started = engine.start_step(0, 0)
    assert started is not None
    now = started[0]
    assert not engine.finish_step(now)
    assert engine.submit(InferenceRequest(1, 1, None, 5, 1, 1, (), 0), now)
    scheduler = engine._prefix_scheduler
    waiting = TrackedWaitingQueue(scheduler._waiting)
    scheduler._waiting = waiting
    for _ in range(2):
        assert scheduler.ready_count == 0 and engine.can_step
        started = engine.start_step(now, engine.num_steps)
        assert started is not None
        now += started[0]
        assert not engine.finish_step(now)
    assert engine._running[0] == 3
    assert waiting.iterations == waiting.pops == waiting.removals == 0


def test_shared_prefetch_only_readies_requests_with_all_dependencies_complete() -> None:
    config = EngineConfig(
        max_running_requests=4,
        max_total_tokens=64,
        page_size=4,
        l2_capacity_tokens=16,
        l3_capacity_tokens=16,
        l3_prefetch_min_tokens=0,
    )
    perf = PerfModelConfig(
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l3_to_l2_bandwidth_bytes_per_ms=2,
        l3_lookup_latency_ms=2,
    )
    engine = Engine(runtime_config(config, perf))
    scheduler = engine._prefix_scheduler
    ids = (encode(90, 0), encode(90, 1))
    for chunk in ids:
        engine._prefix_cache.pools["L3"].insert(chunk)
    for key, blocks in ((0, 1), (1, 2), (2, 1)):
        item = InferenceRequest(key, 90, 90, blocks * 4 + 1, 1, 1, ids[:blocks], blocks)
        assert engine.submit(item, 0)
    assert scheduler.ready_count == 0
    for time, kind, key in engine.drain_events():
        assert kind is EventKind.L3_LOOKUP_DONE
        engine.on_l3_lookup(key, time)
        assert scheduler.ready_count == 0 and not engine.can_step
    prefetch = sorted(engine.drain_events())
    assert len(prefetch) == 2
    assert engine._transfers[prefetch[0][2]].waiting_requests == {0, 1, 2}
    time, kind, key = prefetch[0]
    engine.on_transfer_done(kind, key, time)
    assert scheduler.ready_count == 2 and engine.can_step
    assert set(engine._prefetch_waiting) == {1}
    engine.on_transfer_done(kind, key, time)
    assert scheduler.ready_count == 2
    time, kind, key = prefetch[1]
    engine.on_transfer_done(kind, key, time)
    assert scheduler.ready_count == 3 and not engine._prefetch_waiting
    assert engine.start_step(time, 0) is None
    assert scheduler.ready_count == 0
    loads = sorted(engine.drain_events())
    assert len(loads) == 2
    assert engine.submit(InferenceRequest(3, 3, None, 1, 1, 1, (), 0), time)
    assert scheduler.ready_count == 1 and not engine.can_step
    for index, (time, kind, key) in enumerate(loads):
        engine.on_transfer_done(kind, key, time)
        assert scheduler.ready_count == 1
        assert engine.can_step == (index == 1)
    started = engine.start_step(time, 0)
    assert started is not None and scheduler.ready_count == 1
    time += started[0]
    assert engine.finish_step(time) == (0, 1, 2)
    for key in (0, 1, 2):
        engine.complete(time, key)
    assert engine.start_step(time, engine.num_steps) is not None
    assert scheduler.ready_count == 0


@pytest.mark.parametrize("restoring", [False, True])
@pytest.mark.parametrize("blocked", [None, "lookup", "prefetch", "load"])
def test_prefill_probes_each_ready_candidate_once(restoring: bool, blocked: str | None) -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=32,
        page_size=4,
        l2_capacity_tokens=8,
        l3_capacity_tokens=16,
    )
    engine = Engine(runtime_config(config, PerfModelConfig()))
    cache = engine._prefix_cache
    inherited = (encode(90, 0),)
    cache.pools["L1"].insert(inherited[0])
    child = InferenceRequest(1, 90, 90, 5, 8, 8, inherited, 1)
    assert engine.submit(child, 0)
    if restoring:
        waiting = engine._prefix_scheduler._waiting
        waiting[0] = replace(waiting[0], generated_tokens=4)
        cache.pools["L1"].insert(encode(1, 1))
    if blocked == "lookup":
        engine._schedule_l3_lookup(1, 0)
    elif blocked == "prefetch":
        engine._prefetch_waiting[1] = {0}
        engine._prefix_scheduler.set_ready(1, False)
    elif blocked == "load":
        engine._pending_loads.add(inherited[0])
    with (
        patch.object(cache, "probe", wraps=cache.probe) as probes,
        patch.object(cache, "pin", wraps=cache.pin) as pins,
    ):
        if blocked == "load":
            assert engine._prefix_scheduler.ready_count == 1
            assert not engine.can_step and engine.start_step(0, 0) is None
            probes.assert_not_called()
            pins.assert_not_called()
            return
        selection = engine._prefix_scheduler.select_prefill(
            {},
            accept_probe=lambda item, sources: engine._accept_prefill_probe(item, sources, 0),
        )
    assert engine._prefix_scheduler.ready_count == 0
    assert probes.call_count == pins.call_count == int(blocked is None)
    if blocked is not None:
        assert not selection.admitted and not selection.restored
        assert engine._prefix_scheduler.num_waiting == 1
        assert not cache._pins
    elif restoring:
        assert selection.restored[0].request == child
        assert cache.state(1).k == 8
    else:
        assert selection.admitted[0].reused_tokens == 4
    assert engine._last_local_match[1] == 1


@pytest.mark.parametrize("tier", ["L2", "L3"])
@pytest.mark.parametrize("trace", [False, True])
def test_transfer_completions_preserve_context_order(tier: str, trace: bool) -> None:
    config = EngineConfig(
        max_running_requests=1,
        max_total_tokens=32,
        page_size=4,
        l2_capacity_tokens=16,
        l3_capacity_tokens=16 if tier == "L3" else 0,
        l3_prefetch_min_tokens=0,
    )
    perf = PerfModelConfig(
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l3_to_l2_bandwidth_bytes_per_ms=2,
        l2_latency_ms=3,
        l3_latency_ms=1,
        l3_lookup_latency_ms=2,
    )
    engine = Engine(runtime_config(config, perf))
    cache = engine._prefix_cache
    hashes = (2, 10, 11)
    ids = (
        tuple(f"trace:0:{block}" for block in hashes)
        if trace
        else tuple(encode(90, block) for block in hashes)
    )
    for chunk_id in ids:
        cache.pools[tier].insert(chunk_id)
    item = InferenceRequest(
        request_id=1,
        session_id=0,
        parent_request_id=None,
        prompt_len=13,
        max_new_tokens=1,
        output_len=1,
        inherited_prefix_ids=() if trace else ids,
        ideal_reuse_blocks=3,
        trace_hash_ids=hashes if trace else (),
        is_trace=trace,
    )
    assert engine.submit(item, 0)
    now = 0.0
    if tier == "L3":
        [(now, kind, key)] = engine.drain_events()
        assert kind is EventKind.L3_LOOKUP_DONE
        engine.on_l3_lookup(key, now)
        prefetch = engine.drain_events()
        assert [engine._transfers[event[2]].chunk_id for event in prefetch] == list(ids)
        assert [event[0] for event in prefetch] == [7, 12, 17]
        for now, kind, key in prefetch:
            engine.on_transfer_done(kind, key, now)

    assert engine.start_step(now, 0) is None
    loads = engine.drain_events()
    assert [engine._transfers[event[2]].chunk_id for event in loads] == list(ids)
    assert [event[0] for event in loads] == [now + 5, now + 10, now + 15]
    # 中间页按上下文顺序可见，整批回载完成前不能开始 Prefill。
    for index, (finished_at, kind, key) in enumerate(loads):
        assert kind is EventKind.L2_L1_DONE
        engine.on_transfer_done(kind, key, finished_at)
        assert all(cache.pools["L1"].lookup(chunk) is not None for chunk in ids[: index + 1])
        assert all(cache.pools["L1"].lookup(chunk) is None for chunk in ids[index + 1 :])
        assert not cache.evictable_pages
        if index < len(ids) - 1:
            assert engine.start_step(finished_at, 0) is None
    started = engine.start_step(loads[-1][0], 0)
    assert started is not None
    end = loads[-1][0] + started[0]
    assert engine.finish_step(end) == (1,)
    engine.complete(end, 1)
    result = engine.finalize()
    assert result.restore_bytes == 24 * (2 if tier == "L3" else 1)
    assert result.restore_time_ms == 15 * (2 if tier == "L3" else 1)
    assert result.records[0].l3_hit_tokens == (12 if tier == "L3" else 0)
    assert result.cache_snapshot.active_blocks == 0


def test_l2_hit_waits_for_transfer_before_child_prefill() -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=20,
        page_size=4,
        l2_capacity_tokens=8,
    )
    perf = PerfModelConfig(
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l2_latency_ms=3,
    )
    plan = WorkloadPlan(
        turns={
            0: TurnPlan(0, 0, None, 0, None, 5, 1, 1, 0),
            1: TurnPlan(1, 0, 0, None, 45, 9, 1, 1, 1),
            2: TurnPlan(2, 2, None, 12, None, 12, 1, 1, 0),
            3: TurnPlan(3, 3, None, 26, None, 12, 1, 1, 0),
        },
        root_request_ids=(0, 2, 3),
        children={0: (1,)},
    )
    engine = Engine(runtime_config(config, perf))
    run = Coordinator((engine,), planned_client(plan, config), Router(1)).run()[0]
    child = next(item for item in run.records if item.id == 1)
    parent = next(item for item in run.records if item.id == 0)
    assert child.arrival == parent.finish + 45
    assert child.prefill_start == pytest.approx(child.arrival + 5)
    assert (child.actual_hit_tokens, child.l2_hit_tokens) == (4, 4)
    assert run.restore_time_ms == pytest.approx(5)
    assert run.num_engine_steps == 4


@pytest.mark.parametrize("prefetch_fails", [False, True])
def test_wait_complete_l3_query_prefetch_then_l2_load(prefetch_fails: bool) -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=20,
        page_size=4,
        l2_capacity_tokens=8,
        l3_capacity_tokens=64,
        l3_prefetch_min_tokens=0,
    )
    perf = PerfModelConfig(
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l3_to_l2_bandwidth_bytes_per_ms=2,
        l2_latency_ms=3,
        l3_latency_ms=5,
        l3_lookup_latency_ms=3,
    )
    plan = WorkloadPlan(
        turns={
            0: TurnPlan(0, 0, None, 0, None, 5, 1, 1, 0),
            1: TurnPlan(1, 0, 0, None, 70, 9, 1, 1, 1),
            2: TurnPlan(2, 2, None, 12, None, 12, 1, 1, 0),
            3: TurnPlan(3, 3, None, 26, None, 12, 1, 1, 0),
            4: TurnPlan(4, 4, None, 40, None, 12, 1, 1, 0),
        },
        root_request_ids=(0, 2, 3, 4),
        children={0: (1,)},
    )
    engine = Engine(runtime_config(config, perf))
    if prefetch_fails:
        with patch.object(
            engine._prefix_cache, "reserve_transfer", side_effect=CacheCapacityError()
        ):
            run = Coordinator((engine,), planned_client(plan, config), Router(1)).run()[0]
    else:
        run = Coordinator((engine,), planned_client(plan, config), Router(1)).run()[0]
    child = next(item for item in run.records if item.id == 1)
    parent = next(item for item in run.records if item.id == 0)
    assert child.arrival == parent.finish + 70
    assert child.prefill_start == pytest.approx(child.arrival + (3 if prefetch_fails else 17))
    assert (child.actual_hit_tokens, child.l3_hit_tokens) == ((0, 0) if prefetch_fails else (4, 4))
    assert run.restore_time_ms == pytest.approx(0 if prefetch_fails else 14)


def test_full_l2_restore_does_not_charge_a_second_transfer_step() -> None:
    config = EngineConfig(
        max_running_requests=2, max_total_tokens=16, page_size=4, l2_capacity_tokens=8
    )
    perf = PerfModelConfig(
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l2_latency_ms=3,
    )
    engine = Engine(runtime_config(config, perf))
    item = InferenceRequest(0, 0, None, 4, 2, 2, (), 0)
    engine.submit(item, 0)
    engine._prefix_scheduler.select_prefill({})
    cache = engine._prefix_cache
    cache.admit(0)
    cache.extend(0, 4)
    cache.publish(0)
    cache.withdraw(0)
    filler = InferenceRequest(99, 99, None, 16, 1, 1, (), 0)
    cache.pin(filler, ())
    cache.admit(99)
    cache.extend(99, 16)
    cache.publish(99)
    cache.finish(99)
    cached = cache.page(encode(0, 0))
    assert cached is not None and cached.tier == "L2"
    engine._prefix_scheduler.requeue(item, 1, generated_tokens=1)
    assert engine.start_step(1, 0) is None
    [(end, kind, transfer_id)] = engine.drain_events()
    assert kind is EventKind.L2_L1_DONE
    engine.on_transfer_done(kind, transfer_id, end)
    assert engine.start_step(end, 0) == (0.0, ())
    assert engine.finish_step(end) == ()
    assert engine._running == {0: 1}
    assert engine.num_steps == 0
    assert engine._restore_time_ms == pytest.approx(5)


def test_full_l1_restore_cannot_exhaust_selected_l2_load_capacity() -> None:
    config = EngineConfig(
        max_running_requests=8, max_total_tokens=512, page_size=64, l2_capacity_tokens=512
    )
    perf = PerfModelConfig(
        kv_bytes_per_token=1,
        l2_to_l1_bandwidth_bytes_per_ms=64,
    )
    from simcore.perf_model import PrefillWork, StepWorkload
    from test_engine import duration
    from test_schemas import TOY_HARDWARE, TOY_MODEL

    def reused_prefill(reused: int, new: int) -> float:
        return step_cost(
            StepWorkload("PREFILL", (PrefillWork(0, reused, new, reused + new),), (), 0, 1),
            TOY_MODEL,
            TOY_HARDWARE,
        ).duration_ms

    first_end = duration(prefill=(64,))
    second_end = first_end + duration(prefill=(128,))
    decode_start = second_end + reused_prefill(128, 1)
    third_start = decode_start + duration(decode=(130,))
    fourth_start = third_start + reused_prefill(64, 64)
    fifth_start = fourth_start + duration(prefill=(64,))
    withdrawal_start = fifth_start + duration(prefill=(128,))
    rows = (
        (0, 0, 64, 1, (1,)),
        (1, first_end, 128, 1, (10, 11)),
        (2, second_end, 129, 3, (10, 11)),
        (3, (decode_start + third_start) / 2, 128, 2, (10, 11)),
        (4, (third_start + fourth_start) / 2, 64, 2, (30,)),
        (5, (fourth_start + fifth_start) / 2, 128, 2, (40, 41)),
        (6, withdrawal_start + duration(decode=(131, 65, 129)) / 2,
         448, 1, (1, 20, 21, 22, 23, 24, 25)),
    )
    source = TracePlan(
        block_size=64,
        sessions=("capacity",),
        requests=tuple(
            TraceTurn(
                request_id=key,
                session_id=0,
                arrival_offset_ms=arrival,
                prompt_len=prompt,
                output_len=output,
                hash_ids=hashes,
            )
            for key, arrival, prompt, output, hashes in rows
        ),
    )
    workload = WorkloadConfig(mode="trace", session_rate=1000, trace=TraceWorkloadConfig())

    result = simulate(
        resolved_config(workload=workload, engine=config, perf_model=perf),
        trace_plan=source,
    )

    assert result.meta.num_completed == 7 and result.meta.num_aborted == 0
    records = {item.id: item for item in result.requests}
    assert records[3].retry_count == 1
    assert records[6].actual_hit_tokens == records[6].l2_hit_tokens == 64
    assert records[6].actual_prefill_tokens == 384


def test_arrival_during_gpu_step_cannot_observe_future_published_page() -> None:
    class ObservingEngine(Engine):
        observed_during_step: CacheEntry | None = None

        def submit(self, request: InferenceRequest, now_ms: float) -> bool:
            if request.request_id == 1:
                self.observed_during_step = self._prefix_cache.page("chunk:0:0")
            return super().submit(request, now_ms)

    from test_engine import duration

    config = EngineConfig(max_running_requests=2, max_total_tokens=128, page_size=4)
    end = duration(prefill=(8,))
    plan = WorkloadPlan(
        turns={
            0: TurnPlan(0, 0, None, 0, None, 8, 1, 1, 0),
            1: TurnPlan(1, 1, None, end / 2, None, 4, 1, 1, 0),
        },
        root_request_ids=(0, 1),
        children={},
    )
    engine = ObservingEngine(runtime_config(config, PerfModelConfig()))
    run = Coordinator((engine,), planned_client(plan, config), Router(1)).run()[0]
    assert engine.observed_during_step is None
    assert engine._prefix_cache.page("chunk:0:0") is not None
    assert [item.prefill_start for item in run.records] == pytest.approx([0, end])


def test_zero_cost_multiple_steps_same_timestamp_keep_one_pending_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("simcore.engine.step_cost", lambda *args: StepCost(0, 0, 0, 0, 0))

    class CheckedCoordinator(Coordinator):
        max_pending = 0

        def _schedule_step(self, instance_id: int, now_ms: float) -> None:
            super()._schedule_step(instance_id, now_ms)
            count = sum(item.kind is EventKind.ENGINE_STEP for item in self._heap)
            self.max_pending = max(self.max_pending, count)
            assert count <= 1

    plan = make_plan(ordinary_at=0)
    config = EngineConfig(max_running_requests=5, max_total_tokens=128, page_size=4)
    zero = PerfModelConfig()
    engine = Engine(runtime_config(config, zero))
    coordinator = CheckedCoordinator((engine,), planned_client(plan, config), Router(1))
    run = coordinator.run()[0]
    assert coordinator.max_pending == 1
    assert run.num_engine_steps == 2
    assert [record.id for record in run.records] == [0, 1, 2, 3, 4]
    assert run.records[3].arrival == run.records[0].finish == 0
    assert run.records[4].arrival == run.records[1].finish == 0
    repeated = CheckedCoordinator(
        (Engine(runtime_config(config, zero)),),
        planned_client(plan, config),
        Router(1),
    ).run()[0]
    assert run == repeated
