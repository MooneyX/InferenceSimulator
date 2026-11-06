"""共享资源池的副本、保护、跨实例回载及统计归属。"""

import pytest
from pydantic import ValidationError

from simcore import simulate
from simcore.client import Client
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.events import EventKind
from simcore.kv_cache import CachePool, TieredKVCache
from simcore.prefix_hash import chunk_base
from simcore.radix_cache import RadixTree
from simcore.router import Router
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    InferenceRequest,
    PerfModelConfig,
    SessionConfig,
    SimConfig,
    SimResult,
    SyntheticWorkloadConfig,
    TracePlan,
    TraceTurn,
    TraceWorkloadConfig,
    TurnPlan,
    WorkloadConfig,
    WorkloadPlan,
)
from test_radix_cache import PageView, assert_tree, find_node, page, run_prefill
from test_schemas import resolved_config, runtime_config


def request(key: int, ids: tuple[int, ...] = ()) -> InferenceRequest:
    return InferenceRequest(key, 0, None, len(ids) * 4 + 1, 1, 1, ids, len(ids), chunk_base(key))


def managers(capacity: int = 4) -> tuple[CachePool, TieredKVCache, TieredKVCache]:
    shared = CachePool(capacity)
    return (
        shared,
        TieredKVCache(16, 4, 8, shared_l3=shared),
        TieredKVCache(16, 4, 8, shared_l3=shared, instance_id=1),
    )


def test_local_matching_uses_tree_then_continues_through_shared_l3() -> None:
    shared, first, second = managers()
    ids = tuple(chunk_base(90) + i for i in range(5))
    run_prefill(first, InferenceRequest(9, 0, None, 9, 1, 1, ids[:2], 2, chunk_base(9)))
    first.finish(9)
    for item in ids[2:4]:
        shared.insert(item)
    req = request(0, ids)
    local = first.probe(req, req.prompt_len, include_l3=False)
    assert (local.l1_pages, local.local_pages, local.lower) == (2, 2, ())
    match = first.probe(req, req.prompt_len)
    assert match.lower == tuple((item, "L3") for item in ids[2:4])
    assert second.probe(req, req.prompt_len).pages == 0
    assert first.tree is not second.tree and first.hbm is not second.hbm
    assert first.l3 is second.l3 is shared


def test_two_readers_have_independent_destinations_and_preserve_shared_source() -> None:
    shared, first, second = managers(1)
    chunk = chunk_base(90)
    shared.insert(chunk)
    item = request(0, (chunk,))
    for cache in (first, second):
        assert cache.probe(item, item.prompt_len).l3_pages == 1
        cache.reserve_prefetch(chunk)
        assert cache.tier_snapshot("L2").resident_blocks == 1
        assert find_node(cache.tree, chunk) is None
    assert shared.entries[chunk].ref_count == 2
    first.complete_prefetch(item, ((0, chunk),))
    assert shared.entries[chunk].ref_count == 1
    assert page(first, chunk) == PageView("L2", 0, True)
    assert find_node(second.tree, chunk) is None
    second.complete_prefetch(item, ((0, chunk),))
    assert shared.entries[chunk].ref_count == 0
    assert shared.resident == 1 and shared.evictions == 0
    shared.remove(chunk, evicted=True)
    assert page(first, chunk) == page(second, chunk) == PageView("L2", 0, True)


def test_shared_l3_writes_deduplicate_across_instances() -> None:
    shared = CachePool(1)
    caches = tuple(TieredKVCache(8, 4, 0, shared_l3=shared, instance_id=i) for i in range(2))
    chunk = chunk_base(90)
    for index, cache in enumerate(caches):
        req = request(index, (chunk,))
        cache.pin(req, cache.probe(req, req.prompt_len, include_l3=False))  # 各自重算同一页
        cache.admit(index)
        cache.extend(index, req.prompt_len)
        cache.finish(index)
        assert cache.ensure_free(2) == 1
    assert shared.resident == 1 and shared.entries[chunk].ref_count == 0
    assert all(cache.demotions == 1 for cache in caches)


def test_protected_shared_source_cannot_be_evicted_by_another_instance() -> None:
    shared = CachePool(1)
    reader = TieredKVCache(16, 4, 8, shared_l3=shared)
    writer = TieredKVCache(8, 4, 0, shared_l3=shared, instance_id=1)
    chunk = chunk_base(90)
    shared.insert(chunk)
    reader.reserve_prefetch(chunk)
    run_prefill(writer, request(1, (chunk_base(91),)))
    writer.finish(1)
    writer.ensure_free(2)
    assert page(writer, chunk_base(91)) is None  # L3 已满且唯一页受保护：写入被丢弃
    reader.complete_prefetch(request(0, (chunk,)), ((0, chunk),))
    run_prefill(writer, request(2, (chunk_base(92),)))
    writer.finish(2)
    writer.ensure_free(2)
    assert shared.lookup(chunk) is None and shared.lookup(chunk_base(92)) is not None
    assert page(reader, chunk) == PageView("L2", 0, True)


def test_recomputed_page_merges_with_inflight_l2_prefetch() -> None:
    shared, cache, _ = managers()
    chunk = chunk_base(90)
    shared.insert(chunk)
    cache.reserve_prefetch(chunk)
    recompute = request(1, (chunk,))
    run_prefill(cache, recompute)
    cache.finish(1)
    assert cache.tier_snapshot("L2").resident_blocks == 2  # write-through 副本与预取占位
    cache.complete_prefetch(request(0, (chunk,)), ((0, chunk),))
    snapshot = cache.tier_snapshot("L2")
    assert snapshot.resident_blocks == 1 and snapshot.active_blocks == 1
    assert page(cache, chunk) == PageView("L1", 0, True)
    assert shared.entries[chunk].ref_count == 0
    assert_tree(cache.tree, [cache.hbm])


def settings(*, instances: int = 3, prompt: int = 4) -> SimConfig:
    return resolved_config(
        num_instances=instances,
        workload=WorkloadConfig(
            session_rate=100,
            seed=5,
            synthetic=SyntheticWorkloadConfig(
                num_requests=3,
                prompt_len=FixedLength(value=prompt),
                turn_increment_len=FixedLength(value=1),
                output_len=FixedLength(value=1),
                session=SessionConfig(num_sessions=3),
            ),
        ),
        engine=EngineConfig(
            max_running_requests=1,
            max_total_tokens=16,
            page_size=4,
            l2_capacity_tokens=8,
            l3_capacity_tokens=32,
            l3_prefetch_min_tokens=0,
        ),
        perf_model=PerfModelConfig(
            kv_bytes_per_token=16,
            l2_to_l1_bandwidth_bytes_per_ms=64,
            l3_read_bandwidth_bytes_per_ms=64,
            l3_lookup_latency_ms=2,
        ),
    )


@pytest.mark.parametrize("prompt", [4, 32])
def test_shared_capacity_is_counted_once_even_without_successful_requests(prompt: int) -> None:
    config = settings(prompt=prompt)
    result = simulate(config)
    assert result.metrics.l1_capacity_blocks == 12
    assert result.metrics.l2_capacity_blocks == 6
    assert result.metrics.l3_capacity_blocks == 8
    assert all(item.metrics.l3_capacity_blocks is None for item in result.instances)
    assert SimResult.model_validate_json(result.model_dump_json()) == result
    assert result.meta.num_completed == (3 if prompt == 4 else 0)
    repeat = simulate(config)
    assert result.model_dump(exclude={"meta": {"wall_clock_ms"}}) == repeat.model_dump(
        exclude={"meta": {"wall_clock_ms"}}
    )


def test_engine_prefetches_a_page_written_by_another_instance() -> None:
    shared, _, _ = managers()
    chunk = chunk_base(90)
    shared.insert(chunk)
    config = settings()
    engine = Engine(
        runtime_config(
            EngineConfig(
                max_running_requests=1,
                max_total_tokens=16,
                page_size=4,
                l2_capacity_tokens=8,
                l3_capacity_tokens=16,
                l3_prefetch_min_tokens=0,
            ),
            PerfModelConfig(
                kv_bytes_per_token=16,
                l2_to_l1_bandwidth_bytes_per_ms=64,
                l3_read_bandwidth_bytes_per_ms=64,
                l3_lookup_latency_ms=config.shared_l3.lookup_latency_ms,
            ),
        ),
        shared_l3=shared,
        instance_id=1,
    )
    engine.submit(request(0, (chunk,)), 0)
    assert engine.drain_events() == ((2, EventKind.L3_LOOKUP_DONE, 0),)
    engine.on_l3_lookup(0, 2)
    events = engine.drain_events()
    assert len(events) == 1 and events[0][1] is EventKind.L3_L2_DONE
    engine.on_transfer_done(events[0][2], events[0][0])
    assert page(engine._prefix_cache, chunk) == PageView("L2", 0, True)
    assert shared.lookup(chunk) is not None
    assert engine._restore_bytes == 64


def test_shared_l2_readers_copy_and_release_only_their_own_locks() -> None:
    l3 = CachePool(0)
    tree = RadixTree(4, 1, l3)
    caches = tuple(
        TieredKVCache(16, 4, shared_l2=tree, shared_l3=l3, instance_id=i) for i in range(2)
    )
    chunk = chunk_base(90)
    req = request(0, (chunk,))
    run_prefill(caches[0], req)
    caches[0].finish(0)
    caches[0].ensure_free(4)
    node = find_node(tree, chunk)
    assert node is not None and node.l2 and node.hbm == 0
    for cache in caches:
        match = cache.probe(req, req.prompt_len)
        assert (match.l1_pages, match.local_pages) == (0, 1)
        cache.pin(req, match)
        cache.ensure_free(1)
        assert cache.promote() == 1
    assert node.hbm == 2 and node.locks == 2
    for cache in caches:
        cache.admit(0)
        cache.withdraw(0)
    assert node.locks == 0 and tree.l2_evictable == 0
    caches[0].ensure_free(4)
    assert node.hbm == 1 and not tree.ensure_l2_free(caches[0].hbm, 1)
    caches[1].ensure_free(4)
    assert tree.l2_evictable == 1 and tree.ensure_l2_free(caches[1].hbm, 1)
    assert find_node(tree, chunk) is None and tree.l2_evictions == 1
    assert_tree(tree, [cache.hbm for cache in caches])


@pytest.mark.parametrize("shared_enabled", [True, False])
def test_l3_prefetch_coalesces_only_for_same_destination_tree(shared_enabled: bool) -> None:
    l3 = CachePool(4)
    l2 = RadixTree(4, 4, l3)
    chunk = chunk_base(90)
    l3.insert(chunk)
    runtime = runtime_config(
        EngineConfig(max_running_requests=2, max_total_tokens=16, page_size=4,
                     l2_capacity_tokens=8, l3_capacity_tokens=16, l3_prefetch_min_tokens=0),
        PerfModelConfig(kv_bytes_per_token=16,
                        l2_to_l1_bandwidth_bytes_per_ms=64,
                        l3_read_bandwidth_bytes_per_ms=64, l3_lookup_latency_ms=2),
    )
    engines = tuple(Engine(runtime, shared_l2=l2 if shared_enabled else None,
                           shared_l3=l3, instance_id=i) for i in range(2))
    events = []
    for i, engine in enumerate(engines):
        engine.submit(request(i, (chunk,)), 0)
        engine.drain_events()
        engine.on_l3_lookup(i, 2)
        events.append(engine.drain_events()[0])
    assert l3.lookup(chunk).ref_count == (1 if shared_enabled else 2)
    l3_source = engines[1]._l3_source
    assert l3_source is not None and l3_source.free_ms == (0 if shared_enabled else 3)
    assert events[0][0] == events[1][0] == 3
    for engine, (time, _, ref) in zip(engines, events, strict=True):
        engine.on_transfer_done(ref, time)
    assert sum(engine._restore_bytes for engine in engines) == (64 if shared_enabled else 128)
    assert sum(engine._restore_time_ms for engine in engines) == (1 if shared_enabled else 2)
    assert l3.lookup(chunk).ref_count == 0
    for i, engine in enumerate(engines):
        assert not engine._prefetch_waiting
        assert engine._prefetched_origins[i] == {chunk}
        assert not engine._prefix_cache.l2_incoming_ms
        assert engine._prefix_cache.tree.l2_reserved == 0
        assert engine.start_step(3, 0) is not None
        assert engine.drain_events() == ()
        assert engine._restore_bytes == (128 if i == 0 or not shared_enabled else 64)
    assert (find_node(l2, chunk) is not None) if shared_enabled else l2.l2_pages == 0


@pytest.mark.parametrize("enabled", [True, False])
def test_shared_l2_config_roundtrip_and_metrics_scope(enabled: bool) -> None:
    config = settings().model_copy(update={"shared_l2_enabled": enabled})
    assert SimConfig.model_validate_json(config.model_dump_json()).shared_l2_enabled is enabled
    result = simulate(config)
    assert result.metrics.l2_capacity_blocks == 6
    assert all(item.metrics.l2_capacity_blocks == (None if enabled else 2)
               for item in result.instances)
    assert result.metrics.restore_bytes == sum(
        item.metrics.restore_bytes for item in result.instances
    )


def test_coordinator_completes_shared_prefetch_before_all_reader_notifications() -> None:
    runtime = runtime_config(
        EngineConfig(max_running_requests=2, max_total_tokens=16, page_size=4,
                     l2_capacity_tokens=8, l3_capacity_tokens=8, l3_prefetch_min_tokens=0),
        PerfModelConfig(kv_bytes_per_token=16,
                        l2_to_l1_bandwidth_bytes_per_ms=64,
                        l3_read_bandwidth_bytes_per_ms=64, l3_lookup_latency_ms=2),
    )
    source = TracePlan(sessions=("shared",), block_size=4, requests=tuple(
        TraceTurn(request_id=i, session_id=0, arrival_offset_ms=0,
                  prompt_len=5, output_len=1, hash_ids=(90,)) for i in range(2)
    ))
    workload = WorkloadConfig(mode="trace", arrival_mode="open_loop", session_rate=1,
                              trace=TraceWorkloadConfig())
    l3 = CachePool(2)
    l2 = RadixTree(4, 4, l3)
    l3.insert(chunk_base(0))  # 会话内唯一块 90 重编号为偏移 0
    engines = tuple(Engine(runtime, shared_l2=l2, shared_l3=l3, instance_id=i)
                    for i in range(2))
    runs = Coordinator(engines, Client(workload, runtime.engine, trace_plan=source),
                       Router(2, policy="least_load")).run()
    assert [len(run.records) for run in runs] == [1, 1]
    assert [run.records[0].l3_hit_tokens for run in runs] == [4, 4]
    assert runs[0].records[0].prefill_start == runs[1].records[0].prefill_start
    assert sum(run.restore_bytes for run in runs) == 192
    assert sum(run.restore_time_ms for run in runs) == 3
    assert not l2.l2_incoming_ms and l2.l2_reserved == 0
    node = find_node(l2, chunk_base(0))
    assert node is not None and node.locks == l3.entries[chunk_base(0)].ref_count == 0
    assert all(not engine._transfers and not engine._prefetch_waiting for engine in engines)


def test_shared_prefetch_coalesces_across_storage_batches_and_waits_for_all_sources() -> None:
    from unittest.mock import patch

    runtime = runtime_config(
        EngineConfig(max_running_requests=2, max_total_tokens=1024, page_size=4,
                     l2_capacity_tokens=520, l3_capacity_tokens=520, l3_prefetch_min_tokens=0),
        PerfModelConfig(kv_bytes_per_token=16, l2_to_l1_bandwidth_bytes_per_ms=64000,
                        l3_read_bandwidth_bytes_per_ms=64, l3_lookup_latency_ms=2,
                        l3_latency_ms=1),
    )
    source = TracePlan(sessions=("shared",), block_size=4, requests=tuple(
        TraceTurn(request_id=i, session_id=0, arrival_offset_ms=1000 if i == 3 else 0,
                  prompt_len=count * 4 + 1, output_len=1, hash_ids=tuple(range(count)))
        for i, count in enumerate((129, 128, 130, 130))
    ))
    workload = WorkloadConfig(mode="trace", arrival_mode="open_loop", session_rate=1,
                              trace=TraceWorkloadConfig())
    l3 = CachePool(130)
    l2 = RadixTree(4, 130, l3)
    for index in range(130):
        l3.insert(chunk_base(0) + index)
    engines = tuple(Engine(runtime, shared_l2=l2, shared_l3=l3, instance_id=i)
                    for i in range(2))
    coordinator = Coordinator(engines, Client(workload, runtime.engine, trace_plan=source),
                              Router(2, policy="least_load"))
    first_source, second_source = (engine._l3_source for engine in engines)
    assert first_source is not None and second_source is not None
    with (
        patch.object(engines[1], "on_transfer_done", wraps=engines[1].on_transfer_done) as done,
        patch.object(type(first_source), "submit", autospec=True,
                     side_effect=type(first_source).submit) as loads,
    ):
        runs = coordinator.run()
    records = {record.id: record for run in runs for record in run.records}
    arrival = records[0].arrival
    # 相对到达：首批 128 页于 131 ms 到齐，尾批于 133 ms；另一个来源的新页于 4 ms。
    assert records[1].prefill_start == pytest.approx(arrival + 131)
    assert records[0].prefill_start == records[2].prefill_start == pytest.approx(arrival + 133)
    assert done.call_args_list[0].args[1] == pytest.approx(arrival + 4)
    assert [call.args[2] for call in loads.call_args_list
            if call.args[0] is first_source] == [[128 * 64, 64]]
    assert [call.args[2] for call in loads.call_args_list
            if call.args[0] is second_source] == [[64]]
    assert not l2.l2_incoming_ms and l2.l2_reserved == 0
    assert all(not engine._transfers and not engine._prefetch_waiting for engine in engines)
    assert all(page.ref_count == 0 for page in l3.entries.values())
    assert_tree(l2, [engine._prefix_cache.hbm for engine in engines])
    assert all(engine._prefix_cache.active_pages == 0 for engine in engines)


@pytest.mark.parametrize("owns_prefetch", [True, False])
def test_l2_promotion_overlaps_same_page_prefetch_after_another_instance_writes_it(
    owns_prefetch: bool,
) -> None:
    l3 = CachePool(2)
    l2 = RadixTree(4, 2, l3)
    chunk = chunk_base(90)
    l3.insert(chunk)
    runtime = runtime_config(
        EngineConfig(max_running_requests=2, max_total_tokens=16, page_size=4,
                     l2_capacity_tokens=8, l3_capacity_tokens=8, l3_prefetch_min_tokens=0),
        PerfModelConfig(kv_bytes_per_token=16,
                        l2_to_l1_bandwidth_bytes_per_ms=64,
                        l3_read_bandwidth_bytes_per_ms=32, l3_lookup_latency_ms=2),
    )
    owner, follower = (Engine(runtime, shared_l2=l2, shared_l3=l3, instance_id=i)
                       for i in range(2))
    engine = owner if owns_prefetch else follower
    owner.submit(request(0, (chunk,)), 0)
    owner.drain_events()
    owner.on_l3_lookup(0, 2)
    prefetch_events = [(owner, owner.drain_events()[0])]
    if not owns_prefetch:
        follower.submit(request(1, (chunk,)), 0)
        follower.drain_events()
        follower.on_l3_lookup(1, 2)
        prefetch_events.append((follower, follower.drain_events()[0]))
    writer = TieredKVCache(8, 4, shared_l2=l2, shared_l3=l3, instance_id=6)
    run_prefill(writer, request(9, (chunk,)))
    writer.finish(9)
    assert page(writer, chunk) == PageView("L1", 0, True)  # write-through 到共享 L2
    engine.submit(request(2, (chunk,)), 2)
    assert engine.drain_events() == ()
    started = engine.start_step(2, 0)
    assert started is not None and started[1] == (2,)
    assert engine.drain_events() == ()
    assert page(engine._prefix_cache, chunk) == PageView("L1", 1, True)
    for target, (time, _, ref) in prefetch_events:
        target.on_transfer_done(ref, time)
    assert not engine._transfer_by_chunk
    assert not l2.l2_incoming_ms and l2.l2_reserved == 0
    assert l3.lookup(chunk).ref_count == 0
    assert sum(item._restore_bytes for item in (owner, follower)) == 128
    assert_tree(l2, [owner._prefix_cache.hbm, follower._prefix_cache.hbm, writer.hbm])


@pytest.mark.parametrize("enabled", [True, False])
def test_simulation_reuses_another_instances_kv_only_when_shared(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = WorkloadPlan(
        turns={
            0: TurnPlan(0, 0, None, 0, None, 4, 1, 1, 0),
            1: TurnPlan(1, 1, None, 10, None, 1, 1, 1, 0),
            2: TurnPlan(2, 2, None, 20, None, 8, 1, 1, 0),
            3: TurnPlan(3, 3, None, 30, None, 1, 1, 1, 0),
            4: TurnPlan(4, 4, None, 40, None, 4, 1, 1, 0),
            5: TurnPlan(5, 0, 0, None, 60, 5, 1, 1, 1),
        },
        root_request_ids=(0, 1, 2, 3, 4), children={0: (5,)},
    )
    monkeypatch.setattr("simcore.client.build_synthetic_closed_loop_plan", lambda *_: plan)
    config = resolved_config(
        num_instances=2, workload=settings().workload,
        engine=EngineConfig(max_running_requests=1, max_total_tokens=12, page_size=4,
                            l2_capacity_tokens=8),
        perf_model=PerfModelConfig(
                                   kv_bytes_per_token=16, l2_to_l1_bandwidth_bytes_per_ms=64),
    ).model_copy(update={"shared_l2_enabled": enabled})
    result = simulate(config)
    assert result.meta.num_completed == 6
    assert [item.instance_id for item in result.requests] == [0, 1, 0, 1, 0, 1]
    assert result.requests[5].l2_hit_tokens == (4 if enabled else 0)
    assert result.metrics.l2_hit_tokens == (4 if enabled else 0)
    assert result.metrics.l2_capacity_blocks == 4
    repeat = simulate(config)
    assert result.model_dump(exclude={"meta": {"wall_clock_ms"}}) == repeat.model_dump(
        exclude={"meta": {"wall_clock_ms"}}
    )


@pytest.mark.parametrize("instances", [1, 3])
@pytest.mark.parametrize("enabled", [True, False])
def test_zero_capacity_l2_does_not_change_simulation(instances: int, enabled: bool) -> None:
    config = settings(instances=instances)
    storage = config.inference.storage.model_copy(update={"l2_available_gb": 0})
    config = config.model_copy(update={
        "shared_l2_enabled": enabled, "shared_l3": None,
        "inference": config.inference.model_copy(update={"storage": storage}),
    })
    result = simulate(config)
    assert result.metrics.l2_capacity_blocks == result.metrics.l2_resident_blocks == 0
    assert result.meta.num_completed == 3
    assert result.metrics.restore_bytes == 0


def test_shared_l2_defaults_to_enabled_and_rejects_non_boolean() -> None:
    payload = settings().model_dump()
    payload.pop("shared_l2_enabled")
    assert SimConfig.model_validate(payload).shared_l2_enabled is True
    for value in (None, 1, "false"):
        with pytest.raises(ValidationError):
            SimConfig.model_validate({**payload, "shared_l2_enabled": value})


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_shared_physical_capacity_rejects_invalid_budget(value: float) -> None:
    payload = settings().model_dump()
    payload["shared_l3"]["available_gb"] = value
    with pytest.raises(ValidationError):
        SimConfig.model_validate(payload)
