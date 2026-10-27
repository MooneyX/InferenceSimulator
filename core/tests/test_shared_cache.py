"""共享资源池的副本、保护、跨实例回载及统计归属。"""

import pytest
from pydantic import ValidationError

from simcore import simulate
from simcore.client import Client
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.events import EventKind
from simcore.kv_cache import CacheCapacityError, CachePool, TieredKVCache
from simcore.prefix_hash import encode
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
from test_schemas import resolved_config, runtime_config


def request(key: int, ids: tuple[str, ...] = ()) -> InferenceRequest:
    return InferenceRequest(key, 0, None, len(ids) * 4 + 1, 1, 1, ids, len(ids))


def managers(capacity: int = 4) -> tuple[CachePool, TieredKVCache, TieredKVCache]:
    shared = CachePool("L3", 4, capacity, 4)
    return (
        shared,
        TieredKVCache(16, 4, 8, shared_l3=shared),
        TieredKVCache(
            16,
            4,
            8,
            shared_l3=shared,
            pool_id_base=2,
        ),
    )


def test_local_matching_uses_both_pools_then_continues_through_shared_l3() -> None:
    shared, first, second = managers()
    ids = tuple(encode(90, i) for i in range(5))
    first.pools["L2"].insert(ids[0])
    first.pools["L1"].insert(ids[1])
    shared.insert(ids[2])
    first.pools["L1"].insert(ids[3])
    req = request(0, ids)
    assert first.probe(req, req.prompt_len, include_l3=False) == tuple(
        zip(ids[:2], ("L2", "L1"), strict=True)
    )
    assert first.probe(req, req.prompt_len, start_index=2) == tuple(
        zip(ids[2:4], ("L3", "L1"), strict=True)
    )
    assert second.probe(req, req.prompt_len) == ()
    assert first.pools["L1"] is not second.pools["L1"]
    assert first.pools["L2"] is not second.pools["L2"]
    assert first.pools["L3"] is second.pools["L3"] is shared
    assert len({pool.pool_id for pool in (*first.pools.values(), *second.pools.values())}) == 5


def test_two_readers_have_independent_destinations_and_preserve_shared_source() -> None:
    shared, first, second = managers(1)
    chunk = encode(90, 0)
    shared.insert(chunk)
    for cache in (first, second):
        item = request(0, (chunk,))
        cache.pin(item, cache.probe(item, item.prompt_len))
        assert cache._chunk_pins == {chunk: {0}}
        assert cache.reserve_transfer(chunk, "L2", source="L3") == "L3"
        assert cache.pools["L2"].lookup(chunk) is None
        assert cache.pools["L2"].reserved == 1
    assert shared.lookup(chunk).ref_count == 4
    first.complete_transfer(chunk, "L2")
    assert shared.lookup(chunk).ref_count == 2
    assert first.pools["L2"].lookup(chunk).ref_count == 1
    assert second.pools["L2"].lookup(chunk) is None
    first.reserve_transfer(chunk, "L1", source="L2")
    first.complete_transfer(chunk, "L1")
    first.admit(0)
    assert first._chunk_pins == {}
    assert second._chunk_pins == {chunk: {0}}
    assert second._pin_sources == {0: {chunk: shared}}
    first.withdraw(0)
    second.complete_transfer(chunk, "L2")
    second.cancel_pin(0)
    assert second._chunk_pins == {}
    assert shared.lookup(chunk).ref_count == 0
    assert shared.resident == 1
    assert shared.evictions == 0
    assert first.pools["L1"].lookup(chunk) is not None
    assert second.pools["L2"].lookup(chunk) is not None
    shared.remove(chunk, evicted=True)
    assert first.pools["L1"].lookup(chunk) is not None
    assert second.pools["L2"].lookup(chunk) is not None


def test_shared_writes_deduplicate_without_removing_other_local_copies() -> None:
    shared, first, second = managers(1)
    chunk = encode(90, 0)
    for cache in (first, second):
        cache.pools["L1"].insert(chunk)
        cache.reserve_transfer(chunk, "L3", source="L1")
        cache.complete_transfer(chunk, "L3")
    assert shared.resident == 1 and shared.reserved == 0
    assert shared.lookup(chunk).ref_count == 0
    first.reserve_transfer(chunk, "L2", source="L3")
    first.complete_transfer(chunk, "L2")
    second.pools["L2"].insert(chunk)
    second.pools["L1"].insert(encode(91, 0))
    second.pools["L1"].insert(encode(92, 0))
    second.pools["L1"].insert(encode(93, 0))
    second.pools["L1"].insert(encode(94, 0))
    second.ensure_free(2)
    assert first.pools["L2"].lookup(chunk) is not None
    assert shared.resident == 1


def test_protected_shared_source_cannot_be_evicted_by_another_instance() -> None:
    shared, first, second = managers(1)
    chunk = encode(90, 0)
    shared.insert(chunk)
    first.reserve_transfer(chunk, "L2", source="L3")
    second.pools["L1"].insert(encode(91, 0))
    before = tuple(pool.snapshot() for pool in second.pools.values())
    with pytest.raises(CacheCapacityError):
        second.reserve_transfer(encode(91, 0), "L3", source="L1")
    assert tuple(pool.snapshot() for pool in second.pools.values()) == before
    first.complete_transfer(chunk, "L2")
    second.reserve_transfer(encode(91, 0), "L3", source="L1")
    second.complete_transfer(encode(91, 0), "L3")
    assert shared.lookup(chunk) is None
    assert first.pools["L2"].lookup(chunk) is not None


@pytest.mark.parametrize("first_finisher", [0, 1])
def test_simultaneous_shared_writes_reserve_one_page(first_finisher: int) -> None:
    shared, first, second = managers(1)
    chunk = encode(90, 0)
    for cache in (first, second):
        cache.pools["L1"].insert(chunk)
        cache.reserve_transfer(chunk, "L3", source="L1")
    assert shared.resident == shared.reserved == 1
    assert shared.lookup(chunk) is None
    readers = (first, second)
    readers[first_finisher].complete_transfer(chunk, "L3")
    assert shared.resident == 1 and shared.reserved == 0
    assert shared.lookup(chunk).ref_count == 1
    other = readers[1 - first_finisher]
    other.pools["L1"].insert(encode(91, 0))
    with pytest.raises(CacheCapacityError):
        other.reserve_transfer(encode(91, 0), "L3", source="L1")
    other.complete_transfer(chunk, "L3")
    assert shared.lookup(chunk).ref_count == 0
    assert len(shared.evictable) == shared.resident == 1
    assert shared.evictable == {(shared.lookup(chunk).last_access_seq, chunk)}
    assert not shared.pending_pages


def test_recomputed_l1_rebinds_old_pin_and_merges_inflight_l2_copy() -> None:
    shared, cache, _ = managers()
    chunk = encode(90, 0)
    shared.insert(chunk)
    waiting = request(0, (chunk,))
    cache.pin(waiting, cache.probe(waiting, waiting.prompt_len))
    cache.reserve_transfer(chunk, "L2", source="L3")
    recompute = request(1, (chunk,))
    cache.pin(recompute, ())
    cache.admit(1)
    cache.extend(1, 5)
    cache.publish(1)
    cache.admit(0)
    cache.withdraw(0)
    cache.finish(1)
    cache.complete_transfer(chunk, "L2")
    cache.reserve_transfer(chunk, "L2", source="L1")
    cache.complete_transfer(chunk, "L2")
    assert cache.pools["L1"].lookup(chunk) is None
    assert cache.pools["L2"].lookup(chunk).ref_count == 0
    assert len(cache.pools["L2"].evictable) == len(cache.pools["L2"].entries)
    assert cache.pools["L2"].evictable == {
        (page.last_access_seq, page.chunk_id) for page in cache.pools["L2"].entries.values()
    }
    assert shared.lookup(chunk).ref_count == 0


def test_pool_store_merges_existing_page_references() -> None:
    pool = CachePool("L1", 0, 2, 4)
    pool.insert("x", 1)
    assert not pool.store("x", 1)
    assert pool.resident == 1 and pool.lookup("x").ref_count == 2


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
            l3_to_l2_bandwidth_bytes_per_ms=64,
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
    shared, writer, _ = managers()
    chunk = encode(90, 0)
    writer.pools["L1"].insert(chunk)
    writer.reserve_transfer(chunk, "L3", source="L1")
    writer.complete_transfer(chunk, "L3")
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
                l3_to_l2_bandwidth_bytes_per_ms=64,
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
    engine.on_transfer_done(events[0][1], events[0][2], events[0][0])
    assert engine._prefix_cache.pools["L2"].lookup(chunk) is not None
    assert shared.lookup(chunk) is not None
    assert engine._restore_bytes == 64


@pytest.mark.parametrize("first_finisher", [0, 1])
def test_shared_l2_readers_copy_and_release_only_their_own_refs(first_finisher: int) -> None:
    shared = CachePool("L2", 4, 1, 4)
    caches = tuple(TieredKVCache(16, 4, shared_l2=shared, pool_id_base=i * 2) for i in range(2))
    chunk = encode(90, 0)
    shared.insert(chunk)
    for cache in caches:
        req = request(0, (chunk,))
        cache.pin(req, cache.probe(req, req.prompt_len))
        cache.reserve_transfer(chunk, "L1", source="L2")
    assert shared.lookup(chunk).ref_count == 4
    first, second = caches[first_finisher], caches[1 - first_finisher]
    first.complete_transfer(chunk, "L1")
    first.admit(0)
    first.withdraw(0)
    assert shared.lookup(chunk).ref_count == 2
    assert not first._evict_one("L2")
    second.complete_transfer(chunk, "L1")
    second.admit(0)
    second.withdraw(0)
    assert shared.lookup(chunk).ref_count == 0
    assert shared.resident == shared.peak == 1
    for cache in caches:
        assert cache.pools["L1"].lookup(chunk).ref_count == 0
        assert not cache._transfers
        assert not cache.pools["L1"].pending_pages
        cache._evict_one("L1")
    assert shared.resident == 1 and shared.evictions == 0


def test_shared_l2_recompute_preserves_protected_copy_and_global_eviction() -> None:
    l2, l3 = CachePool("L2", 4, 1, 4), CachePool("L3", 5, 2, 4)
    writer = TieredKVCache(16, 4, shared_l2=l2, shared_l3=l3)
    reader = TieredKVCache(16, 4, shared_l2=l2, shared_l3=l3, pool_id_base=2)
    chunk = encode(90, 0)
    l2.insert(chunk)
    reader.reserve_transfer(chunk, "L1", source="L2")
    req = request(1, (chunk,))
    writer.pin(req, ())
    writer.admit(1)
    writer.extend(1, 5)
    writer.publish(1)
    writer.finish(1)
    assert l2.lookup(chunk).ref_count == 1
    reader.complete_transfer(chunk, "L1")
    assert l2.lookup(chunk).ref_count == 0
    assert writer._evict_one("L2")
    assert l2.lookup(chunk) is None
    assert l3.lookup(chunk) is not None
    assert reader.pools["L1"].lookup(chunk) is not None


@pytest.mark.parametrize("shared_enabled", [True, False])
def test_l3_prefetch_coalesces_only_for_same_destination_pool(shared_enabled: bool) -> None:
    l2, l3 = CachePool("L2", 4, 4, 4), CachePool("L3", 5, 4, 4)
    chunk = encode(90, 0)
    l3.insert(chunk)
    runtime = runtime_config(
        EngineConfig(max_running_requests=2, max_total_tokens=16, page_size=4,
                     l2_capacity_tokens=8, l3_capacity_tokens=16, l3_prefetch_min_tokens=0),
        PerfModelConfig( kv_bytes_per_token=16,
                        l2_to_l1_bandwidth_bytes_per_ms=64,
                        l3_to_l2_bandwidth_bytes_per_ms=64, l3_lookup_latency_ms=2),
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
    assert engines[1]._link_free["L2"] == (0 if shared_enabled else 3)
    assert events[0][0] == events[1][0] == 3
    for engine, (time, kind, ref) in zip(engines, events, strict=True):
        engine.on_transfer_done(kind, ref, time)
    assert sum(engine._restore_bytes for engine in engines) == (64 if shared_enabled else 128)
    assert sum(engine._restore_time_ms for engine in engines) == (1 if shared_enabled else 2)
    assert l3.lookup(chunk).ref_count == 0
    for i, engine in enumerate(engines):
        assert not engine._prefetch_waiting
        assert engine._prefetched_origins[i] == {chunk}
        assert not engine._prefix_cache.pools["L2"].incoming_ready_ms
        assert not engine._prefix_cache.pools["L2"].pending_pages
        assert engine.start_step(3, 0) is None
        time, kind, ref = engine.drain_events()[0]
        engine.on_transfer_done(kind, ref, time)
        assert engine._restore_bytes == (128 if i == 0 or not shared_enabled else 64)
    assert l2.lookup(chunk) is not None if shared_enabled else l2.resident == 0


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
        PerfModelConfig( kv_bytes_per_token=16,
                        l2_to_l1_bandwidth_bytes_per_ms=64,
                        l3_to_l2_bandwidth_bytes_per_ms=64, l3_lookup_latency_ms=2),
    )
    source = TracePlan(sessions=("shared",), block_size=4, requests=tuple(
        TraceTurn(request_id=i, session_id=0, arrival_offset_ms=0,
                  prompt_len=5, output_len=1, hash_ids=(90,)) for i in range(2)
    ))
    workload = WorkloadConfig(mode="trace", arrival_mode="open_loop", session_rate=1,
                              trace=TraceWorkloadConfig())
    l2, l3 = CachePool("L2", 4, 4, 4), CachePool("L3", 5, 2, 4)
    l3.insert("trace:0:90")
    engines = tuple(Engine(runtime, shared_l2=l2, shared_l3=l3, instance_id=i)
                    for i in range(2))
    runs = Coordinator(engines, Client(workload, runtime.engine, trace_plan=source),
                       Router(2, policy="least_load")).run()
    assert [len(run.records) for run in runs] == [1, 1]
    assert [run.records[0].l3_hit_tokens for run in runs] == [4, 4]
    assert runs[0].records[0].prefill_start == runs[1].records[0].prefill_start
    assert sum(run.restore_bytes for run in runs) == 192
    assert sum(run.restore_time_ms for run in runs) == 3
    assert not l2.pending_pages and not l2.incoming_ready_ms
    assert l2.lookup("trace:0:90").ref_count == l3.lookup("trace:0:90").ref_count == 0
    assert all(not engine._pending_loads and not engine._prefetch_waiting for engine in engines)


@pytest.mark.parametrize("owns_prefetch", [True, False])
@pytest.mark.parametrize("prefetch_finishes_first", [True, False])
def test_l2_load_overlaps_same_page_prefetch_after_another_instance_demotes(
    owns_prefetch: bool, prefetch_finishes_first: bool
) -> None:
    l2, l3 = CachePool("L2", 4, 2, 4), CachePool("L3", 5, 2, 4)
    chunk = encode(90, 0)
    l3.insert(chunk)
    runtime = runtime_config(
        EngineConfig(max_running_requests=2, max_total_tokens=16, page_size=4,
                     l2_capacity_tokens=8, l3_capacity_tokens=8, l3_prefetch_min_tokens=0),
        PerfModelConfig( kv_bytes_per_token=16,
                        l2_to_l1_bandwidth_bytes_per_ms=64 if not prefetch_finishes_first else 16,
                        l3_to_l2_bandwidth_bytes_per_ms=32, l3_lookup_latency_ms=2),
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
    writer = TieredKVCache(4, 4, shared_l2=l2, shared_l3=l3, pool_id_base=6)
    writer.pools["L1"].insert(chunk)
    writer.ensure_free(1)
    assert l2.lookup(chunk) is not None
    engine.submit(request(2, (chunk,)), 2)
    assert engine.drain_events() == ()
    assert engine.start_step(2, 0) is None
    (load_event,) = engine.drain_events()
    assert load_event[1] is EventKind.L2_L1_DONE
    events = [*prefetch_events, (engine, load_event)]
    for target, (time, kind, ref) in sorted(events, key=lambda item: item[1][0]):
        target.on_transfer_done(kind, ref, time)
    assert engine._prefix_cache.pools["L1"].lookup(chunk) is not None
    assert not engine._pending_loads
    assert not engine._transfer_by_chunk
    assert not engine._prefix_cache._transfers
    assert not l2.pending_pages and not l2.incoming_ready_ms
    assert l3.lookup(chunk).ref_count == 0
    assert l2.lookup(chunk).ref_count == 0
    assert sum(item._restore_bytes for item in (owner, follower)) == 128


@pytest.mark.parametrize("enabled", [True, False])
def test_simulation_reuses_another_instances_evicted_l1_only_when_shared(
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
