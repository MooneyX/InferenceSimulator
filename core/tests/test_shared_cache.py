"""共享资源池的副本、保护、跨实例回载及统计归属。"""

import pytest
from pydantic import ValidationError

from simcore import simulate
from simcore.engine import Engine
from simcore.events import EventKind
from simcore.kv_cache import CacheCapacityError, CachePool, TieredKVCache
from simcore.prefix_hash import encode
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    InferenceRequest,
    PerfModelConfig,
    SessionConfig,
    SimConfig,
    SimResult,
    SyntheticWorkloadConfig,
    WorkloadConfig,
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
    for cache, key in ((first, 0), (second, 1)):
        cache.pin(request(key, (chunk,)), 4)
        assert cache.reserve_transfer(chunk, "L2") == "L3"
        assert cache.pools["L2"].lookup(chunk) is None
        assert cache.pools["L2"].reserved == 1
    assert shared.lookup(chunk).ref_count == 4
    first.complete_transfer(chunk)
    assert shared.lookup(chunk).ref_count == 2
    assert first.pools["L2"].lookup(chunk).ref_count == 1
    assert second.pools["L2"].lookup(chunk) is None
    first.reserve_transfer(chunk, "L1")
    first.complete_transfer(chunk)
    first.admit(0)
    first.withdraw(0)
    second.complete_transfer(chunk)
    second.cancel_pin(1)
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
        cache.reserve_transfer(chunk, "L3")
        cache.complete_transfer(chunk)
    assert shared.resident == 1 and shared.reserved == 0
    assert shared.lookup(chunk).ref_count == 0
    first.reserve_transfer(chunk, "L2")
    first.complete_transfer(chunk)
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
    first.reserve_transfer(chunk, "L2")
    second.pools["L1"].insert(encode(91, 0))
    before = tuple(pool.snapshot() for pool in second.pools.values())
    with pytest.raises(CacheCapacityError):
        second.reserve_transfer(encode(91, 0), "L3")
    assert tuple(pool.snapshot() for pool in second.pools.values()) == before
    first.complete_transfer(chunk)
    second.reserve_transfer(encode(91, 0), "L3")
    second.complete_transfer(encode(91, 0))
    assert shared.lookup(chunk) is None
    assert first.pools["L2"].lookup(chunk) is not None


@pytest.mark.parametrize("first_finisher", [0, 1])
def test_simultaneous_shared_writes_reserve_one_page(first_finisher: int) -> None:
    shared, first, second = managers(1)
    chunk = encode(90, 0)
    for cache in (first, second):
        cache.pools["L1"].insert(chunk)
        cache.reserve_transfer(chunk, "L3")
    assert shared.resident == shared.reserved == 1
    assert shared.lookup(chunk) is None
    readers = (first, second)
    readers[first_finisher].complete_transfer(chunk)
    assert shared.resident == 1 and shared.reserved == 0
    assert shared.lookup(chunk).ref_count == 1
    other = readers[1 - first_finisher]
    other.pools["L1"].insert(encode(91, 0))
    with pytest.raises(CacheCapacityError):
        other.reserve_transfer(encode(91, 0), "L3")
    other.complete_transfer(chunk)
    assert shared.lookup(chunk).ref_count == 0
    assert shared.evictable == shared.resident == 1
    assert not shared.pending_pages


def test_recomputed_l1_rebinds_old_pin_and_merges_inflight_l2_copy() -> None:
    shared, cache, _ = managers()
    chunk = encode(90, 0)
    shared.insert(chunk)
    waiting = request(0, (chunk,))
    cache.pin(waiting, 4)
    cache.reserve_transfer(chunk, "L2")
    recompute = request(1, (chunk,))
    cache.pin(recompute, 0)
    cache.admit(1)
    cache.extend(1, 5)
    cache.publish(1)
    cache.admit(0)
    cache.withdraw(0)
    cache.finish(1)
    cache.complete_transfer(chunk)
    cache.reserve_transfer(chunk, "L2")
    cache.complete_transfer(chunk)
    assert cache.pools["L1"].lookup(chunk) is None
    assert cache.pools["L2"].lookup(chunk).ref_count == 0
    assert cache.pools["L2"].evictable == len(cache.pools["L2"].entries)
    assert shared.lookup(chunk).ref_count == 0


def test_repeated_pool_insert_is_rejected_without_corrupting_state() -> None:
    pool = CachePool("L1", 0, 2, 4)
    pool.insert("x", 1)
    before = pool.snapshot(), dict(pool.entries), pool.access_seq
    with pytest.raises(ValueError, match="已存在"):
        pool.insert("x")
    assert (pool.snapshot(), pool.entries, pool.access_seq) == before
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
            prefill_c1=1,
            decode_d1=1,
            decode_d2=0,
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
    writer.reserve_transfer(chunk, "L3")
    writer.complete_transfer(chunk)
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
                prefill_c1=1,
                decode_d1=1,
                decode_d2=0,
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


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_shared_physical_capacity_rejects_invalid_budget(value: float) -> None:
    payload = settings().model_dump()
    payload["shared_l3"]["available_gb"] = value
    with pytest.raises(ValidationError):
        SimConfig.model_validate(payload)
