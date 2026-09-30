"""L1 paged KV 的块容量、原子事务、共享、发布、增长与可复现淘汰。"""

from dataclasses import FrozenInstanceError

import pytest

from simcore.kv_cache import (
    AdmissionAllocation,
    CacheCapacityError,
    CacheOOMError,
    HierarchicalKVCache,
)


def admit(
    cache: HierarchicalKVCache,
    key: int,
    hashes: tuple[str, ...],
    prompt: int = 4,
    output: int = 1,
) -> tuple[int, ...]:
    snapshot = cache.snapshot()
    allocation = AdmissionAllocation(key, prompt, output, hashes)
    probe = cache.probe(snapshot, hashes)
    result = cache.commit(cache.prepare_batch(snapshot, (probe,), (allocation,)))
    return result.requests[0].block_ids


def test_snapshot_probe_purity_and_prefix_contiguity() -> None:
    cache = HierarchicalKVCache(5, 2)
    ids = admit(cache, 0, ("a", "b"))
    before = cache.snapshot()
    assert cache.probe(before, ("a", "b")).matched_blocks == 0
    cache.publish(0, ("a", "b"))
    snapshot = cache.snapshot()
    seq = cache.access_seq
    assert cache.probe(snapshot, ("a", "unknown", "b")).matched_blocks == 1
    assert cache.probe(snapshot, ("b",)).matched_blocks == 1
    assert cache.probe(snapshot, ("a", "b")).block_refs == ((ids[0], 1), (ids[1], 1))
    assert cache.snapshot() == snapshot and cache.access_seq == seq
    with pytest.raises((FrozenInstanceError, AttributeError)):
        snapshot.epoch = 100  # type: ignore[misc]
    with pytest.raises(ValueError, match="过期"):
        allocation = AdmissionAllocation(1, 4, 1, ("a", "b"))
        cache.prepare_batch(before, (cache.probe(before, ("a", "b")),), (allocation,))


def test_same_snapshot_batch_miss_and_shared_hit() -> None:
    cache = HierarchicalKVCache(8, 2)
    snapshot = cache.snapshot()
    allocation = (AdmissionAllocation(0, 2, 1, ("a",)), AdmissionAllocation(1, 2, 1, ("a",)))
    probes = tuple(cache.probe(snapshot, item.input_chunk_ids) for item in allocation)
    assert cache.snapshot() == snapshot
    tx = cache.prepare_batch(snapshot, probes, allocation)
    assert cache.snapshot() == snapshot
    first = cache.commit(tx)
    assert first.allocated_blocks == 2
    assert first.requests[0].block_ids != first.requests[1].block_ids
    cache.publish(0, ("a",))
    cache.publish(1, ("a",))
    cache.release(0)
    cache.release(1)
    old = cache.snapshot()
    assert dict(old.hash_index)["a"] == (0, 1)
    snapshot = cache.snapshot()
    allocation = (AdmissionAllocation(2, 2, 1, ("a",)), AdmissionAllocation(3, 2, 1, ("a",)))
    probes = tuple(cache.probe(snapshot, item.input_chunk_ids) for item in allocation)
    result = cache.commit(cache.prepare_batch(snapshot, probes, allocation))
    assert result.allocated_blocks == 0
    assert result.reused_block_ids == (0,)
    assert result.requests[0].matched_cache_tokens == 2
    assert result.requests[0].actual_hit_tokens == 1
    assert result.requests[0].new_blocks == 0
    assert cache.blocks[0].ref_count == 2
    cache.release(2)
    assert cache.blocks[0].ref_count == 1
    cache.release(3)
    assert cache.blocks[0].state == "EVICTABLE"


def test_partial_exclusive_publish_and_reserved_growth() -> None:
    cache = HierarchicalKVCache(4, 2)
    ids = admit(cache, 0, ("a",), prompt=3, output=2)
    cache.publish(0, ("a",))
    assert cache.probe(cache.snapshot(), ("a", "b")).matched_blocks == 1
    assert cache.snapshot().internal_fragmentation_tokens == 1
    assert cache.snapshot().reserved_future_blocks == 1
    cache.advance(0)
    assert cache.snapshot().reserved_future_blocks == 1
    cache.publish(0, ("b",))
    cache.advance(0)
    assert cache.snapshot().reserved_future_blocks == 0
    assert cache.request_context_tokens(0) == 5
    assert cache.snapshot().resident_blocks == 3
    assert cache.blocks[ids[-1]].valid_tokens == 2
    with pytest.raises(ValueError, match="预留"):
        cache.advance(0)
    cache.release(0)
    assert cache.snapshot().resident_blocks == 2
    assert cache.free_blocks == 2


def test_capacity_failure_rollback_and_oom() -> None:
    cache = HierarchicalKVCache(2, 2)
    admit(cache, 0, ("a", "b"), output=0)
    before = cache.snapshot()
    allocation = AdmissionAllocation(1, 2, 1, ("x",))
    probe = cache.probe(before, allocation.input_chunk_ids)
    with pytest.raises(CacheCapacityError):
        cache.prepare_batch(before, (probe,), (allocation,))
    assert cache.snapshot() == before
    with pytest.raises(CacheOOMError):
        admit(cache, 2, ("a", "b", "c"), prompt=6, output=1)
    assert cache.snapshot() == before


def test_lru_eviction_generation_and_stale_transaction() -> None:
    cache = HierarchicalKVCache(2, 2)
    admit(cache, 0, ("old",), prompt=2, output=0)
    cache.publish(0, ("old",))
    cache.release(0)
    old = cache.snapshot()
    block = old.blocks[0]
    allocation = AdmissionAllocation(1, 2, 0, ("new",))
    probe = cache.probe(old, ("new",))
    tx = cache.prepare_batch(old, (probe,), (allocation,))
    admit(cache, 2, ("other",), prompt=2, output=0)
    with pytest.raises(ValueError, match="过期"):
        cache.commit(tx)
    victim = admit(cache, 3, ("latest",), prompt=2, output=0)
    assert victim == (block.block_id,)
    assert cache.blocks[victim[0]].owner_generation > block.owner_generation
    assert cache.evictions == 1
