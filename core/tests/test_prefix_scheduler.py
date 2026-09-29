"""PC3A-A：FCFS 批规划只读 probe、原子缓存准入及实际 Prefill 工作量。"""

import pytest

from simcore.kv_cache import CacheOOMError, HierarchicalKVCache
from simcore.scheduler import PrefixScheduler
from simcore.schemas import EngineConfig, InferenceRequest, PerfModelConfig, PrefixCacheConfig

PERF = PerfModelConfig(prefill_c1=1, prefill_c2=0.1, decode_d1=0, decode_d2=0)


def make(budget: int = 20, block_size: int = 2) -> tuple[PrefixScheduler, HierarchicalKVCache]:
    config = EngineConfig(
        max_batch_size=4,
        kv_token_budget=budget,
        prefix_cache=PrefixCacheConfig(block_size_tokens=block_size),
    )
    cache = HierarchicalKVCache(config.prefix_cache.l1.capacity_blocks, block_size)
    return PrefixScheduler(config, PERF, cache), cache


def request(
    key: int, hashes: tuple[str, ...], prompt: int = 4, output: int = 2
) -> InferenceRequest:
    return InferenceRequest(
        request_id=key,
        session_id=key,
        parent_request_id=None,
        input_token_ids=tuple(range(prompt)),
        output_token_plan=tuple(range(output)),
        input_chunk_hashes=hashes,
        ideal_reuse_blocks=len(hashes),
    )


def test_same_step_new_blocks_not_visible_and_published_hit() -> None:
    scheduler, cache = make()
    scheduler.enqueue(request(0, ("a", "b")), 0)
    scheduler.enqueue(request(1, ("a", "b")), 0)
    first = scheduler.plan_step()
    assert first is not None
    assert first.request_results == {0: (0, 0, 4), 1: (0, 0, 4)}
    assert first.cache_result is not None
    assert first.cache_result.allocated_blocks == 4
    assert first.duration == pytest.approx(2 * (4 + 0.1 * 16))
    cache.publish(0, ("a", "b"))
    cache.publish(1, ("a", "b"))
    cache.release(0)
    cache.release(1)
    scheduler.enqueue(request(2, ("a", "b")), 1)
    later = scheduler.plan_step()
    assert later is not None
    assert later.request_results == {2: (4, 3, 1)}
    assert later.duration == pytest.approx(1 + 0.1 * 7)
    assert later.cache_result is not None
    assert later.cache_result.allocated_blocks == 0
    assert later.cache_result.reused_block_ids == (0, 1)


def test_fcfs_head_waits_without_mutating_cache() -> None:
    scheduler, cache = make(budget=10)
    scheduler.enqueue(request(0, ("a", "b")), 0)
    first = scheduler.plan_step()
    assert first is not None
    snapshot = cache.snapshot()
    scheduler.enqueue(request(1, ("x", "y")), 1)
    scheduler.enqueue(request(2, ("a", "b")), 1)
    waiting = scheduler.plan_step()
    assert waiting is not None
    assert waiting.admitted == ()
    assert scheduler.num_waiting == 2
    assert cache.snapshot() == snapshot
    cache.release(0)
    admitted = scheduler.plan_step()
    assert admitted is not None
    assert admitted.admitted[0].request_id == 1


def test_single_request_oom_is_not_forced() -> None:
    scheduler, cache = make(budget=4)
    scheduler.enqueue(request(0, ("a", "b"), output=1), 0)
    before = cache.snapshot()
    with pytest.raises(CacheOOMError):
        scheduler.plan_step()
    assert cache.snapshot() == before
    assert scheduler.num_waiting == 1
