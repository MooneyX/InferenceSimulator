"""F001 L1 页状态、共享引用和物理容量守恒。"""

import pytest

from simcore.kv_cache import CacheCapacityError, CacheEntry, L1KVCache, TieredKVCache
from simcore.prefix_hash import encode
from simcore.schemas import InferenceRequest


def request(request_id: int, prompt_len: int, inherited: tuple[str, ...] = ()) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        session_id=0,
        parent_request_id=None,
        prompt_len=prompt_len,
        max_new_tokens=128,
        inherited_prefix_ids=inherited,
        ideal_reuse_blocks=len(inherited),
    )


def assert_conserved(cache: TieredKVCache) -> None:
    assert cache.free_pages + cache.resident_pages == cache.capacity_pages
    assert cache.active_pages + cache.evictable_pages == cache.resident_pages
    assert cache.available_tokens == (cache.free_pages + cache.evictable_pages) * cache.page_size


def run_prefill(cache: TieredKVCache, req: InferenceRequest) -> None:
    sources = cache.probe(req, req.prompt_len)
    cache.pin(req, len(sources) * cache.page_size)
    cache.admit(req.request_id)
    cache.extend(req.request_id, req.prompt_len)
    cache.publish(req.request_id)
    assert_conserved(cache)


@pytest.mark.parametrize("prompt_len", [63, 64, 65])
def test_prompt_last_token_is_not_reused(prompt_len: int) -> None:
    cache = L1KVCache(256, 64)
    parent = request(0, 128)
    run_prefill(cache, parent)
    cache.finish(0)
    inherited = (encode(0, 0),) if prompt_len == 65 else ()
    child = request(1, prompt_len, inherited)
    expected = ((encode(0, 0), "L1"),) if prompt_len == 65 else ()
    assert cache.probe(child, prompt_len) == expected
    assert cache.page(encode(0, 0)) is not None
    assert_conserved(cache)


def test_page_size_one_still_requires_last_input_token() -> None:
    cache = L1KVCache(8, 1)
    run_prefill(cache, request(0, 4))
    cache.finish(0)
    assert cache.probe(request(1, 3, tuple(encode(0, i) for i in range(2))), 3) == (
        (encode(0, 0), "L1"),
        (encode(0, 1), "L1"),
    )
    assert cache.probe(request(2, 4, tuple(encode(0, i) for i in range(3))), 4) == (
        (encode(0, 0), "L1"),
        (encode(0, 1), "L1"),
        (encode(0, 2), "L1"),
    )
    assert_conserved(cache)


def test_private_tail_and_decode_full_page_only_published_at_completion() -> None:
    cache = L1KVCache(256, 64)
    req = request(0, 63)
    run_prefill(cache, req)
    assert cache.state(0).k == 63 and cache.state(0).f == 0
    assert cache.private_page_count(0) == 1
    assert cache.page(encode(0, 0)) is None
    assert cache.extend(0, 64) == 0
    assert cache.page(encode(0, 0)) is None
    assert cache.state(0).f == 0
    cache.finish(0)
    page = cache.page(encode(0, 0))
    assert page is not None and page.ref_count == 0 and page.tier == "L1"
    assert page.capacity_pages == 1
    assert cache.private_pages == 0 and cache.evictable_pages == 1
    assert_conserved(cache)


def test_same_batch_private_pages_merge_only_after_publication() -> None:
    cache = L1KVCache(256, 64)
    shared = encode(0, 0)
    first = request(1, 65, (shared,))
    second = request(2, 65, (shared,))
    for req in (first, second):
        assert cache.probe(req, req.prompt_len) == ()
        cache.pin(req, 0)
        cache.admit(req.request_id)
        cache.extend(req.request_id, 65)
    assert cache.resident_pages == 4 and cache.private_pages == 4
    assert cache.page(shared) is None
    assert cache.publish_batch((2, 1)) == 1
    assert cache.resident_pages == 3 and cache.private_pages == 2
    assert cache.state(1).f == cache.state(2).f == 64
    assert cache.page(shared) is not None and cache.page(shared).ref_count == 2  # type: ignore[union-attr]
    assert_conserved(cache)
    cache.finish(1)
    assert cache.page(shared) is not None and cache.page(shared).ref_count == 1  # type: ignore[union-attr]
    assert cache.evictable_pages == 0
    cache.finish(2)
    assert cache.page(shared) is not None and cache.page(shared).ref_count == 0  # type: ignore[union-attr]
    assert_conserved(cache)


def test_failed_candidate_undoes_pin_without_refreshing_lru() -> None:
    cache = L1KVCache(192, 64)
    parent = request(0, 64)
    run_prefill(cache, parent)
    cache.finish(0)
    page = cache.page(encode(0, 0))
    assert page is not None
    seq = page.last_access_seq
    child = request(1, 65, (encode(0, 0),))
    assert cache.probe(child, 65) == ((encode(0, 0), "L1"),)
    cache.pin(child, 64)
    assert cache.available_tokens == 128
    assert cache.evictable_pages == 0
    cache.cancel_pin(1)
    page = cache.page(encode(0, 0))
    assert page is not None and page.ref_count == 0 and page.last_access_seq == seq
    assert cache.available_tokens == 192
    cache.pin(child, 64)
    cache.admit(1)
    page = cache.page(encode(0, 0))
    assert page is not None and page.last_access_seq > seq
    cache.withdraw(1)
    assert_conserved(cache)


def test_multiple_candidates_pin_one_shared_page_once() -> None:
    cache = L1KVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    shared = encode(0, 0)
    first = request(1, 65, (shared,))
    second = request(2, 65, (shared,))
    cache.pin(first, 64)
    cache.pin(second, 64)
    page = cache.page(shared)
    assert page is not None and page.ref_count == 2
    assert cache.available_tokens == 64
    cache.cancel_pin(1)
    assert cache.available_tokens == 64
    cache.cancel_pin(2)
    assert cache.available_tokens == 128
    assert_conserved(cache)


def test_failed_pin_does_not_accumulate_duplicate_eviction_candidates() -> None:
    cache = L1KVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    contender = request(1, 65, (encode(0, 0),))
    for _ in range(100):
        cache.pin(contender, 64)
        cache.cancel_pin(1)
    assert len(cache._evictable_heap) == 1
    assert cache.ensure_free(1) == 0
    assert cache.evictable_pages == 1
    assert cache.ensure_free(2) == 1
    assert cache.page(encode(0, 0)) is None
    assert_conserved(cache)


def test_eviction_is_lru_and_never_evicts_active_pages() -> None:
    cache = L1KVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    run_prefill(cache, request(1, 64))
    cache.finish(1)
    assert cache.free_pages == 0 and cache.evictable_pages == 2
    run_prefill(cache, request(2, 64))
    assert cache.page(encode(0, 0)) is None
    assert cache.page(encode(1, 0)) is not None
    assert cache.evictions == 1
    with pytest.raises(CacheCapacityError):
        cache.extend(2, 129)
    assert cache.state(2).k == 64
    assert cache.page(encode(2, 0)) is not None
    cache.withdraw(2)
    assert_conserved(cache)


def test_successful_reuse_refreshes_lru_and_skips_stale_heap_entry() -> None:
    cache = L1KVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    run_prefill(cache, request(1, 64))
    cache.finish(1)
    hit = request(2, 65, (encode(0, 0),))
    cache.pin(hit, 64)
    cache.admit(2)
    cache.withdraw(2)
    run_prefill(cache, request(3, 64))
    assert cache.page(encode(0, 0)) is not None
    assert cache.page(encode(1, 0)) is None
    assert_conserved(cache)


def test_restore_reuses_complete_context_and_withdraw_discards_private_pages() -> None:
    cache = L1KVCache(256, 64)
    req = request(0, 65)
    run_prefill(cache, req)
    assert cache.extend(0, 128) == 0
    assert cache.page(encode(0, 1)) is None
    cache.withdraw(0)
    assert cache.resident_pages == 1 and cache.private_pages == 0
    assert cache.probe(req, 65) == ((encode(0, 0), "L1"),)
    assert cache.probe(req, 128, restoring=True) == ((encode(0, 0), "L1"),)
    cache.pin(req, 64)
    cache.admit(0)
    cache.extend(0, 128)
    assert cache.state(0).k == 128
    cache.finish(0)
    assert cache.page(encode(0, 1)) is not None
    assert_conserved(cache)


def test_full_hit_restores_without_allocating_or_prefill() -> None:
    cache = L1KVCache(128, 64)
    req = request(0, 64)
    run_prefill(cache, req)
    cache.finish(0)
    assert cache.probe(req, 64, restoring=True) == ((encode(0, 0), "L1"),)
    cache.pin(req, 64)
    state = cache.admit(0)
    assert state.k == state.f == 64
    assert cache.private_pages == 0 and cache.resident_pages == 1
    cache.withdraw(0)
    assert_conserved(cache)


def tier_page(cache: TieredKVCache, chunk_id: str) -> CacheEntry:
    page = cache.page(chunk_id)
    assert page is not None
    return page


def assert_tiers_conserved(cache: TieredKVCache) -> None:
    entries = cache._entries
    assert sum(snapshot.resident_blocks for snapshot in cache.tier_snapshots().values()) == (
        len(entries) + cache.private_pages
    )
    for tier, snapshot in cache.tier_snapshots().items():
        assert snapshot.free_blocks + snapshot.resident_blocks == snapshot.capacity_blocks
        assert snapshot.active_blocks + snapshot.evictable_blocks == snapshot.resident_blocks
        assert snapshot.evictable_blocks == sum(
            entry.tier == tier and entry.ref_count == 0 for entry in entries.values()
        )
        assert snapshot.resident_blocks == sum(
            entry.tier == tier for entry in entries.values()
        ) + (cache.private_pages if tier == "L1" else 0)


def test_tiered_lru_demotes_one_level_at_a_time_and_deletes_last_tier() -> None:
    cache = TieredKVCache(2, 1, 1, 1)
    for request_id in range(5):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
        assert_tiers_conserved(cache)
    assert cache.page(encode(0, 0)) is None
    assert tier_page(cache, encode(1, 0)).tier == "L3"
    assert tier_page(cache, encode(2, 0)).tier == "L2"
    assert tier_page(cache, encode(3, 0)).tier == "L1"
    assert tier_page(cache, encode(4, 0)).tier == "L1"
    assert (cache.evictions, cache.demotions, cache.deletions) == (3, 5, 1)
    assert cache.tier_snapshot("L3").evictions == 1


def test_probe_returns_ordered_tiers_and_stops_at_first_missing_page() -> None:
    cache = TieredKVCache(2, 1, 1, 1)
    for request_id in range(4):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    ids = tuple(encode(i, 0) for i in range(4))
    expected = tuple(zip(ids, ("L3", "L2", "L1", "L1"), strict=True))
    assert cache.probe(request(4, 5, ids), 5) == expected
    assert cache.probe(request(4, 5, ids), 5, restoring=True) == expected
    assert cache.probe(request(5, 5, (ids[0], "missing", *ids[2:])), 5) == expected[:1]
    assert cache.probe(request(6, 4, ids), 4) == expected[:3]
    with pytest.raises(ValueError, match="原始 prompt"):
        cache.probe(request(7, 5, ids), 4)


def test_tiered_shared_low_hit_is_promoted_once_and_preserves_pin_refs() -> None:
    cache = TieredKVCache(2, 1, 1)
    for request_id in range(3):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    shared = encode(0, 0)
    first = request(3, 2, (shared,))
    second = request(4, 2, (shared,))
    sources = cache.probe(first, first.prompt_len)
    assert sources == cache.probe(second, second.prompt_len) == ((shared, "L2"),)
    for item in (first, second):
        cache.pin(item, 1)
    assert tier_page(cache, shared).ref_count == 2
    assert cache.plan((shared, shared), reserve_pages=1) == ((shared, "L2"),)
    assert tier_page(cache, shared).tier == "L2"
    assert cache.prepare_batch((shared, shared), reserve_pages=1) == ((shared, "L2"),)
    assert cache.promotions == 1
    assert tier_page(cache, shared).tier == "L1"
    assert sources == ((shared, "L2"),)  # 来源按查询时冻结，不随迁移改变
    assert tier_page(cache, shared).ref_count == 2
    for item in (first, second):
        cache.admit(item.request_id)
        cache.withdraw(item.request_id)
    assert_tiers_conserved(cache)


def test_tiered_plan_failure_and_cancel_pin_are_side_effect_free() -> None:
    cache = TieredKVCache(1, 1, 1)
    run_prefill(cache, request(0, 1))
    cache.finish(0)
    run_prefill(cache, request(1, 1))
    shared = encode(0, 0)
    contender = request(2, 2, (shared,))
    cache.pin(contender, 1)
    page = tier_page(cache, shared)
    snapshots = cache.tier_snapshots()
    counters = (cache.promotions, cache.demotions, cache.deletions)
    with pytest.raises(CacheCapacityError):
        cache.plan((shared,), reserve_pages=1)
    assert cache.page(shared) == page
    assert cache.tier_snapshots() == snapshots
    assert (cache.promotions, cache.demotions, cache.deletions) == counters
    cache.cancel_pin(contender.request_id)
    assert tier_page(cache, shared).last_access_seq == page.last_access_seq
    cache.withdraw(1)
    assert_tiers_conserved(cache)


def test_tiered_recomputed_low_id_keeps_new_l1_page_without_restore() -> None:
    cache = TieredKVCache(3, 1, 3)
    for request_id in range(4):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    shared = encode(0, 0)
    assert tier_page(cache, shared).tier == "L2"
    recompute = request(4, 2, (shared,))
    cache.pin(recompute, 0)
    cache.admit(recompute.request_id)
    cache.extend(recompute.request_id, 2)
    old_promotions = cache.promotions
    cache.publish(recompute.request_id)
    assert tier_page(cache, shared).tier == "L1"
    assert tier_page(cache, shared).ref_count == 1
    assert cache.promotions == old_promotions
    cache.finish(recompute.request_id)
    assert_tiers_conserved(cache)


def test_tiered_l3_promotion_uses_source_slot_for_two_level_cascade() -> None:
    cache = TieredKVCache(2, 1, 1, 1)
    for request_id in range(4):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    shared = encode(0, 0)
    assert tier_page(cache, shared).tier == "L3"
    contender = request(4, 2, (shared,))
    cache.pin(contender, 1)
    promotions = cache.plan((shared,), reserve_pages=1)
    assert promotions == ((shared, "L3"),)
    assert tier_page(cache, shared).tier == "L3"
    assert cache.promote((shared,), reserve_pages=1) == promotions
    assert tier_page(cache, shared).tier == "L1"
    assert cache.l3_promotions == 1
    assert cache.free_pages == 1
    cache.admit(contender.request_id)
    cache.withdraw(contender.request_id)
    assert_tiers_conserved(cache)
