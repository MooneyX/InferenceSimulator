"""F001 L1 页状态、共享引用和物理容量守恒。"""

import pytest

from simcore.kv_cache import CacheCapacityError, CacheEntry, CachePool, TieredKVCache
from simcore.prefix_hash import encode
from simcore.schemas import InferenceRequest


def request(request_id: int, prompt_len: int, inherited: tuple[str, ...] = ()) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        session_id=0,
        parent_request_id=None,
        prompt_len=prompt_len,
        max_new_tokens=128,
        output_len=128,
        inherited_prefix_ids=inherited,
        ideal_reuse_blocks=len(inherited),
    )


def assert_conserved(cache: TieredKVCache) -> None:
    assert cache.free_pages + cache.resident_pages == cache.capacity_pages
    assert cache.active_pages + cache.evictable_pages == cache.resident_pages
    assert cache.available_tokens == (cache.free_pages + cache.evictable_pages) * cache.page_size


def pin_hit(cache: TieredKVCache, req: InferenceRequest) -> None:
    cache.pin(req, cache.probe(req, req.prompt_len))


def run_prefill(cache: TieredKVCache, req: InferenceRequest) -> None:
    pin_hit(cache, req)
    cache.admit(req.request_id)
    cache.extend(req.request_id, req.prompt_len)
    cache.publish(req.request_id)
    assert_conserved(cache)


@pytest.mark.parametrize("prompt_len", [63, 64, 65])
def test_prompt_last_token_is_not_reused(prompt_len: int) -> None:
    cache = TieredKVCache(256, 64)
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
    cache = TieredKVCache(8, 1)
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
    cache = TieredKVCache(256, 64)
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
    assert cache._private_pages == 0 and cache.evictable_pages == 1
    assert_conserved(cache)


def test_same_batch_private_pages_merge_only_after_publication() -> None:
    cache = TieredKVCache(256, 64)
    shared = encode(0, 0)
    first = request(1, 65, (shared,))
    second = request(2, 65, (shared,))
    for req in (first, second):
        assert cache.probe(req, req.prompt_len) == ()
        cache.pin(req, ())
        cache.admit(req.request_id)
        cache.extend(req.request_id, 65)
    assert cache.resident_pages == 4 and cache._private_pages == 4
    assert cache.page(shared) is None
    assert cache.publish_batch((2, 1)) == 1
    assert cache.resident_pages == 3 and cache._private_pages == 2
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
    cache = TieredKVCache(192, 64)
    parent = request(0, 64)
    run_prefill(cache, parent)
    cache.finish(0)
    page = cache.page(encode(0, 0))
    assert page is not None
    seq = page.last_access_seq
    child = request(1, 65, (encode(0, 0),))
    assert cache.probe(child, 65) == ((encode(0, 0), "L1"),)
    pin_hit(cache, child)
    assert cache.available_tokens == 128
    assert cache.evictable_pages == 0
    cache.cancel_pin(1)
    page = cache.page(encode(0, 0))
    assert page is not None and page.ref_count == 0 and page.last_access_seq == seq
    assert cache.available_tokens == 192
    pin_hit(cache, child)
    cache.admit(1)
    page = cache.page(encode(0, 0))
    assert page is not None and page.last_access_seq > seq
    cache.withdraw(1)
    assert_conserved(cache)


def test_pin_uses_probe_sources_without_second_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TieredKVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    child = request(1, 65, (encode(0, 0),))
    sources = cache.probe(child, child.prompt_len)
    monkeypatch.setattr(cache, "_source_pool", lambda *_args, **_kwargs: None)
    cache.pin(child, sources)
    page = cache.pools["L1"].lookup(sources[0][0])
    assert page is not None and page.ref_count == 1
    cache.cancel_pin(child.request_id)
    assert page.ref_count == 0


def test_multiple_candidates_pin_one_shared_page_once() -> None:
    cache = TieredKVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    shared = encode(0, 0)
    first = request(1, 65, (shared,))
    second = request(2, 65, (shared,))
    pin_hit(cache, first)
    pin_hit(cache, second)
    page = cache.page(shared)
    assert page is not None and page.ref_count == 2
    assert cache.available_tokens == 64
    cache.cancel_pin(1)
    assert cache.available_tokens == 64
    cache.cancel_pin(2)
    assert cache.available_tokens == 128
    assert_conserved(cache)


def test_failed_pin_does_not_accumulate_duplicate_eviction_candidates() -> None:
    cache = TieredKVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    contender = request(1, 65, (encode(0, 0),))
    for _ in range(100):
        pin_hit(cache, contender)
        cache.cancel_pin(1)
    assert len(cache.pools["L1"].heap) == 1
    assert cache.ensure_free(1) == 0
    assert cache.evictable_pages == 1
    assert cache.ensure_free(2) == 1
    assert cache.page(encode(0, 0)) is None
    assert_conserved(cache)


@pytest.mark.parametrize("change", ["queue", "protect", "remove"])
def test_periodic_heap_rebuild_discards_stale_entries(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("simcore.kv_cache._HEAP_REBUILD_CHECK_INTERVAL", 514)
    pool = CachePool("L1", 0, 3, 1)
    pool.insert("old")
    pool.insert("recent")
    for _ in range(511):
        pool.touch("recent")
    assert len(pool.heap) == 513
    if change == "protect":
        pool.add_ref("old")
    elif change == "remove":
        pool.remove("old")
    pool.touch("recent")
    expected = sorted(
        (page.last_access_seq, page.chunk_id)
        for page in pool.entries.values()
        if not page.ref_count
    )
    assert set(pool.heap) == pool.evictable == set(expected)
    assert pool.queued == set(pool.heap)
    assert pool.queued is not pool.evictable
    assert len(pool.heap) == len(pool.evictable)
    assert pool.heap[0] == expected[0]
    if change == "protect":
        pool.drop_ref("old")
        assert pool.heap[0] == (pool.entries["old"].last_access_seq, "old")


@pytest.mark.parametrize("protect_all", [False, True])
def test_heap_rebuild_uses_evictable_set_without_accessing_entries(
    protect_all: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = CachePool("L1", 0, 1026, 1)
    for index in range(1024):
        pool.insert(f"active-{index}", ref_count=1)
    pool.insert("old")
    pool.insert("recent")
    for _ in range(512):
        pool.touch("recent")
    if protect_all:
        pool.add_ref("old")
        pool.add_ref("recent")
    expected = sorted(
        (page.last_access_seq, page.chunk_id)
        for page in pool.entries.values()
        if not page.ref_count
    )
    assert len(pool.heap) == 514
    with monkeypatch.context() as patch:
        patch.setattr(pool, "entries", object())
        pool._maybe_rebuild_heap()
    assert sorted(pool.heap) == expected
    assert pool.evictable == pool.queued == set(expected)
    assert pool.queued is not pool.evictable
    for seq, chunk_id in expected:
        assert pool.pop_lru() is pool.entries[chunk_id]
        assert (seq, chunk_id) in pool.evictable
        pool.remove(chunk_id)
        assert (seq, chunk_id) not in pool.evictable
    assert pool.pop_lru() is None
    assert not pool.evictable


@pytest.mark.parametrize("interval", [64, 4096])
def test_periodic_heap_check_bounds_stale_entries_without_eviction(
    interval: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("simcore.kv_cache._HEAP_REBUILD_CHECK_INTERVAL", interval)
    pool = CachePool("L2", 0, 1, 1)
    pool.insert("page")
    for _ in range(10_000):
        pool.touch("page")
    assert len(pool.heap) <= len(pool.evictable) + 512 + interval
    assert pool.queued == set(pool.heap)
    assert pool.pop_lru() is pool.entries["page"]


@pytest.mark.parametrize("ref_count", [0, 2])
def test_evictable_set_tracks_touch_removal_and_same_id_reinsertion(ref_count: int) -> None:
    pool = CachePool("L1", 0, 1, 1)
    pool.insert("page", ref_count=ref_count)
    assert len(pool.evictable) == int(ref_count == 0)
    pool.touch("page")
    page = pool.entries["page"]
    key = (page.last_access_seq, "page")
    assert pool.evictable == ({key} if ref_count == 0 else set())
    snapshot = pool.snapshot()
    assert snapshot.evictable_blocks == len(pool.evictable)
    assert snapshot.active_blocks == int(ref_count > 0)
    pool.remove("page")
    assert not pool.evictable
    pool.insert("page", ref_count=1)
    pool.touch("page")
    pool.drop_ref("page")
    page = pool.entries["page"]
    assert pool.evictable == {(page.last_access_seq, "page")}
    assert key not in pool.evictable
    assert pool.pop_lru() is page
    pool.remove("page", evicted=True)
    assert not pool.evictable
    assert pool.snapshot().resident_blocks == 0


def test_page_updates_keep_same_entry_and_current_lru_order() -> None:
    pool = CachePool("L1", 0, 2, 1)
    pool.insert("first")
    pool.insert("second")
    page = pool.lookup("first")
    assert page is not None
    first_key = (page.last_access_seq, "first")
    second_key = (pool.entries["second"].last_access_seq, "second")
    pool.add_ref("first")
    assert pool.lookup("first") is page
    assert page.ref_count == 1 and pool.evictable == {second_key}
    pool.add_refs("first", 1)
    pool.drop_ref("first")
    assert page.ref_count == 1 and pool.evictable == {second_key}
    pool.drop_ref("first")
    assert pool.lookup("first") is page
    assert page.ref_count == 0 and pool.evictable == {first_key, second_key}
    pool.touch("first")
    assert pool.lookup("first") is page
    assert page.last_access_seq == pool.access_seq
    assert pool.evictable == {(page.last_access_seq, "first"), second_key}
    assert pool.pop_lru() is pool.lookup("second")
    assert pool.pop_lru() is page


def test_pop_lru_cleans_stale_entries_and_requeues_released_page() -> None:
    pool = CachePool("L1", 0, 3, 1)
    pool.insert("old")
    pool.insert("recent")
    pool.touch("old")
    pool.add_ref("recent")
    assert pool.pop_lru() == pool.lookup("old")
    assert pool.heap == [] and pool.queued == set()
    pool.remove("old", evicted=True)
    pool.drop_ref("recent")
    assert pool.pop_lru() == pool.lookup("recent")
    assert pool.heap == [] and pool.queued == set()


def test_eviction_is_lru_and_never_evicts_active_pages() -> None:
    cache = TieredKVCache(128, 64)
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
    cache = TieredKVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    run_prefill(cache, request(1, 64))
    cache.finish(1)
    hit = request(2, 65, (encode(0, 0),))
    pin_hit(cache, hit)
    cache.admit(2)
    cache.withdraw(2)
    run_prefill(cache, request(3, 64))
    assert cache.page(encode(0, 0)) is not None
    assert cache.page(encode(1, 0)) is None
    assert_conserved(cache)


def test_restore_reuses_complete_context_and_withdraw_discards_private_pages() -> None:
    cache = TieredKVCache(256, 64)
    req = request(0, 65)
    run_prefill(cache, req)
    assert cache.extend(0, 128) == 0
    assert cache.page(encode(0, 1)) is None
    cache.withdraw(0)
    assert cache.resident_pages == 1 and cache._private_pages == 0
    assert cache.probe(req, 65) == ((encode(0, 0), "L1"),)
    assert cache.probe(req, 128, restoring=True) == ((encode(0, 0), "L1"),)
    cache.pin(req, cache.probe(req, 128, restoring=True))
    cache.admit(0)
    cache.extend(0, 128)
    assert cache.state(0).k == 128
    cache.finish(0)
    assert cache.page(encode(0, 1)) is not None
    assert_conserved(cache)


def test_full_hit_restores_without_allocating_or_prefill() -> None:
    cache = TieredKVCache(128, 64)
    req = request(0, 64)
    run_prefill(cache, req)
    cache.finish(0)
    assert cache.probe(req, 64, restoring=True) == ((encode(0, 0), "L1"),)
    cache.pin(req, cache.probe(req, 64, restoring=True))
    state = cache.admit(0)
    assert state.k == state.f == 64
    assert cache._private_pages == 0 and cache.resident_pages == 1
    cache.withdraw(0)
    assert_conserved(cache)


def tier_page(cache: TieredKVCache, chunk_id: str) -> CacheEntry:
    page = cache.page(chunk_id)
    assert page is not None
    return page


def cache_entries(cache: TieredKVCache) -> dict[tuple[str, str], CacheEntry]:
    return {
        (tier, chunk_id): entry
        for tier, pool in cache.pools.items()
        for chunk_id, entry in pool.entries.items()
    }


def entry_values(cache: TieredKVCache) -> dict[tuple[str, str], tuple[int, int]]:
    return {
        key: (page.ref_count, page.last_access_seq) for key, page in cache_entries(cache).items()
    }


def assert_tiers_conserved(cache: TieredKVCache) -> None:
    entries = cache_entries(cache)
    snapshots = {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")}
    assert sum(snapshot.resident_blocks for snapshot in snapshots.values()) == (
        len(entries) + cache._private_pages + sum(state.reserved for state in cache.pools.values())
    )
    for tier, snapshot in snapshots.items():
        assert snapshot.free_blocks + snapshot.resident_blocks == snapshot.capacity_blocks
        assert snapshot.active_blocks + snapshot.evictable_blocks == snapshot.resident_blocks
        assert snapshot.evictable_blocks == sum(
            entry.tier == tier and entry.ref_count == 0 for entry in entries.values()
        )
        assert snapshot.resident_blocks == (
            sum(entry.tier == tier for entry in entries.values())
            + (cache._private_pages if tier == "L1" else 0)
            + cache.pools[tier].reserved
        )


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
    assert (cache.evictions, cache.demotions) == (3, 5)
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


def test_tiered_shared_low_hit_is_promoted_once_and_preserves_pin_refs() -> None:
    cache = TieredKVCache(2, 1, 2)
    for request_id in range(3):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    cache.ensure_free(1)
    shared = encode(0, 0)
    first = request(3, 2, (shared,))
    second = request(4, 2, (shared,))
    sources = cache.probe(first, first.prompt_len)
    assert sources == cache.probe(second, second.prompt_len) == ((shared, "L2"),)
    for item in (first, second):
        cache.pin(item, sources)
    assert tier_page(cache, shared).ref_count == 2
    assert tier_page(cache, shared).tier == "L2"
    assert cache.reserve_transfer(shared, "L1", source="L2") == "L2"
    assert tier_page(cache, shared).tier == "L2"
    assert tier_page(cache, shared).ref_count == 3
    assert cache.pools["L1"].reserved == 1
    assert_tiers_conserved(cache)
    assert cache.complete_transfer(shared) == "L2"
    assert cache.promotions == 1
    assert tier_page(cache, shared).tier == "L1"
    assert sources == ((shared, "L2"),)  # 来源按查询时冻结，不随迁移改变
    assert tier_page(cache, shared).ref_count == 2
    for item in (first, second):
        cache.admit(item.request_id)
        cache.withdraw(item.request_id)
    assert_tiers_conserved(cache)


def test_tiered_insufficient_l1_capacity_and_cancel_pin_are_side_effect_free() -> None:
    cache = TieredKVCache(1, 1, 1)
    run_prefill(cache, request(0, 1))
    cache.finish(0)
    run_prefill(cache, request(1, 1))
    shared = encode(0, 0)
    contender = request(2, 2, (shared,))
    pin_hit(cache, contender)
    page = tier_page(cache, shared)
    ref_count, access_seq = page.ref_count, page.last_access_seq
    snapshots = {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")}
    counters = (cache.promotions, cache.demotions)
    with pytest.raises(CacheCapacityError):
        cache.ensure_free(1)
    assert cache.page(shared) is page
    assert (page.ref_count, page.last_access_seq) == (ref_count, access_seq)
    assert {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")} == snapshots
    assert (cache.promotions, cache.demotions) == counters
    cache.cancel_pin(contender.request_id)
    assert tier_page(cache, shared).last_access_seq == access_seq
    cache.withdraw(1)
    assert_tiers_conserved(cache)


def test_tiered_insufficient_evictable_pages_preserve_cache_and_lru() -> None:
    cache = TieredKVCache(2, 1, 1, 1)
    for request_id in range(4):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    protected = request(4, 2, (encode(3, 0),))
    pin_hit(cache, protected)
    entries = entry_values(cache)
    snapshots = {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")}
    heaps = {tier: list(state.heap) for tier, state in cache.pools.items()}
    queued = {tier: set(state.queued) for tier, state in cache.pools.items()}
    counters = (
        tuple(pool.access_seq for pool in cache.pools.values()),
        cache.promotions,
        cache.demotions,
    )
    with pytest.raises(CacheCapacityError):
        cache.ensure_free(2)
    assert entry_values(cache) == entries
    assert {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")} == snapshots
    assert {tier: state.heap for tier, state in cache.pools.items()} == heaps
    assert {tier: state.queued for tier, state in cache.pools.items()} == queued
    assert (
        tuple(pool.access_seq for pool in cache.pools.values()),
        cache.promotions,
        cache.demotions,
    ) == counters
    cache.cancel_pin(protected.request_id)
    assert cache.ensure_free(2) == 2
    assert_tiers_conserved(cache)


def test_tiered_recomputed_low_id_keeps_new_l1_page_without_restore() -> None:
    cache = TieredKVCache(3, 1, 3)
    for request_id in range(4):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    shared = encode(0, 0)
    assert tier_page(cache, shared).tier == "L2"
    recompute = request(4, 2, (shared,))
    cache.pin(recompute, ())
    cache.admit(recompute.request_id)
    cache.extend(recompute.request_id, 2)
    old_promotions = cache.promotions
    cache.publish(recompute.request_id)
    assert tier_page(cache, shared).tier == "L1"
    assert tier_page(cache, shared).ref_count == 1
    assert cache.promotions == old_promotions
    cache.finish(recompute.request_id)
    assert_tiers_conserved(cache)


def test_l3_transfer_drops_l2_victim_when_source_is_protected() -> None:
    cache = TieredKVCache(2, 1, 1, 1)
    for request_id in range(4):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    shared = encode(0, 0)
    victim = encode(1, 0)
    assert tier_page(cache, shared).tier == "L3"
    contender = request(4, 2, (shared,))
    pin_hit(cache, contender)
    assert cache.reserve_transfer(shared, "L2", source="L3") == "L3"
    assert cache.pools["L3"].lookup(shared) is not None
    assert cache.page(victim) is None
    assert cache.tier_snapshot("L2").evictions >= 1
    assert cache.complete_transfer(shared) == "L3"
    assert tier_page(cache, shared).tier == "L2"
    cache.cancel_pin(contender.request_id)
    assert_tiers_conserved(cache)


@pytest.mark.parametrize("source,destination", [("L2", "L1"), ("L3", "L2")])
def test_transfer_uses_known_source_without_searching_pools(
    source: str, destination: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = TieredKVCache(2, 1, 2, 2)
    chunk = encode(0, 0)
    cache.pools[source].insert(chunk)

    def unexpected_search(*_args: object, **_kwargs: object) -> None:
        pytest.fail("已知传输来源不应重新搜索来源池")

    monkeypatch.setattr(cache, "page", unexpected_search)
    monkeypatch.setattr(cache, "_source_pool", unexpected_search)
    assert cache.reserve_transfer(chunk, destination, source=source) == source
    assert cache.pools[source].entries[chunk].ref_count == 1
    assert cache.pools[destination].reserved == 1
    cache.complete_transfer(chunk)
    assert cache.pools[destination].entries[chunk].ref_count == 0
    assert_tiers_conserved(cache)


def test_transfer_missing_known_source_does_not_reserve_destination() -> None:
    cache = TieredKVCache(2, 1, 2, 2)
    chunk = encode(0, 0)
    cache.pools["L3"].insert(chunk)
    before = tuple(pool.snapshot() for pool in cache.pools.values())
    with pytest.raises(ValueError, match="待传输页不存在"):
        cache.reserve_transfer(chunk, "L1", source="L2")
    assert tuple(pool.snapshot() for pool in cache.pools.values()) == before
    assert not cache._transfers


def test_transfer_reserves_destination_without_exposing_page_or_refreshing_source() -> None:
    cache = TieredKVCache(2, 1, 2)
    for request_id in range(3):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    source = encode(0, 0)
    cache.ensure_free(1)
    before = {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")}
    page = tier_page(cache, source)
    assert page.tier == "L2" and page.ref_count == 0
    assert cache.reserve_transfer(source, "L1", source="L2") == "L2"
    assert cache.page(source) == CacheEntry(
        source, "L2", 1, page.last_access_seq, pool_id=page.pool_id
    )
    assert cache.probe(request(9, 2, (source,)), 2) == ((source, "L2"),)
    assert cache.tier_snapshot("L1").resident_blocks == before["L1"].resident_blocks + 1
    assert cache.tier_snapshot("L1").free_blocks == before["L1"].free_blocks - 1
    assert cache.tier_snapshot("L2").evictable_blocks == before["L2"].evictable_blocks - 1
    assert_tiers_conserved(cache)
    assert cache.complete_transfer(source) == "L2"
    assert tier_page(cache, source).tier == "L1"
    assert tier_page(cache, source).ref_count == 0
    assert tier_page(cache, source).last_access_seq > page.last_access_seq
    assert cache.tier_snapshot("L2").resident_blocks == before["L2"].resident_blocks - 1
    assert_tiers_conserved(cache)


def test_transfer_completion_moves_one_page_into_reserved_slot() -> None:
    cache = TieredKVCache(2, 1, 2)
    for request_id in range(3):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    cache.ensure_free(1)
    source = encode(0, 0)
    old_count = len(cache_entries(cache))
    old_promotions = cache.promotions
    assert cache.reserve_transfer(source, "L1", source="L2") == "L2"
    assert cache.complete_transfer(source) == "L2"
    assert tier_page(cache, source).tier == "L1"
    assert tier_page(cache, source).ref_count == 0
    assert cache.promotions == old_promotions + 1
    assert len(cache_entries(cache)) == old_count
    assert cache.pools["L1"].reserved == 0
    assert cache.pools["L2"].reserved == 0
    assert_tiers_conserved(cache)


@pytest.mark.parametrize("stale_entries", [0, 511])
def test_transfer_failed_reservation_restores_source_reference(stale_entries: int) -> None:
    cache = TieredKVCache(1, 1, 1)
    for request_id in range(2):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    source = encode(0, 0)
    source_pool = cache.pools["L2"]
    for _ in range(stale_entries):
        source_pool.touch(source)
    pinned = request(9, 2, (encode(1, 0),))
    pin_hit(cache, pinned)
    entries = entry_values(cache)
    snapshots = {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")}
    with pytest.raises(CacheCapacityError):
        cache.reserve_transfer(source, "L1", source="L2")
    assert entry_values(cache) == entries
    assert {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")} == snapshots
    assert (source_pool.entries[source].last_access_seq, source) in source_pool.queued
    assert all(pool.queued == set(pool.heap) for pool in cache.pools.values())
    assert cache._transfers == {}
    cache.cancel_pin(pinned.request_id)
    assert_tiers_conserved(cache)


def test_l2_to_l1_load_drops_l1_victim_when_l2_source_is_pinned() -> None:
    cache = TieredKVCache(1, 1, 1)
    for request_id in range(2):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    source, victim = encode(0, 0), encode(1, 0)
    contender = request(2, 2, (source,))
    pin_hit(cache, contender)
    assert cache.free_pages == 0 and cache.evictable_pages == 1
    assert cache.ensure_free(1) == 1
    assert cache.page(victim) is None
    assert tier_page(cache, source).tier == "L2"
    assert cache.reserve_transfer(source, "L1", source="L2") == "L2"
    assert cache.complete_transfer(source) == "L2"
    assert tier_page(cache, source).tier == "L1"
    cache.cancel_pin(contender.request_id)
    assert_tiers_conserved(cache)


def test_full_transfer_destination_evicted_by_lru_before_reservation() -> None:
    cache = TieredKVCache(1, 1, 2, 2)
    for request_id in range(4):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    source = encode(0, 0)
    assert tier_page(cache, source).tier == "L3"
    assert cache.tier_snapshot("L2").free_blocks == 0
    evictions = cache.tier_snapshot("L2").evictions
    assert cache.reserve_transfer(source, "L2", source="L3") == "L3"
    assert cache.tier_snapshot("L2").evictions == evictions + 1
    assert tier_page(cache, encode(1, 0)).tier == "L3"
    assert tier_page(cache, source).tier == "L3"
    assert cache.tier_snapshot("L2").free_blocks == 0
    assert_tiers_conserved(cache)
    assert cache.complete_transfer(source) == "L3"
    assert tier_page(cache, source).tier == "L2"
    assert_tiers_conserved(cache)


def test_full_transfer_drops_l2_victim_instead_of_evicting_l3_source() -> None:
    cache = TieredKVCache(1, 1, 1, 1)
    for request_id in range(3):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    source = encode(0, 0)
    victim = encode(1, 0)
    assert tier_page(cache, source).tier == "L3"
    assert cache.reserve_transfer(source, "L2", source="L3") == "L3"
    assert cache.page(victim) is None
    assert tier_page(cache, source).tier == "L3"
    assert cache.complete_transfer(source) == "L3"
    assert tier_page(cache, source).tier == "L2"
    assert_tiers_conserved(cache)


def test_incremental_promotion_keeps_unaffected_entries_and_heaps() -> None:
    cache = TieredKVCache(3, 1, 3)
    for request_id in range(5):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    source = encode(0, 0)
    unrelated = encode(4, 0)
    page = tier_page(cache, unrelated)
    heaps = {tier: state.heap for tier, state in cache.pools.items()}
    pin_hit(cache, request(8, 2, (source,)))
    assert cache.reserve_transfer(source, "L1", source="L2") == "L2"
    assert tier_page(cache, source).tier == "L2"
    assert_tiers_conserved(cache)
    assert cache.complete_transfer(source) == "L2"
    assert cache.page(unrelated) is page
    assert all(state.heap is heaps[tier] for tier, state in cache.pools.items())
    cache.cancel_pin(8)
    assert_tiers_conserved(cache)


def test_transfer_promotes_pinned_source_and_retains_request_reference() -> None:
    cache = TieredKVCache(2, 1, 2)
    for request_id in range(3):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    cache.ensure_free(1)
    source = encode(0, 0)
    pin_hit(cache, request(9, 2, (source,)))
    page = tier_page(cache, source)
    assert page.tier == "L2" and page.ref_count == 1
    assert cache.reserve_transfer(source, "L1", source="L2") == "L2"
    assert tier_page(cache, source).tier == "L2"
    assert tier_page(cache, source).ref_count == 2
    assert_tiers_conserved(cache)
    assert cache.complete_transfer(source) == "L2"
    assert tier_page(cache, source).tier == "L1"
    assert tier_page(cache, source).ref_count == 1
    cache.admit(9)
    cache.withdraw(9)
    assert_tiers_conserved(cache)


def test_parallel_transfer_reservations_are_counted_and_isolated() -> None:
    cache = TieredKVCache(3, 1, 3)
    for request_id in range(4):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    cache.ensure_free(2)
    first, second = encode(0, 0), encode(1, 0)
    assert cache.reserve_transfer(first, "L1", source="L2") == "L2"
    assert cache.reserve_transfer(second, "L1", source="L2") == "L2"
    assert cache.tier_snapshot("L1").free_blocks == 0
    assert_tiers_conserved(cache)
    snapshots = {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")}
    with pytest.raises(CacheCapacityError):
        cache.ensure_free(2)
    assert {tier: cache.tier_snapshot(tier) for tier in ("L1", "L2", "L3")} == snapshots
    assert cache.complete_transfer(first) == "L2"
    assert cache.tier_snapshot("L1").free_blocks == 0
    assert cache.pools["L1"].reserved == 1
    assert tier_page(cache, first).tier == "L1"
    assert tier_page(cache, second).tier == "L2"
    assert tier_page(cache, second).ref_count == 1
    assert_tiers_conserved(cache)
    assert cache.complete_transfer(second) == "L2"
    assert tier_page(cache, second).tier == "L1"
    assert cache.pools["L1"].reserved == 0
    assert_tiers_conserved(cache)


def test_l3_to_l2_reload_is_promotion_and_preserves_new_request_refs() -> None:
    cache = TieredKVCache(2, 1, 2, 2)
    for request_id in range(5):
        run_prefill(cache, request(request_id, 1))
        cache.finish(request_id)
    cache.ensure_free(1)
    make_room = encode(2, 0)
    pin_hit(cache, request(9, 2, (make_room,)))
    assert cache.reserve_transfer(make_room, "L1", source="L2") == "L2"
    assert cache.complete_transfer(make_room) == "L2"
    cache.cancel_pin(9)
    source = encode(0, 0)
    assert tier_page(cache, source).tier == "L3"
    counters = cache.promotions, cache.demotions
    l3_evictions = cache.tier_snapshot("L3").evictions
    assert cache.reserve_transfer(source, "L2", source="L3") == "L3"
    pin_hit(cache, request(10, 2, (source,)))
    assert_tiers_conserved(cache)
    assert cache.complete_transfer(source) == "L3"
    assert tier_page(cache, source).tier == "L2"
    assert tier_page(cache, source).ref_count == 1
    assert (cache.promotions, cache.demotions) == (counters[0] + 1, counters[1])
    assert cache.tier_snapshot("L3").evictions == l3_evictions
    cache.cancel_pin(10)
    assert_tiers_conserved(cache)
