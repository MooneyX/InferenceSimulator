"""L1/L2 前缀基数树：节点切分、路径锁、叶子 LRU 淘汰、write-through、共享 L2 与 L3 页池。"""

import random
from dataclasses import dataclass

import pytest

from simcore.kv_cache import (
    CacheCapacityError,
    CachePool,
    PrefixMatch,
    TieredKVCache,
    l2_snapshot,
)
from simcore.prefix_hash import chunk_base
from simcore.radix_cache import HbmPool, RadixNode, RadixTree, chunk_id
from simcore.schemas import InferenceRequest


def request(
    request_id: int, prompt_len: int, inherited: tuple[int, ...] = (), *, output: int = 128
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        session_id=0,
        parent_request_id=None,
        prompt_len=prompt_len,
        max_new_tokens=output,
        output_len=output,
        chunk_ids=inherited,
        ideal_reuse_blocks=len(inherited),
        chunk_base=chunk_base(request_id),
    )


@dataclass(frozen=True)
class PageView:
    tier: str  # 本实例可见的最近一层
    ref_count: int  # 本实例经过该页节点的路径锁数；L3 页为 L3 引用数
    l2: bool


def find_node(tree: RadixTree, page: int) -> RadixNode | None:
    stack = list(tree.root.children.values())
    while stack:
        node = stack.pop()
        if any(start <= page < start + length for start, length in node.segments):
            return node
        stack.extend(node.children.values())
    return None


def page(cache: TieredKVCache, page_id: int) -> PageView | None:
    """测试用逐页视图：树中本实例驻留为 L1，否则 L2 副本为 L2，否则查 L3。"""
    node = find_node(cache.tree, page_id)
    if node is not None:
        state = node.holders.get(cache.hbm.instance_id)
        lock = state.lock if state is not None else 0
        if state is not None and state.resident:
            return PageView("L1", lock, node.l2)
        if node.l2:
            return PageView("L2", lock, True)
    entry = cache.l3.lookup(page_id)
    return None if entry is None else PageView("L3", entry.ref_count, False)


def tier(cache: TieredKVCache, page_id: int) -> str | None:
    view = page(cache, page_id)
    return None if view is None else view.tier


def held(cache: TieredKVCache, page_id: int) -> PageView:
    view = page(cache, page_id)
    assert view is not None
    return view


def sources(req: InferenceRequest, match: PrefixMatch) -> tuple[tuple[int, str], ...]:
    return (
        tuple((chunk_id(req, index), "L1") for index in range(match.l1_pages)) + match.lower
    )


def assert_tree(tree: RadixTree, pools: list[HbmPool]) -> None:
    """逐节点重算各计数，校验增量维护的账本与结构不变量。"""
    hbm_pages = {pool.instance_id: 0 for pool in pools}
    evictable = {pool.instance_id: 0 for pool in pools}
    l2_pages = l2_evictable = nodes = 0
    stack = list(tree.root.children.values())
    while stack:
        node = stack.pop()
        nodes += 1
        parent = node.parent
        assert parent is not None and parent.children[node.segments[0][0]] is node
        assert node.pages == sum(length for _, length in node.segments) > 0
        assert node.hbm == sum(state.resident for state in node.holders.values())
        assert node.locks == sum(state.lock for state in node.holders.values())
        assert node.hbm or node.l2 or node.children  # 无副本节点只作为占位存在
        for inst, state in node.holders.items():
            assert state.resident or state.lock
            assert state.kids == sum(
                child.holders.get(inst) is not None and child.holders[inst].resident
                for child in node.children.values()
            )
            if parent is not tree.root:
                parent_state = parent.holders[inst]
                assert parent_state.lock >= state.lock
                assert parent_state.resident or not state.resident  # 本实例驻留集前缀封闭
            if state.resident and inst in hbm_pages:
                hbm_pages[inst] += node.pages
                evictable[inst] += 0 if state.lock else node.pages
        if node.l2:
            l2_pages += node.pages
            l2_evictable += 0 if node.hbm or node.locks else node.pages
        stack.extend(node.children.values())
    assert tree.nodes == nodes
    for pool in pools:
        assert pool.tree_pages == hbm_pages[pool.instance_id]
        assert pool.evictable == evictable[pool.instance_id]
        assert 0 <= pool.tree_pages + pool.private_pages <= pool.capacity
    assert tree.l2_pages == l2_pages and tree.l2_evictable == l2_evictable
    assert tree.l2_pages + tree.l2_reserved <= tree.l2_capacity


def assert_conserved(cache: TieredKVCache) -> None:
    assert cache.free_pages + cache.resident_pages == cache.capacity_pages
    assert cache.active_pages + cache.evictable_pages == cache.resident_pages
    assert cache.available_tokens == (cache.free_pages + cache.evictable_pages) * cache.page_size
    for tier in ("L2", "L3"):
        snapshot = cache.tier_snapshot(tier)
        assert snapshot.free_blocks + snapshot.resident_blocks == snapshot.capacity_blocks
        assert snapshot.active_blocks + snapshot.evictable_blocks == snapshot.resident_blocks
    assert_tree(cache.tree, [cache.hbm])


def pin_hit(cache: TieredKVCache, req: InferenceRequest) -> PrefixMatch:
    match = cache.probe(req, req.prompt_len, include_l3=cache.direct_l3)
    cache.pin(req, match)
    return match


def no_hit(cache: TieredKVCache) -> PrefixMatch:
    """不复用任何页的匹配：请求全部重算，发布时与已有页去重。"""
    return PrefixMatch(cache.tree.root, 0, 0)


def run_prefill(cache: TieredKVCache, req: InferenceRequest) -> None:
    pin_hit(cache, req)
    cache.admit(req.request_id)
    cache.extend(req.request_id, req.prompt_len)
    cache.publish(req.request_id)
    assert_conserved(cache)


def seed(cache: TieredKVCache, ids: tuple[int, ...], key: int = 10_000) -> None:
    """以 ``ids`` 为完整页重算并发布一条已完成前缀（不读低层），用于预置缓存内容。"""
    req = InferenceRequest(
        key, 0, None, len(ids) * cache.page_size + 1, 1, 1, ids, len(ids), chunk_base(key)
    )
    match = cache.probe(req, req.prompt_len, include_l3=False)
    cache.pin(req, match)
    if match.lower:
        cache.ensure_free(len(match.lower))
        cache.promote()
    cache.admit(key)
    cache.extend(key, req.prompt_len)
    cache.finish(key)


def only_child(node: RadixNode) -> RadixNode:
    (child,) = node.children.values()
    return child


# ---------------------------------------------------------------- 匹配、发布与私有页


@pytest.mark.parametrize("prompt_len", [63, 64, 65])
def test_prompt_last_token_is_not_reused(prompt_len: int) -> None:
    cache = TieredKVCache(256, 64)
    run_prefill(cache, request(0, 128))
    cache.finish(0)
    inherited = (chunk_base(0),) if prompt_len == 65 else ()
    child = request(1, prompt_len, inherited)
    expected = ((chunk_base(0), "L1"),) if prompt_len == 65 else ()
    assert sources(child, cache.probe(child, prompt_len)) == expected
    assert page(cache, chunk_base(0)) is not None
    assert_conserved(cache)


def test_page_size_one_still_requires_last_input_token_and_splits_on_match() -> None:
    cache = TieredKVCache(8, 1)
    run_prefill(cache, request(0, 4))
    cache.finish(0)
    assert cache.tree.nodes == 1
    first = request(1, 3, tuple(chunk_base(0) + i for i in range(2)))
    base = chunk_base(0)
    assert sources(first, cache.probe(first, 3)) == ((base, "L1"), (base + 1, "L1"))
    second = request(2, 4, tuple(chunk_base(0) + i for i in range(3)))
    assert cache.probe(second, 4).l1_pages == 3
    # 匹配止于节点内部时切分（SGLang ``_split_node``），切分后不合并
    assert cache.tree.nodes == 3
    assert_conserved(cache)


def test_private_tail_and_decode_full_page_only_published_at_completion() -> None:
    cache = TieredKVCache(256, 64)
    run_prefill(cache, request(0, 63))
    assert cache.state(0).k == 63 and cache.state(0).f == 0
    assert cache.private_page_count(0) == 1
    assert page(cache, chunk_base(0)) is None
    assert cache.extend(0, 64) == 0
    assert page(cache, chunk_base(0)) is None
    cache.finish(0)
    assert page(cache, chunk_base(0)) == PageView("L1", 0, False)
    assert cache.hbm.private_pages == 0 and cache.evictable_pages == 1
    assert_conserved(cache)


def test_same_batch_private_pages_merge_only_after_publication() -> None:
    cache = TieredKVCache(256, 64)
    shared = chunk_base(0)
    first = request(1, 65, (shared,))
    second = request(2, 65, (shared,))
    for req in (first, second):
        assert pin_hit(cache, req).pages == 0
        cache.admit(req.request_id)
        cache.extend(req.request_id, 65)
    assert cache.resident_pages == 4 and cache.hbm.private_pages == 4
    assert page(cache, shared) is None
    assert cache.publish_batch((2, 1)) == 1
    assert cache.resident_pages == 3 and cache.hbm.private_pages == 2
    assert cache.state(1).f == cache.state(2).f == 64
    assert page(cache, shared) == PageView("L1", 2, False)
    assert_conserved(cache)
    cache.finish(1)
    assert page(cache, shared) == PageView("L1", 1, False)
    assert cache.evictable_pages == 0
    cache.finish(2)
    assert page(cache, shared) == PageView("L1", 0, False)
    assert_conserved(cache)


def test_candidates_lock_shared_path_once_each_and_count_unprotected_pages() -> None:
    cache = TieredKVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    shared = chunk_base(0)
    first = request(1, 65, (shared,))
    second = request(2, 65, (shared,))
    assert cache.unprotected_pages(cache.probe(first, 65)) == 1
    pin_hit(cache, first)
    assert cache.unprotected_pages(cache.probe(second, 65)) == 0
    pin_hit(cache, second)
    assert page(cache, shared) == PageView("L1", 2, False)
    assert cache.available_tokens == 64
    for key in (1, 2):
        cache.admit(key)
        cache.withdraw(key)
    assert cache.available_tokens == 128
    assert_conserved(cache)


def test_restore_reuses_complete_context_and_withdraw_discards_private_pages() -> None:
    cache = TieredKVCache(256, 64)
    req = request(0, 65)
    run_prefill(cache, req)
    assert cache.extend(0, 128) == 0
    assert page(cache, chunk_base(0) + 1) is None
    cache.withdraw(0)
    assert cache.resident_pages == 1 and cache.hbm.private_pages == 0
    assert cache.probe(req, 65).l1_pages == 1
    match = cache.probe(req, 128, restoring=True)
    assert match.l1_pages == 1 and match.pages == 1
    cache.pin(req, match)
    cache.admit(0)
    cache.extend(0, 128)
    assert cache.state(0).k == 128
    cache.finish(0)
    assert page(cache, chunk_base(0) + 1) is not None
    assert_conserved(cache)


def test_full_hit_restores_without_allocating_or_prefill() -> None:
    cache = TieredKVCache(128, 64)
    req = request(0, 64)
    run_prefill(cache, req)
    cache.finish(0)
    match = cache.probe(req, 64, restoring=True)
    assert match.l1_pages == 1
    cache.pin(req, match)
    state = cache.admit(0)
    assert state.k == state.f == 64
    assert cache.hbm.private_pages == 0 and cache.resident_pages == 1
    cache.withdraw(0)
    assert_conserved(cache)


# ---------------------------------------------------------------- 节点结构


def test_contiguous_pages_form_one_node_and_partial_match_splits_it() -> None:
    cache = TieredKVCache(512, 64)
    run_prefill(cache, request(0, 257))
    cache.finish(0)
    base = chunk_base(0)
    node = only_child(cache.tree.root)
    assert cache.tree.nodes == 1 and node.segments == ((base, 4),) and node.pages == 4
    child = request(1, 129, (base, base + 1))
    match = cache.probe(child, 129)
    assert match.l1_pages == 2 and cache.tree.nodes == 2
    top = only_child(cache.tree.root)
    assert match.node is top and top.segments == ((base, 2),)
    assert only_child(top) is node and node.segments == ((base + 2, 2),)
    assert top.holders[0].resident and top.holders[0].kids == 1
    assert_conserved(cache)


def test_trace_segments_match_by_binary_search_and_branch_at_divergence() -> None:
    cache = TieredKVCache(1024, 64)
    run_prefill(cache, request(0, 257, (10, 11, 20, 21)))
    cache.finish(0)
    assert only_child(cache.tree.root).segments == ((10, 2), (20, 2))
    branch = request(1, 257, (10, 11, 20, 99))
    assert pin_hit(cache, branch).l1_pages == 3
    cache.admit(1)
    cache.extend(1, 257)
    cache.publish(1)
    top = only_child(cache.tree.root)
    assert top.segments == ((10, 2), (20, 1))
    assert {key: child.segments for key, child in top.children.items()} == {
        21: ((21, 1),),
        99: ((99, 1),),
    }
    assert cache.tree.nodes == 3
    cache.finish(1)
    assert_conserved(cache)


def test_publication_splits_inserted_range_at_prompt_boundary() -> None:
    cache = TieredKVCache(512, 64)
    req = request(0, 128)
    cache.pin(req, cache.probe(req, 192, restoring=True))
    cache.admit(0)
    cache.extend(0, 192)
    cache.publish(0)
    prompt = only_child(cache.tree.root)
    assert prompt.pages == 2 and only_child(prompt).pages == 1
    cache.finish(0)
    assert_conserved(cache)


# ---------------------------------------------------------------- HBM 淘汰


def test_lru_evicts_whole_leaf_node_and_keeps_newer_pages() -> None:
    cache = TieredKVCache(256, 64)
    run_prefill(cache, request(0, 128))
    cache.finish(0)
    run_prefill(cache, request(1, 64))
    cache.finish(1)
    assert cache.hbm.heap is None  # 首次淘汰前不建堆
    run_prefill(cache, request(2, 128))
    assert cache.hbm.heap is not None
    assert cache.evictions == 2
    assert page(cache, chunk_base(0)) is None and page(cache, chunk_base(0) + 1) is None
    assert page(cache, chunk_base(1)) is not None
    assert cache.free_pages == 1
    assert_conserved(cache)


def test_only_leaves_are_evicted_and_parent_follows_its_children() -> None:
    cache = TieredKVCache(192, 64)
    parent = request(0, 128)
    run_prefill(cache, parent)
    cache.finish(0)
    run_prefill(cache, request(1, 192, (chunk_base(0), chunk_base(0) + 1)))
    cache.finish(1)
    assert cache.free_pages == 0 and cache.evictable_pages == 3
    run_prefill(cache, request(2, 64))
    assert cache.evictions == 1
    assert page(cache, chunk_base(0)) is not None and page(cache, chunk_base(1) + 2) is None
    assert cache.ensure_free(cache.free_pages + cache.evictable_pages) == 2
    assert page(cache, chunk_base(0)) is None and cache.evictions == 3
    assert_conserved(cache)


def test_locked_paths_are_never_evicted_and_failure_has_no_side_effects() -> None:
    cache = TieredKVCache(128, 64)
    run_prefill(cache, request(0, 64))
    run_prefill(cache, request(1, 64))
    before = (cache.snapshot(), cache.evictions, cache.tree.nodes)
    with pytest.raises(CacheCapacityError):
        cache.ensure_free(1)
    assert (cache.snapshot(), cache.evictions, cache.tree.nodes) == before
    assert_conserved(cache)


def test_probe_refreshes_lru_like_sglang_match_prefix() -> None:
    cache = TieredKVCache(128, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    run_prefill(cache, request(1, 64))
    cache.finish(1)
    cache.probe(request(2, 65, (chunk_base(0),)), 65)
    run_prefill(cache, request(3, 64))
    assert page(cache, chunk_base(0)) is not None
    assert page(cache, chunk_base(1)) is None
    assert_conserved(cache)


# ---------------------------------------------------------------- L2 write-through 与 H-leaf


def test_publication_writes_through_and_hbm_eviction_keeps_l2_copy() -> None:
    cache = TieredKVCache(128, 64, 256)
    run_prefill(cache, request(0, 128))
    assert page(cache, chunk_base(0)) == PageView("L1", 1, True)
    assert cache.tier_snapshot("L2").resident_blocks == 2
    assert cache.tier_snapshot("L2").evictable_blocks == 0  # 仍有 HBM 副本
    assert cache.demotions == 2
    cache.finish(0)
    run_prefill(cache, request(1, 128))
    assert page(cache, chunk_base(0)) == PageView("L2", 0, True)
    assert cache.evictions == 2
    assert cache.tier_snapshot("L2").resident_blocks == 4
    assert cache.tier_snapshot("L2").evictable_blocks == 2
    assert_conserved(cache)


def test_write_through_is_skipped_without_l2_room_and_unbacked_eviction_drops() -> None:
    cache = TieredKVCache(128, 64, 64)
    run_prefill(cache, request(0, 128))
    assert page(cache, chunk_base(0)) == PageView("L1", 1, False)
    cache.finish(0)
    run_prefill(cache, request(1, 128))
    assert page(cache, chunk_base(0)) is None and cache.tree.nodes == 1
    assert cache.tier_snapshot("L2").resident_blocks == 0
    assert_conserved(cache)


def test_l2_copy_held_in_hbm_is_not_evictable_for_write_through() -> None:
    cache = TieredKVCache(256, 64, 128)
    run_prefill(cache, request(0, 128))
    cache.finish(0)
    run_prefill(cache, request(1, 64))
    assert page(cache, chunk_base(0)) == PageView("L1", 0, True)
    assert page(cache, chunk_base(1)) == PageView("L1", 1, False)
    assert cache.tier_snapshot("L2").evictions == 0
    assert_conserved(cache)


def test_l2_evicts_only_h_leaves_and_writes_them_to_l3() -> None:
    cache = TieredKVCache(128, 64, 128, 256)
    run_prefill(cache, request(0, 128))
    cache.finish(0)
    run_prefill(cache, request(1, 128))
    assert page(cache, chunk_base(0)) == PageView("L3", 0, False)
    assert page(cache, chunk_base(1)) == PageView("L1", 1, True)
    assert cache.tier_snapshot("L2").evictions == 2
    assert cache.tier_snapshot("L3").resident_blocks == 2
    assert cache.tree.nodes == 1
    assert cache.demotions == 6  # 两次 write-through 各 2 页，加 L2→L3 的 2 页
    assert_conserved(cache)


def test_l2_hit_is_copied_to_hbm_before_admission_and_keeps_l2_copy() -> None:
    cache = TieredKVCache(128, 64, 256)
    run_prefill(cache, request(0, 128))
    cache.finish(0)
    run_prefill(cache, request(1, 128))
    cache.finish(1)
    shared = (chunk_base(0), chunk_base(0) + 1)
    child = request(2, 129, shared)
    match = pin_hit(cache, child)
    assert (match.l1_pages, match.local_pages) == (0, 2)
    assert sources(child, match) == tuple((item, "L2") for item in shared)
    assert page(cache, shared[0]) == PageView("L2", 1, True)
    with pytest.raises(CacheCapacityError, match="低层命中页须先提升到 L1"):
        cache.admit(2)
    assert cache.ensure_free(2) == 2
    assert cache.promote() == 2 and cache.promotions == 2
    state = cache.admit(2)
    assert state.k == state.f == 128
    assert page(cache, shared[0]) == PageView("L1", 1, True)
    assert cache.tier_snapshot("L2").resident_blocks == 4
    cache.withdraw(2)
    assert_conserved(cache)


def test_shared_tree_counts_hbm_holders_across_instances() -> None:
    l3 = CachePool(0)
    tree = RadixTree(64, 4, l3)
    first, second = (
        TieredKVCache(128, 64, shared_l3=l3, shared_l2=tree, instance_id=index)
        for index in range(2)
    )
    pools = [first.hbm, second.hbm]
    run_prefill(first, request(0, 128))
    first.finish(0)
    shared = (chunk_base(0), chunk_base(0) + 1)
    # 另一实例的匹配同样切分共享节点，各实例的副本与锁状态随之复制到两半。
    assert second.probe(request(5, 65, shared[:1]), 65).local_pages == 1
    assert tree.nodes == 2
    assert first.probe(request(6, 129, shared), 129).l1_pages == 2
    child = request(1, 129, shared)
    match = pin_hit(second, child)
    assert (match.l1_pages, match.local_pages) == (0, 2)
    second.ensure_free(2)
    assert second.promote() == 2
    second.admit(1)
    second.withdraw(1)
    node = find_node(tree, shared[1])
    assert node is not None and node.hbm == 2
    assert_tree(tree, pools)
    run_prefill(first, request(2, 128))
    assert node.hbm == 1 and l2_snapshot(tree).evictable_blocks == 0
    run_prefill(second, request(3, 128))
    assert find_node(tree, shared[1]) is None  # 两实例都淘汰后成为 H-leaf，被 write-through 挤出
    assert page(first, chunk_base(2)) == PageView("L1", 1, True)
    assert page(second, chunk_base(3)) == PageView("L1", 1, True)
    assert_tree(tree, pools)


# ---------------------------------------------------------------- L3 预取与两层结构


def test_prefetch_reserves_l2_and_inserts_completed_pages_on_request_path() -> None:
    cache = TieredKVCache(256, 64, 256, 256)
    ids = tuple(chunk_base(7) + index for index in range(3))
    for item in ids:
        cache.l3.insert(item)
    req = request(1, 193, ids)
    match = cache.probe(req, 193)
    assert (match.local_pages, match.l3_pages) == (0, 3)
    for item in ids:
        cache.reserve_prefetch(item)
    assert cache.tier_snapshot("L2").resident_blocks == 3
    assert cache.tier_snapshot("L2").active_blocks == 3
    assert all(page(cache, item) == PageView("L3", 1, False) for item in ids)
    cache.complete_prefetch(req, tuple(enumerate(ids)))
    assert cache.tree.nodes == 1 and cache.promotions == 3
    assert cache.tier_snapshot("L2").evictable_blocks == 3
    assert all(page(cache, item) == PageView("L2", 0, True) for item in ids)
    assert cache.probe(req, 193, include_l3=False).local_pages == 3
    assert_conserved(cache)


def test_prefetch_reservation_never_pushes_its_own_source_out_of_full_l3() -> None:
    cache = TieredKVCache(64, 64, 64, 64)
    run_prefill(cache, request(0, 64))
    cache.finish(0)
    cache.ensure_free(1)  # 唯一 L2 页成为 H-leaf
    source = chunk_base(7)
    cache.l3.insert(source)
    cache.reserve_prefetch(source)  # 腾出 L2 时被淘汰页写 L3 失败（来源受保护），直接丢弃
    assert cache.l3.entries[source].ref_count == 1
    assert cache.l3.lookup(chunk_base(0)) is None and cache.tree.nodes == 0
    cache.complete_prefetch(request(1, 65, (source,)), ((0, source),))
    assert page(cache, source) == PageView("L2", 0, True)
    assert_conserved(cache)


def test_prefetch_after_missing_prefix_keeps_placeholder_until_child_is_evicted() -> None:
    cache = TieredKVCache(256, 64, 256, 256)
    ids = tuple(chunk_base(7) + index for index in range(3))
    for item in ids:
        cache.l3.insert(item)
    req = request(1, 193, ids)
    for item in ids[1:]:
        cache.reserve_prefetch(item)
    cache.complete_prefetch(req, ((1, ids[1]), (2, ids[2])))
    placeholder = only_child(cache.tree.root)
    assert cache.tree.nodes == 2 and not placeholder.l2 and not placeholder.hbm
    match = cache.probe(req, 193)
    assert match.local_pages == 0 and match.l3_pages == 3  # 本地前缀止于占位节点
    assert cache.tree.ensure_l2_free(cache.hbm, 4)
    assert cache.tree.nodes == 0
    assert_conserved(cache)


def test_two_tier_writes_hbm_evictions_to_l3_and_inserts_direct_reads() -> None:
    cache = TieredKVCache(128, 64, 0, 256)
    assert cache.direct_l3
    run_prefill(cache, request(0, 128))
    cache.finish(0)
    run_prefill(cache, request(1, 128))
    cache.finish(1)
    shared = (chunk_base(0), chunk_base(0) + 1)
    assert page(cache, shared[0]) == PageView("L3", 0, False)
    assert cache.demotions == 2
    child = request(2, 129, shared)
    match = pin_hit(cache, child)
    assert match.l3_pages == 2 and page(cache, shared[0]) == PageView("L3", 1, False)
    cache.ensure_free(2)
    assert cache.promote() == 2
    cache.admit(2)
    assert page(cache, shared[0]) == PageView("L1", 1, False)
    assert cache.l3.entries[shared[0]].ref_count == 0
    cache.finish(2)
    assert_conserved(cache)


# ---------------------------------------------------------------- 随机操作不变量


@pytest.mark.parametrize("seed", range(6))
def test_random_operations_keep_shared_tree_consistent(seed: int) -> None:
    rng = random.Random(seed)
    l3 = CachePool(24)
    tree = RadixTree(1, 12, l3)
    caches = [
        TieredKVCache(10, 1, shared_l3=l3, shared_l2=tree, instance_id=index)
        for index in range(2)
    ]
    contexts: list[tuple[int, ...]] = [()]
    active: list[tuple[TieredKVCache, int]] = []
    for key in range(300):
        cache = rng.choice(caches)
        action = rng.random()
        if active and action < 0.3:
            owner, done = active.pop(rng.randrange(len(active)))
            (owner.finish if rng.random() < 0.7 else owner.withdraw)(done)
        elif active and action < 0.5:
            owner, running = rng.choice(active)
            try:
                owner.extend(running, owner.state(running).k + 1)
            except CacheCapacityError:
                active.remove((owner, running))
                owner.withdraw(running)
        else:
            base = rng.choice(contexts)
            ids = base[: rng.randint(0, len(base))] + tuple(
                1_000 * (key + 1) + index for index in range(rng.randint(1, 4))
            )
            contexts.append(ids)
            req = InferenceRequest(
                key, 0, None, len(ids) + 1, 8, 8, ids, 0, chunk_base(10_000 + key)
            )
            match = cache.probe(req, req.prompt_len, include_l3=False)
            needed = len(match.lower) + req.prompt_len - match.pages
            available = cache.free_pages + cache.evictable_pages - cache.unprotected_pages(match)
            if needed <= available:
                cache.pin(req, match)
                cache.ensure_free(needed)
                cache.promote()
                cache.admit(key)
                cache.extend(key, req.prompt_len)
                cache.publish(key)
                active.append((cache, key))
        assert_tree(tree, [item.hbm for item in caches])
        for item in caches:
            assert item.free_pages + item.resident_pages == item.capacity_pages
    for owner, key in active:
        owner.finish(key)
    assert_tree(tree, [item.hbm for item in caches])
    assert all(item.active_pages == 0 for item in caches)
    assert tree.l2_evictions and l3.resident and all(item.evictions for item in caches)


# ---------------------------------------------------------------- 共享 L3 页池


def activate_heap(pool: CachePool) -> None:
    pool.heap_active = True
    pool._rebuild_heap()


@pytest.mark.parametrize("change", ["queue", "protect", "remove"])
def test_periodic_heap_rebuild_discards_stale_entries(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("simcore.kv_cache._HEAP_REBUILD_CHECK_INTERVAL", 512)
    pool = CachePool(3)
    pool.insert(0)
    pool.insert(1)
    activate_heap(pool)
    for _ in range(511):
        pool.touch(1)
    assert len(pool.heap) == 513
    if change == "protect":
        pool.add_ref(0)
    elif change == "remove":
        pool.remove(0)
    pool.touch(1)
    expected = sorted(
        (item.last_access_seq, item.chunk_id)
        for item in pool.entries.values()
        if not item.ref_count
    )
    assert set(pool.heap) == pool.evictable == set(expected)
    assert pool.queued == set(pool.heap) and pool.queued is not pool.evictable
    assert pool.heap[0] == expected[0]
    if change == "protect":
        pool.drop_ref(0)
        assert pool.heap[0] == (pool.entries[0].last_access_seq, 0)


@pytest.mark.parametrize("protect_all", [False, True])
def test_heap_rebuild_uses_evictable_set_without_accessing_entries(
    protect_all: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = CachePool(1026)
    for index in range(1024):
        pool.insert(10_000 + index)
        pool.add_ref(10_000 + index)
    pool.insert(0)
    pool.insert(1)
    activate_heap(pool)
    for _ in range(512):
        pool.touch(1)
    if protect_all:
        pool.add_ref(0)
        pool.add_ref(1)
    expected = sorted(
        (item.last_access_seq, item.chunk_id)
        for item in pool.entries.values()
        if not item.ref_count
    )
    assert len(pool.heap) == 514
    with monkeypatch.context() as patch:
        patch.setattr(pool, "entries", object())
        pool._maybe_rebuild_heap()
    assert sorted(pool.heap) == expected
    assert pool.evictable == pool.queued == set(expected)
    for seq, item in expected:
        assert pool.pop_lru() is pool.entries[item]
        assert (seq, item) in pool.evictable
        pool.remove(item)
    assert pool.pop_lru() is None and not pool.evictable


@pytest.mark.parametrize("interval", [64, 4096])
def test_periodic_heap_check_bounds_stale_entries_without_eviction(
    interval: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("simcore.kv_cache._HEAP_REBUILD_CHECK_INTERVAL", interval)
    pool = CachePool(1)
    pool.insert(0)
    activate_heap(pool)
    for _ in range(10_000):
        pool.touch(0)
    assert len(pool.heap) <= len(pool.evictable) + 512 + interval
    assert pool.queued == set(pool.heap)
    assert pool.pop_lru() is pool.entries[0]


def test_heap_rebuild_waits_until_stale_entries_reach_evictable_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("simcore.kv_cache._HEAP_REBUILD_CHECK_INTERVAL", 1)
    pool = CachePool(1024)
    for index in range(1024):
        pool.insert(index)
    activate_heap(pool)
    for index in range(1023):
        pool.touch(index)
    assert len(pool.heap) == 2047
    pool.touch(1023)
    assert len(pool.heap) == len(pool.evictable) == 1024
    assert pool.pop_lru() is pool.entries[0]


def test_l3_heap_is_built_lazily_and_tracks_protection() -> None:
    pool = CachePool(4)
    for item in (0, 1, 2):
        pool.insert(item)
    pool.touch(0)
    pool.add_ref(1)
    assert not pool.heap_active and pool.heap == [] and len(pool.evictable) == 2
    assert pool.pop_lru() is pool.entries[2]
    assert pool.heap_active
    pool.remove(2, evicted=True)
    pool.drop_ref(1)
    assert pool.pop_lru() is pool.entries[1]
    pool.remove(1, evicted=True)
    assert pool.pop_lru() is pool.entries[0]
    snapshot = pool.snapshot()
    assert (snapshot.resident_blocks, snapshot.evictions) == (1, 2)
    assert snapshot.peak_resident_blocks == 3
    with pytest.raises(ValueError, match="已存在"):
        pool.insert(0)
