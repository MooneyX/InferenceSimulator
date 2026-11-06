"""实例请求 KV 管理：L1/L2 前缀基数树与按页管理的共享 L3 池。"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush

from simcore.radix_cache import HbmPool, RadixNode, RadixTree, chunk_id
from simcore.schemas import InferenceRequest

__all__ = [
    "CacheCapacityError",
    "CacheEntry",
    "CachePool",
    "KVRequestState",
    "L1Snapshot",
    "PrefixMatch",
    "TierSnapshot",
    "TieredKVCache",
    "l2_snapshot",
]

_HEAP_REBUILD_CHECK_INTERVAL = 4096


class CacheCapacityError(ValueError):
    """当前物理容量不足，调用方应等待或回撤请求。"""


@dataclass(slots=True)
class CacheEntry:
    chunk_id: int
    ref_count: int
    last_access_seq: int


@dataclass(frozen=True, slots=True)
class KVRequestState:
    k: int  # 请求已有 KV 的总长度，包含私有 KV，单位为 token
    f: int  # 请求已发布到前缀树的 KV 长度，可以被其他请求复用，单位为 token


@dataclass(frozen=True, slots=True)
class L1Snapshot:
    capacity_blocks: int
    resident_blocks: int
    active_blocks: int
    evictable_blocks: int
    free_blocks: int
    internal_fragmentation_tokens: int
    reserved_future_blocks: int = 0


@dataclass(frozen=True, slots=True)
class TierSnapshot:
    capacity_blocks: int
    resident_blocks: int
    active_blocks: int
    evictable_blocks: int
    free_blocks: int
    peak_resident_blocks: int
    evictions: int


def l2_snapshot(tree: RadixTree) -> TierSnapshot:
    """L2 驻留含在途预取占位；可淘汰页为无 HBM 副本且未加锁的 L2 页。"""
    resident = tree.l2_pages + tree.l2_reserved
    return TierSnapshot(
        tree.l2_capacity,
        resident,
        resident - tree.l2_evictable,
        tree.l2_evictable,
        tree.l2_capacity - resident,
        tree.l2_peak,
        tree.l2_evictions,
    )


@dataclass(slots=True)
class CachePool:
    """按页管理的共享 L3：完整页索引、保护引用与稳定 LRU，同 ID 只存一份。"""

    capacity: int
    entries: dict[int, CacheEntry] = field(default_factory=dict, init=False)
    resident: int = field(default=0, init=False)
    evictable: set[tuple[int, int]] = field(default_factory=set, init=False)
    peak: int = field(default=0, init=False)
    evictions: int = field(default=0, init=False)
    access_seq: int = field(default=0, init=False)
    # LRU 堆在首次淘汰时才由 evictable 建立；此前只维护集合，大容量池常整场不淘汰。
    heap_active: bool = field(default=False, init=False)
    heap: list[tuple[int, int]] = field(default_factory=list, init=False)
    queued: set[tuple[int, int]] = field(default_factory=set, init=False)
    queued_since_check: int = field(default=0, init=False)

    def lookup(self, chunk_id: int) -> CacheEntry | None:
        return self.entries.get(chunk_id)

    def snapshot(self) -> TierSnapshot:
        return TierSnapshot(
            self.capacity,
            self.resident,
            self.resident - len(self.evictable),
            len(self.evictable),
            self.capacity - self.resident,
            self.peak,
            self.evictions,
        )

    def insert(self, chunk_id: int) -> None:
        if chunk_id in self.entries:
            raise ValueError("完整页已存在")
        if self.resident >= self.capacity:
            raise CacheCapacityError("L3 容量不足")
        self.resident += 1
        self.peak = max(self.peak, self.resident)
        self.access_seq += 1
        page = CacheEntry(chunk_id, 0, self.access_seq)
        self.entries[chunk_id] = page
        self._queue(page)

    def remove(self, chunk_id: int, *, evicted: bool = False) -> CacheEntry:
        page = self.entries.pop(chunk_id)
        self.resident -= 1
        if not page.ref_count:
            self.evictable.remove((page.last_access_seq, chunk_id))
        self.evictions += int(evicted)
        return page

    def add_ref(self, chunk_id: int) -> None:
        page = self.entries[chunk_id]
        if not page.ref_count:
            self.evictable.remove((page.last_access_seq, chunk_id))
        page.ref_count += 1

    def drop_ref(self, chunk_id: int) -> None:
        page = self.entries[chunk_id]
        page.ref_count -= 1
        if not page.ref_count:
            self._queue(page)

    def touch(self, chunk_id: int) -> None:
        self.access_seq += 1
        page = self.entries[chunk_id]
        if not page.ref_count:
            self.evictable.remove((page.last_access_seq, chunk_id))
        page.last_access_seq = self.access_seq
        if not page.ref_count:
            self._queue(page)

    def _queue(self, page: CacheEntry) -> None:
        key = (page.last_access_seq, page.chunk_id)
        self.evictable.add(key)
        if self.heap_active and key not in self.queued:
            heappush(self.heap, key)
            self.queued.add(key)
            self.queued_since_check += 1
            if self.queued_since_check >= _HEAP_REBUILD_CHECK_INTERVAL:
                self.queued_since_check = 0
                self._maybe_rebuild_heap()

    def _maybe_rebuild_heap(self) -> None:
        # 过期条目不少于有效条目时才重建，每条过期条目摊销约一次建堆成本，堆至多约为两倍大小。
        stale = len(self.heap) - len(self.evictable)
        if stale < 512 or stale < len(self.evictable):
            return
        self._rebuild_heap()

    def _rebuild_heap(self) -> None:
        self.heap = list(self.evictable)
        heapify(self.heap)
        self.queued = self.evictable.copy()

    def pop_lru(self) -> CacheEntry | None:
        """弹出最老的可淘汰页，顺带清理堆顶过期条目。"""
        if not self.heap_active:
            self.heap_active = True
            self._rebuild_heap()
        while self.heap:
            seq, chunk_id = heappop(self.heap)
            self.queued.remove((seq, chunk_id))
            page = self.lookup(chunk_id)
            if page is not None and not page.ref_count and page.last_access_seq == seq:
                return page
        return None


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    """连续前缀命中：本实例 HBM 页在前，其后依次为 L2 与 L3 页。

    ``node`` 是本地（HBM+L2）连续前缀在前缀树中的终点；``lower`` 列出 HBM 之后各页的
    ``(chunk_id, tier)``。
    """

    node: RadixNode
    l1_pages: int
    local_pages: int
    lower: tuple[tuple[int, str], ...] = ()

    @property
    def pages(self) -> int:
        return self.l1_pages + len(self.lower)

    @property
    def l3_pages(self) -> int:
        return self.pages - self.local_pages

    def without_l3(self) -> "PrefixMatch":
        local = self.lower[: self.local_pages - self.l1_pages]
        return PrefixMatch(self.node, self.l1_pages, self.local_pages, local)


@dataclass(slots=True)
class _Pin:
    request: InferenceRequest
    match: PrefixMatch
    node: RadixNode  # 当前锁终点；两层结构插入 L3 命中页后后移
    promoted: bool = False


class TieredKVCache:
    """单实例 KV 管理：请求私有页、L1/L2 前缀树中的本实例视图，以及共享 L3。"""

    def __init__(
        self,
        max_total_tokens: int,
        page_size: int,
        l2_capacity_tokens: int = 0,
        l3_capacity_tokens: int = 0,
        *,
        shared_l3: CachePool | None = None,
        shared_l2: RadixTree | None = None,
        instance_id: int = 0,
    ) -> None:
        if page_size < 1 or max_total_tokens < page_size or max_total_tokens % page_size:
            raise ValueError("max_total_tokens 必须至少一页且是 page_size 的整数倍")
        if (
            l2_capacity_tokens < 0
            or l3_capacity_tokens < 0
            or (l2_capacity_tokens % page_size or l3_capacity_tokens % page_size)
        ):
            raise ValueError("低层容量必须是 page_size 的非负整数倍")
        l3 = shared_l3 if shared_l3 is not None else CachePool(l3_capacity_tokens // page_size)
        if shared_l2 is not None:
            if shared_l2.page_size != page_size or shared_l2.l3 is not l3:
                raise ValueError("共享 L2 的页大小和 L3 必须与实例一致")
            tree = shared_l2
        else:
            tree = RadixTree(page_size, l2_capacity_tokens // page_size, l3)
        self.tree = tree
        self.l3 = l3
        self.page_size = page_size
        self.capacity_pages = max_total_tokens // page_size
        # L2 关闭而 L3 启用时为两层结构：L1 淘汰直接写入 L3，L3 命中页直接读入 L1。
        self.direct_l3 = bool(l3.capacity) and not tree.l2_capacity
        self.hbm = HbmPool(instance_id, self.capacity_pages)
        self._requests: dict[int, InferenceRequest] = {}
        self._states: dict[int, KVRequestState] = {}
        self._endpoints: dict[int, RadixNode] = {}
        self._pins: dict[int, _Pin] = {}

    @property
    def resident_pages(self) -> int:
        return self.hbm.resident

    @property
    def free_pages(self) -> int:
        return self.capacity_pages - self.hbm.resident

    @property
    def evictable_pages(self) -> int:
        return self.hbm.evictable

    @property
    def active_pages(self) -> int:
        return self.hbm.resident - self.hbm.evictable

    @property
    def evictions(self) -> int:
        return self.hbm.evictions

    @property
    def peak_resident_pages(self) -> int:
        return self.hbm.peak

    @property
    def promotions(self) -> int:
        return self.hbm.promotions

    @property
    def demotions(self) -> int:
        return self.hbm.demotions

    @property
    def available_tokens(self) -> int:
        return (self.free_pages + self.evictable_pages) * self.page_size

    @property
    def l2_incoming_ms(self) -> dict[int, float]:
        return self.tree.l2_incoming_ms

    def tier_snapshot(self, tier: str) -> TierSnapshot:
        return l2_snapshot(self.tree) if tier == "L2" else self.l3.snapshot()

    def snapshot(self) -> L1Snapshot:
        private_tokens = sum(state.k - state.f for state in self._states.values())
        return L1Snapshot(
            self.capacity_pages,
            self.resident_pages,
            self.active_pages,
            self.evictable_pages,
            self.free_pages,
            self.hbm.private_pages * self.page_size - private_tokens,
        )

    def state(self, request_id: int) -> KVRequestState:
        return self._states[request_id]

    def private_page_count(self, request_id: int) -> int:
        state = self.state(request_id)
        return (state.k + self.page_size - 1) // self.page_size - state.f // self.page_size

    def probe(
        self,
        request: InferenceRequest,
        target_tokens: int,
        *,
        restoring: bool = False,
        include_l3: bool = True,
    ) -> PrefixMatch:
        """前缀树中的本地连续前缀，之后可接共享 L3 逐页连续命中；匹配刷新树中路径的 LRU。"""
        if target_tokens < request.prompt_len:
            raise ValueError("重算目标不得小于原始 prompt")
        limit = (
            target_tokens // self.page_size
            if restoring
            else (request.prompt_len - 1) // self.page_size
        )
        node, l1_pages, local_pages = self.tree.match(self.hbm, request, limit)
        lower = [(chunk_id(request, index), "L2") for index in range(l1_pages, local_pages)]
        if include_l3 and self.l3.capacity:
            entries = self.l3.entries
            for index in range(local_pages, limit):
                page = chunk_id(request, index)
                if page not in entries:
                    break
                lower.append((page, "L3"))
        return PrefixMatch(node, l1_pages, local_pages, tuple(lower))

    def unprotected_pages(self, match: PrefixMatch) -> int:
        """命中路径上本实例驻留但尚未加锁的页数；加锁后它们不再可淘汰。"""
        return self.tree.unlocked_resident_pages(self.hbm, match.node)

    def pin(self, request: InferenceRequest, match: PrefixMatch) -> None:
        """沿命中路径加锁，并保护其后的 L3 命中页。"""
        if request.request_id in self._pins or request.request_id in self._states:
            raise ValueError("请求已被保护或驻留")
        self.tree.lock(self.hbm, match.node)
        for page, tier in match.lower[match.local_pages - match.l1_pages :]:
            if tier == "L3":
                self.l3.add_ref(page)
        self._pins[request.request_id] = _Pin(request, match, match.node)

    def promote(self) -> int:
        """把全部已保护候选的低层命中复制到本实例 HBM；调用方已腾出容量。

        L2 命中沿加锁路径复制并保留 L2 副本；两层结构的 L3 命中页插入前缀树并后移锁终点。
        同批候选共享的节点只复制一次。返回复制页数。
        """
        pages = 0
        for pin in self._pins.values():
            if pin.promoted:
                continue
            pin.promoted = True
            match = pin.match
            if match.local_pages > match.l1_pages:
                pages += self.tree.promote_path(self.hbm, pin.node)
            if match.l3_pages:
                start, end = match.local_pages, match.pages
                pin.node, duplicate = self.tree.attach(
                    self.hbm, pin.request, start, end, pin.node
                )
                self.hbm.promotions += end - start - duplicate
                pages += end - start - duplicate
                for page, tier in match.lower[start - match.l1_pages :]:
                    if tier == "L3":
                        self.l3.touch(page)
        return pages

    def admit(self, request_id: int) -> KVRequestState:
        """候选转为驻留请求：命中页须已全部在本实例 HBM，锁转由请求持有。"""
        pin = self._pins[request_id]
        if (pin.match.pages > pin.match.l1_pages and not pin.promoted) or not self.tree.resident(
            self.hbm, pin.node
        ):
            raise CacheCapacityError("低层命中页须先提升到 L1 才能接纳请求")
        del self._pins[request_id]
        match = pin.match
        for page, tier in match.lower[match.local_pages - match.l1_pages :]:
            if tier == "L3":
                self.l3.drop_ref(page)
        tokens = match.pages * self.page_size
        self._requests[request_id] = pin.request
        self._endpoints[request_id] = pin.node
        state = KVRequestState(tokens, tokens)
        self._states[request_id] = state
        return state

    def extend(self, request_id: int, target_tokens: int) -> int:
        state = self.state(request_id)
        if target_tokens <= state.k:
            raise ValueError("KV 目标长度必须大于当前 K")
        new_pages = (target_tokens + self.page_size - 1) // self.page_size - (
            state.k + self.page_size - 1
        ) // self.page_size
        self.ensure_free(new_pages)
        self.hbm.grow(new_pages)
        self._states[request_id] = KVRequestState(target_tokens, state.f)
        return new_pages

    def publish(self, request_id: int) -> int:
        """发布 ``[F, K)`` 内的完整私有页；已在本实例树中的重复页释放，返回释放页数。"""
        state = self.state(request_id)
        request = self._requests[request_id]
        start = state.f // self.page_size
        end = state.k // self.page_size
        self.hbm.private_pages -= max(end - start, 0)
        endpoint, duplicate = self.tree.publish(
            self.hbm,
            request,
            start,
            end,
            self._endpoints[request_id],
            request.prompt_len // self.page_size,
        )
        self._endpoints[request_id] = endpoint
        if end > start:
            self._states[request_id] = KVRequestState(state.k, end * self.page_size)
        return duplicate

    def publish_batch(self, request_ids: Sequence[int]) -> int:
        return sum(self.publish(request_id) for request_id in sorted(request_ids))

    def finish(self, request_id: int) -> None:
        self.publish(request_id)
        self._release(request_id)

    def withdraw(self, request_id: int) -> None:
        self._release(request_id)

    def _release(self, request_id: int) -> None:
        count = self.private_page_count(request_id)
        del self._states[request_id], self._requests[request_id]
        self.tree.unlock(self.hbm, self._endpoints.pop(request_id))
        self.hbm.private_pages -= count

    def ensure_free(self, count: int) -> int:
        """确保空闲页不少于 ``count``，按 LRU 整节点淘汰本实例 D-leaf；返回淘汰页数。"""
        free = self.free_pages
        if count <= free:
            return 0
        if count > free + self.hbm.evictable:
            raise CacheCapacityError("L1 容量不足且没有可淘汰页")
        return self.tree.evict_hbm(self.hbm, count)

    def reserve_prefetch(self, page: int) -> None:
        """为一页 L3→L2 预取保护 L3 来源并预留 L2 目的容量；腾挪 L2 写入 L3 时不会挤掉来源。"""
        if self.l3.lookup(page) is None:
            raise ValueError("待传输页不存在")
        self.l3.add_ref(page)
        if not self.tree.reserve_l2(self.hbm):
            self.l3.drop_ref(page)
            raise CacheCapacityError("L2 容量不足且没有可淘汰页")

    def complete_prefetch(
        self, request: InferenceRequest, pages: Sequence[tuple[int, int]]
    ) -> None:
        """预取完成：按请求页序号把连续段插入 L2，释放 L3 来源保护。"""
        index = 0
        while index < len(pages):
            stop = index + 1
            while stop < len(pages) and pages[stop][0] == pages[stop - 1][0] + 1:
                stop += 1
            self.tree.insert_l2(request, pages[index][0], pages[stop - 1][0] + 1)
            index = stop
        for _, page in pages:
            self.l3.drop_ref(page)
            self.l3.touch(page)
        self.hbm.promotions += len(pages)
