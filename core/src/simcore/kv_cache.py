"""实例请求 KV 管理与可独占或共享的分层资源池。"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush

from simcore.prefix_hash import logical_chunk_id
from simcore.schemas import InferenceRequest

__all__ = [
    "CacheCapacityError",
    "CacheEntry",
    "CachePool",
    "KVRequestState",
    "L1Snapshot",
    "TierSnapshot",
    "TieredKVCache",
]

_TIER_ORDER = ("L1", "L2", "L3")


class CacheCapacityError(ValueError):
    """当前物理容量不足，调用方应等待或回撤请求。"""


@dataclass(frozen=True, slots=True)
class CacheEntry:
    chunk_id: str
    tier: str
    ref_count: int
    last_access_seq: int
    pool_id: int = 0


@dataclass(frozen=True, slots=True)
class KVRequestState:
    k: int
    f: int


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


@dataclass(slots=True)
class CachePool:
    """一个物理资源池；完整页、容量占位与保护状态均归该池所有。"""

    level: str
    pool_id: int
    capacity: int
    page_size: int
    entries: dict[str, CacheEntry] = field(default_factory=dict, init=False)
    resident: int = field(default=0, init=False)
    reserved: int = field(default=0, init=False)
    evictable: int = field(default=0, init=False)
    peak: int = field(default=0, init=False)
    evictions: int = field(default=0, init=False)
    access_seq: int = field(default=0, init=False)
    heap: list[tuple[int, str]] = field(default_factory=list, init=False)
    queued: set[tuple[int, str]] = field(default_factory=set, init=False)
    pending_pages: dict[str, int] = field(default_factory=dict, init=False)

    def lookup(self, chunk_id: str) -> CacheEntry | None:
        return self.entries.get(chunk_id)

    def snapshot(self) -> TierSnapshot:
        return TierSnapshot(
            self.capacity,
            self.resident,
            self.resident - self.evictable,
            self.evictable,
            self.capacity - self.resident,
            self.peak,
            self.evictions,
        )

    def reserve(self, count: int) -> None:
        if count > self.capacity - self.resident:
            raise CacheCapacityError(f"{self.level} 容量不足")
        self.resident += count
        self.reserved += count
        self.peak = max(self.peak, self.resident)

    def release_reservation(self, count: int) -> None:
        self.reserved -= count
        self.resident -= count

    def reserve_page(self, chunk_id: str) -> None:
        if self.lookup(chunk_id) is not None:
            self.add_ref(chunk_id)
        elif chunk_id not in self.pending_pages:
            self.reserve(1)
        self.pending_pages[chunk_id] = self.pending_pages.get(chunk_id, 0) + 1

    def finish_page(self, chunk_id: str) -> None:
        remaining = self.pending_pages[chunk_id] - 1
        if remaining:
            self.pending_pages[chunk_id] = remaining
        else:
            del self.pending_pages[chunk_id]
        self.drop_ref(chunk_id)

    def store(self, chunk_id: str, ref_count: int = 0, *, occupied: bool = False) -> bool:
        """合并副本和引用；在途同 ID 只占一页，首次完成后仍保护其他传输。"""
        existing = self.lookup(chunk_id)
        if existing is not None:
            if occupied:
                self.resident -= 1
            self.add_refs(chunk_id, ref_count)
            self.touch(chunk_id)
            return False
        pending = self.pending_pages.get(chunk_id, 0)
        if pending:
            if occupied:
                self.release_reservation(1)
            else:
                self.reserved -= 1
                occupied = True
        self.insert(chunk_id, ref_count + pending, occupied=occupied)
        return True

    def insert(self, chunk_id: str, ref_count: int = 0, *, occupied: bool = False) -> None:
        if chunk_id in self.entries:
            raise ValueError("完整页已存在，重复写入须使用 store 合并")
        if not occupied:
            self.reserve(1)
            self.reserved -= 1
        self.access_seq += 1
        page = CacheEntry(chunk_id, self.level, ref_count, self.access_seq, pool_id=self.pool_id)
        self.entries[chunk_id] = page
        if not ref_count:
            self.evictable += 1
            self._queue(page)

    def remove(self, chunk_id: str, *, evicted: bool = False) -> CacheEntry:
        page = self.entries.pop(chunk_id)
        self.resident -= 1
        if not page.ref_count:
            self.evictable -= 1
        self.evictions += int(evicted)
        return page

    def add_refs(self, chunk_id: str, count: int) -> None:
        if not count:
            return
        page = self.entries[chunk_id]
        if not page.ref_count:
            self.evictable -= 1
        self.entries[chunk_id] = replace(page, ref_count=page.ref_count + count)

    def add_ref(self, chunk_id: str, *, refresh: bool = False) -> None:
        self.add_refs(chunk_id, 1)
        if refresh:
            self.touch(chunk_id)

    def drop_ref(self, chunk_id: str) -> None:
        page = self.entries[chunk_id]
        page = replace(page, ref_count=page.ref_count - 1)
        self.entries[chunk_id] = page
        if not page.ref_count:
            self.evictable += 1
            self._queue(page)

    def touch(self, chunk_id: str) -> None:
        self.access_seq += 1
        page = replace(self.entries[chunk_id], last_access_seq=self.access_seq)
        self.entries[chunk_id] = page
        if not page.ref_count:
            self._queue(page)

    def _queue(self, page: CacheEntry) -> None:
        key = (page.last_access_seq, page.chunk_id)
        if key not in self.queued:
            heappush(self.heap, key)
            self.queued.add(key)

    def candidates(self) -> Iterator[tuple[int, str]]:
        """按稳定 LRU 顺序只读遍历堆，不复制整个索引或修改真实堆。"""
        frontier: list[tuple[int, str, int]] = []
        if self.heap:
            heappush(frontier, (*self.heap[0], 0))
        while frontier:
            seq, chunk_id, index = heappop(frontier)
            for child in (2 * index + 1, 2 * index + 2):
                if child < len(self.heap):
                    heappush(frontier, (*self.heap[child], child))
            yield seq, chunk_id


class TieredKVCache:
    """按逻辑页组织请求；本地 L1/L2 迁移，L3 与本地允许独立副本。"""

    def __init__(
        self,
        max_total_tokens: int,
        page_size: int,
        l2_capacity_tokens: int = 0,
        l3_capacity_tokens: int = 0,
        *,
        shared_l3: CachePool | None = None,
        pool_id_base: int = 0,
    ) -> None:
        if page_size < 1 or max_total_tokens < page_size or max_total_tokens % page_size:
            raise ValueError("max_total_tokens 必须至少一页且是 page_size 的整数倍")
        if (
            l2_capacity_tokens < 0
            or l3_capacity_tokens < 0
            or (l2_capacity_tokens % page_size or l3_capacity_tokens % page_size)
        ):
            raise ValueError("低层容量必须是 page_size 的非负整数倍")
        l3 = (
            shared_l3
            if shared_l3 is not None
            else CachePool(
                "L3",
                pool_id_base + 2,
                l3_capacity_tokens // page_size,
                page_size,
            )
        )
        if l3.level != "L3" or l3.page_size != page_size:
            raise ValueError("共享 L3 的层级和页大小必须与实例一致")
        if l3.capacity and not l2_capacity_tokens:
            raise ValueError("L3 启用时须启用 L2")
        self.page_size = page_size
        self.capacity_pages = max_total_tokens // page_size
        self.pools: dict[str, CachePool] = {
            "L1": CachePool("L1", pool_id_base, self.capacity_pages, page_size),
            "L2": CachePool("L2", pool_id_base + 1, l2_capacity_tokens // page_size, page_size),
            "L3": l3,
        }
        self._requests: dict[int, InferenceRequest] = {}
        self._states: dict[int, KVRequestState] = {}
        self._pins: dict[int, tuple[InferenceRequest, int]] = {}
        self._pin_sources: dict[int, dict[str, CachePool]] = {}
        self._transfers: dict[str, tuple[str, str]] = {}
        self._private_pages = 0
        self.promotions = 0
        self.demotions = 0

    @property
    def resident_pages(self) -> int:
        return self.pools["L1"].resident

    @property
    def free_pages(self) -> int:
        return self.capacity_pages - self.resident_pages

    @property
    def evictable_pages(self) -> int:
        return self.pools["L1"].evictable

    @property
    def active_pages(self) -> int:
        return self.resident_pages - self.evictable_pages

    @property
    def evictions(self) -> int:
        return self.pools["L1"].evictions

    @property
    def peak_resident_pages(self) -> int:
        return self.pools["L1"].peak

    def tier_snapshot(self, tier: str) -> TierSnapshot:
        return self.pools[tier].snapshot()

    @property
    def available_tokens(self) -> int:
        return (self.free_pages + self.evictable_pages) * self.page_size

    def snapshot(self) -> L1Snapshot:
        private_tokens = sum(state.k - state.f for state in self._states.values())
        return L1Snapshot(
            self.capacity_pages,
            self.resident_pages,
            self.active_pages,
            self.evictable_pages,
            self.free_pages,
            self._private_pages * self.page_size - private_tokens,
        )

    def _source_pool(self, chunk_id: str, *, include_l3: bool = True) -> CachePool | None:
        for tier in _TIER_ORDER if include_l3 else _TIER_ORDER[:2]:
            pool = self.pools[tier]
            if pool.lookup(chunk_id) is not None:
                return pool
        return None

    def page(self, chunk_id: str) -> CacheEntry | None:
        pool = self._source_pool(chunk_id)
        return pool.lookup(chunk_id) if pool is not None else None

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
        start_index: int = 0,
    ) -> tuple[tuple[str, str], ...]:
        """按页选择最近来源；首个所有可访问池均缺失的页结束连续匹配。"""
        if target_tokens < request.prompt_len:
            raise ValueError("重算目标不得小于原始 prompt")
        limit = (
            target_tokens // self.page_size
            if restoring
            else (request.prompt_len - 1) // self.page_size
        )
        sources: list[tuple[str, str]] = []
        for index in range(start_index, limit):
            chunk_id = self._chunk_id(request, index)
            pool = self._source_pool(chunk_id, include_l3=include_l3)
            if pool is None:
                break
            sources.append((chunk_id, pool.level))
        return tuple(sources)

    def pin(self, request: InferenceRequest, matched_tokens: int) -> None:
        if request.request_id in self._pins or request.request_id in self._states:
            raise ValueError("请求已被保护或驻留")
        if matched_tokens < 0 or matched_tokens % self.page_size:
            raise ValueError("命中前缀必须是非负整页长度")
        bindings: dict[str, CachePool] = {}
        for index in range(matched_tokens // self.page_size):
            chunk_id = self._chunk_id(request, index)
            pool = self._source_pool(chunk_id)
            if pool is None:
                raise ValueError("命中前缀包含未发布页")
            bindings[chunk_id] = pool
        for chunk_id, pool in bindings.items():
            pool.add_ref(chunk_id)
        self._pins[request.request_id] = (request, matched_tokens)
        self._pin_sources[request.request_id] = bindings

    def cancel_pin(self, request_id: int) -> None:
        self._pins.pop(request_id)
        for chunk_id, pool in self._pin_sources.pop(request_id).items():
            pool.drop_ref(chunk_id)

    def admit(self, request_id: int) -> KVRequestState:
        request, matched_tokens = self._pins[request_id]
        bindings = self._pin_sources[request_id]
        l1 = self.pools["L1"]
        for chunk_id, pool in bindings.items():
            if pool is not l1 and l1.lookup(chunk_id) is not None:
                pool.drop_ref(chunk_id)
                l1.add_ref(chunk_id)
                bindings[chunk_id] = l1
        if any(pool.level != "L1" for pool in bindings.values()):
            raise CacheCapacityError("低层命中页须先提升到 L1 才能接纳请求")
        del self._pins[request_id], self._pin_sources[request_id]
        for chunk_id in bindings:
            self.pools["L1"].touch(chunk_id)
        self._requests[request_id] = request
        state = KVRequestState(matched_tokens, matched_tokens)
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
        self.pools["L1"].reserve(new_pages)
        self.pools["L1"].reserved -= new_pages
        self._private_pages += new_pages
        self._states[request_id] = KVRequestState(target_tokens, state.f)
        return new_pages

    def publish(self, request_id: int) -> int:
        state = self.state(request_id)
        request = self._requests[request_id]
        l1, l2 = self.pools["L1"], self.pools["L2"]
        ids = tuple(
            self._chunk_id(request, i)
            for i in range(state.f // self.page_size, state.k // self.page_size)
        )
        if any((page := l2.lookup(chunk_id)) is not None and page.ref_count for chunk_id in ids):
            raise CacheCapacityError("正在被保护的低层页不能被重算替换")
        freed = 0
        for chunk_id in ids:
            self._private_pages -= 1
            if l2.lookup(chunk_id) is not None:
                l2.remove(chunk_id, evicted=True)
            freed += int(not l1.store(chunk_id, 1, occupied=True))
        self._states[request_id] = KVRequestState(
            state.k, state.k // self.page_size * self.page_size
        )
        return freed

    def publish_batch(self, request_ids: Sequence[int]) -> int:
        return sum(self.publish(request_id) for request_id in sorted(request_ids))

    def finish(self, request_id: int) -> None:
        self.publish(request_id)
        self._release(request_id)

    def withdraw(self, request_id: int) -> None:
        self._release(request_id)

    def _release(self, request_id: int) -> None:
        state = self.state(request_id)
        request = self._requests.pop(request_id)
        for index in range(state.f // self.page_size):
            self.pools["L1"].drop_ref(self._chunk_id(request, index))
        count = self.private_page_count(request_id)
        self._private_pages -= count
        self.pools["L1"].resident -= count
        del self._states[request_id]

    def _migration_plan(
        self,
        promote_ids: Iterable[str],
        reserve_pages: int,
        *,
        reserve_tier: str = "L1",
        protected: tuple[str, str] | None = None,
    ) -> tuple[list[tuple[str, str, str | None]], tuple[tuple[str, str], ...]]:
        """按资源池分别试算增量；共享副本不会借用来源容量。"""
        if reserve_pages < 0:
            raise ValueError("所需页数不得为负")
        needed = tuple(sorted(set(promote_ids)))
        for chunk_id in needed:
            page = self.page(chunk_id)
            if page is None:
                raise ValueError("待提升页不存在")
            if chunk_id in self._transfers:
                raise ValueError("在途页不能同步提升")
            if page.tier != "L1" and not page.ref_count:
                raise ValueError("待提升的低层页必须先 pin")
        counts = {tier: pool.resident for tier, pool in self.pools.items()}
        seqs = {tier: pool.access_seq for tier, pool in self.pools.items()}
        changed: dict[tuple[str, str], CacheEntry | None] = {}
        actions: list[tuple[str, str, str | None]] = []
        promoted: list[tuple[str, str]] = []
        candidates = {tier: pool.candidates() for tier, pool in self.pools.items()}
        pending: dict[str, tuple[int, str] | None] = dict.fromkeys(self.pools)
        added: dict[str, list[tuple[int, str]]] = {tier: [] for tier in self.pools}

        def lookup(tier: str, chunk_id: str) -> CacheEntry | None:
            return changed.get((tier, chunk_id), self.pools[tier].lookup(chunk_id))

        def next_victim(tier: str) -> CacheEntry:
            while True:
                if pending[tier] is None:
                    pending[tier] = next(candidates[tier], None)
                original = pending[tier]
                if original is not None and (not added[tier] or original <= added[tier][0]):
                    access_seq, chunk_id = original
                    pending[tier] = None
                elif added[tier]:
                    access_seq, chunk_id = heappop(added[tier])
                else:
                    raise CacheCapacityError(f"{tier} 容量不足且没有可淘汰页")
                page = lookup(tier, chunk_id)
                if (
                    page is not None
                    and not page.ref_count
                    and page.last_access_seq == access_seq
                    and (tier, chunk_id) != protected
                ):
                    return page

        def place(chunk_id: str, destination: str, refs: int = 0) -> None:
            existing = lookup(destination, chunk_id)
            if existing is not None:
                seqs[destination] += 1
                page = replace(existing, ref_count=existing.ref_count + refs,
                               last_access_seq=seqs[destination])
                changed[destination, chunk_id] = page
                if not page.ref_count:
                    heappush(added[destination], (page.last_access_seq, chunk_id))
                return
            pending_refs = self.pools[destination].pending_pages.get(chunk_id, 0)
            if not pending_refs:
                free_slot(destination)
                counts[destination] += 1
            seqs[destination] += 1
            page = CacheEntry(
                chunk_id,
                destination,
                refs + pending_refs,
                seqs[destination],
                pool_id=self.pools[destination].pool_id,
            )
            changed[destination, chunk_id] = page
            if not page.ref_count:
                heappush(added[destination], (page.last_access_seq, chunk_id))

        def evict_one(tier: str) -> None:
            victim = next_victim(tier)
            lower = {"L1": "L2", "L2": "L3", "L3": None}[tier]
            destination = lower if lower is not None and self.pools[lower].capacity else None
            changed[tier, victim.chunk_id] = None
            counts[tier] -= 1
            if destination is not None:
                place(victim.chunk_id, destination)
            actions.append((victim.chunk_id, tier, destination))

        def free_slot(tier: str) -> None:
            while counts[tier] >= self.pools[tier].capacity:
                evict_one(tier)

        for chunk_id in needed:
            source = next(
                (tier for tier in _TIER_ORDER if lookup(tier, chunk_id) is not None), None
            )
            if source is None or source == "L1":
                continue
            page = lookup(source, chunk_id)
            if source != "L3":
                counts[source] -= 1
                changed[source, chunk_id] = None
            place(chunk_id, "L1", page.ref_count if page is not None else 0)
            actions.append((chunk_id, source, "L1"))
            promoted.append((chunk_id, source))
        while self.pools[reserve_tier].capacity - counts[reserve_tier] < reserve_pages:
            evict_one(reserve_tier)
        return actions, tuple(promoted)

    def _apply_move(self, chunk_id: str, source: str, destination: str | None) -> None:
        old = self.pools[source]
        page = old.entries[chunk_id]
        upward = destination is not None and _TIER_ORDER.index(destination) < _TIER_ORDER.index(
            source
        )
        if source != "L3" or destination is None:
            old.remove(chunk_id, evicted=not upward)
        if destination is None:
            return
        target = self.pools[destination]
        target.store(chunk_id, page.ref_count if source != "L3" else 0)
        for bindings in self._pin_sources.values():
            if bindings.get(chunk_id) is old:
                if source == "L3":
                    old.drop_ref(chunk_id)
                    target.add_ref(chunk_id)
                bindings[chunk_id] = target
        if upward:
            self.promotions += 1
        else:
            self.demotions += 1

    def plan(
        self, promote_ids: Iterable[str] = (), reserve_pages: int = 0
    ) -> tuple[tuple[str, str], ...]:
        return self._migration_plan(promote_ids, reserve_pages)[1]

    def reserve_transfer(self, chunk_id: str, destination: str) -> str:
        page = self.page(chunk_id)
        if page is None:
            raise ValueError("待传输页不存在")
        if chunk_id in self._transfers:
            raise ValueError("同一页已有在途传输")
        if destination not in self.pools or not self.pools[destination].capacity:
            raise ValueError("目的层不存在或未启用")
        source = page.tier
        if source == destination:
            raise ValueError("来源与目的层不能相同")
        if _TIER_ORDER.index(destination) > _TIER_ORDER.index(source) and page.ref_count:
            raise CacheCapacityError("有活动引用的页不能降级")
        target = self.pools[destination]
        duplicate = target.lookup(chunk_id) is not None or chunk_id in target.pending_pages
        actions, _ = self._migration_plan(
            (), int(not duplicate), reserve_tier=destination, protected=(source, chunk_id)
        )
        for victim, source_tier, destination_tier in actions:
            self._apply_move(victim, source_tier, destination_tier)
        self.pools[source].add_ref(chunk_id)
        target.reserve_page(chunk_id)
        self._transfers[chunk_id] = (source, destination)
        return source

    def complete_transfer(self, chunk_id: str) -> str:
        source, destination = self._transfers[chunk_id]
        old, target = self.pools[source], self.pools[destination]
        page = old.entries[chunk_id]
        if _TIER_ORDER.index(destination) > _TIER_ORDER.index(source) and page.ref_count > 1:
            raise CacheCapacityError("在途降级页获得活动引用，须先释放活动引用")
        if destination == "L3":
            old.drop_ref(chunk_id)
        self._apply_move(chunk_id, source, destination)
        if source == "L3":
            old.drop_ref(chunk_id)
            old.touch(chunk_id)
        elif destination != "L3":
            target.drop_ref(chunk_id)
        target.finish_page(chunk_id)
        del self._transfers[chunk_id]
        return source

    def ensure_free(self, count: int) -> int:
        if count <= self.free_pages:
            return 0
        evictions_before = self.evictions
        actions, _ = self._migration_plan((), count)
        for chunk_id, source, destination in actions:
            self._apply_move(chunk_id, source, destination)
        return self.evictions - evictions_before

    @staticmethod
    def _chunk_id(request: InferenceRequest, index: int) -> str:
        if request.is_trace and index < len(request.trace_hash_ids):
            return f"trace:{request.session_id}:{request.trace_hash_ids[index]}"
        return logical_chunk_id(request.request_id, index, request.inherited_prefix_ids)
