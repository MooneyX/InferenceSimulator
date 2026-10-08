"""单实例 L1/L2/L3 KV 元数据管理：共享完整页、保护引用与稳定 LRU。"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from heapq import heappush

from simcore.prefix_hash import logical_chunk_id
from simcore.schemas import InferenceRequest

__all__ = [
    "CacheCapacityError", "CacheEntry", "KVRequestState", "L1KVCache", "L1Page",
    "L1Snapshot", "TierSnapshot", "TieredKVCache",
]


class CacheCapacityError(ValueError):
    """当前物理容量不足，调用方应等待或回撤请求。"""


@dataclass(frozen=True, slots=True)
class CacheEntry:
    chunk_id: str
    tier: str
    ref_count: int
    last_access_seq: int
    capacity_pages: int = 1


L1Page = CacheEntry


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
class _TierState:
    capacity: int
    resident: int = 0
    evictable: int = 0
    peak: int = 0
    evictions: int = 0
    ids: set[str] = field(default_factory=set)
    heap: list[tuple[int, str]] = field(default_factory=list)
    queued: set[tuple[int, str]] = field(default_factory=set)


class TieredKVCache:
    """每页只在全局索引中出现一次；各层只管理身份、计数和 LRU。"""

    def __init__(
        self, max_total_tokens: int, page_size: int,
        l2_capacity_tokens: int = 0, l3_capacity_tokens: int = 0,
    ) -> None:
        if page_size < 1 or max_total_tokens < page_size or max_total_tokens % page_size:
            raise ValueError("max_total_tokens 必须至少一页且是 page_size 的整数倍")
        if (l2_capacity_tokens < 0 or l3_capacity_tokens < 0
                or l2_capacity_tokens % page_size or l3_capacity_tokens % page_size
                or (l3_capacity_tokens and not l2_capacity_tokens)):
            raise ValueError("低层容量必须是 page_size 的非负整数倍，L3 启用时须启用 L2")
        self.page_size = page_size
        self.capacity_pages = max_total_tokens // page_size
        self._tiers = {
            "L1": _TierState(self.capacity_pages),
            "L2": _TierState(l2_capacity_tokens // page_size),
            "L3": _TierState(l3_capacity_tokens // page_size),
        }
        self._entries: dict[str, CacheEntry] = {}
        self._pages = self._entries  # 兼容旧测试对完整页索引的只读检查
        self._requests: dict[int, InferenceRequest] = {}
        self._states: dict[int, KVRequestState] = {}
        self._pins: dict[int, tuple[InferenceRequest, int]] = {}
        self._evictable_heap = self._tiers["L1"].heap
        self._queued_evictions = self._tiers["L1"].queued
        self._private_pages = 0
        self._access_seq = 0
        self.promotions = 0
        self.demotions = 0
        self.deletions = 0
        self.l2_promotions = 0
        self.l3_promotions = 0
        self.l1_to_l2_demotions = 0
        self.l2_to_l3_demotions = 0

    @property
    def resident_pages(self) -> int:
        return self._tiers["L1"].resident

    @property
    def free_pages(self) -> int:
        return self.capacity_pages - self.resident_pages

    @property
    def evictable_pages(self) -> int:
        return self._tiers["L1"].evictable

    @property
    def active_pages(self) -> int:
        return self.resident_pages - self.evictable_pages

    @property
    def evictions(self) -> int:
        return self._tiers["L1"].evictions

    @property
    def peak_resident_pages(self) -> int:
        return self._tiers["L1"].peak

    def tier_snapshot(self, tier: str) -> TierSnapshot:
        state = self._tiers[tier]
        return TierSnapshot(
            state.capacity, state.resident, state.resident - state.evictable,
            state.evictable, state.capacity - state.resident, state.peak, state.evictions,
        )

    def tier_snapshots(self) -> dict[str, TierSnapshot]:
        return {tier: self.tier_snapshot(tier) for tier in self._tiers}

    @property
    def private_pages(self) -> int:
        return self._private_pages

    @property
    def available_tokens(self) -> int:
        return (self.free_pages + self.evictable_pages) * self.page_size

    def snapshot(self) -> L1Snapshot:
        """只汇总计数与活动请求，不复制或排序驻留完整页。"""
        private_tokens = sum(state.k - state.f for state in self._states.values())
        return L1Snapshot(
            capacity_blocks=self.capacity_pages,
            resident_blocks=self.resident_pages,
            active_blocks=self.active_pages,
            evictable_blocks=self.evictable_pages,
            free_blocks=self.free_pages,
            internal_fragmentation_tokens=self._private_pages * self.page_size - private_tokens,
        )

    def page(self, chunk_id: str) -> L1Page | None:
        """只读查找不刷新 LRU。"""
        return self._pages.get(chunk_id)

    def state(self, request_id: int) -> KVRequestState:
        return self._states[request_id]

    def private_page_count(self, request_id: int) -> int:
        state = self.state(request_id)
        return (state.k + self.page_size - 1) // self.page_size - state.f // self.page_size

    def probe(
        self, request: InferenceRequest, target_tokens: int, *, restoring: bool = False
    ) -> int:
        """逐页查连续已发布前缀；只读查找不会刷新 LRU。"""
        if target_tokens < request.prompt_len:
            raise ValueError("重算目标不得小于原始 prompt")
        limit = (
            target_tokens // self.page_size
            if restoring
            else (request.prompt_len - 1) // self.page_size
        )
        for index in range(limit):
            if self._chunk_id(request, index) not in self._entries:
                return index * self.page_size
        return limit * self.page_size

    def probe_sources(
        self, request: InferenceRequest, matched_tokens: int
    ) -> tuple[tuple[str, str], ...]:
        """返回当前命中前缀的逐页来源；应在 pin 及迁移之前调用。"""
        if matched_tokens < 0 or matched_tokens % self.page_size:
            raise ValueError("命中前缀必须是非负整页长度")
        sources = []
        for index in range(matched_tokens // self.page_size):
            chunk_id = self._chunk_id(request, index)
            if chunk_id not in self._entries:
                raise ValueError("命中前缀包含未发布页")
            sources.append((chunk_id, self._entries[chunk_id].tier))
        return tuple(sources)

    def pin(self, request: InferenceRequest, matched_tokens: int) -> None:
        """临时保护候选命中页；失败撤销不改变访问序号。"""
        if request.request_id in self._pins or request.request_id in self._states:
            raise ValueError("请求已被保护或驻留")
        if matched_tokens < 0 or matched_tokens % self.page_size:
            raise ValueError("命中前缀必须是非负整页长度")
        ids = tuple(
            self._chunk_id(request, index) for index in range(matched_tokens // self.page_size)
        )
        if any(chunk_id not in self._pages for chunk_id in ids):
            raise ValueError("命中前缀包含未发布页")
        for chunk_id in ids:
            self._add_ref(chunk_id, refresh=False)
        self._pins[request.request_id] = (request, matched_tokens)

    def cancel_pin(self, request_id: int) -> None:
        request, matched_tokens = self._pins.pop(request_id)
        for index in range(matched_tokens // self.page_size):
            self._drop_ref(self._chunk_id(request, index))

    def admit(self, request_id: int) -> KVRequestState:
        """确认候选并刷新已获得的引用；低层引用必须先提升到 L1。"""
        request, matched_tokens = self._pins[request_id]
        if any(
            self._entries[self._chunk_id(request, index)].tier != "L1"
            for index in range(matched_tokens // self.page_size)
        ):
            raise CacheCapacityError("低层命中页须先提升到 L1 才能接纳请求")
        del self._pins[request_id]
        for index in range(matched_tokens // self.page_size):
            self._touch(self._chunk_id(request, index))
        self._requests[request_id] = request
        state = KVRequestState(matched_tokens, matched_tokens)
        self._states[request_id] = state
        return state

    def extend(self, request_id: int, target_tokens: int) -> int:
        """申请真正新增的页并将 K 推进到目标；未来输出不占容量。"""
        state = self.state(request_id)
        if target_tokens <= state.k:
            raise ValueError("KV 目标长度必须大于当前 K")
        new_pages = (target_tokens + self.page_size - 1) // self.page_size - (
            state.k + self.page_size - 1
        ) // self.page_size
        self.ensure_free(new_pages)
        self._private_pages += new_pages
        l1 = self._tiers["L1"]
        l1.resident += new_pages
        l1.peak = max(l1.peak, l1.resident)
        self._states[request_id] = KVRequestState(target_tokens, state.f)
        return new_pages

    def publish(self, request_id: int) -> int:
        """按逻辑顺序发布完整私有页；低层同 ID 重算保留新算的 L1 页。"""
        state = self.state(request_id)
        request = self._requests[request_id]
        for index in range(state.f // self.page_size, state.k // self.page_size):
            existing = self._entries.get(self._chunk_id(request, index))
            if existing is not None and existing.tier != "L1" and existing.ref_count:
                raise CacheCapacityError("正在被保护的低层页不能被重算替换")
        freed = 0
        for index in range(state.f // self.page_size, state.k // self.page_size):
            chunk_id = self._chunk_id(request, index)
            existing = self._entries.get(chunk_id)
            self._private_pages -= 1
            if existing is None:
                self._access_seq += 1
                self._entries[chunk_id] = CacheEntry(chunk_id, "L1", 1, self._access_seq)
                self._tiers["L1"].ids.add(chunk_id)
            elif existing.tier == "L1":
                self._add_ref(chunk_id, refresh=True)
                self._tiers["L1"].resident -= 1
                freed += 1
            else:
                lower = self._tiers[existing.tier]
                lower.ids.remove(chunk_id)
                lower.resident -= 1
                lower.evictable -= 1
                lower.evictions += 1
                self.deletions += 1
                self._access_seq += 1
                self._entries[chunk_id] = CacheEntry(chunk_id, "L1", 1, self._access_seq)
                self._tiers["L1"].ids.add(chunk_id)
        self._states[request_id] = KVRequestState(
            state.k, state.k // self.page_size * self.page_size
        )
        return freed

    def publish_batch(self, request_ids: Sequence[int]) -> int:
        """结果处理按请求 ID 稳定顺序发布同批私有页。"""
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("同批请求 ID 不得重复")
        for request_id in request_ids:
            self.state(request_id)
        return sum(self.publish(request_id) for request_id in sorted(request_ids))

    def finish(self, request_id: int) -> None:
        """完成时先发布剩余完整页，再按逻辑位置逐页解除全部共享引用。"""
        self.publish(request_id)
        self._release(request_id)

    def withdraw(self, request_id: int) -> None:
        """回撤时丢弃未发布完整页和尾页，已发布共享页保留为缓存。"""
        self._release(request_id)

    def _release(self, request_id: int) -> None:
        state = self.state(request_id)
        request = self._requests.pop(request_id)
        for index in range(state.f // self.page_size):
            self._drop_ref(self._chunk_id(request, index))
        count = self.private_page_count(request_id)
        self._private_pages -= count
        self._tiers["L1"].resident -= count
        del self._states[request_id]

    def _add_ref(self, chunk_id: str, *, refresh: bool) -> None:
        page = self._entries[chunk_id]
        if page.ref_count == 0:
            self._tiers[page.tier].evictable -= 1
        self._entries[chunk_id] = replace(page, ref_count=page.ref_count + 1)
        if refresh:
            self._touch(chunk_id)

    def _drop_ref(self, chunk_id: str) -> None:
        page = self._entries[chunk_id]
        self._entries[chunk_id] = replace(page, ref_count=page.ref_count - 1)
        if page.ref_count == 1:
            tier = self._tiers[page.tier]
            tier.evictable += 1
            key = (page.last_access_seq, chunk_id)
            if key not in tier.queued:
                heappush(tier.heap, key)
                tier.queued.add(key)

    def _touch(self, chunk_id: str) -> None:
        self._access_seq += 1
        self._entries[chunk_id] = replace(
            self._entries[chunk_id], last_access_seq=self._access_seq
        )

    def _migration_plan(
        self, promote_ids: Iterable[str], reserve_pages: int,
    ) -> tuple[
        dict[str, CacheEntry], list[tuple[str, str, str | None]],
        tuple[tuple[str, str], ...], int,
    ]:
        """在局部副本上模拟完整批次，不碰全局项、LRU 堆及计数。"""
        if reserve_pages < 0:
            raise ValueError("所需页数不得为负")
        needed = tuple(sorted(set(promote_ids)))
        for chunk_id in needed:
            page = self._entries.get(chunk_id)
            if page is None:
                raise ValueError("待提升页不存在")
            if page.tier != "L1" and not page.ref_count:
                raise ValueError("待提升的低层页必须先 pin")
        if reserve_pages <= self.free_pages and all(
            self._entries[chunk_id].tier == "L1" for chunk_id in needed
        ):
            return self._entries, [], (), self._access_seq
        entries = dict(self._entries)
        ids = {tier: set(state.ids) for tier, state in self._tiers.items()}
        seq = self._access_seq
        private = self._private_pages
        actions: list[tuple[str, str, str | None]] = []
        promoted: list[tuple[str, str]] = []

        def evict_one(tier: str) -> None:
            nonlocal seq
            victims = (entries[key] for key in ids[tier] if not entries[key].ref_count)
            victim = min(
                victims, key=lambda page: (page.last_access_seq, page.chunk_id),
                default=None,
            )
            if victim is None:
                raise CacheCapacityError(f"{tier} 容量不足且没有可淘汰页")
            lower = {"L1": "L2", "L2": "L3", "L3": None}[tier]
            if lower is not None and self._tiers[lower].capacity:
                free_slot(lower)
            ids[tier].remove(victim.chunk_id)
            if lower is None or not self._tiers[lower].capacity:
                del entries[victim.chunk_id]
            else:
                seq += 1
                entries[victim.chunk_id] = replace(victim, tier=lower, last_access_seq=seq)
                ids[lower].add(victim.chunk_id)
            destination = lower if lower and self._tiers[lower].capacity else None
            actions.append((victim.chunk_id, tier, destination))

        def free_slot(tier: str) -> None:
            if len(ids[tier]) + (private if tier == "L1" else 0) >= self._tiers[tier].capacity:
                evict_one(tier)

        for chunk_id in needed:
            page = entries[chunk_id]
            if page.tier == "L1":
                continue
            source = page.tier
            ids[source].remove(chunk_id)
            free_slot("L1")
            seq += 1
            entries[chunk_id] = replace(page, tier="L1", last_access_seq=seq)
            ids["L1"].add(chunk_id)
            promoted.append((chunk_id, source))
        while self._tiers["L1"].capacity - len(ids["L1"]) - private < reserve_pages:
            evict_one("L1")
        return entries, actions, tuple(promoted), seq

    def plan(
        self, promote_ids: Iterable[str] = (), reserve_pages: int = 0,
    ) -> tuple[tuple[str, str], ...]:
        """预检提升和新增私有页所需容量；失败无任何状态副作用。"""
        return self._migration_plan(promote_ids, reserve_pages)[2]

    def promote(
        self, promote_ids: Iterable[str] = (), reserve_pages: int = 0,
    ) -> tuple[tuple[str, str], ...]:
        """全批次容量预检通过后一次性提交提升、降级及末层删除。"""
        entries, actions, promoted, seq = self._migration_plan(promote_ids, reserve_pages)
        if not actions and not promoted:
            return promoted
        self._entries.clear()
        self._entries.update(entries)
        self._access_seq = seq
        for _, source, dest in actions:
            self._tiers[source].evictions += 1
            if dest is None:
                self.deletions += 1
            else:
                self.demotions += 1
                if source == "L1":
                    self.l1_to_l2_demotions += 1
                else:
                    self.l2_to_l3_demotions += 1
        for _, source in promoted:
            self.promotions += 1
            if source == "L2":
                self.l2_promotions += 1
            else:
                self.l3_promotions += 1
        for tier, state in self._tiers.items():
            state.ids.clear()
            state.heap.clear()
            state.queued.clear()
            state.resident = self._private_pages if tier == "L1" else 0
            state.evictable = 0
        for chunk_id, page in self._entries.items():
            state = self._tiers[page.tier]
            state.ids.add(chunk_id)
            state.resident += 1
            if not page.ref_count:
                state.evictable += 1
                key = (page.last_access_seq, chunk_id)
                heappush(state.heap, key)
                state.queued.add(key)
        for state in self._tiers.values():
            state.peak = max(state.peak, state.resident)
        return promoted

    def prepare_batch(
        self, promote_ids: Iterable[str] = (), reserve_pages: int = 0,
    ) -> tuple[tuple[str, str], ...]:
        """与 promote 等价，供引擎原子提交本批容量及恢复页。"""
        return self.promote(promote_ids, reserve_pages)

    def ensure_free(self, count: int) -> int:
        """原子腾出 L1 实际页；容量不足不执行逐级迁移。"""
        if count < 0:
            raise ValueError("所需页数不得为负")
        if count <= self.free_pages:
            return 0
        evictions_before = self.evictions
        self.promote(reserve_pages=count)
        return self.evictions - evictions_before

    @staticmethod
    def _chunk_id(request: InferenceRequest, index: int) -> str:
        return logical_chunk_id(request.request_id, index, request.inherited_prefix_ids)


class L1KVCache(TieredKVCache):
    """零低层容量兼容入口；不维护独立缓存状态。"""

    def __init__(self, max_total_tokens: int, page_size: int) -> None:
        super().__init__(max_total_tokens, page_size)
