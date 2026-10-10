"""单实例 L1/L2/L3 KV 元数据管理：共享完整页、保护引用与稳定 LRU。"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush

from simcore.prefix_hash import logical_chunk_id
from simcore.schemas import InferenceRequest

__all__ = [
    "CacheCapacityError",
    "CacheEntry",
    "KVRequestState",
    "L1KVCache",
    "L1Page",
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
    reserved: int = 0
    evictable: int = 0
    peak: int = 0
    evictions: int = 0
    ids: set[str] = field(default_factory=set)
    heap: list[tuple[int, str]] = field(default_factory=list)
    queued: set[tuple[int, str]] = field(default_factory=set)


class TieredKVCache:
    """每页只在全局索引中出现一次；各层只管理身份、计数和 LRU。"""

    def __init__(
        self,
        max_total_tokens: int,
        page_size: int,
        l2_capacity_tokens: int = 0,
        l3_capacity_tokens: int = 0,
    ) -> None:  # L1/L2/L3 token 容量与页大小 → 初始化三级页池；无返回值
        if page_size < 1 or max_total_tokens < page_size or max_total_tokens % page_size:
            raise ValueError("max_total_tokens 必须至少一页且是 page_size 的整数倍")
        if (
            l2_capacity_tokens < 0
            or l3_capacity_tokens < 0
            or l2_capacity_tokens % page_size
            or l3_capacity_tokens % page_size
            or (l3_capacity_tokens and not l2_capacity_tokens)
        ):
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
        self._transfers: dict[str, str] = {}
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
    def resident_pages(self) -> int:  # L1 计数 → 当前驻留页数
        return self._tiers["L1"].resident

    @property
    def free_pages(self) -> int:  # L1 容量与驻留数 → 可用空页数
        return self.capacity_pages - self.resident_pages

    @property
    def evictable_pages(self) -> int:  # L1 计数 → 无活动引用的可淘汰页数
        return self._tiers["L1"].evictable

    @property
    def active_pages(self) -> int:  # L1 驻留数与可淘汰数 → 活动页数
        return self.resident_pages - self.evictable_pages

    @property
    def evictions(self) -> int:  # L1 计数 → 累计驱逐次数
        return self._tiers["L1"].evictions

    @property
    def peak_resident_pages(self) -> int:  # L1 计数 → 历史驻留页峰值
        return self._tiers["L1"].peak

    def tier_snapshot(self, tier: str) -> TierSnapshot:  # 层名 → 容量/驱逐快照；在途目的层计入预留占位
        state = self._tiers[tier]
        return TierSnapshot(
            state.capacity,
            state.resident,
            state.resident - state.evictable,
            state.evictable,
            state.capacity - state.resident,
            state.peak,
            state.evictions,
        )

    def tier_snapshots(self) -> dict[str, TierSnapshot]:  # 各层计数 → 层名到容量快照的映射
        return {tier: self.tier_snapshot(tier) for tier in self._tiers}

    @property
    def private_pages(self) -> int:  # 私有页计数 → 未发布页数
        return self._private_pages

    @property
    def available_tokens(self) -> int:  # 空页与可淘汰页 → 可供新请求使用的 L1 token 数
        return (self.free_pages + self.evictable_pages) * self.page_size

    def snapshot(self) -> L1Snapshot:  # L1 页计数与请求状态 → 容量及内部碎片快照
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

    def page(self, chunk_id: str) -> L1Page | None:  # 完整页 ID → 驻留页或 None，不刷新 LRU
        """只读查找不刷新 LRU。"""
        return self._pages.get(chunk_id)

    def state(self, request_id: int) -> KVRequestState:  # 请求 ID → 已写入 K 与已发布前缀 F 状态
        return self._states[request_id]

    def private_page_count(self, request_id: int) -> int:  # 请求 ID → 未发布部分占用的物理页数
        state = self.state(request_id)
        return (state.k + self.page_size - 1) // self.page_size - state.f // self.page_size

    def probe(
        self, request: InferenceRequest, target_tokens: int, *, restoring: bool = False
    ) -> tuple[tuple[str, str], ...]:  # 请求、目标长度及是否恢复 → 连续命中页的 ID/层级元组
        """逐页返回连续已发布前缀的查询时来源；不刷新 LRU。"""
        if target_tokens < request.prompt_len:
            raise ValueError("重算目标不得小于原始 prompt")
        limit = (
            target_tokens // self.page_size
            if restoring
            else (request.prompt_len - 1) // self.page_size
        )
        sources: list[tuple[str, str]] = []
        for index in range(limit):
            chunk_id = self._chunk_id(request, index)
            entry = self._entries.get(chunk_id)
            if entry is None:
                break
            sources.append((chunk_id, entry.tier))
        return tuple(sources)

    def pin(self, request: InferenceRequest, matched_tokens: int) -> None:  # 请求及整页命中长度 → 暂时保护命中页；无返回值
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

    def cancel_pin(self, request_id: int) -> None:  # 请求 ID → 释放候选命中页的临时保护；无返回值
        request, matched_tokens = self._pins.pop(request_id)
        for index in range(matched_tokens // self.page_size):
            self._drop_ref(self._chunk_id(request, index))

    def admit(self, request_id: int) -> KVRequestState:  # 已保护的请求 ID → 驻留请求的初始 KV 状态
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

    def extend(self, request_id: int, target_tokens: int) -> int:  # 请求 ID 与目标 KV 长度 → 实际新申请的私有页数
        """申请真正新增的页并将 K 推进到目标；未来输出不占容量。"""
        state = self.state(request_id)  # k 是已写入量，f 是已发布的整页前缀
        if target_tokens <= state.k:
            raise ValueError("KV 目标长度必须大于当前 K")
        new_pages = (target_tokens + self.page_size - 1) // self.page_size - (
            state.k + self.page_size - 1
        ) // self.page_size  # 尾部私有页未写满时继续使用，不再申请新页
        self.ensure_free(new_pages)
        self._private_pages += new_pages
        l1 = self._tiers["L1"]
        l1.resident += new_pages
        l1.peak = max(l1.peak, l1.resident)
        self._states[request_id] = KVRequestState(target_tokens, state.f)
        return new_pages

    def publish(self, request_id: int) -> int:  # 请求 ID → 将完整私有页转为共享缓存，返回释放的重复页数
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
            elif existing.tier == "L1":  # 同 ID 私有页并入已有共享页，退还多占的一页
                self._add_ref(chunk_id, refresh=True)
                self._tiers["L1"].resident -= 1
                freed += 1
            else:  # 低层旧页被新计算出的 L1 页替换
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

    def publish_batch(self, request_ids: Sequence[int]) -> int:  # 同批请求 ID → 稳定顺序发布并返回释放的重复页总数
        """结果处理按请求 ID 稳定顺序发布同批私有页。"""
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("同批请求 ID 不得重复")
        for request_id in request_ids:
            self.state(request_id)
        return sum(self.publish(request_id) for request_id in sorted(request_ids))

    def finish(self, request_id: int) -> None:  # 完成请求 ID → 发布整页并归还活动引用；无返回值
        """完成时先发布剩余完整页，再按逻辑位置逐页解除全部共享引用。"""
        self.publish(request_id)
        self._release(request_id)

    def withdraw(self, request_id: int) -> None:  # 回撤请求 ID → 释放私有页并保留共享页；无返回值
        """回撤时丢弃未发布完整页和尾页，已发布共享页保留为缓存。"""
        self._release(request_id)

    def _release(self, request_id: int) -> None:  # 驻留请求 ID → 解除共享引用并归还私有页；无返回值
        state = self.state(request_id)
        request = self._requests.pop(request_id)
        for index in range(state.f // self.page_size):
            self._drop_ref(self._chunk_id(request, index))
        count = self.private_page_count(request_id)
        self._private_pages -= count
        self._tiers["L1"].resident -= count
        del self._states[request_id]

    def _add_ref(self, chunk_id: str, *, refresh: bool) -> None:  # 页 ID 与是否刷新 LRU → 增加保护引用；无返回值
        page = self._entries[chunk_id]
        if page.ref_count == 0:
            self._tiers[page.tier].evictable -= 1
        self._entries[chunk_id] = replace(page, ref_count=page.ref_count + 1)
        if refresh:
            self._touch(chunk_id)

    def _drop_ref(self, chunk_id: str) -> None:  # 页 ID → 释放引用，归零时加入可淘汰堆；无返回值
        page = self._entries[chunk_id]
        self._entries[chunk_id] = replace(page, ref_count=page.ref_count - 1)
        if page.ref_count == 1:
            tier = self._tiers[page.tier]
            tier.evictable += 1
            key = (page.last_access_seq, chunk_id)
            if key not in tier.queued:
                heappush(tier.heap, key)
                tier.queued.add(key)

    def _touch(self, chunk_id: str) -> None:  # 页 ID → 更新最近访问顺序；无返回值
        self._access_seq += 1
        self._entries[chunk_id] = replace(self._entries[chunk_id], last_access_seq=self._access_seq)

    def _migration_plan(
        self,
        promote_ids: Iterable[str],
        reserve_pages: int,
        *,
        reserve_tier: str = "L1",
        protected_id: str | None = None,
    ) -> tuple[list[tuple[str, str, str | None]], tuple[tuple[str, str], ...]]:  # 待提升页、保留页数、目标层及保护页 → 原子迁移动作/提升来源
        """仅试算受影响页：增量覆盖表和堆前沿，不复制驻留索引或修改真实 LRU。"""
        if reserve_pages < 0:
            raise ValueError("所需页数不得为负")
        needed = tuple(sorted(set(promote_ids)))
        for chunk_id in needed:
            page = self._entries.get(chunk_id)
            if page is None:
                raise ValueError("待提升页不存在")
            if chunk_id in self._transfers:
                raise ValueError("在途页不能同步提升")
            if page.tier != "L1" and not page.ref_count:
                raise ValueError("待提升的低层页必须先 pin")
        target = self._tiers[reserve_tier]
        if reserve_pages <= target.capacity - target.resident and all(
            self._entries[chunk_id].tier == "L1" for chunk_id in needed
        ):
            return [], ()

        counts = {tier: state.resident for tier, state in self._tiers.items()}  # 试算期间各层预计驻留页数
        changed: dict[str, CacheEntry | None] = {}  # 页 ID → 试算覆盖状态；None 表示将被删除
        seq = self._access_seq  # 仅在试算中推进的 LRU 序号
        actions: list[tuple[str, str, str | None]] = []  # 页 ID、源层、目的层的提交顺序
        promoted: list[tuple[str, str]] = []  # 提升页及其原层，供回载计费
        # 用二叉堆下标遍历原堆的最小候选；无论成功或失败，原堆均不被弹出。
        frontiers: dict[str, list[tuple[int, str, int]]] = {}
        added: dict[str, list[tuple[int, str]]] = {}

        def next_victim(tier: str) -> CacheEntry:  # 目标层 → 从原堆与试算新增堆选出最久未用且未保护的页
            state = self._tiers[tier]
            frontier = frontiers.setdefault(tier, [])
            local = added.setdefault(tier, [])
            if not frontier and tier not in traversed:
                traversed.add(tier)
                if state.heap:
                    key = state.heap[0]
                    heappush(frontier, (*key, 0))
            while frontier or local:
                if frontier and (not local or frontier[0][:2] <= local[0]):
                    access_seq, chunk_id, index = heappop(frontier)
                    for child in (2 * index + 1, 2 * index + 2):
                        if child < len(state.heap):
                            heappush(frontier, (*state.heap[child], child))
                else:
                    access_seq, chunk_id = heappop(local)
                page = changed.get(chunk_id, self._entries.get(chunk_id))
                if (
                    page is not None
                    and page.tier == tier
                    and not page.ref_count
                    and page.last_access_seq == access_seq
                    and chunk_id != protected_id
                ):
                    return page
            raise CacheCapacityError(f"{tier} 容量不足且没有可淘汰页")

        traversed: set[str] = set()

        def evict_one(tier: str) -> None:  # 满额层名 → 试算一次淘汰/降级动作；无返回值
            nonlocal seq
            victim = next_victim(tier)
            lower = {"L1": "L2", "L2": "L3", "L3": None}[tier]
            destination = lower if lower is not None and self._tiers[lower].capacity else None
            if destination is not None:
                free_slot(destination)
            counts[tier] -= 1
            if destination is None:
                changed[victim.chunk_id] = None
            else:
                seq += 1
                moved = replace(victim, tier=destination, last_access_seq=seq)
                changed[victim.chunk_id] = moved
                counts[destination] += 1
                heappush(added.setdefault(destination, []), (seq, victim.chunk_id))
            actions.append((victim.chunk_id, tier, destination))

        def free_slot(tier: str) -> None:  # 目标层名 → 递归腾出一个可用页位；无返回值
            while counts[tier] >= self._tiers[tier].capacity:
                evict_one(tier)

        for chunk_id in needed:
            page = changed.get(chunk_id, self._entries[chunk_id])
            if page is None or page.tier == "L1":
                continue
            source = page.tier
            counts[source] -= 1
            changed[chunk_id] = None
            free_slot("L1")
            seq += 1
            changed[chunk_id] = replace(page, tier="L1", last_access_seq=seq)
            counts["L1"] += 1
            actions.append((chunk_id, source, "L1"))
            promoted.append((chunk_id, source))
        while target.capacity - counts[reserve_tier] < reserve_pages:
            evict_one(reserve_tier)
        return actions, tuple(promoted)

    def _apply_move(
        self, chunk_id: str, source: str, destination: str | None, *, detached: bool = False
    ) -> None:  # 页 ID、源/目的层及预先移除标志 → 提交迁移计数和 LRU；无返回值
        """仅提交一页的索引、计数和 LRU 差量。"""
        page = self._entries[chunk_id]
        old = self._tiers[source]
        if not detached:
            old.ids.remove(chunk_id)
            old.resident -= 1
            if not page.ref_count:
                old.evictable -= 1
        if destination is not None and _TIER_ORDER.index(destination) < _TIER_ORDER.index(source):
            self.promotions += 1
            if source == "L2":
                self.l2_promotions += 1
            else:
                self.l3_promotions += 1
        else:
            old.evictions += 1
            if destination is None:
                self.deletions += 1
            else:
                self.demotions += 1
                if (source, destination) == ("L1", "L2"):
                    self.l1_to_l2_demotions += 1
                elif (source, destination) == ("L2", "L3"):
                    self.l2_to_l3_demotions += 1
        if destination is None:
            del self._entries[chunk_id]
            return
        self._access_seq += 1
        moved = replace(page, tier=destination, last_access_seq=self._access_seq)
        self._entries[chunk_id] = moved
        target = self._tiers[destination]
        target.ids.add(chunk_id)
        target.resident += 1
        target.peak = max(target.peak, target.resident)
        if not moved.ref_count:
            target.evictable += 1
            key = (moved.last_access_seq, chunk_id)
            heappush(target.heap, key)
            target.queued.add(key)

    def plan(
        self,
        promote_ids: Iterable[str] = (),
        reserve_pages: int = 0,
    ) -> tuple[tuple[str, str], ...]:  # 待提升页及预留页数 → 可行时的页 ID/原层元组，不改变状态
        """预检提升和新增私有页所需容量；失败无任何状态副作用。"""
        return self._migration_plan(promote_ids, reserve_pages)[1]

    def promote(
        self,
        promote_ids: Iterable[str] = (),
        reserve_pages: int = 0,
    ) -> tuple[tuple[str, str], ...]:  # 待提升页及预留页数 → 执行迁移并返回页 ID/原层元组
        """局部试算通过后增量提交批次迁移。"""
        actions, promoted = self._migration_plan(promote_ids, reserve_pages)
        detached = {chunk_id for chunk_id, _ in promoted}
        for chunk_id, source in promoted:
            origin = self._tiers[source]
            origin.ids.remove(chunk_id)
            origin.resident -= 1
        for chunk_id, source, destination in actions:
            self._apply_move(
                chunk_id, source, destination, detached=chunk_id in detached and destination == "L1"
            )
        return promoted

    def reserve_transfer(self, chunk_id: str, destination: str) -> str:  # 页 ID 与目的层 → 锁定源页并预留容量，返回来源层
        """保护来源并占用目的容量；查询仍只观察来源页。"""
        page = self._entries.get(chunk_id)
        if page is None:
            raise ValueError("待传输页不存在")
        if chunk_id in self._transfers:
            raise ValueError("同一页已有在途传输")
        if destination not in self._tiers or not self._tiers[destination].capacity:
            raise ValueError("目的层不存在或未启用")
        source = page.tier
        if source == destination:
            raise ValueError("来源与目的层不能相同")
        if _TIER_ORDER.index(destination) > _TIER_ORDER.index(source) and page.ref_count:
            raise CacheCapacityError("有活动引用的页不能降级")
        actions, _ = self._migration_plan((), 1, reserve_tier=destination, protected_id=chunk_id)
        for victim, source_tier, destination_tier in actions:
            self._apply_move(victim, source_tier, destination_tier)
        self._add_ref(chunk_id, refresh=False)
        target = self._tiers[destination]
        target.resident += 1
        target.reserved += 1
        target.peak = max(target.peak, target.resident)
        self._transfers[chunk_id] = destination
        return source

    def complete_transfer(self, chunk_id: str) -> str:  # 在途页 ID → 提交迁入、释放保护并返回来源层
        """传输完成时将来源页迁入占位，最后释放传输 pin。"""
        destination = self._transfers[chunk_id]
        page = self._entries[chunk_id]
        source = page.tier
        if _TIER_ORDER.index(destination) > _TIER_ORDER.index(source) and page.ref_count > 1:
            raise CacheCapacityError("在途降级页获得活动引用，须先取消传输")
        target = self._tiers[destination]
        target.reserved -= 1
        target.resident -= 1
        self._apply_move(chunk_id, source, destination)
        self._drop_ref(chunk_id)
        del self._transfers[chunk_id]
        return source

    def cancel_transfer(self, chunk_id: str) -> None:  # 在途页 ID → 撤销目的占位与来源保护；无返回值
        """撤销目的层占位和来源页保护，不刷新来源 LRU。"""
        destination = self._transfers.pop(chunk_id)
        target = self._tiers[destination]
        target.reserved -= 1
        target.resident -= 1
        self._drop_ref(chunk_id)

    def prepare_batch(
        self,
        promote_ids: Iterable[str] = (),
        reserve_pages: int = 0,
    ) -> tuple[tuple[str, str], ...]:  # 提升页 ID 与所需新页数 → 已迁入页的 ID/原层元组
        """与 promote 等价，供引擎原子提交本批容量及恢复页。"""
        return self.promote(promote_ids, reserve_pages)

    def ensure_free(self, count: int) -> int:  # 需要的 L1 空页数 → 原子腾页后的驱逐次数
        """原子腾出 L1 实际页；容量不足不执行逐级迁移。"""
        if count < 0:
            raise ValueError("所需页数不得为负")
        if count <= self.free_pages:
            return 0
        evictions_before = self.evictions
        self.promote(reserve_pages=count)
        return self.evictions - evictions_before

    @staticmethod
    def _chunk_id(request: InferenceRequest, index: int) -> str:  # 请求与逻辑页序号 → 可继承的稳定页身份
        return logical_chunk_id(request.request_id, index, request.inherited_prefix_ids)


class L1KVCache(TieredKVCache):
    """零低层容量兼容入口；不维护独立缓存状态。"""

    def __init__(self, max_total_tokens: int, page_size: int) -> None:  # L1 token 容量与页大小 → 初始化仅有 L1 的页池；无返回值
        super().__init__(max_total_tokens, page_size)
