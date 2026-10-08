"""F001 单池 L1 KV 页管理：共享完整页、物理容量与 LRU。"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from heapq import heappop, heappush

from simcore.prefix_hash import logical_chunk_id
from simcore.schemas import InferenceRequest

__all__ = ["CacheCapacityError", "KVRequestState", "L1KVCache", "L1Page", "L1Snapshot"]


class CacheCapacityError(ValueError):
    """当前 L1 物理容量不足，调用方应等待或回撤请求。"""


@dataclass(frozen=True, slots=True)
class L1Page:
    chunk_id: str
    tier: str
    ref_count: int
    last_access_seq: int
    capacity_pages: int = 1


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


class L1KVCache:
    """F001 单池 KV：共享完整页索引 + 每请求 K/F，不保存物理页表。

    候选先 pin 命中页；失败 cancel_pin 不改变 LRU；成功 admit 后逐步 extend。
    发布前完整页仍是请求私有页，同批按 ID 顺序 publish 才合并相同身份。
    """

    def __init__(self, max_total_tokens: int, page_size: int) -> None:
        if page_size < 1 or max_total_tokens < page_size or max_total_tokens % page_size:
            raise ValueError("max_total_tokens 必须至少一页且是 page_size 的整数倍")
        self.page_size = page_size
        self.capacity_pages = max_total_tokens // page_size
        self._pages: dict[str, L1Page] = {}
        self._requests: dict[int, InferenceRequest] = {}
        self._states: dict[int, KVRequestState] = {}
        self._pins: dict[int, tuple[InferenceRequest, int]] = {}
        self._evictable_heap: list[tuple[int, str]] = []
        self._queued_evictions: set[tuple[int, str]] = set()
        self._evictable_pages = 0
        self._private_pages = 0
        self._resident_pages = 0
        self._access_seq = 0
        self.evictions = 0
        self.peak_resident_pages = 0

    @property
    def resident_pages(self) -> int:
        return self._resident_pages

    @property
    def free_pages(self) -> int:
        return self.capacity_pages - self._resident_pages

    @property
    def evictable_pages(self) -> int:
        return self._evictable_pages

    @property
    def active_pages(self) -> int:
        return self._resident_pages - self._evictable_pages

    @property
    def private_pages(self) -> int:
        return self._private_pages

    @property
    def available_tokens(self) -> int:
        return (self.free_pages + self._evictable_pages) * self.page_size

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
        """逐页查连续已发布前缀；首次始终留最后一个输入 token 计算。"""
        if target_tokens < request.prompt_len:
            raise ValueError("重算目标不得小于原始 prompt")
        limit = (
            target_tokens // self.page_size
            if restoring
            else (request.prompt_len - 1) // self.page_size
        )
        for index in range(limit):
            if self._chunk_id(request, index) not in self._pages:
                return index * self.page_size
        return limit * self.page_size

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
        """确认候选并刷新已获得的引用；其余 KV 待 extend 时实际分配。"""
        request, matched_tokens = self._pins.pop(request_id)
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
        self._resident_pages += new_pages
        self.peak_resident_pages = max(self.peak_resident_pages, self._resident_pages)
        self._states[request_id] = KVRequestState(target_tokens, state.f)
        return new_pages

    def publish(self, request_id: int) -> int:
        """将请求私有完整页依次发布；相同 ID 合并并立即归还重复页。"""
        state = self.state(request_id)
        request = self._requests[request_id]
        freed = 0
        for index in range(state.f // self.page_size, state.k // self.page_size):
            chunk_id = self._chunk_id(request, index)
            existing = self._pages.get(chunk_id)
            self._private_pages -= 1
            if existing is None:
                self._access_seq += 1
                self._pages[chunk_id] = L1Page(chunk_id, "L1", 1, self._access_seq)
            else:
                self._add_ref(chunk_id, refresh=True)
                self._resident_pages -= 1
                freed += 1
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
        self._resident_pages -= count
        del self._states[request_id]

    def _add_ref(self, chunk_id: str, *, refresh: bool) -> None:
        page = self._pages[chunk_id]
        if page.ref_count == 0:
            self._evictable_pages -= 1
        self._pages[chunk_id] = replace(page, ref_count=page.ref_count + 1)
        if refresh:
            self._touch(chunk_id)

    def _drop_ref(self, chunk_id: str) -> None:
        page = self._pages[chunk_id]
        self._pages[chunk_id] = replace(page, ref_count=page.ref_count - 1)
        if page.ref_count == 1:
            self._evictable_pages += 1
            key = (page.last_access_seq, chunk_id)
            if key not in self._queued_evictions:
                heappush(self._evictable_heap, key)
                self._queued_evictions.add(key)

    def _touch(self, chunk_id: str) -> None:
        self._access_seq += 1
        self._pages[chunk_id] = replace(self._pages[chunk_id], last_access_seq=self._access_seq)

    def ensure_free(self, count: int) -> int:
        """执行前腾出所需实际页；不足则保持原状态供上层回撤。"""
        if count < 0:
            raise ValueError("所需页数不得为负")
        if count > self.free_pages + self._evictable_pages:
            raise CacheCapacityError("L1 当前容量不足以分配新的 KV 页")
        evictions_before = self.evictions
        while self.free_pages < count:
            seq, chunk_id = heappop(self._evictable_heap)
            self._queued_evictions.remove((seq, chunk_id))
            page = self._pages.get(chunk_id)
            if page is None or page.ref_count or page.last_access_seq != seq:
                continue
            del self._pages[chunk_id]
            self._evictable_pages -= 1
            self._resident_pages -= 1
            self.evictions += 1
        return self.evictions - evictions_before

    @staticmethod
    def _chunk_id(request: InferenceRequest, index: int) -> str:
        return logical_chunk_id(request.request_id, index, request.inherited_prefix_ids)
