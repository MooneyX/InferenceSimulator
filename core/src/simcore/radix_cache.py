"""L1（各实例 HBM）与 L2 共用的前缀基数树。

节点是一段不分叉的连续完整页，key 是若干段连续 chunk ID ``(起始 ID, 页数)``；子节点按首页
ID 索引。节点直接记录 L2 副本状态，并按实例记录 HBM 副本、路径锁与访问时间。共享 L2 时全场
一棵树（每个实例把整块共享 L2 视为自己的 L2），否则每个实例一棵。

行为对照 SGLang（提交 ``188cf7da``）``mem_cache/radix_cache.py`` 与
``mem_cache/unified_cache/unified_tree_core.py``：

- 匹配或插入止于节点内部时切分节点（``_match_prefix_helper`` / ``_insert_helper``），
  发布时另在 prompt 边界切分（``checkpoint(split_prompt=True)``）；切分后不合并。
- 请求从终点节点沿路径到根加锁（``inc_lock_ref``）；锁同时阻止 HBM 与 L2 淘汰。
- HBM 只淘汰本实例的 D-leaf（本实例驻留、未加锁、无本实例驻留子节点），整节点释放；
  L2 只淘汰 H-leaf（L2 驻留、无任何 HBM 副本、未加锁、无子节点）。按访问时间 LRU。
- 写入策略为 ``write_through``：发布即把本路径上尚无 L2 副本的节点写入 L2，L2 无可淘汰
  空间时跳过；HBM 淘汰有 L2 副本的节点只丢弃 HBM 副本，L2→HBM 回载保留 L2 副本。

chunk ID 同时编码整条前缀，因此节点内与请求的公共页数可二分确定。
"""

from collections.abc import Iterator
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING

from simcore.schemas import InferenceRequest

if TYPE_CHECKING:
    from simcore.kv_cache import CachePool

__all__ = ["HbmPool", "HbmState", "RadixNode", "RadixTree", "chunk_id", "request_segments"]

_HEAP_SLACK = 1024

type Segments = tuple[tuple[int, int], ...]
type HeapEntry = tuple[int, int, RadixNode]


def chunk_id(request: InferenceRequest, index: int) -> int:
    """请求第 ``index`` 个完整页的 ID：显式 ``chunk_ids`` 之后为 ``chunk_base + index``。"""
    ids = request.chunk_ids
    return ids[index] if index < len(ids) else request.chunk_base + index


def request_segments(request: InferenceRequest, start: int, end: int) -> Segments:
    """请求页 ``[start, end)`` 的连续 ID 段。"""
    ids = request.chunk_ids
    count = len(ids)
    segments: list[tuple[int, int]] = []
    index = start
    while index < end and index < count:
        first = ids[index]
        stop = index + 1
        while stop < end and stop < count and ids[stop] == first + stop - index:
            stop += 1
        segments.append((first, stop - index))
        index = stop
    if index < end:
        first = request.chunk_base + index
        if segments and sum(segments[-1]) == first:
            segments[-1] = (segments[-1][0], segments[-1][1] + end - index)
        else:
            segments.append((first, end - index))
    return tuple(segments)


def _split_segments(segments: Segments, offset: int) -> tuple[Segments, Segments]:
    top: list[tuple[int, int]] = []
    for position, (start, length) in enumerate(segments):
        if offset < length:
            if offset:
                top.append((start, offset))
                return tuple(top), ((start + offset, length - offset), *segments[position + 1 :])
            return tuple(top), segments[position:]
        top.append((start, length))
        offset -= length
    raise ValueError("切分位置必须位于节点内部")


class HbmState:
    """某实例在一个节点上的 HBM 副本、路径锁、访问时间与本实例驻留子节点数。"""

    __slots__ = ("access", "kids", "lock", "resident")

    def __init__(self) -> None:
        self.resident = False
        self.lock = 0
        self.access = 0
        self.kids = 0


class RadixNode:
    """连续完整页；``parent is None`` 表示根或已删除。"""

    __slots__ = (
        "children",
        "hbm",
        "holders",
        "l2",
        "l2_access",
        "locks",
        "node_id",
        "pages",
        "parent",
        "segments",
    )

    def __init__(
        self, node_id: int, parent: "RadixNode | None", segments: Segments, pages: int
    ) -> None:
        self.node_id = node_id
        self.parent = parent
        self.children: dict[int, RadixNode] = {}
        self.segments = segments
        self.pages = pages
        self.l2 = False
        self.l2_access = 0
        self.locks = 0  # 全部实例的路径锁之和，阻止 L2 淘汰
        self.hbm = 0  # 持有 HBM 副本的实例数
        self.holders: dict[int, HbmState] = {}

    def page_id(self, offset: int) -> int:
        for start, length in self.segments:
            if offset < length:
                return start + offset
            offset -= length
        raise IndexError(offset)

    def page_ids(self) -> Iterator[int]:
        for start, length in self.segments:
            yield from range(start, start + length)


class HbmPool:
    """一个实例的 HBM（L1）页账本与 D-leaf 淘汰堆；堆在首次淘汰时才建立。"""

    __slots__ = (
        "capacity",
        "demotions",
        "evictable",
        "evictions",
        "heap",
        "instance_id",
        "peak",
        "private_pages",
        "promotions",
        "tree_pages",
    )

    def __init__(self, instance_id: int, capacity: int) -> None:
        self.instance_id = instance_id
        self.capacity = capacity
        self.tree_pages = 0  # 已发布到树中的本实例驻留页
        self.private_pages = 0
        self.evictable = 0  # 本实例驻留且未加锁的树页
        self.peak = 0
        self.evictions = 0
        self.promotions = 0
        self.demotions = 0
        self.heap: list[HeapEntry] | None = None

    @property
    def resident(self) -> int:
        return self.tree_pages + self.private_pages

    def grow(self, pages: int) -> None:
        self.private_pages += pages
        if self.tree_pages + self.private_pages > self.peak:
            self.peak = self.tree_pages + self.private_pages


class RadixTree:
    """L1/L2 前缀树；L2 容量与淘汰全树共享，HBM 账本由各实例的 ``HbmPool`` 持有。"""

    def __init__(self, page_size: int, l2_capacity: int, l3: "CachePool") -> None:
        self.page_size = page_size
        self.l2_capacity = l2_capacity
        self.l3 = l3
        self.root = RadixNode(0, None, (), 0)
        self.nodes = 0
        self.clock = 0
        self._next_id = 1
        self.l2_pages = 0
        self.l2_reserved = 0  # 在途 L3→L2 预取的目的占位
        self.l2_evictable = 0  # L2 驻留、无 HBM 副本且未加锁的页
        self.l2_peak = 0
        self.l2_evictions = 0
        self.l2_incoming_ms: dict[int, float] = {}  # 在途预取页 → 完成时刻，供其他读者合并等待
        self._l2_heap: list[HeapEntry] | None = None

    # ------------------------------------------------------------ 匹配与结构

    def match(
        self, hbm: HbmPool, request: InferenceRequest, limit: int
    ) -> tuple[RadixNode, int, int]:
        """本地连续前缀（HBM 后接 L2）：返回终点节点、HBM 页数与本地页数，并刷新路径访问时间。"""
        inst = hbm.instance_id
        node = self.root
        index = 0
        l1 = 0
        if limit <= 0:
            return node, 0, 0
        self.clock += 1
        now = self.clock
        while index < limit:
            child = node.children.get(chunk_id(request, index))
            if child is None:
                break
            state = child.holders.get(inst)
            resident = state is not None and state.resident
            if not resident and not child.l2:
                break
            matched = self._common(child, request, index, limit - index)
            partial = matched < child.pages
            if partial:
                child = self._split(child, matched)
                state = child.holders.get(inst)
            index += matched
            if resident:
                l1 = index
            if resident and state is not None:
                state.access = now
                if hbm.heap is not None and not state.lock and not state.kids:
                    self._push_hbm(hbm, child, now)
            if child.l2:
                child.l2_access = now
                if (
                    self._l2_heap is not None
                    and not child.hbm
                    and not child.locks
                    and not child.children
                ):
                    self._push_l2(child)
            node = child
            if partial:
                break
        return node, l1, index

    @staticmethod
    def _common(node: RadixNode, request: InferenceRequest, index: int, cap: int) -> int:
        """首页已相同；ID 编码整条前缀，相同页必为连续前缀，故可二分。"""
        count = node.pages if node.pages < cap else cap
        if node.page_id(count - 1) == chunk_id(request, index + count - 1):
            return count
        low, high = 1, count - 1
        while low < high:
            middle = (low + high) // 2
            if node.page_id(middle) == chunk_id(request, index + middle):
                low = middle + 1
            else:
                high = middle
        return low

    def _split(self, node: RadixNode, offset: int) -> RadixNode:
        """在 ``offset`` 页处切分：新建上半节点并返回，原对象保留为下半节点（锁终点不变）。"""
        top_segments, bottom_segments = _split_segments(node.segments, offset)
        parent = node.parent
        assert parent is not None
        top = RadixNode(self._next_id, parent, top_segments, offset)
        self._next_id += 1
        self.nodes += 1
        top.l2 = node.l2
        top.l2_access = node.l2_access
        top.locks = node.locks
        top.hbm = node.hbm
        for inst, state in node.holders.items():
            copy = HbmState()
            copy.resident = state.resident
            copy.lock = state.lock
            copy.access = state.access
            copy.kids = 1 if state.resident else 0
            top.holders[inst] = copy
        parent.children[top_segments[0][0]] = top
        node.parent = top
        node.segments = bottom_segments
        node.pages -= offset
        top.children[bottom_segments[0][0]] = node
        return top

    def _new_node(self, parent: RadixNode, segments: Segments, pages: int) -> RadixNode:
        node = RadixNode(self._next_id, parent, segments, pages)
        self._next_id += 1
        self.nodes += 1
        parent.children[segments[0][0]] = node
        return node

    def ensure_path(self, request: InferenceRequest, start: int, end: int) -> list[RadixNode]:
        """保证请求路径在 ``start`` 与 ``end`` 页处有节点边界并补齐缺失结构，返回覆盖
        ``[start, end)`` 的节点；``start`` 之前缺失的部分建为无副本的占位节点。"""
        node = self.root
        index = 0
        nodes: list[RadixNode] = []
        while index < end:
            stop = start if index < start else end
            child = node.children.get(chunk_id(request, index))
            if child is None:
                child = self._new_node(node, request_segments(request, index, stop), stop - index)
                matched = stop - index
            else:
                matched = self._common(child, request, index, stop - index)
                if matched < child.pages:
                    child = self._split(child, matched)
            if index >= start:
                nodes.append(child)
            index += matched
            node = child
        return nodes

    # ------------------------------------------------------------ 路径锁

    def lock(self, hbm: HbmPool, node: RadixNode) -> None:
        inst = hbm.instance_id
        root = self.root
        while node is not root:
            state = node.holders.get(inst)
            if state is None:
                state = node.holders[inst] = HbmState()
            if not state.lock and state.resident:
                hbm.evictable -= node.pages
            if not node.locks and node.l2 and not node.hbm:
                self.l2_evictable -= node.pages
            state.lock += 1
            node.locks += 1
            parent = node.parent
            assert parent is not None
            node = parent

    def unlock(self, hbm: HbmPool, node: RadixNode) -> None:
        inst = hbm.instance_id
        root = self.root
        while node is not root:
            state = node.holders[inst]
            state.lock -= 1
            node.locks -= 1
            if not state.lock:
                if state.resident:
                    hbm.evictable += node.pages
                    if not state.kids and hbm.heap is not None:
                        self._push_hbm(hbm, node, state.access)
                else:
                    del node.holders[inst]
            if not node.locks and node.l2 and not node.hbm:
                self.l2_evictable += node.pages
                if not node.children and self._l2_heap is not None:
                    self._push_l2(node)
            parent = node.parent
            assert parent is not None
            node = parent

    def unlocked_resident_pages(self, hbm: HbmPool, node: RadixNode) -> int:
        """路径上本实例驻留且未加锁的页数，即加锁后不再可淘汰的页数。"""
        inst = hbm.instance_id
        pages = 0
        while node is not self.root:
            state = node.holders.get(inst)
            if state is not None and state.resident and not state.lock:
                pages += node.pages
            parent = node.parent
            assert parent is not None
            node = parent
        return pages

    def resident(self, hbm: HbmPool, node: RadixNode) -> bool:
        """本实例 HBM 驻留集对前缀封闭：终点驻留即整条路径驻留。"""
        state = node.holders.get(hbm.instance_id)
        return node is self.root or (state is not None and state.resident)

    # ------------------------------------------------------------ HBM 副本

    def _add_hbm(self, hbm: HbmPool, node: RadixNode) -> None:
        inst = hbm.instance_id
        state = node.holders.get(inst)
        if state is None:
            state = node.holders[inst] = HbmState()
        state.resident = True
        state.access = self.clock
        if not node.hbm and node.l2 and not node.locks:
            self.l2_evictable -= node.pages
        node.hbm += 1
        hbm.tree_pages += node.pages
        if hbm.tree_pages + hbm.private_pages > hbm.peak:
            hbm.peak = hbm.tree_pages + hbm.private_pages
        parent = node.parent
        assert parent is not None
        if parent is not self.root:
            parent.holders[inst].kids += 1
        if not state.lock:
            hbm.evictable += node.pages
            if hbm.heap is not None:
                self._push_hbm(hbm, node, state.access)

    def attach(
        self,
        hbm: HbmPool,
        request: InferenceRequest,
        start: int,
        end: int,
        endpoint: RadixNode,
    ) -> tuple[RadixNode, int]:
        """把请求页 ``[start, end)`` 作为本实例 HBM 副本插入并把锁终点移到末节点。

        ``start`` 须为当前锁终点 ``endpoint`` 的末页；返回新终点与已在本实例驻留的重复页数。
        """
        inst = hbm.instance_id
        self.clock += 1
        nodes = self.ensure_path(request, start, end)
        last = nodes[-1]
        self.lock(hbm, last)
        duplicate = 0
        for node in nodes:
            state = node.holders[inst]
            if state.resident:
                duplicate += node.pages
            else:
                self._add_hbm(hbm, node)
        self.unlock(hbm, endpoint)
        return last, duplicate

    def publish(
        self,
        hbm: HbmPool,
        request: InferenceRequest,
        start: int,
        end: int,
        endpoint: RadixNode,
        prompt_pages: int,
    ) -> tuple[RadixNode, int]:
        """发布已计算页：插入、prompt 边界切分、刷新路径并 write-through 到 L2。"""
        last = endpoint
        duplicate = 0
        if end > start:
            last, duplicate = self.attach(hbm, request, start, end, endpoint)
        if 0 < prompt_pages < end:
            self.ensure_path(request, prompt_pages, prompt_pages)
        if end <= start:
            return last, duplicate
        inst = hbm.instance_id
        now = self.clock
        unbacked: list[RadixNode] = []
        node = last
        while node is not self.root:
            node.holders[inst].access = now
            if node.l2:
                node.l2_access = now
            else:
                unbacked.append(node)
            parent = node.parent
            assert parent is not None
            node = parent
        if unbacked and self.l2_capacity:
            unbacked.reverse()
            self._write_through(hbm, unbacked)
        return last, duplicate

    def promote_path(self, hbm: HbmPool, node: RadixNode) -> int:
        """把已加锁路径上本实例未驻留的 L2 节点自上而下复制到 HBM，L2 副本保留。"""
        inst = hbm.instance_id
        chain: list[RadixNode] = []
        while node is not self.root:
            state = node.holders.get(inst)
            if state is not None and state.resident:
                break
            chain.append(node)
            parent = node.parent
            assert parent is not None
            node = parent
        if not chain:
            return 0
        self.clock += 1
        pages = 0
        for item in reversed(chain):
            self._add_hbm(hbm, item)
            pages += item.pages
        hbm.promotions += pages
        return pages

    def evict_hbm(self, hbm: HbmPool, count: int) -> int:
        """按 LRU 淘汰本实例 D-leaf 直至空闲页不少于 ``count``；返回淘汰页数。"""
        inst = hbm.instance_id
        heap = hbm.heap
        if heap is None:
            heap = hbm.heap = self._collect_hbm(inst)
        evicted = 0
        while hbm.capacity - hbm.tree_pages - hbm.private_pages < count:
            victim: RadixNode | None = None
            while heap:
                access, _, node = heappop(heap)
                state = node.holders.get(inst)
                if (
                    node.parent is not None
                    and state is not None
                    and state.resident
                    and not state.lock
                    and not state.kids
                    and state.access == access
                ):
                    victim = node
                    break
            if victim is None:
                break
            evicted += victim.pages
            self._evict_hbm_node(hbm, victim)
            heap = hbm.heap
            assert heap is not None
        return evicted

    def _evict_hbm_node(self, hbm: HbmPool, node: RadixNode) -> None:
        inst = hbm.instance_id
        del node.holders[inst]
        node.hbm -= 1
        hbm.tree_pages -= node.pages
        hbm.evictable -= node.pages
        hbm.evictions += node.pages
        parent = node.parent
        assert parent is not None
        if parent is not self.root:
            parent_state = parent.holders[inst]
            parent_state.kids -= 1
            if not parent_state.kids and not parent_state.lock and hbm.heap is not None:
                self._push_hbm(hbm, parent, parent_state.access)
        if not self.l2_capacity and self.l3.capacity:
            self._write_l3(hbm, node)  # 两层结构：L1 淘汰直接写入 L3
        if node.hbm:
            return
        if node.l2:
            if not node.locks:
                self.l2_evictable += node.pages
                if not node.children and self._l2_heap is not None:
                    self._push_l2(node)
            return
        if not node.children:
            self._delete(node)

    # ------------------------------------------------------------ L2 副本

    def ensure_l2_free(self, hbm: HbmPool, count: int) -> bool:
        """按 LRU 淘汰 H-leaf 直至 L2 空闲页不少于 ``count``；不足时返回 False，已淘汰的不恢复。"""
        if count > self.l2_capacity:
            return False
        while self.l2_capacity - self.l2_pages - self.l2_reserved < count:
            heap = self._l2_heap
            if heap is None:
                heap = self._l2_heap = self._collect_l2()
            victim: RadixNode | None = None
            while heap:
                access, _, node = heappop(heap)
                if (
                    node.parent is not None
                    and node.l2
                    and not node.hbm
                    and not node.locks
                    and not node.children
                    and node.l2_access == access
                ):
                    victim = node
                    break
            if victim is None:
                return False
            self._evict_l2_node(hbm, victim)
        return True

    def _evict_l2_node(self, hbm: HbmPool, node: RadixNode) -> None:
        if self.l3.capacity:
            self._write_l3(hbm, node)
        node.l2 = False
        self.l2_pages -= node.pages
        self.l2_evictable -= node.pages
        self.l2_evictions += node.pages
        self._delete(node)

    def _write_through(self, hbm: HbmPool, nodes: list[RadixNode]) -> None:
        pages = sum(node.pages for node in nodes)
        if not self.ensure_l2_free(hbm, pages):
            return
        now = self.clock
        for node in nodes:
            node.l2 = True
            node.l2_access = now
        self.l2_pages += pages
        hbm.demotions += pages
        self.l2_peak = max(self.l2_peak, self.l2_pages + self.l2_reserved)

    def reserve_l2(self, hbm: HbmPool) -> bool:
        """为一页 L3→L2 预取预留目的容量。"""
        if not self.ensure_l2_free(hbm, 1):
            return False
        self.l2_reserved += 1
        self.l2_peak = max(self.l2_peak, self.l2_pages + self.l2_reserved)
        return True

    def insert_l2(self, request: InferenceRequest, start: int, end: int) -> None:
        """预取完成：把请求页 ``[start, end)`` 作为 L2 副本插入，释放对应目的占位。"""
        self.clock += 1
        now = self.clock
        self.l2_reserved -= end - start
        for node in self.ensure_path(request, start, end):
            if node.l2:
                continue
            node.l2 = True
            node.l2_access = now
            self.l2_pages += node.pages
            if not node.hbm and not node.locks:
                self.l2_evictable += node.pages
                if not node.children and self._l2_heap is not None:
                    self._push_l2(node)

    # ------------------------------------------------------------ 删除、L3 写入与堆

    def _delete(self, node: RadixNode) -> None:
        """删除无副本、无子节点的节点，并向上级联删除同样变为空叶的占位节点。"""
        while True:
            parent = node.parent
            assert parent is not None
            del parent.children[node.segments[0][0]]
            node.parent = None
            self.nodes -= 1
            if parent is self.root or parent.children:
                return
            if parent.hbm or parent.l2:
                if parent.l2 and not parent.hbm and not parent.locks and self._l2_heap is not None:
                    self._push_l2(parent)
                return
            node = parent

    def _write_l3(self, hbm: HbmPool, node: RadixNode) -> None:
        """逐页写入共享 L3：已有副本只刷新 LRU；L3 满时先淘汰 LRU，无可淘汰页则丢弃。"""
        l3 = self.l3
        entries = l3.entries
        for page in node.page_ids():
            if page in entries:
                l3.touch(page)
            else:
                if l3.resident >= l3.capacity:
                    victim = l3.pop_lru()
                    if victim is None:
                        continue
                    l3.remove(victim.chunk_id, evicted=True)
                l3.insert(page)
            hbm.demotions += 1

    def _push_hbm(self, hbm: HbmPool, node: RadixNode, access: int) -> None:
        heap = hbm.heap
        assert heap is not None
        heappush(heap, (access, node.node_id, node))
        if len(heap) > 2 * self.nodes + _HEAP_SLACK:
            hbm.heap = self._collect_hbm(hbm.instance_id)

    def _push_l2(self, node: RadixNode) -> None:
        heap = self._l2_heap
        assert heap is not None
        heappush(heap, (node.l2_access, node.node_id, node))
        if len(heap) > 2 * self.nodes + _HEAP_SLACK:
            self._l2_heap = self._collect_l2()

    def _collect_hbm(self, inst: int) -> list[HeapEntry]:
        """本实例 D-leaf 候选；本实例未驻留的子树不会有本实例副本，直接剪枝。"""
        heap: list[HeapEntry] = []
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            state = node.holders.get(inst)
            if state is None or not state.resident:
                continue
            if state.kids:
                stack.extend(node.children.values())
            elif not state.lock:
                heap.append((state.access, node.node_id, node))
        heapify(heap)
        return heap

    def _collect_l2(self) -> list[HeapEntry]:
        heap: list[HeapEntry] = []
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            elif node.l2 and not node.hbm and not node.locks:
                heap.append((node.l2_access, node.node_id, node))
        heapify(heap)
        return heap
