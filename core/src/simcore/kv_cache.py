"""L1 paged KV block manager：硬物理容量、只读快照及原子 batch 准入。"""

from bisect import insort
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush

__all__ = [
    "AdmissionAllocation",
    "BatchCacheResult",
    "CacheCapacityError",
    "CacheOOMError",
    "CacheSnapshot",
    "CacheTransaction",
    "HierarchicalKVCache",
    "KVBlock",
    "PrefixProbe",
    "RequestCacheResult",
]


class CacheCapacityError(ValueError):
    """当前活动引用或未来预留占满 L1；请求应留在 FCFS 队头。"""


class CacheOOMError(CacheCapacityError):
    """单个请求即使独占空 L1 也无法完成。"""


@dataclass(frozen=True, slots=True)
class KVBlock:
    block_id: int
    content_hash: str | None
    valid_tokens: int
    tier: str
    state: str
    ref_count: int
    last_access_seq: int
    owner_generation: int


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    cache_id: int = field(compare=False, repr=False)
    epoch: int
    capacity_blocks: int
    block_size_tokens: int
    blocks: tuple[KVBlock, ...]
    hash_index: tuple[tuple[str, tuple[int, ...]], ...]
    reserved_future_blocks: int

    @property
    def resident_blocks(self) -> int:
        return len(self.blocks)

    @property
    def active_blocks(self) -> int:
        return sum(block.ref_count > 0 for block in self.blocks)

    @property
    def evictable_blocks(self) -> int:
        return sum(block.state == "EVICTABLE" for block in self.blocks)

    @property
    def free_blocks(self) -> int:
        return self.capacity_blocks - self.resident_blocks - self.reserved_future_blocks

    @property
    def valid_tokens(self) -> int:
        return sum(block.valid_tokens for block in self.blocks)

    @property
    def block_capacity_tokens(self) -> int:
        return self.resident_blocks * self.block_size_tokens

    @property
    def internal_fragmentation_tokens(self) -> int:
        return self.block_capacity_tokens - self.valid_tokens


@dataclass(frozen=True, slots=True)
class PrefixProbe:
    snapshot_epoch: int
    input_chunk_ids: tuple[str, ...]
    block_refs: tuple[tuple[int, int], ...]  # (block_id, owner_generation)
    source_tiers: tuple[str, ...]
    block_size_tokens: int

    @property
    def matched_blocks(self) -> int:
        return len(self.block_refs)

    @property
    def matched_cache_tokens(self) -> int:
        return self.matched_blocks * self.block_size_tokens


@dataclass(frozen=True, slots=True)
class AdmissionAllocation:
    request_id: int
    prompt_len: int
    output_len: int
    input_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CacheTransaction:
    cache_id: int = field(compare=False, repr=False)
    snapshot_epoch: int
    probes: tuple[PrefixProbe, ...]
    allocations: tuple[AdmissionAllocation, ...]
    victim_ids: tuple[int, ...]
    new_block_count: int
    future_block_count: int


@dataclass(frozen=True, slots=True)
class RequestCacheResult:
    request_id: int
    block_ids: tuple[int, ...]
    matched_cache_tokens: int
    actual_hit_tokens: int
    new_blocks: int
    reserved_future_blocks: int


@dataclass(frozen=True, slots=True)
class BatchCacheResult:
    requests: tuple[RequestCacheResult, ...]
    evicted_block_ids: tuple[int, ...]
    allocated_blocks: int
    reused_block_ids: tuple[int, ...]
    epoch: int


class HierarchicalKVCache:
    """单 Engine 的 L1 paged KV；多级存储预留相同事务接口。

    准入预分配输入全部块，保留尚未生成的输出增长容量；只有 publish 的完整块
    才能被之后的 snapshot/probe 发现。所有块 ID 回收后提升 generation。
    """

    def __init__(self, capacity_blocks: int, block_size_tokens: int = 1) -> None:
        if capacity_blocks < 1 or block_size_tokens < 1:
            raise ValueError("capacity_blocks 与 block_size_tokens 必须 >= 1")
        self.capacity_blocks = capacity_blocks
        self.block_size_tokens = block_size_tokens
        self.blocks: dict[int, KVBlock] = {}
        self.request_blocks: dict[int, list[int]] = {}
        self.hash_index: dict[str, list[int]] = {}
        self._free_ids: list[int] = []
        self._generations: dict[int, int] = {}
        self._next_id = 0
        self._future_reserved: dict[int, int] = {}
        self._output_remaining: dict[int, int] = {}
        self._published_count: dict[int, int] = {}
        self._request_tokens: dict[int, int] = {}
        self.reserved_future_blocks = 0
        self.epoch = 0
        self.access_seq = 0
        self.evictions = 0

    @property
    def resident_blocks(self) -> int:
        return len(self.blocks)

    @property
    def free_blocks(self) -> int:
        return self.capacity_blocks - self.resident_blocks - self.reserved_future_blocks

    @property
    def logical_context_tokens(self) -> int:
        """所有活动请求的逻辑 token 总数；共享块按各请求分别计入。"""
        return sum(self._request_tokens.values())

    def request_context_tokens(self, request_id: int) -> int:
        self._blocks_for(request_id)
        return self._request_tokens[request_id]

    def snapshot(self) -> CacheSnapshot:
        """完整不可变复制，不触碰 LRU 和引用计数。"""
        return CacheSnapshot(
            id(self),
            self.epoch,
            self.capacity_blocks,
            self.block_size_tokens,
            tuple(self.blocks[key] for key in sorted(self.blocks)),
            tuple((key, tuple(value)) for key, value in sorted(self.hash_index.items())),
            self.reserved_future_blocks,
        )

    def probe(self, snapshot: CacheSnapshot, input_chunk_ids: Sequence[str]) -> PrefixProbe:
        """仅对给定 snapshot 查找连续的完整前缀，同 hash 选择最小物理 ID。"""
        self._check_snapshot_owner(snapshot)
        index = dict(snapshot.hash_index)
        blocks = {block.block_id: block for block in snapshot.blocks}
        found: list[tuple[int, int]] = []
        tiers: list[str] = []
        for content_hash in input_chunk_ids:
            candidates = index.get(content_hash)
            if not candidates:
                break
            block = blocks[candidates[0]]
            found.append((block.block_id, block.owner_generation))
            tiers.append(block.tier)
        return PrefixProbe(
            snapshot.epoch,
            tuple(input_chunk_ids),
            tuple(found),
            tuple(tiers),
            self.block_size_tokens,
        )

    def prepare_batch(
        self,
        snapshot: CacheSnapshot,
        probes: Sequence[PrefixProbe],
        allocations: Sequence[AdmissionAllocation],
    ) -> CacheTransaction:
        """同一快照上只读规划整个 batch；不足时不更改任何缓存状态。"""
        self._check_snapshot_owner(snapshot)
        if snapshot.epoch != self.epoch:
            raise ValueError("过期 cache snapshot")
        requests = tuple(allocations)
        matched = tuple(probes)
        if not requests or len(requests) != len(matched):
            raise ValueError("probes 与 allocations 必须是非空、等长的 batch")
        if len({item.request_id for item in requests}) != len(requests):
            raise ValueError("batch 内 request_id 重复")
        block_map = {block.block_id: block for block in snapshot.blocks}
        snapshot_index = dict(snapshot.hash_index)
        protected: set[int] = set()
        new_count = 0
        future_count = 0
        for probe, item in zip(matched, requests, strict=True):
            if item.request_id in self.request_blocks:
                raise ValueError(f"请求 {item.request_id} 重复准入")
            if item.prompt_len < 1 or item.output_len < 0:
                raise ValueError("prompt_len 必须 >= 1，output_len 必须 >= 0")
            prompt_blocks = (item.prompt_len + self.block_size_tokens - 1) // self.block_size_tokens
            total_blocks = (
                item.prompt_len + item.output_len + self.block_size_tokens - 1
            ) // self.block_size_tokens
            if len(item.input_chunk_ids) != item.prompt_len // self.block_size_tokens:
                raise ValueError("input_chunk_ids 数量必须等于完整输入块数")
            if item.input_chunk_ids != probe.input_chunk_ids:
                raise ValueError("probe 与 allocation 的输入块 ID 不一致")
            if (
                probe.snapshot_epoch != snapshot.epoch
                or probe.block_size_tokens != self.block_size_tokens
            ):
                raise ValueError("probe 不属于本次 snapshot")
            if probe.matched_blocks > len(item.input_chunk_ids):
                raise ValueError("probe 匹配块数不合法")
            for position, (block_id, generation) in enumerate(probe.block_refs):
                block = block_map.get(block_id)
                if (
                    block is None
                    or block.owner_generation != generation
                    or block.content_hash != item.input_chunk_ids[position]
                    or block_id not in snapshot_index.get(block.content_hash, ())
                ):
                    raise ValueError("probe 包含已失效的块引用")
                protected.add(block_id)
            if total_blocks > self.capacity_blocks:
                raise CacheOOMError(
                    f"请求 {item.request_id} 的输入和未来输出需要 {total_blocks} 块，"
                    f"单独运行也超过 L1 容量 {self.capacity_blocks}"
                )
            new_count += prompt_blocks - probe.matched_blocks
            future_count += total_blocks - prompt_blocks
        required = (
            snapshot.resident_blocks
            + snapshot.reserved_future_blocks
            + new_count
            + future_count
            - snapshot.capacity_blocks
        )
        victims = sorted(
            (
                block
                for block in snapshot.blocks
                if block.ref_count == 0 and block.block_id not in protected
            ),
            key=lambda block: (block.last_access_seq, block.block_id),
        )
        if required > len(victims):
            raise CacheCapacityError(
                f"L1 暂时不足：需要淘汰 {required} 块，仅有 {len(victims)} 块可淘汰"
            )
        return CacheTransaction(
            id(self),
            snapshot.epoch,
            matched,
            requests,
            tuple(block.block_id for block in victims[: max(required, 0)]),
            new_count,
            future_count,
        )

    def commit(self, transaction: CacheTransaction) -> BatchCacheResult:
        """事务准备阶段已完成校验；epoch 不变时一次提交全部引用及分配。"""
        if transaction.cache_id != id(self) or transaction.snapshot_epoch != self.epoch:
            raise ValueError("过期或不属于本缓存的 cache transaction")
        for block_id in transaction.victim_ids:
            self._remove_block(block_id)
            self.evictions += 1
        self.reserved_future_blocks += transaction.future_block_count
        results: list[RequestCacheResult] = []
        reused: set[int] = set()
        for probe, item in zip(transaction.probes, transaction.allocations, strict=True):
            ids: list[int] = []
            for block_id, generation in probe.block_refs:
                block = self.blocks[block_id]
                if block.owner_generation != generation:
                    raise RuntimeError("cache transaction 的 block generation 已改变")
                self.access_seq += 1
                self.blocks[block_id] = replace(
                    block,
                    ref_count=block.ref_count + 1,
                    state="ACTIVE",
                    last_access_seq=self.access_seq,
                )
                ids.append(block_id)
                reused.add(block_id)
            prompt_blocks = (item.prompt_len + self.block_size_tokens - 1) // self.block_size_tokens
            for position in range(probe.matched_blocks, prompt_blocks):
                block_id = self._allocate_id()
                valid = min(
                    self.block_size_tokens, item.prompt_len - position * self.block_size_tokens
                )
                self.access_seq += 1
                self.blocks[block_id] = KVBlock(
                    block_id,
                    None,
                    valid,
                    "L1",
                    "ACTIVE",
                    1,
                    self.access_seq,
                    self._generations[block_id],
                )
                ids.append(block_id)
            reserved = (
                item.prompt_len + item.output_len + self.block_size_tokens - 1
            ) // self.block_size_tokens - prompt_blocks
            self.request_blocks[item.request_id] = ids
            self._future_reserved[item.request_id] = reserved
            self._output_remaining[item.request_id] = item.output_len
            self._request_tokens[item.request_id] = item.prompt_len
            self._published_count[item.request_id] = probe.matched_blocks
            results.append(
                RequestCacheResult(
                    item.request_id,
                    tuple(ids),
                    probe.matched_cache_tokens,
                    min(probe.matched_cache_tokens, item.prompt_len - 1),
                    prompt_blocks - probe.matched_blocks,
                    reserved,
                )
            )
        self.epoch += 1
        return BatchCacheResult(
            tuple(results),
            transaction.victim_ids,
            transaction.new_block_count,
            tuple(sorted(reused)),
            self.epoch,
        )

    def publish(self, request_id: int, completed_block_hashes: Sequence[str]) -> None:
        """按逻辑顺序发布完整块 ID；参数名为历史遗留，不要求真实内容 hash。"""
        ids = self._blocks_for(request_id)
        hashes = tuple(completed_block_hashes)
        start = self._published_count[request_id]
        if start + len(hashes) > len(ids):
            raise ValueError("发布的块 ID 数量超过请求已有块数")
        for position, content_hash in enumerate(hashes, start):
            block = self.blocks[ids[position]]
            if block.valid_tokens != self.block_size_tokens or block.content_hash is not None:
                raise ValueError("仅可发布尚未发布的完整块")
            if not isinstance(content_hash, str) or not content_hash:
                raise ValueError("完整块 ID 必须为非空字符串")
        for position, content_hash in enumerate(hashes, start):
            block_id = ids[position]
            self.blocks[block_id] = replace(self.blocks[block_id], content_hash=content_hash)
            insort(self.hash_index.setdefault(content_hash, []), block_id)
        if hashes:
            self._published_count[request_id] += len(hashes)
            self.epoch += 1

    def advance(self, request_id: int, added_tokens: int = 1) -> None:
        """追加已计算的 output token，填充 partial 或消耗预留分配新块。"""
        ids = self._blocks_for(request_id)
        if added_tokens < 1 or added_tokens > self._output_remaining[request_id]:
            raise ValueError("输出增长超过准入时预留的 output_len")
        for _ in range(added_tokens):
            tokens = self._request_tokens[request_id]
            if tokens % self.block_size_tokens == 0:
                if self._future_reserved[request_id] < 1:
                    raise RuntimeError("输出增长缺少未来 KV 块预留")
                block_id = self._allocate_id()
                self.access_seq += 1
                self.blocks[block_id] = KVBlock(
                    block_id,
                    None,
                    1,
                    "L1",
                    "ACTIVE",
                    1,
                    self.access_seq,
                    self._generations[block_id],
                )
                ids.append(block_id)
                self._future_reserved[request_id] -= 1
                self.reserved_future_blocks -= 1
            else:
                block_id = ids[-1]
                block = self.blocks[block_id]
                if block.ref_count != 1 or block.content_hash is not None:
                    raise RuntimeError("partial KV 块不能跨请求共享或覆盖已发布块")
                self.blocks[block_id] = replace(block, valid_tokens=block.valid_tokens + 1)
            self._request_tokens[request_id] += 1
            self._output_remaining[request_id] -= 1
        self.epoch += 1

    def release(self, request_id: int) -> None:
        """归还未来预留和每处逻辑引用；未发布块及 partial 块直接回收。"""
        ids = self._blocks_for(request_id)
        self.reserved_future_blocks -= self._future_reserved.pop(request_id)
        del self._output_remaining[request_id]
        del self._request_tokens[request_id]
        del self._published_count[request_id]
        del self.request_blocks[request_id]
        for block_id in ids:
            block = self.blocks[block_id]
            if block.ref_count > 1:
                self.blocks[block_id] = replace(block, ref_count=block.ref_count - 1)
            elif block.content_hash is not None and block.valid_tokens == self.block_size_tokens:
                self.blocks[block_id] = replace(block, ref_count=0, state="EVICTABLE")
            else:
                self._remove_block(block_id)
        self.epoch += 1

    def _blocks_for(self, request_id: int) -> list[int]:
        try:
            return self.request_blocks[request_id]
        except KeyError:
            raise KeyError(f"请求 {request_id} 未驻留于 KV cache") from None

    def _check_snapshot_owner(self, snapshot: CacheSnapshot) -> None:
        if snapshot.cache_id != id(self) or snapshot.block_size_tokens != self.block_size_tokens:
            raise ValueError("snapshot 不属于本缓存")

    def _allocate_id(self) -> int:
        if self._free_ids:
            block_id = heappop(self._free_ids)
            self._generations[block_id] += 1
            return block_id
        block_id = self._next_id
        self._next_id += 1
        self._generations[block_id] = 1
        return block_id

    def _remove_block(self, block_id: int) -> None:
        block = self.blocks.pop(block_id)
        if block.content_hash is not None:
            indexed = self.hash_index[block.content_hash]
            indexed.remove(block_id)
            if not indexed:
                del self.hash_index[block.content_hash]
        heappush(self._free_ids, block_id)
