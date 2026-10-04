"""PrefixScheduler 基于 L1 快照原子准入 FCFS 队头，并计算每个引擎步的时长。"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from simcore.kv_cache import (
    AdmissionAllocation,
    BatchCacheResult,
    CacheCapacityError,
    CacheOOMError,
    CacheSnapshot,
    HierarchicalKVCache,
    PrefixProbe,
)
from simcore.perf_model import step_time
from simcore.prefix_hash import logical_chunk_id
from simcore.schemas import EngineConfig, InferenceRequest, PerfModelConfig

__all__ = ["CachedStepPlan", "PrefixScheduler"]


@dataclass(frozen=True, slots=True)
class CachedStepPlan:
    """同一快照上原子准入的队头连续段及实际 Prefill 工作量。"""

    admitted: tuple[InferenceRequest, ...]
    probes: tuple[PrefixProbe, ...]
    cache_result: BatchCacheResult | None
    cache_before: CacheSnapshot | None
    cache_after: CacheSnapshot | None
    num_decoding: int
    batch_total_tokens: int
    duration: float

    @property
    def request_results(self) -> Mapping[int, tuple[int, int, int]]:
        """request_id -> (matched, reused, new_prefill) token 数。"""
        if self.cache_result is None:
            return {}
        return {
            result.request_id: (
                result.matched_cache_tokens,
                result.actual_hit_tokens,
                request.prompt_len - result.actual_hit_tokens,
            )
            for request, result in zip(self.admitted, self.cache_result.requests, strict=True)
        }


class PrefixScheduler:
    """有缓存的 FCFS：同一步所有候选基于同一快照，成功才统一提交。"""

    def __init__(
        self,
        engine: EngineConfig,
        perf: PerfModelConfig,
        cache: HierarchicalKVCache,
        *,
        record_trace: bool = True,
    ) -> None:
        if engine.prefix_cache.l1.capacity_blocks != cache.capacity_blocks:
            raise ValueError("L1 块容量与引擎配置不一致")
        if engine.prefix_cache.block_size_tokens != cache.block_size_tokens:
            raise ValueError("L1 block_size_tokens 与引擎配置不一致")
        self._max_batch_size = engine.max_batch_size
        self._perf = perf
        self._cache = cache
        self._record_trace = record_trace
        self._blocked_on_release = False
        self._waiting: deque[InferenceRequest] = deque()
        self._last_enqueued_time = float("-inf")

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    def enqueue(self, request: InferenceRequest, now_ms: float) -> None:
        if now_ms < self._last_enqueued_time:
            raise ValueError("FCFS 的请求提交时间不得倒退")
        self._last_enqueued_time = now_ms
        self._waiting.append(request)

    def on_request_released(self) -> None:
        """真实物理引用归还后重新探测 FCFS 队头。"""
        self._blocked_on_release = False

    def plan_step(self) -> CachedStepPlan | None:
        cache = self._cache
        num_decoding = len(cache.request_blocks)
        if not num_decoding and not self._waiting:
            return None
        capacity = self._max_batch_size - num_decoding
        can_admit = bool(self._waiting) and capacity > 0 and not self._blocked_on_release
        before = cache.snapshot() if (self._record_trace or can_admit) else None
        selected: list[InferenceRequest] = []
        probes: list[PrefixProbe] = []
        allocations: list[AdmissionAllocation] = []
        transaction = None
        candidates = list(self._waiting)[:capacity] if can_admit else ()
        for request in candidates:
            assert before is not None
            input_chunk_ids = tuple(
                logical_chunk_id(request.request_id, index, request.inherited_prefix_ids)
                for index in range(request.prompt_len // cache.block_size_tokens)
            )
            probe = cache.probe(before, input_chunk_ids)
            allocation = AdmissionAllocation(
                request.request_id,
                request.prompt_len,
                request.max_new_tokens,
                input_chunk_ids,
            )
            try:
                proposal = cache.prepare_batch(before, (*probes, probe), (*allocations, allocation))
            except CacheCapacityError as exc:
                if isinstance(exc, CacheOOMError):
                    raise
                if not selected and not num_decoding:
                    raise RuntimeError("空 L1 无法准入可执行请求，容量规划不一致") from exc
                self._blocked_on_release = True
                break
            selected.append(request)
            probes.append(probe)
            allocations.append(allocation)
            transaction = proposal
        result = cache.commit(transaction) if transaction is not None else None
        for _ in selected:
            self._waiting.popleft()
        after = cache.snapshot() if self._record_trace else None
        reused = [entry.actual_hit_tokens for entry in result.requests] if result else []
        new_prefill = [
            request.prompt_len - hit for request, hit in zip(selected, reused, strict=True)
        ]
        logical_tokens = cache.logical_context_tokens
        duration = step_time(
            new_prefill,
            logical_tokens,
            self._perf,
            has_decode=num_decoding > 0,
            prefill_contexts=reused,
        )
        return CachedStepPlan(
            tuple(selected),
            tuple(probes),
            result,
            before,
            after,
            num_decoding,
            logical_tokens,
            duration,
        )
