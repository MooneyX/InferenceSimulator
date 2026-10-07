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
    L1KVCache,
    PrefixProbe,
)
from simcore.perf_model import step_time
from simcore.prefix_hash import logical_chunk_id
from simcore.schemas import EngineConfig, InferenceRequest, PerfModelConfig

__all__ = [
    "AdmissionBatch",
    "CachedStepPlan",
    "F001Scheduler",
    "PrefillCandidate",
    "PrefixScheduler",
    "RunningRequest",
    "WaitingRequest",
]


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


@dataclass(frozen=True, slots=True)
class WaitingRequest:
    request: InferenceRequest
    arrival: float
    generated_tokens: int = 0

    @property
    def restoring(self) -> bool:
        return self.generated_tokens > 0

    @property
    def target_tokens(self) -> int:
        return self.request.prompt_len + max(self.generated_tokens - 1, 0)


@dataclass(frozen=True, slots=True)
class RunningRequest:
    request: InferenceRequest
    generated_tokens: int


@dataclass(frozen=True, slots=True)
class PrefillCandidate:
    waiting: WaitingRequest
    reused_tokens: int
    new_tokens: int
    required_pages: int

    @property
    def context_after(self) -> int:
        return self.waiting.target_tokens


@dataclass(frozen=True, slots=True)
class AdmissionBatch:
    """已接纳候选仍持有临时引用，执行层须依次调用 cache.admit。"""

    admitted: tuple[PrefillCandidate, ...]
    restored: tuple[WaitingRequest, ...]
    aborted: tuple[WaitingRequest, ...]
    blocked: bool
    decode_new_tokens: int
    current_offset: int
    total_offset: float


class F001Scheduler:
    """按当前 KV 状态逐队头准入；预算只估算未来页，不预分配。"""

    def __init__(self, engine: EngineConfig, cache: L1KVCache) -> None:
        if cache.page_size != engine.page_size or cache.capacity_pages != engine.capacity_pages:
            raise ValueError("L1 页配置与引擎配置不一致")
        self._engine = engine
        self._cache = cache
        self._waiting: deque[WaitingRequest] = deque()
        self._queued: set[int] = set()
        self._last_arrival = float("-inf")
        self._ratio = 0.7

    @property
    def ratio(self) -> float:
        return self._ratio

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    def enqueue(
        self, request: InferenceRequest, now_ms: float, *, generated_tokens: int = 0
    ) -> bool:
        """返回 False 表示 M<1，调用方须在发送时立即记录请求失败。"""
        if now_ms < self._last_arrival:
            raise ValueError("FCFS 请求发送时刻不得倒退")
        if generated_tokens < 0 or generated_tokens > request.max_new_tokens:
            raise ValueError("已生成 token 数不合法")
        if request.request_id in self._queued:
            raise ValueError("请求已在等待队列中")
        self._last_arrival = now_ms
        if request.max_new_tokens < 1:
            return False
        self._insert(WaitingRequest(request, now_ms, generated_tokens))
        return True

    def requeue(self, request: InferenceRequest, now_ms: float, generated_tokens: int) -> None:
        """回撤请求于步终重新等待；不改变尚在计算中到达请求的发送时钟。"""
        if not 1 <= generated_tokens < request.max_new_tokens:
            raise ValueError("回撤请求的生成进度不合法")
        self._insert(WaitingRequest(request, now_ms, generated_tokens))

    def _insert(self, item: WaitingRequest) -> None:
        key = item.request.request_id
        if key in self._queued:
            raise ValueError("请求已在等待队列中")
        index = len(self._waiting)
        while index:
            previous = self._waiting[index - 1]
            if (previous.arrival, previous.request.request_id) <= (item.arrival, key):
                break
            index -= 1
        self._waiting.insert(index, item)
        self._queued.add(key)

    def select_prefill(
        self,
        running: Mapping[int, RunningRequest],
        decode_ids: tuple[int, ...] = (),
    ) -> AdmissionBatch:
        """冻结本步 D，依次临时 pin 候选，预算失败撤销且不跳过队头。"""
        if len(set(decode_ids)) != len(decode_ids) or not set(decode_ids) <= running.keys():
            raise ValueError("Decode 集合必须是步首运行请求的不重复子集")
        if self._engine.pd_mutually_exclusive and decode_ids:
            raise ValueError("互斥模式不得同时规划 Prefill 和 Decode")
        cache = self._cache
        block_size = cache.page_size
        for request_id, active in running.items():
            if (
                request_id != active.request.request_id
                or not 1 <= active.generated_tokens <= active.request.max_new_tokens
            ):
                raise ValueError("运行请求身份或生成进度不合法")
            cache.state(request_id)
        decode_new = block_size * sum(cache.state(key).k % block_size == 0 for key in decode_ids)
        running_future = sum(
            min(max(item.request.max_new_tokens - item.generated_tokens, 0), 4096) * self._ratio
            for item in running.values()
        )
        active_count = len(running)
        selected: list[PrefillCandidate] = []
        restored: list[WaitingRequest] = []
        aborted: list[WaitingRequest] = []
        current_offset = decode_new
        selected_future = 0
        selected_pages_tokens = 0
        blocked = False
        while self._waiting and active_count + len(selected) < self._engine.max_running_requests:
            item = self._waiting[0]
            request = item.request
            target = item.target_tokens
            hit = cache.probe(request, target, restoring=item.restoring)
            cache.pin(request, hit)
            new_tokens = target - hit
            if item.restoring and new_tokens == 0:
                cache.admit(request.request_id)
                self._pop_head()
                restored.append(item)
                active_count += 1
                running_future += (
                    min(max(request.max_new_tokens - item.generated_tokens, 0), 4096) * self._ratio
                )
                continue
            required_tokens = ((new_tokens + block_size - 1) // block_size) * block_size
            future = min(max(request.max_new_tokens - item.generated_tokens, 0), 4096)
            proposed_current = current_offset + required_tokens + block_size
            proposed_total = proposed_current + selected_future + future + running_future
            compute_ok = not selected or (
                required_tokens
                < self._engine.max_prefill_tokens - len(decode_ids) - selected_pages_tokens
            )
            if (
                proposed_current > cache.available_tokens
                or proposed_total > cache.available_tokens
                or not compute_ok
            ):
                cache.cancel_pin(request.request_id)
                if (
                    not running
                    and not selected
                    and not restored
                    and not decode_ids
                    and (
                        (target + block_size - 1) // block_size * block_size + block_size + future
                        > self._engine.max_total_tokens
                    )
                ):
                    aborted.append(self._pop_head())
                    continue
                blocked = True
                break
            selected.append(PrefillCandidate(item, hit, new_tokens, required_tokens // block_size))
            self._pop_head()
            current_offset = proposed_current
            selected_future += future
            selected_pages_tokens += required_tokens
        if self._waiting and active_count + len(selected) >= self._engine.max_running_requests:
            blocked = True
        return AdmissionBatch(
            tuple(selected),
            tuple(restored),
            tuple(aborted),
            blocked,
            decode_new,
            current_offset,
            current_offset + selected_future + running_future,
        )

    def _pop_head(self) -> WaitingRequest:
        item = self._waiting.popleft()
        self._queued.remove(item.request.request_id)
        return item

    def advance_ratio(
        self,
        running: Mapping[int, RunningRequest],
        *,
        decoded: bool = False,
        withdrew: bool = False,
        idle: bool = False,
    ) -> float:
        """每步完成后更新未来估算比例；纯 Prefill 不衰减。"""
        if idle and not running:
            self._ratio = 0.7
        elif withdrew and running:
            generated = sum(item.generated_tokens for item in running.values())
            limits = sum(item.request.max_new_tokens for item in running.values())
            self._ratio = min(1.0, (generated + 20 * len(running)) / (limits + 1))
        elif decoded and not withdrew:
            self._ratio = max(0.098, self._ratio - (0.7 - 0.098) / 600)
        return self._ratio
