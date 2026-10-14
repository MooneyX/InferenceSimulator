"""F001 调度器：按 L1 页状态逐队头规划 FCFS 准入。"""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from simcore.kv_cache import CacheCapacityError, TieredKVCache
from simcore.schemas import EngineConfig, InferenceRequest

__all__ = [
    "AdmissionBatch",
    "F001Scheduler",
    "PrefillCandidate",
    "RunningRequest",
    "WaitingRequest",
]


@dataclass(frozen=True, slots=True)
class WaitingRequest:
    request: InferenceRequest
    arrival: float
    generated_tokens: int = 0

    @property
    def restoring(self) -> bool:  # 已生成 token 数 → 是否为回撤后恢复的请求
        return self.generated_tokens > 0

    @property
    def target_tokens(self) -> int:  # 输入与已生成输出 → 重建 KV 长度；末个输出尚未写入 KV
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
    hit_sources: tuple[tuple[str, str], ...] = ()

    @property
    def context_after(self) -> int:  # 当前候选的目标长度 → Prefill 后上下文 token 数
        return self.waiting.target_tokens


@dataclass(frozen=True, slots=True)
class AdmissionBatch:
    """已接纳候选仍持有临时引用，执行层须依次调用 cache.admit。"""

    admitted: tuple[PrefillCandidate, ...]
    restored: tuple[WaitingRequest, ...]
    aborted: tuple[WaitingRequest, ...]
    restore_only: tuple[WaitingRequest, ...] = ()
    promote_ids: tuple[str, ...] = ()


class F001Scheduler:
    """按当前 KV 状态逐队头准入；只用生成上限与进度估算，不以 output_len 调度。"""

    def __init__(self, engine: EngineConfig, cache: TieredKVCache) -> None:  # 引擎配置与页池 → 初始化 FCFS 队列和预算；无返回值
        if cache.page_size != engine.page_size or cache.capacity_pages != engine.capacity_pages:
            raise ValueError("L1 页配置与引擎配置不一致")
        if (
            cache.tier_snapshot("L2").capacity_blocks
            != engine.l2_capacity_tokens // engine.page_size
            or cache.tier_snapshot("L3").capacity_blocks
            != engine.l3_capacity_tokens // engine.page_size
        ):
            raise ValueError("低层页配置与引擎配置不一致")
        self._engine = engine
        self._cache = cache
        self._waiting: deque[WaitingRequest] = deque()
        self._queued: set[int] = set()
        self._last_arrival = float("-inf")
        self._ratio = 0.7

    @property
    def num_waiting(self) -> int:  # FCFS 等待队列 → 等待请求数
        return len(self._waiting)

    @property
    def head_request_id(self) -> int | None:  # FCFS 队头 → 请求 ID，空队列返回 None
        return self._waiting[0].request.request_id if self._waiting else None

    def enqueue(
        self, request: InferenceRequest, now_ms: float, *, generated_tokens: int = 0
    ) -> bool:  # 请求、到达时刻及已生成数 → 是否成功排入 FCFS 队列
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

    def requeue(self, request: InferenceRequest, now_ms: float, generated_tokens: int) -> None:  # 回撤请求、步终时刻与输出进度 → 重新按 FCFS 排队；无返回值
        """回撤请求于步终重新等待；不改变尚在计算中到达请求的发送时钟。"""
        if not 1 <= generated_tokens < request.max_new_tokens:
            raise ValueError("回撤请求的生成进度不合法")
        self._insert(WaitingRequest(request, now_ms, generated_tokens))

    def _insert(self, item: WaitingRequest) -> None:  # 待调度请求 → 按到达时刻与 ID 插入队列；无返回值
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
        *,
        is_ready: Callable[[int], bool] | None = None,
        can_reuse: Callable[[int, str, str], bool] | None = None,
    ) -> AdmissionBatch:  # 运行集与就绪/复用条件 → 本步准入、恢复和失败批次
        """逐个保护候选命中页，预算失败不跳过队头。"""
        cache = self._cache
        block_size = cache.page_size
        for request_id, active in running.items():
            if (
                request_id != active.request.request_id
                or not 1 <= active.generated_tokens <= active.request.max_new_tokens
            ):
                raise ValueError("运行请求身份或生成进度不合法")
            cache.state(request_id)
        running_future = sum(
            min(item.request.max_new_tokens - item.generated_tokens, 4096) * self._ratio
            for item in running.values()
        )  # 未来输出只用于预算估算，不预占物理页
        active_count = len(running)
        selected: list[PrefillCandidate] = []
        restored: list[WaitingRequest] = []
        restore_only: list[WaitingRequest] = []
        promotions: set[str] = set()
        aborted: list[WaitingRequest] = []
        current_offset = 0  # 本步新页、增长余量及页提升的容量预算（不预占）
        selected_future = 0  # 候选请求将来输出的预算，不预留物理页
        selected_pages_tokens = 0  # 本批已选 Prefill 的整页计算额度
        tiered = self._engine.l2_capacity_tokens > 0
        while self._waiting and active_count + len(selected) < self._engine.max_running_requests:
            item = self._waiting[0]
            request = item.request
            if is_ready is not None and not is_ready(request.request_id):
                break
            target = item.target_tokens
            sources = cache.probe(request, target, restoring=item.restoring)
            if can_reuse is not None:
                sources = sources[
                    : next(
                        (
                            index
                            for index, (chunk_id, tier) in enumerate(sources)
                            if not can_reuse(request.request_id, chunk_id, tier)
                        ),
                        len(sources),
                    )
                ]
            hit = len(sources) * block_size
            cache.pin(request, hit)
            new_tokens = target - hit
            new_promotions = {chunk_id for chunk_id, tier in sources if tier != "L1"} - promotions
            if item.restoring and new_tokens == 0 and not any(tier != "L1" for _, tier in sources):
                cache.admit(request.request_id)
                self._pop_head()
                restored.append(item)
                active_count += 1
                running_future += (
                    min(request.max_new_tokens - item.generated_tokens, 4096) * self._ratio
                )
                continue
            required_tokens = ((new_tokens + block_size - 1) // block_size) * block_size
            future = min(request.max_new_tokens - item.generated_tokens, 4096)
            proposed_current = (
                current_offset
                + required_tokens
                + block_size  # 为后续增长再留一页
                + len(new_promotions) * block_size  # 低层命中页先提升到 L1
            )
            proposed_total = proposed_current + selected_future + future + running_future
            compute_ok = (
                not selected
                or not new_tokens
                or (
                    required_tokens
                    < self._engine.max_prefill_tokens - selected_pages_tokens
                )
            )
            planned_promotions = tuple(sorted(promotions | new_promotions))
            planned_new_pages = (
                sum(candidate.required_pages for candidate in selected)
                + required_tokens // block_size
            )
            capacity_ok = True
            if tiered:
                try:
                    cache.plan(planned_promotions, planned_new_pages)
                except CacheCapacityError:
                    capacity_ok = False
            future_ok = proposed_total <= cache.available_tokens or not (
                running or selected or restored or restore_only
            )
            if (
                proposed_current > cache.available_tokens
                or not future_ok
                or not compute_ok
                or not capacity_ok
            ):
                cache.cancel_pin(request.request_id)  # 未准入，撤销对命中页的临时保护
                if (
                    not running
                    and not selected
                    and not restored
                    and not restore_only
                    and (
                        (target + block_size - 1) // block_size * block_size + block_size
                        > self._engine.max_total_tokens
                    )
                ):
                    aborted.append(self._pop_head())
                    continue
                break
            if item.restoring and new_tokens == 0:
                restore_only.append(item)
                active_count += 1
                running_future += future * self._ratio
            else:
                selected.append(
                    PrefillCandidate(item, hit, new_tokens, required_tokens // block_size, sources)
                )
                selected_future += future
                selected_pages_tokens += required_tokens
            self._pop_head()
            promotions.update(new_promotions)
            current_offset = proposed_current
        return AdmissionBatch(
            tuple(selected),
            tuple(restored),
            tuple(aborted),
            tuple(sorted(restore_only, key=lambda item: item.request.request_id)),
            tuple(sorted(promotions)),
        )

    def rollback_selection(self, selection: AdmissionBatch) -> None:  # 已选准入批次 → 取消保护并还原 FCFS 队列；无返回值
        """回载容量不足时撤销候选引用，并按原 FCFS 顺序放回队列。"""
        for candidate in selection.admitted:
            self._cache.cancel_pin(candidate.waiting.request.request_id)
            self._insert(candidate.waiting)
        for item in selection.restore_only:
            self._cache.cancel_pin(item.request.request_id)
            self._insert(item)

    def _pop_head(self) -> WaitingRequest:  # FCFS 队列 → 移除并返回队头请求
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
    ) -> float:  # 运行集和解码/回撤/空闲标志 → 更新后的未来输出预留比例
        """每步完成后更新未来估算比例；纯 Prefill 不衰减。"""
        if idle and not running:
            self._ratio = 0.7
        elif withdrew and running:
            generated = sum(item.generated_tokens for item in running.values())
            limits = sum(item.request.max_new_tokens for item in running.values())
            # 根据回撤后仍运行请求的生成进度重新估计未来输出预留比例。
            self._ratio = min(1.0, (generated + 20 * len(running)) / (limits + 1))
        elif decoded and not withdrew:
            self._ratio = max(0.098, self._ratio - (0.7 - 0.098) / 600)  # 稳定解码时逐步放宽
        return self._ratio
