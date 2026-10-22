"""F001 调度器：按到达顺序考虑候选，跳过暂未就绪的请求。"""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from simcore.kv_cache import TieredKVCache
from simcore.schemas import EngineConfig, InferenceRequest

__all__ = [
    "AdmissionBatch",
    "AdmissionScheduler",
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


class AdmissionScheduler:
    """按队列顺序尝试就绪请求；只用生成上限与进度估算，不以 output_len 调度。"""

    def __init__(self, engine: EngineConfig, cache: TieredKVCache) -> None:
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
    def num_waiting(self) -> int:  # 等待队列 → 等待请求数
        return len(self._waiting)

    @property
    def head_request_id(self) -> int | None:  # 队列首项 → 请求 ID，空队列返回 None
        return self._waiting[0].request.request_id if self._waiting else None

    def has_ready(self, is_ready: Callable[[int], bool]) -> bool:
        return any(is_ready(item.request.request_id) for item in self._waiting)

    def enqueue(
        self, request: InferenceRequest, now_ms: float, *, generated_tokens: int = 0
    ) -> bool:  # 请求、到达时刻及已生成数 → 是否成功排入等待队列
        """返回 False 表示 M<1，调用方须在发送时立即记录请求失败。"""
        if now_ms < self._last_arrival:
            raise ValueError("请求发送时刻不得倒退")
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
        *,
        is_ready: Callable[[int], bool] | None = None,
        can_reuse: Callable[[int, str, str], bool] | None = None,
    ) -> AdmissionBatch:  # 运行集与就绪/复用条件 → 本步准入、恢复和失败批次
        """按队列顺序选批；跳过等待预取的请求，预算失败则停止选批。"""
        cache = self._cache
        block_size = cache.page_size
        for request_id, active in running.items():
            if (
                request_id != active.request.request_id
                or not 1 <= active.generated_tokens <= active.request.max_new_tokens
            ):
                raise ValueError("运行请求身份或生成进度不合法")
            _ = cache.state(request_id) # 调用state函数, 尝试取出状态信息, 若没有则会报错
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
        for item in tuple(self._waiting): # 遍历等待队列中的所有请求
            if active_count + len(selected) >= self._engine.max_running_requests: # 同时运行请求数限制
                break
            request = item.request
            if is_ready is not None and not is_ready(request.request_id): # 请求未就绪(prefetch未完成, 或者本地命中率比到达时更低, 原因是排队期间有KV被驱逐)
                continue
            target = item.target_tokens
            sources = cache.probe(request, target, restoring=item.restoring)
            if can_reuse is not None:
                sources = sources[
                    : next(
                        (
                            index
                            for index, (chunk_id, tier) in enumerate(sources)
                            if not can_reuse(request.request_id, chunk_id, tier)
                            # 排除位于L3或者之前因为传输失败而被标记为不可复用的块, 因此上面的probe可以改用忽略L3, 可以复用上次probe的结果
                            # "传输失败就标记该页无法使用"的机制需要修改
                        ),
                        len(sources),
                    )
                ]
            hit = len(sources) * block_size
            cache.pin(request, sources)
            new_tokens = target - hit
            new_promotions = {chunk_id for chunk_id, tier in sources if tier != "L1"} - promotions
            if item.restoring and new_tokens == 0 and not any(tier != "L1" for _, tier in sources): # 回撤后恢复的请求, 所有KV都可复用且在L1
                cache.admit(request.request_id) # 直接准入
                self._pop_item(item) # 从等待队列中移除
                restored.append(item)
                active_count += 1 # 活跃请求数
                running_future += (
                    min(request.max_new_tokens - item.generated_tokens, 4096) * self._ratio
                ) # 该请求的未来输出预算
                continue
            required_tokens = ((new_tokens + block_size - 1) // block_size) * block_size # 该请求重算KV所需的空间(单位为token)
            future = min(request.max_new_tokens - item.generated_tokens, 4096)
            proposed_current = (
                current_offset
                + required_tokens
                + block_size  # 为后续增长再留一页
                + len(new_promotions) * block_size  # 低层命中页先提升到 L1
            )
            proposed_total = proposed_current + selected_future + future + running_future
            compute_ok = (
                not selected # 当前没有选中的请求
                or not new_tokens # 当前请求不需要重算KV
                or ( # 或者预算还足够
                    required_tokens
                    < self._engine.max_prefill_tokens - selected_pages_tokens
                ) # 预算不足时整批只会准入这一个超预算请求, 和sglang的chunked prefill不一致
            )# 预算判断时要求严格小于, 也就是总token数<=预算 - 1, 但是reuired_tokens为0时还是可以准入, 即使总token数==预算, 和sglang中的处理一致
            # 当前page和默认max_prefill_tokens设定下不可能恰好剩余1个token的预算, 但是后续cunked prefill可能出现这种情况
            future_ok = proposed_total <= cache.available_tokens or not (
                running or selected or restored or restore_only
            )
            if (
                proposed_current > cache.available_tokens
                or not future_ok
                or not compute_ok
            ): # 容量不足, 这个请求无法准入, 结束这次组批, 将已选入的请求提交计算
                cache.cancel_pin(request.request_id)  # 未准入，撤销对命中页的临时保护
                if (
                    not running
                    and not selected
                    and not restored
                    and not restore_only
                    and (
                        (target + block_size - 1) // block_size * block_size + block_size
                        > self._engine.max_total_tokens # 若整个L1都无法容纳目标长度的KV, abort掉(sglang中的这个判断发生在请求到达时, 需要修复)
                    )
                ):
                    aborted.append(self._pop_item(item))
                    continue
                break
            if item.restoring and new_tokens == 0: # 回撤后恢复的请求, 所有KV都可复用, 若全在L1, 则在185行就直接准入了, 此处表示还有KV处于L2
                restore_only.append(item)
                active_count += 1
                running_future += future * self._ratio
            else: # 非回撤恢复请求, 即新到达的请求; 或者回撤恢复但是KV不全的请求
                selected.append(
                    PrefillCandidate(item, hit, new_tokens, required_tokens // block_size, sources)
                )
                selected_future += future
                selected_pages_tokens += required_tokens
            self._pop_item(item)
            promotions.update(new_promotions)
            current_offset = proposed_current
        return AdmissionBatch(
            tuple(selected),
            tuple(restored),
            tuple(aborted),
            tuple(sorted(restore_only, key=lambda item: item.request.request_id)),
            tuple(sorted(promotions)),
        )

    def rollback_selection(self, selection: AdmissionBatch) -> None:
        """回载容量不足时撤销候选引用，并按原到达顺序放回队列。"""
        for candidate in selection.admitted:
            self._cache.cancel_pin(candidate.waiting.request.request_id)
            self._insert(candidate.waiting)
        for item in selection.restore_only:
            self._cache.cancel_pin(item.request.request_id)
            self._insert(item)

    def _pop_item(self, item: WaitingRequest) -> WaitingRequest:
        self._waiting.remove(item)
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
