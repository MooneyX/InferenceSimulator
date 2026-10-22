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
    # 回载按候选准入顺序、候选内部上下文顺序提交；块 ID 只作身份，不作排序键。
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
        self._blocked: set[int] = set()
        self._ready_count: int = 0
        self._last_arrival = float("-inf")
        self._ratio = 0.7

    @property
    def num_waiting(self) -> int:  # 等待队列 → 等待请求数
        return len(self._waiting)

    @property
    def head_request_id(self) -> int | None:  # 队列首项 → 请求 ID，空队列返回 None
        return self._waiting[0].request.request_id if self._waiting else None

    @property
    def ready_count(self) -> int:
        """等待队列中不受请求级查询、预取阻塞的请求数，不代表预算可准入。"""
        return self._ready_count

    def has_ready(self) -> bool:
        return self._ready_count > 0

    def set_ready(self, request_id: int, ready: bool) -> None:
        """更新已入队请求的就绪状态；重复通知不重复计数。"""
        if ready:
            if request_id in self._blocked:
                self._blocked.remove(request_id)
                self._ready_count += 1
        elif request_id not in self._blocked:
            self._blocked.add(request_id)
            self._ready_count -= 1

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
        self._ready_count += 1

    def select_prefill(
        self,
        running: Mapping[int, RunningRequest],
        *,
        accept_probe: Callable[[WaitingRequest, tuple[tuple[str, str], ...]], bool] | None = None,
    ) -> AdmissionBatch:  # 运行集与就绪条件 → 本步准入、恢复和失败批次
        """按队列顺序选批；跳过等待预取的请求，预算失败则停止选批。"""
        cache = self._cache
        block_size = cache.page_size
        for request_id, active in running.items():
            if (
                request_id != active.request.request_id
                or not 1 <= active.generated_tokens <= active.request.max_new_tokens
            ):
                raise ValueError("运行请求身份或生成进度不合法")
            _ = cache.state(request_id)  # 调用state函数, 尝试取出状态信息, 若没有则会报错
        active_count = len(running)
        if active_count >= self._engine.max_running_requests or not self._ready_count:
            return AdmissionBatch((), (), ())
        running_future = sum(
            min(item.request.max_new_tokens - item.generated_tokens, 4096) * self._ratio
            for item in running.values()
        )  # 未来输出只用于预算估算，不预占物理页
        selected: list[PrefillCandidate] = []
        restored: list[WaitingRequest] = []
        restore_only: list[WaitingRequest] = []
        # 列表保留提交顺序，集合只负责跨候选共享页去重和唯一页容量预算。
        promotions: list[str] = []
        promoted: set[str] = set()
        aborted: list[WaitingRequest] = []
        current_offset = 0  # 本步新页、增长余量及页提升的容量预算（不预占）
        selected_future = 0  # 候选请求将来输出的预算，不预留物理页
        selected_pages_tokens = 0  # 本批已选 Prefill 的整页计算额度
        skipped: list[WaitingRequest] = []
        pending_item: WaitingRequest | None = None
        try:
            while self._waiting and self._ready_count:
                if (
                    active_count + len(selected) >= self._engine.max_running_requests
                ):  # 同时运行请求数限制
                    break
                item = self._waiting.popleft()
                pending_item = item
                request = item.request
                if request.request_id in self._blocked:
                    skipped.append(item)
                    pending_item = None
                    continue
                target = item.target_tokens
                sources = cache.probe(request, target, restoring=item.restoring, include_l3=False)
                if accept_probe is not None and not accept_probe(item, sources):
                    self.set_ready(request.request_id, False)
                    skipped.append(item)
                    pending_item = None
                    continue
                hit = len(sources) * block_size
                cache.pin(request, sources)
                new_tokens = target - hit
                new_promotions = [
                    chunk_id
                    for chunk_id, tier in sources
                    if tier != "L1" and chunk_id not in promoted
                ]
                direct_restore = (
                    item.restoring and new_tokens == 0 and all(tier == "L1" for _, tier in sources)
                )
                required_tokens = (
                    (new_tokens + block_size - 1) // block_size
                ) * block_size  # 该请求重算KV所需的空间(单位为token)
                future = min(request.max_new_tokens - item.generated_tokens, 4096)
                proposed_current = (
                    current_offset
                    + required_tokens
                    + (0 if direct_restore else block_size)
                    + len(new_promotions) * block_size  # 低层命中页先提升到 L1
                )
                proposed_total = proposed_current + selected_future + future + running_future
                compute_ok = (
                    not selected  # 当前没有选中的请求
                    or not new_tokens  # 当前请求不需要重算KV
                    or (  # 或者预算还足够
                        required_tokens < self._engine.max_prefill_tokens - selected_pages_tokens
                    )  # 预算不足时整批只会准入这一个超预算请求, 和sglang的chunked prefill不一致
                )  # 预算判断时要求严格小于, 也就是总token数<=预算 - 1,
                # 但是reuired_tokens为0时还是可以准入, 即使总token数==预算,
                # 和sglang中的处理一致
                # 当前page和默认max_prefill_tokens设定下不可能恰好剩余1个token的预算,
                # 但是后续cunked prefill可能出现这种情况
                future_ok = (
                    direct_restore
                    or proposed_total <= cache.available_tokens
                    or not (running or selected or restored or restore_only)
                )
                if (
                    proposed_current > cache.available_tokens or not future_ok or not compute_ok
                ):  # 容量不足, 这个请求无法准入, 结束这次组批, 将已选入的请求提交计算
                    cache.cancel_pin(request.request_id)  # 未准入，撤销对命中页的临时保护
                    if (
                        not running
                        and not selected
                        and not restored
                        and not restore_only
                        and (
                            # 若整个L1都无法容纳目标长度的KV, abort掉
                            # (sglang中的这个判断发生在请求到达时, 需要修复)
                            (target + block_size - 1) // block_size * block_size + block_size
                            > self._engine.max_total_tokens
                        )
                    ):
                        self._queued.remove(request.request_id)
                        self._ready_count -= 1
                        pending_item = None
                        aborted.append(item)
                        continue
                    break
                if item.restoring and new_tokens == 0:
                    if direct_restore:
                        cache.admit(request.request_id)
                        restored.append(item)
                    else:
                        restore_only.append(item)
                    active_count += 1
                    running_future += future * self._ratio
                else:  # 非回撤恢复请求, 即新到达的请求; 或者回撤恢复但是KV不全的请求
                    selected.append(
                        PrefillCandidate(
                            item, hit, new_tokens, required_tokens // block_size, sources
                        )
                    )
                    selected_future += future
                    selected_pages_tokens += required_tokens
                self._queued.remove(request.request_id)
                self._ready_count -= 1
                pending_item = None
                # 预算通过后才提交候选贡献的页，失败候选不能污染去重集合或传输顺序。
                promotions.extend(new_promotions)
                promoted.update(new_promotions)
                current_offset = proposed_current
        finally:
            if pending_item is not None:
                self._waiting.appendleft(pending_item)
            self._waiting.extendleft(reversed(skipped))
        return AdmissionBatch(
            tuple(selected),
            tuple(restored),
            tuple(aborted),
            tuple(sorted(restore_only, key=lambda item: item.request.request_id)),
            tuple(promotions),
        )

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
