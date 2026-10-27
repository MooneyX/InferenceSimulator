"""单引擎执行器：请求通过 F001 调度器与 L1 页池顺序执行。

事件堆、时钟推进与客户端回调由 ``Coordinator`` 独占。
"""

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Literal

from simcore.events import EventKind
from simcore.kv_cache import CacheCapacityError, CachePool, L1Snapshot, TieredKVCache, TierSnapshot
from simcore.perf_model import DecodeWork, PrefillWork, StepWorkload, step_cost
from simcore.runtime import RuntimeConfig
from simcore.scheduler import AdmissionBatch, AdmissionScheduler, RunningRequest, WaitingRequest
from simcore.schemas import (
    AbortedRequest,
    AnalyticalSummary,
    Arrival,
    InferenceRequest,
    RequestRecord,
    SkippedRequest,
)

__all__ = ["Engine", "EngineRun"]


@dataclass(frozen=True, slots=True)
class EngineRun:
    """单场仿真的原始产出。

    Attributes:
        records: 每请求一条区间记录，按 ``request_id`` 升序。
        warnings: 告警文本，按发生时刻排序。
        num_engine_steps: 执行的引擎步数。
        sim_start: 首请求到达时刻（ms）；无请求时为 0。
        sim_end: 最晚请求完成或失败时刻（ms）；无请求时为 0。
        cache_snapshot: 仿真结束时的物理 L1 容量快照。
    """

    records: tuple[RequestRecord, ...]
    warnings: tuple[str, ...]
    num_engine_steps: int
    sim_start: float
    sim_end: float
    cache_snapshot: L1Snapshot
    l2_snapshot: TierSnapshot
    l3_snapshot: TierSnapshot
    l1_evictions: int = 0
    l1_peak_resident_blocks: int = 0
    aborted_requests: tuple[AbortedRequest, ...] = ()
    skipped_requests: tuple[SkippedRequest, ...] = ()
    replayed_prefill_tokens: int = 0
    promotions: int = 0
    demotions: int = 0
    restore_bytes: int = 0
    restore_time_ms: float = 0.0
    analysis: AnalyticalSummary = field(default_factory=AnalyticalSummary)


@dataclass(slots=True)
class _TransferTask:
    chunk_id: str
    waiting_requests: set[int]
    service_ms: float
    owns_transfer: bool = True


class Engine:
    """维护即时请求与调度/KV 状态，不预存未来请求或持有事件队列。"""

    def __init__(
        self, runtime: RuntimeConfig, *, shared_l3: CachePool | None = None,
        shared_l2: CachePool | None = None, instance_id: int = 0
    ) -> None:
        engine, perf = runtime.engine, runtime.perf_model
        self._links = runtime.links
        self._analytical = runtime.analytical
        self._analysis = AnalyticalSummary()
        self._engine_config = engine
        self._perf = perf
        self._prefix_cache = TieredKVCache(
            engine.max_total_tokens,
            engine.page_size,
            engine.l2_capacity_tokens,
            engine.l3_capacity_tokens,
            shared_l3=shared_l3,
            shared_l2=shared_l2,
            pool_id_base=instance_id * 2,
        )
        self._prefix_scheduler = AdmissionScheduler(engine, self._prefix_cache)
        self._admission_stats: dict[int, tuple[int, int]] = {}
        self._admission_tiers: dict[
            int, tuple[int, int, int]
        ] = {}  # 请求 ID → 首次准入时 L1/L2/L3 命中 token 数
        self._restore_bytes = 0
        self._restore_time_ms = 0.0
        self._requests: dict[int, InferenceRequest] = {}
        self._arrivals: dict[int, Arrival] = {}
        self._arrival_times: dict[int, float] = {}
        self._aborted: dict[int, AbortedRequest] = {}
        self._recent_aborts: list[int] = []
        self._skipped: dict[int, SkippedRequest] = {}
        self._retry_counts: dict[int, int] = {}
        self._replayed_tokens: dict[int, int] = {}
        # request_id -> 已采样输出数；Decode 只处理步首已有请求
        self._running: dict[int, int] = {}
        self._pending_done: dict[int, float] = {}
        self._prefill_start: dict[int, float] = {}
        self._prefill_end: dict[int, float] = {}
        self._finish: dict[int, float] = {}
        self._num_steps = 0
        self._lookup_waiting: set[int] = set()
        self._last_local_match: dict[int, int] = {}
        self._prefetch_waiting: dict[int, set[int]] = {}
        self._prefetched_origins: dict[int, set[str]] = {}
        self._pending_selection: AdmissionBatch | None = None  # 等待 L2 回载后执行的准入批次
        self._pending_selection_prompt_tokens: int = 0
        self._pending_requeue_prompt_tokens: int = 0
        self._pending_loads: set[str] = set()  # 当前批次尚未回载到 L1 的页 ID
        self._pending_publish: tuple[int, ...] = ()
        self._pending_finished: tuple[int, ...] = ()
        self._scheduled_events: list[tuple[float, EventKind, int]] = []
        self._transfers: dict[int, _TransferTask] = {}
        self._transfer_by_chunk: dict[tuple[str, str], int] = {}
        self._next_transfer_id = 0
        self._link_free = {"L2": 0.0, "L1": 0.0}

    @property
    def has_work(self) -> bool:  # 等待队列或运行集 → 是否仍有可执行请求
        """是否有等待或执行中的请求（不含等待完成结算的请求）。"""
        return bool(self._running or self._prefix_scheduler.num_waiting)

    @property
    def load_tokens(self) -> int:
        """即时物理 L1 占用加逻辑等待输入量；不读取预推进的生成结果。"""
        return (
            self._prefix_cache.active_pages * self._prefix_cache.page_size
            + self._prefix_scheduler.waiting_prompt_tokens
            + self._pending_selection_prompt_tokens
            - self._pending_requeue_prompt_tokens
        )

    @property
    def num_steps(self) -> int:  # 无显式输入 → 累计 GPU 步数
        return self._num_steps

    @property
    def can_step(self) -> bool:  # 当前运行、选批与回载状态 → 是否可安排下一步
        if self._running:
            return True
        if self._pending_selection is not None or self._pending_loads:
            return self._pending_selection is not None and not self._pending_loads
        return self._prefix_scheduler.has_ready()

    def drain_events(
        self,
    ) -> tuple[
        tuple[float, EventKind, int], ...
    ]:  # 待派发事件列表 → 时刻、类型、引用元组并清空队列
        events = tuple(self._scheduled_events)
        self._scheduled_events.clear()
        return events

    def _start_transfer(
        self, chunk_id: str, destination: str, now_ms: float, request_id: int, *, source: str
    ) -> int:  # 页 ID、目的层、时刻及请求 ID → 去重后的传输任务 ID
        existing = self._transfer_by_chunk.get((chunk_id, destination))
        if existing is not None:  # 查询当前id是否正在被传输
            self._transfers[existing].waiting_requests.add(
                request_id
            )  # transfer_id -> 等待该页的请求id
            return existing
        target = self._prefix_cache.pools[destination]
        end = target.incoming_ready_ms.get(chunk_id) if destination == "L2" else None
        owns_transfer = end is None
        service_ms = 0.0
        if end is None:
            link = self._links.l3_to_l2 if destination == "L2" else self._links.l2_to_l1
            if link is None:
                raise ValueError("传输链路须配置有效带宽和 KV 字节数")
            self._prefix_cache.reserve_transfer(chunk_id, destination, source=source)
            service_ms = link.latency_ms + link.transfer_ms
            start = max(now_ms, self._link_free[destination])
            end = start + service_ms
            self._link_free[destination] = end
            if destination == "L2":
                target.incoming_ready_ms[chunk_id] = end
        transfer_id = self._next_transfer_id
        self._next_transfer_id += 1
        self._transfers[transfer_id] = _TransferTask(
            chunk_id, {request_id}, service_ms, owns_transfer
        )
        self._transfer_by_chunk[chunk_id, destination] = transfer_id
        kind = EventKind.L3_L2_DONE if destination == "L2" else EventKind.L2_L1_DONE
        self._scheduled_events.append((end, kind, transfer_id))
        return transfer_id

    def submit(
        self, request: InferenceRequest, now_ms: float
    ) -> bool:  # 推理请求与发送时刻 → 是否成功入队
        """校验输出限制并按 FCFS 入队；不按 KV 余量截短实际输出目标。"""
        key = request.request_id
        if key in self._requests:
            raise ValueError(f"request_id 重复: {key}")
        output_len = request.output_len
        if request.is_trace and output_len == 0:
            self._requests[key] = request
            self._arrival_times[key] = now_ms
            self._abort(key, now_ms, "invalid_trace_output")
            return False
        if output_len > self._engine_config.max_output_tokens:
            self._requests[key] = request
            self._arrival_times[key] = now_ms
            self._abort(key, now_ms, "output_exceeds_engine_limit")
            return False
        if not 1 <= request.max_new_tokens <= self._engine_config.max_output_tokens:
            raise ValueError("请求生成上限超出引擎输出限制")
        if not 1 <= output_len <= request.max_new_tokens:
            raise ValueError("请求实际输出长度与可见上限不一致")
        self._requests[key] = request
        self._arrival_times[key] = now_ms
        if not self._prefix_scheduler.enqueue(request, now_ms):
            self._abort(key, now_ms, "input_exceeds_kv_capacity")
            return False
        self._arrivals[key] = Arrival(
            request_id=key,
            sim_time=now_ms,
            prompt_len=request.prompt_len,
            output_len=output_len,
            max_new_tokens=request.max_new_tokens,
        )
        if self._engine_config.l3_capacity_tokens:
            local = self._prefix_cache.probe(request, request.prompt_len, include_l3=False)
            self._last_local_match[key] = len(local)
            if len(local) < (request.prompt_len - 1) // self._prefix_cache.page_size:
                self._schedule_l3_lookup(key, now_ms)
        return True

    def _schedule_l3_lookup(self, request_id: int, now_ms: float) -> None:
        self._lookup_waiting.add(request_id)
        self._prefix_scheduler.set_ready(request_id, False)
        self._scheduled_events.append(
            (now_ms + self._perf.l3_lookup_latency_ms, EventKind.L3_LOOKUP_DONE, request_id)
        )

    def _accept_prefill_probe(
        self, item: WaitingRequest, sources: tuple[tuple[str, str], ...], now_ms: float
    ) -> bool:
        request = item.request
        request_id = request.request_id
        if request_id in self._last_local_match:
            local_match = min(
                len(sources), (request.prompt_len - 1) // self._prefix_cache.page_size
            )
            if local_match < self._last_local_match[request_id]:
                self._last_local_match[request_id] = local_match
                self._schedule_l3_lookup(request_id, now_ms)
                return False
        return True

    def on_l3_lookup(
        self, request_id: int, now_ms: float
    ) -> None:  # 请求 ID 与查询完成时刻 → 触发连续 L3 命中页预取；无返回值
        self._lookup_waiting.remove(request_id)
        request = self._requests[request_id]
        local = self._prefix_cache.probe(request, request.prompt_len, include_l3=False)
        self._last_local_match[request_id] = len(local)
        sources = self._prefix_cache.probe(request, request.prompt_len, start_index=len(local))
        remote = [chunk_id for chunk_id, tier in sources if tier == "L3"]
        if len(remote) * self._prefix_cache.page_size <= self._engine_config.l3_prefetch_min_tokens:
            self._prefix_scheduler.set_ready(request_id, True)
            return
        pending: set[int] = set()
        for chunk_id in remote:
            try:
                pending.add(self._start_transfer(chunk_id, "L2", now_ms, request_id, source="L3"))
            except (CacheCapacityError, ValueError):
                break
        if pending:
            self._prefetch_waiting[request_id] = pending
        else:
            self._prefix_scheduler.set_ready(request_id, True)

    def on_transfer_done(
        self, kind: EventKind, transfer_id: int, now_ms: float
    ) -> None:  # 传输事件、任务 ID 和时刻 → 结算回载并唤醒等待；无返回值
        task = self._transfers.pop(transfer_id, None)
        if task is None:
            return
        destination = "L2" if kind is EventKind.L3_L2_DONE else "L1"
        del self._transfer_by_chunk[task.chunk_id, destination]
        if task.owns_transfer:
            self._prefix_cache.complete_transfer(task.chunk_id, destination)
            if kind is EventKind.L3_L2_DONE:
                del self._prefix_cache.pools["L2"].incoming_ready_ms[task.chunk_id]
            page_bytes = self._prefix_cache.page_size * (self._perf.kv_bytes_per_token or 0)
            self._restore_bytes += page_bytes
            self._restore_time_ms += task.service_ms
            self._analysis.restore_ms += task.service_ms
        if kind is EventKind.L3_L2_DONE:
            for request_id in task.waiting_requests:
                self._prefetched_origins.setdefault(request_id, set()).add(task.chunk_id)
                remaining = self._prefetch_waiting.get(request_id)
                if remaining is not None:
                    remaining.discard(transfer_id)
                    if not remaining:
                        del self._prefetch_waiting[request_id]
                        request = self._requests[request_id]
                        local = self._prefix_cache.probe(
                            request, request.prompt_len, include_l3=False
                        )
                        self._last_local_match[request_id] = len(local)
                        self._prefix_scheduler.set_ready(request_id, True)
        else:
            self._pending_loads.discard(task.chunk_id)

    def _abort(
        self,
        request_id: int,
        now_ms: float,
        reason: Literal[
            "input_exceeds_kv_capacity",
            "decode_kv_exhausted",
            "invalid_trace_output",
            "output_exceeds_engine_limit",
        ],
    ) -> None:  # 请求 ID、失败时刻和容量原因 → 登记中止记录；无返回值
        self._aborted[request_id] = AbortedRequest(
            id=request_id,
            arrival=self._arrival_times[request_id],
            failed_at=now_ms,
            reason=reason,
            generated_tokens=self._running.get(request_id, 0),
            retry_count=self._retry_counts.get(request_id, 0),
        )
        self._recent_aborts.append(request_id)

    def drain_aborts(self) -> tuple[int, ...]:  # 最近失败记录 → 按 ID 排序的待通知请求元组并清空
        """供客户端标记后代 skipped；每个失败仅通知一次。"""
        result = tuple(sorted(self._recent_aborts))
        self._recent_aborts.clear()
        return result

    def register_skipped(
        self, skipped: tuple[SkippedRequest, ...]
    ) -> None:  # 未发送子轮记录 → 登记跳过请求；无返回值
        for item in skipped:
            self._skipped[item.id] = item

    def _prepare_decode(
        self, running: dict[int, RunningRequest], now_ms: float
    ) -> tuple[
        dict[int, RunningRequest], tuple[tuple[int, int], ...]
    ]:  # 运行请求与时刻 → 可解码请求及回撤的 ID/进度
        """先驱逐，再按 g、负输入长、ID 回撤；单请求仍无法进展则终止。"""
        remaining = dict(running)
        withdrawn: list[tuple[int, int]] = []
        cache = self._prefix_cache
        while remaining:
            # 满页的请求本步需要新页。
            needed = sum(cache.state(key).k % cache.page_size == 0 for key in remaining)
            try:
                cache.ensure_free(needed)
                break
            except CacheCapacityError:
                # 下层被保护时无法降级，仍须继续回撤活动请求。
                with suppress(CacheCapacityError):
                    cache.ensure_free(cache.free_pages + cache.evictable_pages)
                key = min(
                    remaining,
                    key=lambda candidate: (
                        remaining[candidate].generated_tokens,
                        -remaining[candidate].request.prompt_len,
                        candidate,
                    ),
                )  # 进度少者先回撤；同进度优先释放长输入
                generated = remaining[key].generated_tokens
                if len(remaining) == 1:
                    self._abort(key, now_ms, "decode_kv_exhausted")  # 单请求也放不下则失败
                else:
                    self._retry_counts[key] = self._retry_counts.get(key, 0) + 1
                    withdrawn.append((key, generated))
                cache.withdraw(key)
                del self._running[key]
                del remaining[key]
        return remaining, tuple(withdrawn)

    def start_step(
        self, now_ms: float, step_index: int
    ) -> tuple[float, tuple[int, ...]] | None:  # 步起点与序号 → 耗时/完成 ID，等待回载时返回 None
        """回载未就绪时仍可处理已有请求的 Decode；就绪后再执行准入批次。"""
        selection = self._pending_selection
        if self._pending_loads:
            if not self._running:  # 没有等待decode的请求
                return None
            selection = AdmissionBatch((), (), ())
        elif (
            selection is not None
        ):  #  之前选中了一批请求准备开始prefill, 且load_back已完成, 可以开始prefill
            self._pending_selection = None
            self._pending_selection_prompt_tokens = 0
        else:  # 之前没有尝试组织prefill批次, 现在组批
            running = {
                key: RunningRequest(self._requests[key], generated)
                for key, generated in self._running.items()
            }
            selection = self._prefix_scheduler.select_prefill(
                running,
                accept_probe=lambda item, sources: self._accept_prefill_probe(
                    item, sources, now_ms
                ),
            )
            if (
                not self._running
                and not selection.admitted
                and not selection.restored
                and not selection.restore_only
                and not selection.aborted  # 没有运行中的请求, 也没有可以开始prefill的请求
                and (self._lookup_waiting or self._prefetch_waiting)  # 有请求在等待L3查询或prefetch
            ):
                return None
            if selection.promote_ids:
                self._pending_selection = selection  # 将选择准入请求放入等待准入中
                self._pending_selection_prompt_tokens = sum(
                    candidate.waiting.request.prompt_len for candidate in selection.admitted
                ) + sum(
                    item.request.prompt_len
                    for item in (*selection.restored, *selection.restore_only)
                )
                new_pages = sum(candidate.required_pages for candidate in selection.admitted)
                _ = self._prefix_cache.ensure_free(len(selection.promote_ids) + new_pages)
                # 调度器已保留上下文首次遇到顺序；串行链路按此顺序安排页完成时刻。
                for chunk_id in selection.promote_ids:
                    _ = self._start_transfer(
                        chunk_id,
                        "L1",
                        now_ms,
                        selection.admitted[0].waiting.request.request_id
                        if selection.admitted
                        else selection.restore_only[0].request.request_id,
                        source="L2",
                    )
                    self._pending_loads.add(chunk_id)
                return None  # 本次无法直接开始计算, 需要等待load_back完成
        return self._execute_step(now_ms, step_index, selection)  # 所有KV都在L1, 可以开始计算

    def finish_step(
        self, now_ms: float
    ) -> tuple[int, ...]:  # GPU 步完成时刻 → 发布前缀并返回已完成请求 ID
        """仅在模拟的 GPU 步完成时公开本步产生的共享前缀。"""
        if self._pending_publish:
            self._prefix_cache.publish_batch(self._pending_publish)
            self._pending_publish = ()
        finished = self._pending_finished
        self._pending_finished = ()
        self._pending_requeue_prompt_tokens = 0
        return finished

    def _execute_step(
        self,
        now_ms: float,
        step_index: int,
        selection: AdmissionBatch,
    ) -> tuple[float, tuple[int, ...]]:  # 步时刻、序号和已选批次 → 耗时及完成请求 ID
        """执行已就绪批次；前缀公开与完成通知统一延后至步末。"""
        cache = self._prefix_cache
        running = {
            key: RunningRequest(self._requests[key], generated)
            for key, generated in self._running.items()
        }
        withdrawn: tuple[tuple[int, int], ...] = ()
        if self._engine_config.pd_mutually_exclusive:
            if not selection.admitted and not selection.restore_only and running:
                running, withdrawn = self._prepare_decode(running, now_ms)
                if withdrawn:
                    self._prefix_scheduler.advance_ratio(running, withdrew=True)
            decode_ids = (
                () if selection.admitted or selection.restore_only else tuple(sorted(running))
            )
        else:
            running, withdrawn = self._prepare_decode(running, now_ms)
            if withdrawn:
                self._prefix_scheduler.advance_ratio(running, withdrew=True)
            decode_ids = tuple(sorted(running))
        for item in selection.aborted:
            self._abort(item.request.request_id, now_ms, "input_exceeds_kv_capacity")
        for item in selection.restored:
            self._running[item.request.request_id] = item.generated_tokens
        if selection.restore_only and not selection.admitted and not decode_ids:
            for item in selection.restore_only:
                cache.admit(item.request.request_id)
                self._running[item.request.request_id] = item.generated_tokens
            return 0.0, ()
        prefill = tuple(
            sorted(
                (
                    PrefillWork(
                        candidate.waiting.request.request_id,
                        candidate.reused_tokens,
                        candidate.new_tokens,
                        candidate.context_after,
                    )
                    for candidate in selection.admitted
                ),
                key=lambda entry: entry.request_id,
            )
        )
        decode = tuple(
            DecodeWork(key, self._requests[key].prompt_len + running[key].generated_tokens)
            for key in decode_ids
        )
        if not prefill and not decode and not selection.restore_only:
            for key, generated in withdrawn:
                self._prefix_scheduler.requeue(self._requests[key], now_ms, generated)
            if not self.has_work and not self._pending_done:
                self._prefix_scheduler.advance_ratio({}, idle=True)
            if selection.aborted or selection.restored or withdrawn or self._recent_aborts:
                return 0.0, ()
            raise RuntimeError(f"step {step_index} @{now_ms}: 没有可执行的 Prefill/Decode")
        phase: Literal["PREFILL", "DECODE", "MIXED"] = (
            "MIXED" if prefill and decode else "PREFILL" if prefill else "DECODE"
        )
        decode_new_pages = sum(cache.state(key).k % cache.page_size == 0 for key in decode_ids)
        allocated_pages = decode_new_pages + sum(
            candidate.required_pages for candidate in selection.admitted
        )
        cache.ensure_free(allocated_pages)
        workload = StepWorkload(
            phase,
            prefill,
            decode,
            allocated_pages,
            len(prefill) + len(decode) + len(selection.restore_only),
            tuple(item.request.request_id for item in selection.restore_only),
        )
        cost = step_cost(
            workload,
            self._analytical.model,
            self._analytical.hardware,
            self._analytical.tp_size,
            self._analytical.ep_size,
        )
        duration = cost.duration_ms
        self._analysis.steps += 1
        self._analysis.flops += cost.flops
        self._analysis.hbm_bytes += cost.hbm_bytes
        self._analysis.compute_ms += cost.compute_ms
        self._analysis.memory_ms += cost.memory_ms
        self._analysis.device_ms += duration
        self._analysis.communication_ms += cost.communication_ms
        self._analysis.communication_bytes += cost.communication_bytes
        self._analysis.collectives += cost.collectives
        self._analysis.tp_communication_ms += cost.tp_communication_ms
        self._analysis.tp_communication_bytes += cost.tp_communication_bytes
        self._analysis.ep_dispatch_ms += cost.ep_dispatch_ms
        self._analysis.ep_combine_ms += cost.ep_combine_ms
        self._analysis.ep_dispatch_bytes += cost.ep_dispatch_bytes
        self._analysis.ep_combine_bytes += cost.ep_combine_bytes
        self._analysis.ep_exchanges += cost.ep_exchanges
        end = now_ms + duration
        for candidate in selection.admitted:
            cache.admit(candidate.waiting.request.request_id)
        for item in selection.restore_only:
            cache.admit(item.request.request_id)
            self._running[item.request.request_id] = item.generated_tokens
        finished: list[int] = []
        for key in decode_ids:
            previous = running[key].generated_tokens
            cache.extend(key, cache.state(key).k + 1)
            self._running[key] = previous + 1
            if self._running[key] == self._requests[key].output_len:
                finished.append(key)
        for candidate in selection.admitted:
            item = candidate.waiting
            key = item.request.request_id
            cache.extend(key, item.target_tokens)
            if item.restoring:
                self._running[key] = item.generated_tokens  # 重算 KV 不会额外生成输出
                self._replayed_tokens[key] = (
                    self._replayed_tokens.get(key, 0) + candidate.new_tokens
                )
            else:
                self._prefill_start[key] = now_ms
                self._prefill_end[key] = end
                self._running[key] = 1  # 首个输出在 Prefill 步完成时已生成
                if self._requests[key].output_len == 1:
                    finished.append(key)
            if key not in self._admission_stats:
                self._admission_stats[key] = (candidate.reused_tokens, candidate.reused_tokens)
                prefetched = self._prefetched_origins.get(key, set())
                self._admission_tiers[key] = (
                    sum(
                        tier == "L1" and chunk_id not in prefetched
                        for chunk_id, tier in candidate.hit_sources
                    )
                    * cache.page_size
                    if candidate.hit_sources
                    else candidate.reused_tokens,
                    sum(
                        tier == "L2" and chunk_id not in prefetched
                        for chunk_id, tier in candidate.hit_sources
                    )
                    * cache.page_size,
                    sum(
                        tier == "L3" or chunk_id in prefetched
                        for chunk_id, tier in candidate.hit_sources
                    )
                    * cache.page_size,
                )
        # 新计算页尚未完成前只占私有容量，不向后来到达的请求发布。
        self._pending_publish = tuple(
            item.waiting.request.request_id for item in selection.admitted
        )
        finished.sort()
        self._pending_finished = tuple(finished)
        for key in finished:
            del self._running[key]
            self._pending_done[key] = end  # KV 等 Done 事件结算时才释放
        for key, generated in withdrawn:
            self._prefix_scheduler.requeue(self._requests[key], end, generated)
            self._pending_requeue_prompt_tokens += self._requests[key].prompt_len
        if not withdrawn:
            _ = self._prefix_scheduler.advance_ratio(
                {key: RunningRequest(self._requests[key], g) for key, g in self._running.items()},
                decoded=bool(decode_ids),
            )
        self._num_steps += 1
        return duration, tuple(finished)

    def complete(self, now_ms: float, request_id: int) -> None:  # Done 时刻与 ID → 结算
        """在 Done 事件时发布剩余完整页、释放 KV 并记录完成时间。"""
        expected = self._pending_done.get(request_id)
        if expected is None or now_ms != expected:
            raise ValueError(f"请求 {request_id} 未待结算或完成时刻不符")
        self._prefix_cache.finish(request_id)
        del self._pending_done[request_id]
        self._finish[request_id] = now_ms
        if not self.has_work and not self._pending_done:
            _ = self._prefix_scheduler.advance_ratio({}, idle=True)

    # ------------------------------------------------------------ 结算

    def finalize(self) -> EngineRun:  # 已结算的请求、缓存及成本统计 → 单实例仿真原始产出
        """所有请求完成后形成稳定输出，包含即时提交时刻。"""
        arrivals = self._arrivals
        records = tuple(
            RequestRecord(
                id=request_id,
                arrival=arrivals[request_id].sim_time,
                prefill_start=self._prefill_start[request_id],
                prefill_end=self._prefill_end[request_id],
                finish=self._finish[request_id],
                prompt_len=arrivals[request_id].prompt_len,
                output_len=arrivals[request_id].output_len,
                session_id=self._requests[request_id].session_id,
                parent_request_id=self._requests[request_id].parent_request_id,
                ideal_reuse_tokens=(
                    self._requests[request_id].ideal_reuse_blocks * self._prefix_cache.page_size
                ),
                matched_cache_tokens=self._admission_stats.get(request_id, (0, 0))[0],
                actual_hit_tokens=self._admission_stats.get(request_id, (0, 0))[1],
                actual_prefill_tokens=(
                    arrivals[request_id].prompt_len - self._admission_stats[request_id][1]
                ),
                replayed_prefill_tokens=self._replayed_tokens.get(request_id, 0),
                retry_count=self._retry_counts.get(request_id, 0),
                l1_hit_tokens=self._admission_tiers.get(request_id, (0, 0, 0))[0],
                l2_hit_tokens=self._admission_tiers.get(request_id, (0, 0, 0))[1],
                l3_hit_tokens=self._admission_tiers.get(request_id, (0, 0, 0))[2],
                logical_context_tokens=(
                    arrivals[request_id].prompt_len + arrivals[request_id].output_len
                ),
            )
            for request_id in sorted(self._finish)
        )
        sim_start = min(self._arrival_times.values(), default=0.0)
        sim_end = max(
            (*self._finish.values(), *(item.failed_at for item in self._aborted.values())),
            default=sim_start,
        )
        return EngineRun(
            records=records,
            warnings=(),
            num_engine_steps=self._num_steps,
            sim_start=sim_start,
            sim_end=sim_end,
            cache_snapshot=self._prefix_cache.snapshot(),
            l1_evictions=self._prefix_cache.evictions,
            l1_peak_resident_blocks=self._prefix_cache.peak_resident_pages,
            aborted_requests=tuple(self._aborted[key] for key in sorted(self._aborted)),
            skipped_requests=tuple(self._skipped[key] for key in sorted(self._skipped)),
            replayed_prefill_tokens=sum(self._replayed_tokens.values()),
            l2_snapshot=self._prefix_cache.tier_snapshot("L2"),
            l3_snapshot=self._prefix_cache.tier_snapshot("L3"),
            promotions=self._prefix_cache.promotions,
            demotions=self._prefix_cache.demotions,
            restore_bytes=self._restore_bytes,
            restore_time_ms=self._restore_time_ms,
            analysis=self._analysis,
        )
