"""单引擎执行器：请求通过 F001 调度器与 L1 页池顺序执行。

事件堆、时钟推进与客户端回调由 ``Coordinator`` 独占。
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from simcore.events import ArrivalTrace, DoneTrace, EventLog, StepTrace
from simcore.kv_cache import CacheCapacityError, L1Snapshot, TieredKVCache, TierSnapshot
from simcore.perf_model import (
    DecodeWork,
    PrefillWork,
    StepWorkload,
    restore_bytes,
    restore_time,
    workload_time,
)
from simcore.scheduler import F001Scheduler, RunningRequest
from simcore.schemas import (
    AbortedRequest,
    Arrival,
    EngineConfig,
    InferenceRequest,
    PerfModelConfig,
    RequestCompletion,
    RequestRecord,
    SkippedRequest,
    validate_kv_config,
)

__all__ = ["Engine", "EngineRun", "run_engine"]


@dataclass(frozen=True, slots=True)
class EngineRun:
    """单场仿真的原始产出。

    Attributes:
        records: 每请求一条区间记录，按 ``request_id`` 升序。
        warnings: 告警文本，按发生时刻排序。
        num_engine_steps: 执行的引擎步数。
        sim_start: 首请求到达时刻（ms）；无请求时为 0。
        sim_end: 末请求完成时刻（ms）；无请求时为 0。
        trace: 完整事件流（``record_trace=False`` 时为空）。
        cache_snapshot: 仿真结束时的物理 L1 容量快照。
    """

    records: tuple[RequestRecord, ...]
    warnings: tuple[str, ...]
    num_engine_steps: int
    sim_start: float
    sim_end: float
    trace: EventLog
    cache_snapshot: L1Snapshot
    l1_evictions: int = 0
    l1_peak_resident_blocks: int = 0
    aborted_requests: tuple[AbortedRequest, ...] = ()
    skipped_requests: tuple[SkippedRequest, ...] = ()
    replayed_prefill_tokens: int = 0
    l2_snapshot: TierSnapshot | None = None
    l3_snapshot: TierSnapshot | None = None
    l2_evictions: int = 0
    l3_evictions: int = 0
    l2_peak_resident_blocks: int = 0
    l3_peak_resident_blocks: int = 0
    promotions: int = 0
    demotions: int = 0
    l2_promotions: int = 0
    l3_promotions: int = 0
    l1_to_l2_demotions: int = 0
    l2_to_l3_demotions: int = 0
    cache_deletions: int = 0
    restore_bytes: int = 0
    restore_time_ms: float = 0.0


def run_engine(
    engine: EngineConfig,
    perf: PerfModelConfig,
    arrivals: Iterable[Arrival],
    *,
    record_trace: bool = True,
) -> EngineRun:
    """执行一场仿真。

    Args:
        engine: 引擎配置。
        perf: 性能模型系数。
        arrivals: 到达序列，``request_id`` 须唯一。经事件堆排序，故不要求输入有序；
            同刻到达按请求 ID 排序处理。
        record_trace: 是否保存事件流；基准测试可关闭以剥离记录开销。

    Raises:
        ValueError: ``request_id`` 重复。
    """
    from simcore.client import ClientDriver
    from simcore.coordinator import Coordinator
    from simcore.workload import plan_from_arrivals

    validate_kv_config(engine, perf)
    plan = plan_from_arrivals(arrivals, engine)
    client = ClientDriver(plan, engine)
    return Coordinator(
        Engine(
            engine,
            perf,
            output_targets={key: turn.output_len for key, turn in plan.turns.items()},
            record_trace=record_trace,
        ),
        client,
    ).run()


class Engine:
    """维护即时请求与调度/KV 状态，不预存未来请求或持有事件队列。"""

    def __init__(
        self,
        engine: EngineConfig,
        perf: PerfModelConfig,
        *,
        output_targets: Mapping[int, int],
        record_trace: bool = True,
    ) -> None:
        validate_kv_config(engine, perf)
        self._output_targets = dict(output_targets)
        self._engine_config = engine
        self._perf = perf
        self._prefix_cache = TieredKVCache(
            engine.max_total_tokens, engine.page_size,
            engine.l2_capacity_tokens, engine.l3_capacity_tokens,
        )
        self._prefix_scheduler = F001Scheduler(engine, self._prefix_cache)
        self._admission_stats: dict[int, tuple[int, int]] = {}
        self._admission_tiers: dict[int, tuple[int, int, int]] = {}
        self._restore_bytes = 0
        self._restore_time_ms = 0.0
        self._log = EventLog(enabled=record_trace)
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

    @property
    def has_work(self) -> bool:
        """是否有等待或执行中的请求（不含等待完成结算的请求）。"""
        return bool(self._running or self._prefix_scheduler.num_waiting)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def submit(self, request: InferenceRequest, now_ms: float) -> bool:
        """发送时 M<1 直接失败；其他请求按 FCFS 入队。"""
        key = request.request_id
        if key in self._requests:
            raise ValueError(f"request_id 重复: {key}")
        output_len = self._output_targets.get(key)
        if output_len is None:
            raise ValueError("请求缺少引擎持有的输出目标")
        if request.max_new_tokens > 0 and not 1 <= output_len <= request.max_new_tokens:
            raise ValueError("请求实际输出长度与可见上限不一致")
        self._requests[key] = request
        self._arrival_times[key] = now_ms
        self._log.record(ArrivalTrace(now_ms, key, request.prompt_len, output_len))
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
        return True

    def _abort(
        self,
        request_id: int,
        now_ms: float,
        reason: Literal["input_exceeds_kv_capacity", "decode_kv_exhausted"],
    ) -> None:
        self._aborted[request_id] = AbortedRequest(
            id=request_id,
            arrival=self._arrival_times[request_id],
            failed_at=now_ms,
            reason=reason,
            generated_tokens=self._running.get(request_id, 0),
            retry_count=self._retry_counts.get(request_id, 0),
        )
        self._recent_aborts.append(request_id)

    def drain_aborts(self) -> tuple[int, ...]:
        """供客户端标记后代 skipped；每个失败仅通知一次。"""
        result = tuple(sorted(self._recent_aborts))
        self._recent_aborts.clear()
        return result

    def register_skipped(self, skipped: tuple[SkippedRequest, ...]) -> None:
        for item in skipped:
            self._skipped[item.id] = item

    def _prepare_decode(
        self, running: dict[int, RunningRequest], now_ms: float
    ) -> tuple[dict[int, RunningRequest], tuple[tuple[int, int], ...]]:
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
                cache.ensure_free(cache.free_pages + cache.evictable_pages)  # 先腾尽可驱逐页
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

    def step(self, now_ms: float, step_index: int) -> tuple[float, tuple[int, ...]]:
        """只推进步首 Decode 和本步 Prefill；结果发布后下一步才可复用。"""
        cache = self._prefix_cache
        running = {
            key: RunningRequest(self._requests[key], generated)
            for key, generated in self._running.items()
        }
        before = cache.snapshot() if self._log.enabled else None
        evictions_before = cache.evictions
        l2_evictions_before = cache.tier_snapshot("L2").evictions
        l3_evictions_before = cache.tier_snapshot("L3").evictions
        promotions_before = cache.promotions
        demotions_before = cache.demotions
        aborts_before = len(self._recent_aborts)
        withdrawn: tuple[tuple[int, int], ...] = ()
        if self._engine_config.pd_mutually_exclusive:
            # 先试 Prefill，成功则本步不 Decode。
            selection = self._prefix_scheduler.select_prefill(running)
            if not selection.admitted and not selection.restore_only and running:
                running, withdrawn = self._prepare_decode(running, now_ms)
                if withdrawn:
                    self._prefix_scheduler.advance_ratio(running, withdrew=True)
            decode_ids = (
                () if selection.admitted or selection.restore_only else tuple(sorted(running))
            )
        else:
            running, withdrawn = self._prepare_decode(running, now_ms)  # 混合模式先为 Decode 腾页
            if withdrawn:
                self._prefix_scheduler.advance_ratio(running, withdrew=True)
            decode_ids = tuple(sorted(running))
            selection = self._prefix_scheduler.select_prefill(running, decode_ids)
        for item in selection.aborted:
            self._abort(item.request.request_id, now_ms, "input_exceeds_kv_capacity")
        for item in selection.restored:
            self._running[item.request.request_id] = item.generated_tokens
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
        phase: Literal["PREFILL", "DECODE", "MIXED", "RESTORE"] = (
            "MIXED" if prefill and decode else
            "PREFILL" if prefill else
            "DECODE" if decode else "RESTORE"
        )
        decode_new_pages = sum(cache.state(key).k % cache.page_size == 0 for key in decode_ids)
        allocated_pages = decode_new_pages + sum(
            candidate.required_pages for candidate in selection.admitted
        )
        if self._engine_config.l2_capacity_tokens:
            cache.plan(selection.promote_ids, allocated_pages)
            promoted = cache.promote(selection.promote_ids, allocated_pages)
        else:
            cache.ensure_free(allocated_pages)
            promoted = ()
        workload = StepWorkload(
            phase, prefill, decode, allocated_pages,
            len(prefill) + len(decode) + len(selection.restore_only),
            tuple(sorted(promoted)), cache.page_size if promoted else 0,
            tuple(item.request.request_id for item in selection.restore_only),
        )
        transfer_time = restore_time(workload, self._perf)
        transferred_bytes = restore_bytes(workload, self._perf)
        duration = workload_time(workload, self._perf) + transfer_time
        self._restore_time_ms += transfer_time
        self._restore_bytes += transferred_bytes
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
            if self._running[key] == self._output_targets[key]:
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
                if self._output_targets[key] == 1:
                    finished.append(key)
            if key not in self._admission_stats:
                self._admission_stats[key] = (candidate.reused_tokens, candidate.reused_tokens)
                self._admission_tiers[key] = (
                    tuple(tier for _, tier in candidate.hit_sources).count("L1") * cache.page_size
                    if candidate.hit_sources else candidate.reused_tokens,
                    tuple(tier for _, tier in candidate.hit_sources).count("L2") * cache.page_size,
                    tuple(tier for _, tier in candidate.hit_sources).count("L3") * cache.page_size,
                )
        # 本步新页到步末才发布，留给后续步骤复用。
        cache.publish_batch(tuple(item.waiting.request.request_id for item in selection.admitted))
        finished.sort()
        if before is not None:
            cache_after = cache.snapshot()
            self._log.record(
                StepTrace(
                    time=now_ms,
                    step_index=step_index,
                    duration=duration,
                    admitted=tuple(
                        candidate.waiting.request.request_id for candidate in selection.admitted
                    ),
                    forced=(),
                    num_decoding=len(decode),
                    batch_total_tokens=sum(item.context_after for item in prefill)
                    + sum(item.context_after_decode for item in decode),
                    kv_used=sum(cache.state(key).k for key in self._running)
                    + sum(cache.state(key).k for key in self._pending_done),
                    kv_overflow=0,
                    finished=tuple(finished),
                    probes=len(selection.admitted),
                    matched_cache_tokens=sum(item.reused_tokens for item in prefill),
                    actual_hit_tokens=sum(item.reused_tokens for item in prefill),
                    allocated_blocks=allocated_pages,
                    evicted_blocks=cache.evictions - evictions_before,
                    l1_active_before=before.active_blocks,
                    l1_evictable_before=before.evictable_blocks,
                    l1_free_before=before.free_blocks,
                    l1_active_after=cache_after.active_blocks,
                    l1_evictable_after=cache_after.evictable_blocks,
                    l1_free_after=cache_after.free_blocks,
                    l1_internal_fragmentation_tokens=cache_after.internal_fragmentation_tokens,
                    phase=workload.phase,
                    withdrawn=tuple(key for key, _ in withdrawn),
                    aborted=tuple(sorted(self._recent_aborts[aborts_before:])),
                    restored=tuple(sorted(
                        item.request.request_id
                        for item in (*selection.restored, *selection.restore_only)
                    )),
                    restore_sources=workload.restore_sources,
                    restore_bytes=transferred_bytes,
                    restore_time_ms=transfer_time,
                    l1_evictions=cache.evictions - evictions_before,
                    l2_evictions=cache.tier_snapshot("L2").evictions - l2_evictions_before,
                    l3_evictions=cache.tier_snapshot("L3").evictions - l3_evictions_before,
                    promotions=cache.promotions - promotions_before,
                    demotions=cache.demotions - demotions_before,
                )
            )
        for key in finished:
            del self._running[key]
            self._pending_done[key] = end  # KV 等 Done 事件结算时才释放
        for key, generated in withdrawn:
            self._prefix_scheduler.requeue(self._requests[key], end, generated)
        if not withdrawn:
            self._prefix_scheduler.advance_ratio(
                {key: RunningRequest(self._requests[key], g) for key, g in self._running.items()},
                decoded=bool(decode_ids),
            )
        self._num_steps += 1
        return duration, tuple(finished)

    def complete(self, now_ms: float, request_id: int) -> RequestCompletion:
        """在 Done 事件时释放 KV，只返回请求身份和完成时间。"""
        expected = self._pending_done.get(request_id)
        if expected is None or now_ms != expected:
            raise ValueError(f"请求 {request_id} 未待结算或完成时刻不符")
        released = self._prefix_cache.state(request_id).k
        self._prefix_cache.finish(request_id)
        self._log.record(DoneTrace(now_ms, request_id, released))
        del self._pending_done[request_id]
        self._finish[request_id] = now_ms
        if not self.has_work and not self._pending_done:
            self._prefix_scheduler.advance_ratio({}, idle=True)
        return RequestCompletion(request_id, now_ms, self._output_targets[request_id])

    # ------------------------------------------------------------ 结算

    def finalize(self) -> EngineRun:
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
            trace=self._log,
            cache_snapshot=self._prefix_cache.snapshot(),
            l1_evictions=self._prefix_cache.evictions,
            l1_peak_resident_blocks=self._prefix_cache.peak_resident_pages,
            aborted_requests=tuple(self._aborted[key] for key in sorted(self._aborted)),
            skipped_requests=tuple(self._skipped[key] for key in sorted(self._skipped)),
            replayed_prefill_tokens=sum(self._replayed_tokens.values()),
            l2_snapshot=self._prefix_cache.tier_snapshot("L2"),
            l3_snapshot=self._prefix_cache.tier_snapshot("L3"),
            l2_evictions=self._prefix_cache.tier_snapshot("L2").evictions,
            l3_evictions=self._prefix_cache.tier_snapshot("L3").evictions,
            l2_peak_resident_blocks=self._prefix_cache.tier_snapshot("L2").peak_resident_blocks,
            l3_peak_resident_blocks=self._prefix_cache.tier_snapshot("L3").peak_resident_blocks,
            promotions=self._prefix_cache.promotions,
            demotions=self._prefix_cache.demotions,
            l2_promotions=self._prefix_cache.l2_promotions,
            l3_promotions=self._prefix_cache.l3_promotions,
            l1_to_l2_demotions=self._prefix_cache.l1_to_l2_demotions,
            l2_to_l3_demotions=self._prefix_cache.l2_to_l3_demotions,
            cache_deletions=self._prefix_cache.deletions,
            restore_bytes=self._restore_bytes,
            restore_time_ms=self._restore_time_ms,
        )
