"""单引擎执行器：请求通过 F001 调度器与 L1 页池顺序执行。

事件堆、时钟推进与客户端回调由 ``Coordinator`` 独占。
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from simcore.events import ArrivalTrace, DoneTrace, EventLog, StepTrace
from simcore.kv_cache import L1KVCache, L1Snapshot
from simcore.perf_model import DecodeWork, PrefillWork, StepWorkload, workload_time
from simcore.scheduler import F001Scheduler, RunningRequest
from simcore.schemas import (
    Arrival,
    EngineConfig,
    InferenceRequest,
    PerfModelConfig,
    RequestCompletion,
    RequestRecord,
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
        self._output_targets = dict(output_targets)
        self._engine_config = engine
        self._perf = perf
        self._prefix_cache = L1KVCache(engine.max_total_tokens, engine.page_size)
        self._prefix_scheduler = F001Scheduler(engine, self._prefix_cache)
        self._admission_stats: dict[int, tuple[int, int]] = {}
        self._last_workload: StepWorkload | None = None
        self._log = EventLog(enabled=record_trace)
        self._requests: dict[int, InferenceRequest] = {}
        self._arrivals: dict[int, Arrival] = {}
        # request_id -> 已采样输出数；Decode 只处理步首已有请求
        self._running: dict[int, int] = {}
        self._pending_done: dict[int, float] = {}
        self._prefill_start: dict[int, float] = {}
        self._prefill_end: dict[int, float] = {}
        self._finish: dict[int, float] = {}
        self._num_steps = 0
        self._warnings: list[str] = []

    @property
    def has_work(self) -> bool:
        """是否有等待或执行中的请求（不含等待完成结算的请求）。"""
        return bool(self._running or self._prefix_scheduler.num_waiting)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def last_workload(self) -> StepWorkload | None:
        return self._last_workload

    def submit(self, request: InferenceRequest, now_ms: float) -> None:
        """只在客户端真正发送时入缓存调度队列并记录提交时刻。"""
        if request.request_id in self._requests:
            raise ValueError(f"request_id 重复: {request.request_id}")
        output_len = self._output_targets.get(request.request_id)
        if output_len is None or not 1 <= output_len <= request.max_new_tokens:
            raise ValueError("请求实际输出长度与可见上限不一致")
        arrival = Arrival(
            request_id=request.request_id,
            sim_time=now_ms,
            prompt_len=request.prompt_len,
            output_len=output_len,
            max_new_tokens=request.max_new_tokens,
        )
        self._prefix_scheduler.enqueue(request, now_ms)
        self._requests[request.request_id] = request
        self._arrivals[request.request_id] = arrival
        self._log.record(ArrivalTrace.of(arrival))

    def step(self, now_ms: float, step_index: int) -> tuple[float, tuple[int, ...]]:
        """只推进步首 Decode 和本步 Prefill；结果发布后下一步才可复用。"""
        if step_index != self._num_steps:
            raise ValueError(f"预期 step_index={self._num_steps}，得到 {step_index}")
        cache = self._prefix_cache
        running = {
            key: RunningRequest(self._requests[key], generated)
            for key, generated in self._running.items()
        }
        before = cache.snapshot() if self._log.enabled else None
        if self._engine_config.pd_mutually_exclusive:
            selection = self._prefix_scheduler.select_prefill(running)
            decode_ids = () if selection.admitted else tuple(sorted(running))
        else:
            decode_ids = tuple(sorted(running))
            selection = self._prefix_scheduler.select_prefill(running, decode_ids)
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
        if not prefill and not decode:
            raise RuntimeError(f"step {step_index} @{now_ms}: 没有可执行的 Prefill/Decode")
        phase: Literal["PREFILL", "DECODE", "MIXED"] = (
            "MIXED" if prefill and decode else "PREFILL" if prefill else "DECODE"
        )
        decode_new_pages = sum(cache.state(key).k % cache.page_size == 0 for key in decode_ids)
        allocated_pages = decode_new_pages + sum(
            candidate.required_pages for candidate in selection.admitted
        )
        workload = StepWorkload(phase, prefill, decode, allocated_pages, len(prefill) + len(decode))
        duration = workload_time(workload, self._perf)
        end = now_ms + duration
        evictions_before = cache.evictions
        cache.ensure_free(workload.allocated_pages)
        for candidate in selection.admitted:
            cache.admit(candidate.waiting.request.request_id)
        finished: list[int] = []
        for key in decode_ids:
            previous = running[key].generated_tokens
            if cache.state(key).k != self._requests[key].prompt_len + previous - 1:
                raise RuntimeError("Decode 步首 KV 长度与已采样输出数不一致")
            cache.extend(key, cache.state(key).k + 1)
            self._running[key] = previous + 1
            if self._running[key] == self._output_targets[key]:
                finished.append(key)
        for candidate in selection.admitted:
            item = candidate.waiting
            key = item.request.request_id
            cache.extend(key, item.target_tokens)
            if not item.restoring:
                self._prefill_start[key] = now_ms
                self._prefill_end[key] = end
                self._running[key] = 1
                if self._output_targets[key] == 1:
                    finished.append(key)
            self._admission_stats.setdefault(
                key, (candidate.reused_tokens, candidate.reused_tokens)
            )
        cache.publish_batch(tuple(item.waiting.request.request_id for item in selection.admitted))
        self._last_workload = workload
        finished.sort()
        if self._log.enabled:
            assert before is not None
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
                )
            )
        for key in finished:
            del self._running[key]
            self._pending_done[key] = end
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
        unfinished = arrivals.keys() - self._finish.keys()
        if unfinished:  # pragma: no cover - 进展保证下不可达
            raise RuntimeError(f"仿真结束但仍有 {len(unfinished)} 个请求未完成")

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
                l1_hit_tokens=self._admission_stats.get(request_id, (0, 0))[1],
                logical_context_tokens=(
                    arrivals[request_id].prompt_len + arrivals[request_id].output_len
                ),
            )
            for request_id in sorted(arrivals)
        )
        sim_start = min((a.sim_time for a in arrivals.values()), default=0.0)
        sim_end = max(self._finish.values(), default=0.0)
        return EngineRun(
            records=records,
            warnings=tuple(self._warnings),
            num_engine_steps=self._num_steps,
            sim_start=sim_start,
            sim_end=sim_end,
            trace=self._log,
            cache_snapshot=self._prefix_cache.snapshot(),
            l1_evictions=self._prefix_cache.evictions,
            l1_peak_resident_blocks=self._prefix_cache.peak_resident_pages,
        )
