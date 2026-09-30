"""单引擎执行器：内容请求始终通过 L1 块缓存调度与执行。

事件堆、时钟推进与客户端回调由 ``Coordinator`` 独占。
"""

from collections.abc import Iterable
from dataclasses import dataclass

from simcore.events import ArrivalTrace, DoneTrace, EventLog, StepTrace
from simcore.kv_cache import CacheSnapshot, HierarchicalKVCache
from simcore.prefix_hash import encode
from simcore.scheduler import PrefixScheduler
from simcore.schemas import (
    Arrival,
    EngineConfig,
    InferenceRequest,
    PerfModelConfig,
    PrefixCacheConfig,
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
    cache_snapshot: CacheSnapshot
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
            同刻到达按输入顺序处理。
        record_trace: 是否保存事件流；基准测试可关闭以剥离记录开销。

    Raises:
        ValueError: ``request_id`` 重复。
    """
    from simcore.client import ClientDriver
    from simcore.coordinator import Coordinator
    from simcore.workload import plan_from_arrivals

    plan = plan_from_arrivals(arrivals, engine.prefix_cache)
    client = ClientDriver(plan, engine.prefix_cache)
    return Coordinator(Engine(engine, perf, record_trace=record_trace), client).run()


class Engine:
    """维护即时请求与调度/KV 状态，不预存未来请求或持有事件队列。"""

    def __init__(
        self, engine: EngineConfig, perf: PerfModelConfig, record_trace: bool = True
    ) -> None:
        self._cache_config: PrefixCacheConfig = engine.prefix_cache
        self._prefix_cache = HierarchicalKVCache(
            engine.prefix_cache.l1.capacity_blocks, engine.prefix_cache.block_size_tokens
        )
        self._prefix_scheduler = PrefixScheduler(
            engine, perf, self._prefix_cache, record_trace=record_trace
        )
        self._admission_stats: dict[int, tuple[int, int]] = {}
        self._l1_peak_resident_blocks = 0
        self._log = EventLog(enabled=record_trace)
        self._requests: dict[int, InferenceRequest] = {}
        self._arrivals: dict[int, Arrival] = {}
        # request_id -> 已产出 token 数；插入顺序即准入顺序，决定 decode 推进顺序
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

    def submit(self, request: InferenceRequest, now_ms: float) -> None:
        """只在客户端真正发送时入缓存调度队列并记录提交时刻。"""
        if request.request_id in self._requests:
            raise ValueError(f"request_id 重复: {request.request_id}")
        arrival = Arrival(
            request_id=request.request_id,
            sim_time=now_ms,
            prompt_len=request.prompt_len,
            output_len=request.output_len,
        )
        self._prefix_scheduler.enqueue(request, now_ms)
        self._requests[request.request_id] = request
        self._arrivals[request.request_id] = arrival
        self._log.record(ArrivalTrace.of(arrival))

    def step(self, now_ms: float, step_index: int) -> tuple[float, tuple[int, ...]]:
        """在同一 L1 快照上原子准入，步末发布真正算完的完整块。"""
        if step_index != self._num_steps:
            raise ValueError(f"预期 step_index={self._num_steps}，得到 {step_index}")
        scheduler = self._prefix_scheduler
        cache = self._prefix_cache
        plan = scheduler.plan_step()
        if plan is None:
            raise RuntimeError(f"step {step_index} @{now_ms}: 引擎步被调度但无工作")
        end = now_ms + plan.duration
        if plan.cache_result is not None:
            for result in plan.cache_result.requests:
                self._admission_stats[result.request_id] = (
                    result.matched_cache_tokens,
                    result.actual_hit_tokens,
                )
        self._l1_peak_resident_blocks = max(self._l1_peak_resident_blocks, cache.resident_blocks)
        decoding = list(self._running)
        finished: list[int] = []
        for request_id in decoding:
            produced = self._running[request_id] + 1
            self._running[request_id] = produced
            self._publish_output(request_id, produced)
            if produced == self._requests[request_id].output_len:
                finished.append(request_id)

        for request, probe in zip(plan.admitted, plan.probes, strict=True):
            request_id = request.request_id
            self._prefill_start[request_id] = now_ms
            self._prefill_end[request_id] = end
            cache.publish(request_id, request.input_chunk_ids[probe.matched_blocks :])
            self._running[request_id] = 1
            self._publish_output(request_id, 1)
            if request.output_len == 1:
                finished.append(request_id)

        self._l1_peak_resident_blocks = max(self._l1_peak_resident_blocks, cache.resident_blocks)
        if self._log.enabled:
            cache_after = cache.snapshot()
            before = plan.cache_before
            assert before is not None
            self._log.record(
                StepTrace(
                    time=now_ms,
                    step_index=step_index,
                    duration=plan.duration,
                    admitted=tuple(request.request_id for request in plan.admitted),
                    forced=(),
                    num_decoding=plan.num_decoding,
                    batch_total_tokens=plan.batch_total_tokens,
                    kv_used=cache.logical_context_tokens,
                    kv_overflow=0,
                    finished=tuple(finished),
                    probes=len(plan.probes),
                    matched_cache_tokens=sum(p.matched_cache_tokens for p in plan.probes),
                    actual_hit_tokens=sum(
                        item.actual_hit_tokens for item in plan.cache_result.requests
                    )
                    if plan.cache_result
                    else 0,
                    allocated_blocks=plan.cache_result.allocated_blocks if plan.cache_result else 0,
                    evicted_blocks=len(plan.cache_result.evicted_block_ids)
                    if plan.cache_result
                    else 0,
                    l1_active_before=before.active_blocks,
                    l1_evictable_before=before.evictable_blocks,
                    l1_free_before=before.free_blocks,
                    l1_active_after=cache_after.active_blocks,
                    l1_evictable_after=cache_after.evictable_blocks,
                    l1_free_after=cache_after.free_blocks,
                    l1_internal_fragmentation_tokens=cache_after.internal_fragmentation_tokens,
                )
            )
        for request_id in finished:
            del self._running[request_id]
            self._pending_done[request_id] = end
        self._num_steps += 1
        return plan.duration, tuple(finished)

    def _publish_output(self, request_id: int, produced: int) -> None:
        cache = self._prefix_cache
        request = self._requests[request_id]
        cache.advance(request_id)
        total_tokens = request.prompt_len + produced
        block_size = self._cache_config.block_size_tokens
        if total_tokens % block_size == 0:
            cache.publish(request_id, (encode(request_id, total_tokens // block_size - 1),))

    def complete(self, now_ms: float, request_id: int) -> RequestCompletion:
        """在 Done 事件时释放 KV，只返回请求身份和完成时间。"""
        expected = self._pending_done.get(request_id)
        if expected is None or now_ms != expected:
            raise ValueError(f"请求 {request_id} 未待结算或完成时刻不符")
        released = self._prefix_cache.request_context_tokens(request_id)
        self._prefix_cache.release(request_id)
        self._prefix_scheduler.on_request_released()
        self._log.record(DoneTrace(now_ms, request_id, released))
        del self._pending_done[request_id]
        self._finish[request_id] = now_ms
        return RequestCompletion(request_id, now_ms)

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
                    self._requests[request_id].ideal_reuse_blocks
                    * self._cache_config.block_size_tokens
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
            l1_peak_resident_blocks=self._l1_peak_resident_blocks,
        )
