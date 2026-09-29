"""单引擎执行器：只处理即时提交、引擎步与请求完成。

事件堆、时钟推进、step 调度和客户端回调由 ``SimulationCoordinator`` 独占。
当前保留无缓存 ``KVCache``/``FCFSScheduler`` 作为 PC1-B 过渡路径；
请求内容在提交时保留，以便返回实际生成的 ``RequestCompletion``。
"""

from collections.abc import Iterable
from dataclasses import dataclass

from simcore.events import ArrivalTrace, DoneTrace, EventLog, StepTrace
from simcore.kv_cache import HierarchicalKVCache, KVCache, KVStats
from simcore.prefix_hash import cache_namespace, next_chunk_hash
from simcore.scheduler import FCFSScheduler, PrefixScheduler
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
        kv_stats: 仿真结束时的 KV 账本统计。
        trace: 完整事件流（``record_trace=False`` 时为空）。
    """

    records: tuple[RequestRecord, ...]
    warnings: tuple[str, ...]
    num_engine_steps: int
    sim_start: float
    sim_end: float
    kv_stats: KVStats
    trace: EventLog


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
        self,
        engine: EngineConfig,
        perf: PerfModelConfig,
        record_trace: bool = True,
        *,
        use_prefix_cache: bool = False,
    ) -> None:
        self._kv = KVCache(engine.kv_token_budget)
        self._scheduler = FCFSScheduler(engine, perf, self._kv)
        self._cache_config: PrefixCacheConfig = engine.prefix_cache
        self._prefix_cache = (
            HierarchicalKVCache(
                engine.prefix_cache.l1.capacity_blocks, engine.prefix_cache.block_size_tokens
            )
            if use_prefix_cache
            else None
        )
        self._prefix_scheduler = (
            PrefixScheduler(engine, perf, self._prefix_cache)
            if self._prefix_cache is not None
            else None
        )
        self._namespace = cache_namespace(engine.prefix_cache)
        self._last_hash: dict[int, str] = {}
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
        self._overflow_episodes_seen = 0
        self._warnings: list[str] = []

    @property
    def has_work(self) -> bool:
        """是否有等待或执行中的请求（不含等待完成结算的请求）。"""
        waiting = (
            self._prefix_scheduler.num_waiting
            if self._prefix_scheduler is not None
            else self._scheduler.num_waiting
        )
        return bool(self._running or waiting)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def submit(self, request: InferenceRequest, now_ms: float) -> None:
        """只在客户端真正发送时转换为旧调度器接受的长度到达。"""
        if request.request_id in self._requests:
            raise ValueError(f"request_id 重复: {request.request_id}")
        arrival = Arrival(
            request_id=request.request_id,
            sim_time=now_ms,
            prompt_len=len(request.input_token_ids),
            output_len=len(request.output_token_plan),
        )
        if self._prefix_scheduler is not None:
            self._prefix_scheduler.enqueue(request, now_ms)
        else:
            self._scheduler.enqueue(arrival)
        self._requests[request.request_id] = request
        self._arrivals[request.request_id] = arrival
        self._log.record(ArrivalTrace.of(arrival))

    def step(self, now_ms: float, step_index: int) -> tuple[float, tuple[int, ...]]:
        """执行一步，返回步长和待在步末结算的完成 ID。"""
        if step_index != self._num_steps:
            raise ValueError(f"预期 step_index={self._num_steps}，得到 {step_index}")
        if self._prefix_scheduler is not None:
            return self._step_with_cache(now_ms, step_index)
        plan = self._scheduler.plan_step()
        if plan is None:  # pragma: no cover - 调度逻辑保证有工作才压步
            raise RuntimeError(f"step {step_index} @{now_ms}: 引擎步被调度但无工作")

        end = now_ms + plan.duration
        running = self._running
        arrivals = self._arrivals

        # decode：步开始时的驻留请求各产出 1 token，KV 各 +1
        decoding = list(running)
        finished: list[int] = []
        for request_id in decoding:
            produced = running[request_id] + 1
            running[request_id] = produced
            if produced >= arrivals[request_id].output_len:
                finished.append(request_id)
        overflow = self._kv.advance(decoding)

        # prefill：本步准入请求产出首 token
        for arrival in plan.admitted:
            request_id = arrival.request_id
            self._prefill_start[request_id] = now_ms
            self._prefill_end[request_id] = end
            running[request_id] = 1
            if arrival.output_len == 1:
                finished.append(request_id)

        for request_id in plan.forced:
            prompt_len = arrivals[request_id].prompt_len
            self._warnings.append(
                f"t={now_ms:.3f}ms step {step_index}: 请求 {request_id} "
                f"的 prompt 长度 {prompt_len} 超过 KV 预算 {self._kv.budget}，"
                "已在空 batch 中强制准入"
            )
        if overflow > 0:
            # 强制准入引起的超限已有专门告警，不再重复报超限段
            self._warn_overflow(now_ms, step_index, announce=not plan.forced)

        self._log.record(
            StepTrace(
                time=now_ms,
                step_index=step_index,
                duration=plan.duration,
                admitted=tuple(a.request_id for a in plan.admitted),
                forced=plan.forced,
                num_decoding=plan.num_decoding,
                batch_total_tokens=plan.batch_total_tokens,
                kv_used=self._kv.used,
                kv_overflow=overflow,
                finished=tuple(finished),
            )
        )

        for request_id in finished:
            del running[request_id]
            self._pending_done[request_id] = end

        self._num_steps += 1
        return plan.duration, tuple(finished)

    def _step_with_cache(self, now_ms: float, step_index: int) -> tuple[float, tuple[int, ...]]:
        scheduler = self._prefix_scheduler
        cache = self._prefix_cache
        assert scheduler is not None and cache is not None
        plan = scheduler.plan_step()
        if plan is None:
            raise RuntimeError(f"step {step_index} @{now_ms}: 引擎步被调度但无工作")
        end = now_ms + plan.duration
        decoding = list(self._running)
        finished: list[int] = []
        for request_id in decoding:
            produced = self._running[request_id] + 1
            self._running[request_id] = produced
            self._publish_output(request_id, produced)
            if produced == len(self._requests[request_id].output_token_plan):
                finished.append(request_id)

        for request, probe in zip(plan.admitted, plan.probes, strict=True):
            request_id = request.request_id
            self._prefill_start[request_id] = now_ms
            self._prefill_end[request_id] = end
            new_hashes = request.input_chunk_hashes[probe.matched_blocks :]
            cache.publish(request_id, new_hashes)
            self._last_hash[request_id] = (
                request.input_chunk_hashes[-1] if request.input_chunk_hashes else "ROOT"
            )
            self._running[request_id] = 1
            self._publish_output(request_id, 1)
            if len(request.output_token_plan) == 1:
                finished.append(request_id)

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
            )
        )
        for request_id in finished:
            del self._running[request_id]
            self._pending_done[request_id] = end
        self._num_steps += 1
        return plan.duration, tuple(finished)

    def _publish_output(self, request_id: int, produced: int) -> None:
        cache = self._prefix_cache
        assert cache is not None
        request = self._requests[request_id]
        cache.advance(request_id)
        total_tokens = len(request.input_token_ids) + produced
        block_size = self._cache_config.block_size_tokens
        if total_tokens % block_size == 0:
            context = request.input_token_ids + request.output_token_plan[:produced]
            digest = next_chunk_hash(
                self._namespace, self._last_hash[request_id], context[-block_size:]
            )
            cache.publish(request_id, (digest,))
            self._last_hash[request_id] = digest

    def complete(self, now_ms: float, request_id: int) -> RequestCompletion:
        """在 Done 事件时释放 KV，返回本请求实际已生成的内容。"""
        expected = self._pending_done.get(request_id)
        if expected is None or now_ms != expected:
            raise ValueError(f"请求 {request_id} 未待结算或完成时刻不符")
        if self._prefix_cache is not None:
            released = self._prefix_cache.request_context_tokens(request_id)
            self._prefix_cache.release(request_id)
            del self._last_hash[request_id]
        else:
            released = self._kv.release(request_id)
        self._log.record(DoneTrace(now_ms, request_id, released))
        del self._pending_done[request_id]
        self._finish[request_id] = now_ms
        request = self._requests[request_id]
        return RequestCompletion(request_id, request.output_token_plan, now_ms)

    def _warn_overflow(self, time: float, step_index: int, *, announce: bool) -> None:
        """每个超限段只告警一次，避免逐步刷屏。"""
        stats = self._kv.stats()
        if stats.overflow_episodes > self._overflow_episodes_seen:
            self._overflow_episodes_seen = stats.overflow_episodes
            if not announce:
                return
            self._warnings.append(
                f"t={time:.3f}ms step {step_index}: KV 占用 {stats.used} 超出预算 {stats.budget}，"
                "在跑请求继续执行（不做 OOM stall），新请求暂停准入"
            )

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
            kv_stats=self._kv.stats(),
            trace=self._log,
        )
