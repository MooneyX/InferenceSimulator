"""DES 内核：事件队列、时钟推进与 ``EngineStep`` 编排。

事件循环（见 ``docs/design/kernel.md``）：

- 启动时将全部到达物化为 ``ARRIVAL`` 事件压入堆。
- ``ARRIVAL``：请求入等待队列；若无待执行的 ``ENGINE_STEP``，在 ``max(当前时刻, 上一步终点)``
  调度一步——上一步可能已清空 batch 但尚未结束，新步不得与之重叠。
- ``ENGINE_STEP``：调度准入 → 推进在跑请求 decode（KV +1）→ 判定完成。完成请求在步末
  时刻压入 ``REQUEST_DONE``；仍有工作时在步末自再调度下一步，否则引擎转为空闲，
  时钟由堆直接跳跃到下一个 ``ARRIVAL``。
- ``REQUEST_DONE``：归还 KV 额度。同刻优先级保证它先于步末的下一个 ``ENGINE_STEP`` 处理，
  故每步开始时驻留请求均处于 decode 阶段。

堆操作次数 ≈ 到达数 + 引擎步数 + 完成数，与 batch 内请求步数无关。

所有权：``KVCache``、``FCFSScheduler``、``EventFactory``、``EventLog`` 均由本模块构造并持有；
``EventLog`` 随 ``EngineRun`` 移交调用方。

依赖层级：依赖 ``schemas``、``kv_cache``、``scheduler``、``events``。
"""

import heapq
from collections.abc import Iterable
from dataclasses import dataclass

from simcore.events import (
    ArrivalTrace,
    DoneTrace,
    Event,
    EventFactory,
    EventKind,
    EventLog,
    StepTrace,
)
from simcore.kv_cache import KVCache, KVStats
from simcore.scheduler import FCFSScheduler
from simcore.schemas import Arrival, EngineConfig, PerfModelConfig, RequestRecord

__all__ = ["EngineRun", "run_engine"]


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
    return _Engine(engine, perf, arrivals, record_trace=record_trace).run()


def _first_duplicate(ids: Iterable[int]) -> int:
    seen: set[int] = set()
    for request_id in ids:
        if request_id in seen:
            return request_id
        seen.add(request_id)
    raise AssertionError("未找到重复 ID")  # pragma: no cover


class _Engine:
    """单场仿真的可变状态。一次性使用：``run`` 只能调用一次。"""

    def __init__(
        self,
        engine: EngineConfig,
        perf: PerfModelConfig,
        arrivals: Iterable[Arrival],
        *,
        record_trace: bool,
    ) -> None:
        self._kv = KVCache(engine.kv_token_budget)
        self._scheduler = FCFSScheduler(engine, perf, self._kv)
        self._factory = EventFactory()
        self._log = EventLog(enabled=record_trace)
        self._heap: list[Event] = []

        self._arrivals: dict[int, Arrival] = {}
        for arrival in arrivals:
            self._arrivals[arrival.request_id] = arrival
            self._heap.append(self._factory.arrival(arrival.sim_time, arrival.request_id))
        # 重复 ID 会在 dict 中静默覆盖：循环内不逐个查，事后一次计数比对，出错才定位
        if len(self._arrivals) != len(self._heap):
            raise ValueError(f"request_id 重复: {_first_duplicate(e.ref for e in self._heap)}")
        heapq.heapify(self._heap)

        # request_id -> 已产出 token 数；插入顺序即准入顺序，决定 decode 推进顺序
        self._running: dict[int, int] = {}
        self._prefill_start: dict[int, float] = {}
        self._prefill_end: dict[int, float] = {}
        self._finish: dict[int, float] = {}

        self._step_pending = False
        # 最近一步的终点：引擎在此之前忙碌，空闲期到达的请求须等到此刻才能开步
        self._busy_until = float("-inf")
        self._num_steps = 0
        self._overflow_episodes_seen = 0
        self._warnings: list[str] = []

    # ------------------------------------------------------------ 主循环

    def run(self) -> EngineRun:
        heap = self._heap
        pop = heapq.heappop
        while heap:
            event = pop(heap)
            kind = event.kind
            if kind is EventKind.ENGINE_STEP:
                self._on_engine_step(event.time, event.ref)
            elif kind is EventKind.ARRIVAL:
                self._on_arrival(event.time, event.ref)
            else:
                self._on_request_done(event.time, event.ref)
        return self._finalize()

    def _schedule_step(self, time: float) -> None:
        heapq.heappush(self._heap, self._factory.engine_step(time, self._num_steps))
        self._step_pending = True

    # ------------------------------------------------------------ 事件处理

    def _on_arrival(self, time: float, request_id: int) -> None:
        arrival = self._arrivals[request_id]
        self._scheduler.enqueue(arrival)
        self._log.record(ArrivalTrace.of(arrival))
        if not self._step_pending:
            # 上一步刚清空 batch 但尚未结束时，不得在其步内开新步
            self._schedule_step(max(time, self._busy_until))

    def _on_engine_step(self, time: float, step_index: int) -> None:
        self._step_pending = False
        plan = self._scheduler.plan_step()
        if plan is None:  # pragma: no cover - 调度逻辑保证有工作才压步
            raise RuntimeError(f"step {step_index} @{time}: 引擎步被调度但无工作")

        end = time + plan.duration
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
            self._prefill_start[request_id] = time
            self._prefill_end[request_id] = end
            running[request_id] = 1
            if arrival.output_len == 1:
                finished.append(request_id)

        for request_id in plan.forced:
            prompt_len = arrivals[request_id].prompt_len
            self._warnings.append(
                f"t={time:.3f}ms step {step_index}: 请求 {request_id} 的 prompt 长度 {prompt_len} "
                f"超过 KV 预算 {self._kv.budget}，已在空 batch 中强制准入"
            )
        if overflow > 0:
            # 强制准入引起的超限已有专门告警，不再重复报超限段
            self._warn_overflow(time, step_index, announce=not plan.forced)

        self._log.record(
            StepTrace(
                time=time,
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

        factory = self._factory
        push = heapq.heappush
        for request_id in finished:
            del running[request_id]
            self._finish[request_id] = end
            push(self._heap, factory.request_done(end, request_id))

        self._num_steps += 1
        self._busy_until = end
        if running or self._scheduler.num_waiting:
            self._schedule_step(end)

    def _on_request_done(self, time: float, request_id: int) -> None:
        released = self._kv.release(request_id)
        self._log.record(DoneTrace(time, request_id, released))

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

    def _finalize(self) -> EngineRun:
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
