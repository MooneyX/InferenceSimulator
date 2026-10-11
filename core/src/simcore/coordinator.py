"""一条全局事件时间线编排多个独占 KV 的推理实例。"""

import heapq
import random
from collections.abc import Iterable, Sequence
from typing import cast

from simcore.client import ClientDriver, OpenLoopClient
from simcore.engine import Engine, EngineRun
from simcore.events import Event, EventFactory, EventKind
from simcore.prefix_hash import stable_derive
from simcore.router import Router
from simcore.schemas import ClientWakeup, InferenceRequest, RequestCompletion

__all__ = ["Coordinator", "SimulationCoordinator"]


class Coordinator:
    """持有唯一客户端和事件堆；每个实例独立安排步进与完成结算。"""

    def __init__(
        self, engines: Sequence[Engine], client: ClientDriver | OpenLoopClient, router: Router
    ) -> None:  # 引擎副本、客户端、选路器 → 建立统一事件队列；无返回值
        self.engines = tuple(engines)
        self.client = client
        self.router = router
        self._factory = EventFactory()
        self._heap: list[Event] = []
        self._wakeups: dict[int, ClientWakeup] = {}
        self._completions: dict[int, RequestCompletion] = {}
        self._step_pending = [False] * len(self.engines)
        self._busy_until = [float("-inf")] * len(self.engines)
        self._transfer_owners: dict[int, tuple[int, int]] = {}
        self._next_transfer_id = 0
        if isinstance(client, OpenLoopClient):
            arrival_rng = random.Random(stable_derive(client.config.seed, "open-loop-arrival"))
            rate_per_ms = cast(float, client.config.arrival_rate) / 1000.0
            clock = 0.0
            for index in range(client.config.num_requests):
                clock += arrival_rng.expovariate(rate_per_ms)
                heapq.heappush(self._heap, self._factory.open_loop_tick(clock, index))
        else:
            self._enqueue_wakeups(client.initial_wakeups())

    def _enqueue_wakeups(
        self, wakeups: Iterable[ClientWakeup], *, completed_at_ms: float | None = None
    ) -> None:  # 唤醒序列与可选父轮完成时刻 → 按事件优先级入堆；无返回值
        for wakeup in wakeups:
            self._wakeups[wakeup.turn_id] = wakeup
            if completed_at_ms is not None and wakeup.send_time_ms == completed_at_ms:
                event = self._factory.client_child_send(wakeup.send_time_ms, wakeup.turn_id)
            else:
                event = self._factory.client_send(wakeup.send_time_ms, wakeup.turn_id)
            heapq.heappush(self._heap, event)

    def _process_aborts(self, instance_id: int) -> None:  # 副本 ID → 通知客户端失败并标记后代；无返回值
        engine = self.engines[instance_id]
        for request_id in engine.drain_aborts():
            if isinstance(self.client, OpenLoopClient):
                self.client.on_request_aborted(request_id)
            else:
                engine.register_skipped(self.client.on_request_aborted(request_id))

    def _schedule_step(self, instance_id: int, now_ms: float) -> None:  # 副本 ID 与当前时刻 → 安排可运行的引擎步；无返回值
        if not self._step_pending[instance_id] and self.engines[instance_id].can_step:
            time = max(now_ms, self._busy_until[instance_id])
            heapq.heappush(self._heap, self._factory.engine_step(time, instance_id))
            self._step_pending[instance_id] = True

    def _drain_engine_events(self, instance_id: int) -> None:  # 副本 ID → 将局部页传输/查询事件映射进全局堆；无返回值
        for time, kind, local_ref in self.engines[instance_id].drain_events():
            if kind is EventKind.L3_LOOKUP_DONE:
                event = self._factory.l3_lookup_done(time, local_ref)
            elif kind in (EventKind.L3_L2_DONE, EventKind.L2_L1_DONE):
                transfer_id = self._next_transfer_id
                self._next_transfer_id += 1
                self._transfer_owners[transfer_id] = (instance_id, local_ref)
                event = (
                    self._factory.l3_l2_done(time, transfer_id)
                    if kind is EventKind.L3_L2_DONE
                    else self._factory.l2_l1_done(time, transfer_id)
                )
            else:
                raise ValueError(f"未知的引擎事件类型: {kind}")
            heapq.heappush(self._heap, event)

    def _submit(self, request: InferenceRequest, output_len: int, now_ms: float) -> None:  # 请求、实际输出长度和时刻 → 选路并投放副本；无返回值
        instance_id = self.router.select(request)
        engine = self.engines[instance_id]
        engine.register_output_target(request.request_id, output_len)
        accepted = engine.submit(request, now_ms)
        self._process_aborts(instance_id)
        self._drain_engine_events(instance_id)
        if accepted:
            self._schedule_step(instance_id, now_ms)

    def run(self) -> tuple[EngineRun, ...]:  # 消费全局事件堆直至清空 → 每副本的原始仿真结果
        while self._heap:
            event = heapq.heappop(self._heap)
            now_ms = event.time
            if event.kind is EventKind.STEP_DONE:
                instance_id = event.ref
                for request_id in self.engines[instance_id].finish_step(now_ms): #finish_step提交本步的prefill计算过程中产生的新的KV块
                    heapq.heappush(self._heap, self._factory.request_done(now_ms, request_id))
                self._drain_engine_events(instance_id)
                self._schedule_step(instance_id, now_ms)
            elif event.kind is EventKind.REQUEST_DONE:
                instance_id = self.router.instance_of(event.ref)
                engine = self.engines[instance_id]
                completion = engine.complete(now_ms, event.ref)
                self._completions[event.ref] = completion
                heapq.heappush(self._heap, self._factory.client_completion(now_ms, event.ref))
                self._schedule_step(instance_id, now_ms) # 可能冗余
            elif event.kind is EventKind.CLIENT_COMPLETION:
                completion = self._completions.pop(event.ref)
                if isinstance(self.client, OpenLoopClient):
                    self.client.on_request_completed(completion)
                else:
                    self._enqueue_wakeups(
                        self.client.on_request_completed(completion), completed_at_ms=now_ms
                    )
            elif event.kind is EventKind.L3_LOOKUP_DONE:
                instance_id = self.router.instance_of(event.ref)
                self.engines[instance_id].on_l3_lookup(event.ref, now_ms)
                self._drain_engine_events(instance_id)
                self._schedule_step(instance_id, now_ms)
            elif event.kind in (EventKind.L3_L2_DONE, EventKind.L2_L1_DONE):
                instance_id, local_ref = self._transfer_owners.pop(event.ref)
                self.engines[instance_id].on_transfer_done(event.kind, local_ref, now_ms)
                self._drain_engine_events(instance_id)
                self._schedule_step(instance_id, now_ms)
            elif event.kind in (EventKind.CLIENT_CHILD_SEND, EventKind.CLIENT_SEND):
                wakeup = self._wakeups.pop(event.ref)
                if not isinstance(self.client, ClientDriver):
                    raise RuntimeError("开环客户端不接受闭环唤醒")
                request = self.client.on_wakeup(wakeup)
                self._submit(request, self.client.plan.turns[request.request_id].output_len, now_ms)
            elif event.kind is EventKind.OPEN_LOOP_TICK:
                if not isinstance(self.client, OpenLoopClient):
                    raise RuntimeError("闭环客户端不接受开环时钟")
                request, output_len = self.client.send(now_ms)
                self._submit(request, output_len, now_ms)
            elif event.kind is EventKind.ENGINE_STEP:
                instance_id = event.ref
                engine = self.engines[instance_id]
                self._step_pending[instance_id] = False
                if not engine.can_step:
                    continue
                started = engine.start_step(now_ms, engine.num_steps)
                self._process_aborts(instance_id)
                self._drain_engine_events(instance_id)
                if started is None:
                    self._schedule_step(instance_id, now_ms)
                    continue
                duration, _ = started
                end = now_ms + duration
                self._busy_until[instance_id] = end
                heapq.heappush(self._heap, self._factory.step_done(end, instance_id))
        return tuple(engine.finalize() for engine in self.engines)


SimulationCoordinator = Coordinator
