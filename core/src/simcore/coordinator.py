"""唯一仿真时钟及事件堆：先完成、回调，再按计划到达发送，最后引擎步。"""

import heapq
import random
from collections.abc import Iterable
from typing import cast

from simcore.client import ClientDriver, OpenLoopClient
from simcore.engine import Engine, EngineRun
from simcore.events import Event, EventFactory, EventKind
from simcore.prefix_hash import stable_derive
from simcore.schemas import ClientWakeup, RequestCompletion

__all__ = ["Coordinator", "SimulationCoordinator"]


class Coordinator:
    """编排开环/闭环客户端和引擎，不持有缓存或请求内容。"""

    def __init__(self, engine: Engine, client: ClientDriver | OpenLoopClient) -> None:
        self.engine = engine
        self.client = client
        self._factory = EventFactory()
        self._heap: list[Event] = []
        self._wakeups: dict[int, ClientWakeup] = {}
        self._completions: dict[int, RequestCompletion] = {}
        self._step_pending = False
        self._next_step = 0
        self._busy_until = float("-inf")
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
    ) -> None:
        for wakeup in wakeups:
            self._wakeups[wakeup.turn_id] = wakeup
            if completed_at_ms is not None and wakeup.send_time_ms == completed_at_ms:
                event = self._factory.client_child_send(wakeup.send_time_ms, wakeup.turn_id)
            else:
                event = self._factory.client_send(wakeup.send_time_ms, wakeup.turn_id)
            heapq.heappush(self._heap, event)

    def _process_aborts(self) -> None:
        for request_id in self.engine.drain_aborts():
            if isinstance(self.client, OpenLoopClient):
                self.client.on_request_aborted(request_id)
            else:
                self.engine.register_skipped(self.client.on_request_aborted(request_id))

    def _schedule_step(self, now_ms: float) -> None:
        if not self._step_pending:
            time = max(now_ms, self._busy_until)  # 引擎忙时到达的请求等当前步结束
            heapq.heappush(self._heap, self._factory.engine_step(time, self._next_step))
            self._step_pending = True

    def run(self) -> EngineRun:
        while self._heap:
            event = heapq.heappop(self._heap)
            now_ms = event.time
            if event.kind is EventKind.REQUEST_DONE:
                completion = self.engine.complete(now_ms, event.ref)  # 先结算并释放 KV
                self._completions[event.ref] = completion
                heapq.heappush(
                    self._heap,
                    self._factory.client_completion(now_ms, event.ref),
                )
            elif event.kind is EventKind.CLIENT_COMPLETION:  # 回调可在同刻唤醒子轮
                completion = self._completions.pop(event.ref)
                if isinstance(self.client, OpenLoopClient):
                    self.client.on_request_completed(completion)
                else:
                    self._enqueue_wakeups(
                        self.client.on_request_completed(completion), completed_at_ms=now_ms
                    )
            elif event.kind in (EventKind.CLIENT_CHILD_SEND, EventKind.CLIENT_SEND):
                wakeup = self._wakeups.pop(event.ref)
                if not isinstance(self.client, ClientDriver):
                    raise RuntimeError("开环客户端不接受闭环唤醒")
                accepted = self.engine.submit(self.client.on_wakeup(wakeup), now_ms)
                self._process_aborts()
                if accepted:
                    self._schedule_step(now_ms)  # 同刻发送先全部入队，再执行引擎步
            elif event.kind is EventKind.OPEN_LOOP_TICK:
                if not isinstance(self.client, OpenLoopClient):
                    raise RuntimeError("闭环客户端不接受开环时钟")
                request, output_len = self.client.send(now_ms)
                self.engine.register_output_target(request.request_id, output_len)
                accepted = self.engine.submit(request, now_ms)
                self._process_aborts()
                if accepted:
                    self._schedule_step(now_ms)
            elif event.kind is EventKind.ENGINE_STEP:
                self._step_pending = False
                duration, finished = self.engine.step(now_ms, event.ref)
                self._process_aborts()
                end = now_ms + duration
                self._busy_until = end
                self._next_step = self.engine.num_steps
                for request_id in finished:
                    heapq.heappush(self._heap, self._factory.request_done(end, request_id))
                if self.engine.has_work:
                    self._schedule_step(end)
        return self.engine.finalize()


SimulationCoordinator = Coordinator
