"""唯一仿真时钟及事件堆：先完成、回调、发送，最后引擎步。"""

import heapq
from collections.abc import Iterable

from simcore.client import ClientDriver
from simcore.engine import Engine, EngineRun
from simcore.events import Event, EventFactory, EventKind
from simcore.schemas import ClientWakeup, RequestCompletion

__all__ = ["Coordinator", "SimulationCoordinator"]


class Coordinator:
    """编排闭环客户端和引擎，不持有缓存或请求内容。"""

    def __init__(self, engine: Engine, client: ClientDriver) -> None:
        self.engine = engine
        self.client = client
        self._factory = EventFactory()
        self._heap: list[Event] = []
        self._wakeups: dict[int, ClientWakeup] = {}
        self._completions: dict[int, RequestCompletion] = {}
        self._step_pending = False
        self._next_step = 0
        self._busy_until = float("-inf")
        self._enqueue_wakeups(client.initial_wakeups())

    def _enqueue_wakeups(self, wakeups: Iterable[ClientWakeup]) -> None:
        for wakeup in wakeups:
            if wakeup.turn_id in self._wakeups:
                raise ValueError(f"轮次 {wakeup.turn_id} 重复安排发送")
            self._wakeups[wakeup.turn_id] = wakeup
            heapq.heappush(
                self._heap,
                self._factory.client_send(wakeup.send_time_ms, wakeup.turn_id),
            )

    def _process_aborts(self) -> None:
        for request_id in self.engine.drain_aborts():
            self.engine.register_skipped(self.client.on_request_aborted(request_id))

    def _schedule_step(self, now_ms: float) -> None:
        if not self._step_pending:
            time = max(now_ms, self._busy_until)
            heapq.heappush(self._heap, self._factory.engine_step(time, self._next_step))
            self._step_pending = True

    def run(self) -> EngineRun:
        while self._heap:
            event = heapq.heappop(self._heap)
            now_ms = event.time
            if event.kind is EventKind.REQUEST_DONE:
                completion = self.engine.complete(now_ms, event.ref)
                self._completions[event.ref] = completion
                heapq.heappush(
                    self._heap,
                    self._factory.client_completion(now_ms, event.ref),
                )
            elif event.kind is EventKind.CLIENT_COMPLETION:
                completion = self._completions.pop(event.ref)
                self._enqueue_wakeups(self.client.on_request_completed(completion))
            elif event.kind is EventKind.CLIENT_SEND:
                wakeup = self._wakeups.pop(event.ref)
                accepted = self.engine.submit(self.client.on_wakeup(wakeup), now_ms)
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
            else:
                raise AssertionError(f"未知事件类型 {event.kind}")
        return self.engine.finalize()


SimulationCoordinator = Coordinator
