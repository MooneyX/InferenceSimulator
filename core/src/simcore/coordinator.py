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
    ) -> None:
        self.engines = tuple(engines)
        self.client = client
        self.router = router
        self._factory = EventFactory()
        self._heap: list[Event] = []
        self._wakeups: dict[int, ClientWakeup] = {}
        self._completions: dict[int, RequestCompletion] = {}
        self._step_pending = [False] * len(self.engines)
        self._busy_until = [float("-inf")] * len(self.engines)
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

    def _process_aborts(self, instance_id: int) -> None:
        engine = self.engines[instance_id]
        for request_id in engine.drain_aborts():
            if isinstance(self.client, OpenLoopClient):
                self.client.on_request_aborted(request_id)
            else:
                engine.register_skipped(self.client.on_request_aborted(request_id))

    def _schedule_step(self, instance_id: int, now_ms: float) -> None:
        if not self._step_pending[instance_id]:
            time = max(now_ms, self._busy_until[instance_id])
            heapq.heappush(self._heap, self._factory.engine_step(time, instance_id))
            self._step_pending[instance_id] = True

    def _submit(self, request: InferenceRequest, output_len: int, now_ms: float) -> None:
        instance_id = self.router.select(request)
        engine = self.engines[instance_id]
        engine.register_output_target(request.request_id, output_len)
        accepted = engine.submit(request, now_ms)
        self._process_aborts(instance_id)
        if accepted:
            self._schedule_step(instance_id, now_ms)

    def run(self) -> tuple[EngineRun, ...]:
        while self._heap:
            event = heapq.heappop(self._heap)
            now_ms = event.time
            if event.kind is EventKind.REQUEST_DONE:
                engine = self.engines[self.router.instance_of(event.ref)]
                completion = engine.complete(now_ms, event.ref)
                self._completions[event.ref] = completion
                heapq.heappush(self._heap, self._factory.client_completion(now_ms, event.ref))
            elif event.kind is EventKind.CLIENT_COMPLETION:
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
                duration, finished = engine.step(now_ms, engine.num_steps)
                self._process_aborts(instance_id)
                end = now_ms + duration
                self._busy_until[instance_id] = end
                for request_id in finished:
                    heapq.heappush(self._heap, self._factory.request_done(end, request_id))
                if engine.has_work:
                    self._schedule_step(instance_id, end)
        return tuple(engine.finalize() for engine in self.engines)


SimulationCoordinator = Coordinator
