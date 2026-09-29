"""events 模块单测：同刻全序、序号分配、事件流时序不变式与查询。"""

import heapq
import random

import pytest

from simcore.events import (
    ArrivalTrace,
    DoneTrace,
    Event,
    EventFactory,
    EventKind,
    EventLog,
    StepTrace,
    TraceRecord,
)
from simcore.schemas import Arrival


def step_trace(
    time: float,
    step_index: int = 0,
    admitted: tuple[int, ...] = (),
    finished: tuple[int, ...] = (),
) -> StepTrace:
    return StepTrace(
        time=time,
        step_index=step_index,
        duration=5.0,
        admitted=admitted,
        forced=(),
        num_decoding=0,
        batch_total_tokens=10,
        kv_used=10,
        kv_overflow=0,
        finished=finished,
    )


class TestOrdering:
    def test_time_dominates(self) -> None:
        f = EventFactory()
        later_done = f.request_done(2.0, 0)
        earlier_step = f.engine_step(1.0, 0)
        assert earlier_step < later_done

    def test_same_time_kind_priority(self) -> None:
        """同刻：完成结算 → 到达 → 引擎步，与创建顺序无关。"""
        f = EventFactory()
        step = f.engine_step(5.0, 0)
        arrival = f.arrival(5.0, 1)
        done = f.request_done(5.0, 2)
        assert sorted([step, arrival, done]) == [done, arrival, step]

    def test_client_callbacks_precede_send_and_step(self) -> None:
        f = EventFactory()
        step = f.engine_step(5.0, 0)
        send = f.client_send(5.0, 3)
        callback = f.client_completion(5.0, 2)
        done = f.request_done(5.0, 2)
        assert sorted((step, send, callback, done)) == [done, callback, send, step]
        assert f.arrival(5.0, 3).kind == send.kind

    def test_same_time_same_kind_by_creation(self) -> None:
        f = EventFactory()
        a = f.arrival(5.0, 9)
        b = f.arrival(5.0, 3)
        assert a < b  # 按 seq，而非 request_id

    def test_heap_pop_order_is_deterministic(self) -> None:
        """随机打乱入堆顺序，出堆序列恒定 —— 可复现性的直接保证。"""
        f = EventFactory()
        events = [
            f.request_done(1.0, 0),
            f.arrival(1.0, 1),
            f.engine_step(1.0, 0),
            f.arrival(0.5, 2),
            f.arrival(1.0, 3),
            f.engine_step(2.0, 1),
        ]
        expected = sorted(events)
        for trial in range(20):
            shuffled = events[:]
            random.Random(trial).shuffle(shuffled)
            heap: list[Event] = []
            for e in shuffled:
                heapq.heappush(heap, e)
            assert [heapq.heappop(heap) for _ in range(len(heap))] == expected

    def test_kind_priority_values(self) -> None:
        assert EventKind.REQUEST_DONE < EventKind.ARRIVAL < EventKind.ENGINE_STEP


class TestEventFactory:
    def test_seq_monotonic_and_unique(self) -> None:
        f = EventFactory()
        events = [f.arrival(0.0, 0), f.engine_step(0.0, 0), f.request_done(0.0, 0)]
        assert [e.seq for e in events] == [0, 1, 2]
        assert f.issued == 3

    def test_kind_and_ref(self) -> None:
        f = EventFactory()
        assert (f.arrival(1.0, 7).kind, f.arrival(1.0, 7).ref) == (EventKind.ARRIVAL, 7)
        assert f.engine_step(1.0, 4).kind is EventKind.ENGINE_STEP
        assert f.request_done(1.0, 2).kind is EventKind.REQUEST_DONE

    def test_independent_factories(self) -> None:
        assert EventFactory().arrival(0.0, 0).seq == EventFactory().arrival(0.0, 0).seq == 0

    def test_event_is_immutable(self) -> None:
        e = EventFactory().arrival(0.0, 0)
        with pytest.raises(AttributeError):
            e.time = 1.0  # type: ignore[misc]


class TestTraceRecords:
    def test_arrival_trace_from_arrival(self) -> None:
        a = Arrival(request_id=3, sim_time=12.5, prompt_len=100, output_len=8)
        assert ArrivalTrace.of(a) == ArrivalTrace(12.5, 3, 100, 8)

    def test_step_trace_end(self) -> None:
        assert step_trace(10.0).end == 15.0


class TestEventLog:
    def test_records_in_order(self) -> None:
        log = EventLog()
        entries: list[TraceRecord] = [
            ArrivalTrace(1.0, 0, 10, 2),
            step_trace(1.0),
            DoneTrace(6.0, 0, 12),
        ]
        for e in entries:
            log.record(e)
        assert list(log) == entries and len(log) == 3

    def test_equal_time_accepted(self) -> None:
        log = EventLog()
        log.record(DoneTrace(3.0, 0, 5))
        log.record(ArrivalTrace(3.0, 1, 5, 1))
        assert len(log) == 2

    def test_time_regression_rejected(self) -> None:
        log = EventLog()
        log.record(ArrivalTrace(5.0, 0, 10, 2))
        with pytest.raises(ValueError, match="倒退"):
            log.record(ArrivalTrace(4.0, 1, 10, 2))

    def test_disabled_log_still_validates_but_stores_nothing(self) -> None:
        log = EventLog(enabled=False)
        log.record(ArrivalTrace(5.0, 0, 10, 2))
        assert len(log) == 0 and not log.enabled
        with pytest.raises(ValueError):
            log.record(ArrivalTrace(4.0, 1, 10, 2))

    def test_steps_filter(self) -> None:
        log = EventLog()
        log.record(ArrivalTrace(0.0, 0, 10, 2))
        log.record(step_trace(0.0, 0))
        log.record(step_trace(5.0, 1))
        assert [s.step_index for s in log.steps()] == [0, 1]

    def test_for_request(self) -> None:
        log = EventLog()
        log.record(ArrivalTrace(0.0, 0, 10, 2))
        log.record(ArrivalTrace(0.0, 1, 10, 2))
        log.record(step_trace(0.0, 0, admitted=(0, 1)))
        log.record(step_trace(5.0, 1, finished=(0,)))
        log.record(DoneTrace(10.0, 0, 12))
        log.record(step_trace(10.0, 2, finished=(1,)))
        history = log.for_request(0)
        assert [type(r).__name__ for r in history] == [
            "ArrivalTrace",
            "StepTrace",
            "StepTrace",
            "DoneTrace",
        ]
        assert len(log.for_request(1)) == 3
        assert log.for_request(99) == []
