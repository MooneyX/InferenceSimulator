"""events 模块单测：同刻全序、序号分配、事件流时序不变式与查询。"""

import heapq
import random

import pytest

from simcore.events import Event, EventFactory, EventKind


class TestOrdering:
    def test_time_dominates(self) -> None:
        f = EventFactory()
        later_done = f.request_done(2.0, 0)
        earlier_step = f.engine_step(1.0, 0)
        assert earlier_step < later_done

    def test_same_time_kind_priority(self) -> None:
        """同刻先发布与回调，再传输、子轮发送、普通发送和开环时钟，最后步进。"""
        f = EventFactory()
        step = f.engine_step(5.0, 0)
        send = f.client_send(5.0, 1)
        done = f.request_done(5.0, 2)
        end = f.step_done(5.0, 0)
        callback = f.client_completion(5.0, 2)
        lookup = f.l3_lookup_done(5.0, 1)
        l3_l2 = f.l3_l2_done(5.0, 3)
        l2_l1 = f.l2_l1_done(5.0, 4)
        child = f.client_child_send(5.0, 5)
        tick = f.open_loop_tick(5.0, 0)
        assert sorted([step, send, done, end, callback, lookup, l3_l2, l2_l1, child, tick]) == [
            end,
            done,
            callback,
            lookup,
            l3_l2,
            l2_l1,
            child,
            send,
            tick,
            step,
        ]

    def test_client_callbacks_and_immediate_children_precede_normal_send(self) -> None:
        f = EventFactory()
        step = f.engine_step(5.0, 0)
        send = f.client_send(5.0, 3)
        child = f.client_child_send(5.0, 4)
        callback = f.client_completion(5.0, 2)
        done = f.request_done(5.0, 2)
        assert sorted((step, send, child, callback, done)) == [done, callback, child, send, step]

    def test_same_time_same_kind_by_request_id_then_creation(self) -> None:
        f = EventFactory()
        high_id = f.client_send(5.0, 9)
        low_id = f.client_send(5.0, 3)
        assert low_id < high_id
        assert f.request_done(5.0, 1) < f.request_done(5.0, 2)
        first = f.client_send(5.0, 7)
        second = f.client_send(5.0, 7)
        assert first < second

    def test_heap_pop_order_is_deterministic(self) -> None:
        """随机打乱入堆顺序，出堆序列恒定 —— 可复现性的直接保证。"""
        f = EventFactory()
        events = [
            f.request_done(1.0, 0),
            f.client_send(1.0, 1),
            f.engine_step(1.0, 0),
            f.client_send(0.5, 2),
            f.client_send(1.0, 3),
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
        assert (
            EventKind.REQUEST_DONE
            < EventKind.CLIENT_COMPLETION
            < EventKind.CLIENT_CHILD_SEND
            < EventKind.CLIENT_SEND
            < EventKind.ENGINE_STEP
        )


class TestEventFactory:
    def test_seq_monotonic_and_unique(self) -> None:
        f = EventFactory()
        events = [f.client_send(0.0, 0), f.engine_step(0.0, 0), f.request_done(0.0, 0)]
        assert [e.seq for e in events] == [0, 1, 2]
        assert f.client_send(0.0, 0).seq == 3

    def test_kind_and_ref(self) -> None:
        f = EventFactory()
        assert (f.client_send(1.0, 7).kind, f.client_send(1.0, 7).ref) == (EventKind.CLIENT_SEND, 7)
        assert f.engine_step(1.0, 4).kind is EventKind.ENGINE_STEP
        assert f.request_done(1.0, 2).kind is EventKind.REQUEST_DONE
        assert f.client_child_send(1.0, 3).kind is EventKind.CLIENT_CHILD_SEND

    def test_independent_factories(self) -> None:
        assert EventFactory().client_send(0.0, 0).seq == EventFactory().client_send(0.0, 0).seq == 0

    def test_event_is_immutable(self) -> None:
        e = EventFactory().client_send(0.0, 0)
        with pytest.raises(AttributeError):
            e.time = 1.0  # type: ignore[misc]
