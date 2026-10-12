"""F001 同刻完成、客户端回调、子轮发送和单步调度全序验收。"""

from unittest.mock import patch

import pytest

from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.events import EventKind
from simcore.kv_cache import CacheCapacityError, CacheEntry
from simcore.prefix_hash import encode
from simcore.router import Router
from simcore.schemas import EngineConfig, InferenceRequest, PerfModelConfig, TurnPlan, WorkloadPlan
from test_client import planned_client
from test_schemas import runtime_config


def make_plan(*, ordinary_at: float = 8, think: float = 0) -> WorkloadPlan:
    turns = {
        0: TurnPlan(0, 0, None, 0, None, 4, 1, 1, 0),
        1: TurnPlan(1, 1, None, 0, None, 4, 1, 1, 0),
        2: TurnPlan(2, 2, None, ordinary_at, None, 4, 1, 1, 0),
        3: TurnPlan(3, 0, 0, None, think, 5, 1, 1, 1),
        4: TurnPlan(4, 1, 1, None, think, 5, 1, 1, 1),
    }
    return WorkloadPlan(turns, (2, 1, 0), {0: (3,), 1: (4,)})


def make_run(plan: WorkloadPlan, *, zero_cost: bool = False) -> tuple[Coordinator, Engine]:
    config = EngineConfig(max_running_requests=5, max_total_tokens=128, page_size=4)
    perf = PerfModelConfig(
        prefill_c1=0 if zero_cost else 1,
        decode_d1=0,
        decode_d2=0,
    )
    engine = Engine(
        runtime_config(config, perf), output_targets={key: item.output_len for key, item in plan.turns.items()}
    )
    return Coordinator((engine,), planned_client(plan, config), Router(1)), engine


def test_parent_done_callback_and_immediate_children_precede_other_send() -> None:
    coordinator, engine = make_run(make_plan())
    with (
        patch.object(
            coordinator.client, "on_request_completed",
            wraps=coordinator.client.on_request_completed,
        ) as callbacks,
        patch.object(engine, "submit", wraps=engine.submit) as submissions,
    ):
        run = coordinator.run()[0]
    assert [
        call.args[0].request_id for call in submissions.call_args_list if call.args[1] == 8
    ] == [3, 4, 2]
    assert [
        call.args[0].request_id
        for call in callbacks.call_args_list
        if call.args[0].finish_time_ms == 8
    ] == [0, 1]
    assert [run.records[key].prefill_start for key in (2, 3, 4)] == [8, 8, 8]
    assert run.records[3].actual_hit_tokens == run.records[4].actual_hit_tokens == 4
    assert run.records[3].arrival == run.records[0].finish == 8
    assert run.num_engine_steps == 2


def test_future_think_time_child_is_ordered_like_other_same_time_send() -> None:
    coordinator, engine = make_run(make_plan(ordinary_at=12, think=4))
    with patch.object(engine, "submit", wraps=engine.submit) as submissions:
        run = coordinator.run()[0]
    assert [
        call.args[0].request_id for call in submissions.call_args_list if call.args[1] == 12
    ] == [2, 3, 4]
    assert [run.records[key].arrival for key in (2, 3, 4)] == [12, 12, 12]
    assert [run.records[key].prefill_start for key in (2, 3, 4)] == [12, 12, 12]
    assert run.num_engine_steps == 2
    assert run.records[3].arrival == run.records[0].finish + 4


def test_same_time_roots_are_sent_in_id_order_even_if_created_in_reverse() -> None:
    coordinator, _ = make_run(make_plan(ordinary_at=0))
    run = coordinator.run()[0]
    assert [run.records[key].arrival for key in (0, 1, 2)] == [0, 0, 0]
    assert [run.records[key].prefill_start for key in (0, 1, 2)] == [0, 0, 0]


@pytest.mark.parametrize("load_fails", [False, True])
def test_l2_hit_waits_for_transfer_before_child_prefill(load_fails: bool) -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=20,
        page_size=4,
        l2_capacity_tokens=8,
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=0,
        decode_d2=0,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l2_latency_ms=3,
    )
    plan = WorkloadPlan(
        turns={
            0: TurnPlan(0, 0, None, 0, None, 5, 1, 1, 0),
            1: TurnPlan(1, 0, 0, None, 45, 9, 1, 1, 1),
            2: TurnPlan(2, 2, None, 12, None, 12, 1, 1, 0),
            3: TurnPlan(3, 3, None, 26, None, 12, 1, 1, 0),
        },
        root_request_ids=(0, 2, 3),
        children={0: (1,)},
    )
    engine = Engine(runtime_config(config, perf), output_targets={key: 1 for key in plan.turns})
    if load_fails:
        with patch.object(
            engine._prefix_cache, "reserve_transfer", side_effect=CacheCapacityError()
        ):
            run = Coordinator((engine,), planned_client(plan, config), Router(1)).run()[0]
    else:
        run = Coordinator((engine,), planned_client(plan, config), Router(1)).run()[0]
    child = next(item for item in run.records if item.id == 1)
    assert child.arrival == 50
    assert child.prefill_start == pytest.approx(50 if load_fails else 55)
    assert (child.actual_hit_tokens, child.l2_hit_tokens) == ((0, 0) if load_fails else (4, 4))
    assert run.restore_time_ms == pytest.approx(0 if load_fails else 5)
    assert run.num_engine_steps == 4


@pytest.mark.parametrize("prefetch_fails", [False, True])
def test_wait_complete_l3_query_prefetch_then_l2_load(prefetch_fails: bool) -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=20,
        page_size=4,
        l2_capacity_tokens=8,
        l3_capacity_tokens=64,
        l3_prefetch_min_tokens=0,
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=0,
        decode_d2=0,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l3_to_l2_bandwidth_bytes_per_ms=2,
        l2_latency_ms=3,
        l3_latency_ms=5,
        l3_lookup_latency_ms=3,
    )
    plan = WorkloadPlan(
        turns={
            0: TurnPlan(0, 0, None, 0, None, 5, 1, 1, 0),
            1: TurnPlan(1, 0, 0, None, 70, 9, 1, 1, 1),
            2: TurnPlan(2, 2, None, 12, None, 12, 1, 1, 0),
            3: TurnPlan(3, 3, None, 26, None, 12, 1, 1, 0),
            4: TurnPlan(4, 4, None, 40, None, 12, 1, 1, 0),
        },
        root_request_ids=(0, 2, 3, 4),
        children={0: (1,)},
    )
    engine = Engine(runtime_config(config, perf), output_targets={key: 1 for key in plan.turns})
    if prefetch_fails:
        with patch.object(
            engine._prefix_cache, "reserve_transfer", side_effect=CacheCapacityError()
        ):
            run = Coordinator((engine,), planned_client(plan, config), Router(1)).run()[0]
    else:
        run = Coordinator((engine,), planned_client(plan, config), Router(1)).run()[0]
    child = next(item for item in run.records if item.id == 1)
    assert child.arrival == 78
    assert child.prefill_start == pytest.approx(81 if prefetch_fails else 95)
    assert (child.actual_hit_tokens, child.l3_hit_tokens) == ((0, 0) if prefetch_fails else (4, 4))
    assert run.restore_time_ms == pytest.approx(0 if prefetch_fails else 14)


def test_full_l2_restore_does_not_charge_a_second_transfer_step() -> None:
    config = EngineConfig(
        max_running_requests=2, max_total_tokens=16, page_size=4, l2_capacity_tokens=8
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=5,
        decode_d2=0,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l2_latency_ms=3,
    )
    engine = Engine(runtime_config(config, perf), output_targets={0: 2})
    item = InferenceRequest(0, 0, None, 4, 2, (), 0)
    engine.submit(item, 0)
    engine._prefix_scheduler.select_prefill({})
    cache = engine._prefix_cache
    cache.admit(0)
    cache.extend(0, 4)
    cache.publish(0)
    cache.withdraw(0)
    filler = InferenceRequest(99, 99, None, 16, 1, (), 0)
    cache.pin(filler, 0)
    cache.admit(99)
    cache.extend(99, 16)
    cache.publish(99)
    cache.finish(99)
    cached = cache.page(encode(0, 0))
    assert cached is not None and cached.tier == "L2"
    engine._prefix_scheduler.requeue(item, 1, generated_tokens=1)
    assert engine.start_step(1, 0) is None
    [(end, kind, transfer_id)] = engine.drain_events()
    assert kind is EventKind.L2_L1_DONE
    engine.on_transfer_done(kind, transfer_id, end)
    assert engine.start_step(end, 0) == (0.0, ())
    assert engine.finish_step(end) == ()
    assert engine._running == {0: 1}
    assert engine.num_steps == 0
    assert engine._restore_time_ms == pytest.approx(5)


def test_partial_load_failure_waits_for_inflight_page_then_recomputes() -> None:
    config = EngineConfig(
        max_running_requests=1, max_total_tokens=20, page_size=4, l2_capacity_tokens=12
    )
    perf = PerfModelConfig(
        prefill_c1=1,
        decode_d1=0,
        decode_d2=0,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
    )
    engine = Engine(runtime_config(config, perf), output_targets={1: 1})
    cache = engine._prefix_cache
    parent = InferenceRequest(0, 0, None, 8, 1, (), 0)
    cache.pin(parent, 0)
    cache.admit(0)
    cache.extend(0, 8)
    cache.publish(0)
    cache.finish(0)
    filler = InferenceRequest(99, 99, None, 20, 1, (), 0)
    cache.pin(filler, 0)
    cache.admit(99)
    cache.extend(99, 20)
    cache.publish(99)
    cache.finish(99)
    first, second = encode(0, 0), encode(0, 1)
    first_page, second_page = cache.page(first), cache.page(second)
    assert first_page is not None and second_page is not None
    assert first_page.tier == second_page.tier == "L2"
    child = InferenceRequest(1, 0, 0, 9, 1, (first, second), 2)
    engine.submit(child, 1)
    reserve = cache.reserve_transfer
    attempts = 0

    def fail_second(chunk_id: str, destination: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise CacheCapacityError("L1 空间不足")
        return reserve(chunk_id, destination)

    with patch.object(cache, "reserve_transfer", side_effect=fail_second):
        assert engine.start_step(1, 0) is None
    [(end, kind, transfer_id)] = engine.drain_events()
    assert kind is EventKind.L2_L1_DONE and not engine.can_step
    engine.on_transfer_done(kind, transfer_id, end)
    assert engine.can_step
    started = engine.start_step(end, 0)
    assert started is not None
    duration, _ = started
    assert engine.finish_step(end + duration) == (1,)
    assert engine._admission_stats[1] == (4, 4)


def test_arrival_during_gpu_step_cannot_observe_future_published_page() -> None:
    class ObservingEngine(Engine):
        observed_during_step: CacheEntry | None = None

        def submit(self, request: InferenceRequest, now_ms: float) -> bool:
            if request.request_id == 1:
                self.observed_during_step = self._prefix_cache.page("chunk:0:0")
            return super().submit(request, now_ms)

    config = EngineConfig(max_running_requests=2, max_total_tokens=128, page_size=4)
    perf = PerfModelConfig(prefill_c1=1, decode_d1=0, decode_d2=0)
    plan = WorkloadPlan(
        turns={
            0: TurnPlan(0, 0, None, 0, None, 8, 1, 1, 0),
            1: TurnPlan(1, 1, None, 1, None, 4, 1, 1, 0),
        },
        root_request_ids=(0, 1),
        children={},
    )
    engine = ObservingEngine(runtime_config(config, perf), output_targets={0: 1, 1: 1})
    run = Coordinator((engine,), planned_client(plan, config), Router(1)).run()[0]
    assert engine.observed_during_step is None
    assert engine._prefix_cache.page("chunk:0:0") is not None
    assert [item.prefill_start for item in run.records] == [0, 8]


def test_zero_cost_multiple_steps_same_timestamp_keep_one_pending_at_a_time() -> None:
    class CheckedCoordinator(Coordinator):
        max_pending = 0

        def _schedule_step(self, instance_id: int, now_ms: float) -> None:
            super()._schedule_step(instance_id, now_ms)
            count = sum(item.kind is EventKind.ENGINE_STEP for item in self._heap)
            self.max_pending = max(self.max_pending, count)
            assert count <= 1

    plan = make_plan(ordinary_at=0)
    config = EngineConfig(max_running_requests=5, max_total_tokens=128, page_size=4)
    zero = PerfModelConfig(prefill_c1=0, decode_d1=0, decode_d2=0)
    engine = Engine(
        runtime_config(config, zero), output_targets={key: item.output_len for key, item in plan.turns.items()}
    )
    coordinator = CheckedCoordinator((engine,), planned_client(plan, config), Router(1))
    run = coordinator.run()[0]
    assert coordinator.max_pending == 1
    assert run.num_engine_steps == 2
    assert [record.id for record in run.records] == [0, 1, 2, 3, 4]
    assert run.records[3].arrival == run.records[0].finish == 0
    assert run.records[4].arrival == run.records[1].finish == 0
    repeated = CheckedCoordinator(
        (Engine(runtime_config(config, zero), output_targets={}),),
        planned_client(plan, config), Router(1),
    ).run()[0]
    assert run == repeated
