"""F001 同刻完成、客户端回调、子轮发送和单步调度全序验收。"""

from unittest.mock import patch

from simcore.client import ClientDriver
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.events import ArrivalTrace, DoneTrace, EventKind, StepTrace
from simcore.router import Router
from simcore.schemas import EngineConfig, PerfModelConfig, TurnPlan, WorkloadPlan


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
        config, perf, output_targets={key: item.output_len for key, item in plan.turns.items()}
    )
    return Coordinator((engine,), ClientDriver(plan, config), Router(1)), engine


def test_parent_done_callback_and_immediate_children_precede_other_send() -> None:
    coordinator, _ = make_run(make_plan())
    with patch.object(
        coordinator.client,
        "on_request_completed",
        wraps=coordinator.client.on_request_completed,
    ) as callbacks:
        run = coordinator.run()[0]
    assert [
        call.args[0].request_id
        for call in callbacks.call_args_list
        if call.args[0].finish_time_ms == 8
    ] == [0, 1]
    same_time = [entry for entry in run.trace if entry.time == 8]
    assert [
        (type(item), item.request_id) for item in same_time if not isinstance(item, StepTrace)
    ] == [
        (DoneTrace, 0),
        (DoneTrace, 1),
        (ArrivalTrace, 3),
        (ArrivalTrace, 4),
        (ArrivalTrace, 2),
    ]
    assert isinstance(same_time[-1], StepTrace)
    assert same_time[-1].admitted == (2, 3, 4)
    assert run.records[3].actual_hit_tokens == run.records[4].actual_hit_tokens == 4
    assert run.records[3].arrival == run.records[0].finish == 8
    assert run.num_engine_steps == 2


def test_future_think_time_child_is_ordered_like_other_same_time_send() -> None:
    coordinator, _ = make_run(make_plan(ordinary_at=12, think=4))
    run = coordinator.run()[0]
    assert [
        item.request_id for item in run.trace if isinstance(item, ArrivalTrace) and item.time == 12
    ] == [2, 3, 4]
    assert [step.time for step in run.trace.steps()] == [0, 12]
    assert run.records[3].arrival == run.records[0].finish + 4


def test_same_time_roots_are_sent_in_id_order_even_if_created_in_reverse() -> None:
    coordinator, _ = make_run(make_plan(ordinary_at=0))
    run = coordinator.run()[0]
    assert [
        entry.request_id
        for entry in run.trace
        if isinstance(entry, ArrivalTrace) and entry.time == 0
    ] == [0, 1, 2]
    assert next(run.trace.steps()).admitted == (0, 1, 2)


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
        config, zero, output_targets={key: item.output_len for key, item in plan.turns.items()}
    )
    coordinator = CheckedCoordinator((engine,), ClientDriver(plan, config), Router(1))
    run = coordinator.run()[0]
    assert coordinator.max_pending == 1
    assert run.num_engine_steps == 2
    assert [(item.time, item.step_index) for item in run.trace.steps()] == [(0, 0), (0, 1)]
    assert [item.request_id for item in run.trace if isinstance(item, DoneTrace)] == [0, 1, 2, 3, 4]
    assert run.records[3].arrival == run.records[0].finish == 0
    assert run.records[4].arrival == run.records[1].finish == 0
    assert list(run.trace) == list(
        CheckedCoordinator(
            (Engine(config, zero, output_targets={}),),
            ClientDriver(plan, config),
            Router(1),
        )
        .run()[0]
        .trace
    )
