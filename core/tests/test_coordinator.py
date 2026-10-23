"""真实完成触发会话后续轮次与统一时钟同刻事件顺序。"""

from itertools import pairwise
from unittest.mock import patch

import pytest

from simcore import simulate
from simcore.capacity import prepare_runtime
from simcore.client import Client
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.events import EventFactory, EventKind
from simcore.router import Router
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    PerfModelConfig,
    SessionConfig,
    SimConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)
from simcore.workload import build_synthetic_closed_loop_plan
from test_client import planned_client, synthetic_workload, trace_plan
from test_schemas import resolved_config, runtime_config


def config(think: float = 0) -> SimConfig:
    return resolved_config(
        num_instances=1,
        workload=WorkloadConfig(
            session_rate=10,
            seed=21,
            synthetic=SyntheticWorkloadConfig(
                num_requests=3,
                prompt_len=FixedLength(value=4),
                turn_increment_len=FixedLength(value=1),
                output_len=FixedLength(value=2),
                session=SessionConfig(num_sessions=1, think_time_ms=think),
            ),
        ),
        engine=EngineConfig(max_running_requests=1, max_total_tokens=200, page_size=4),
        perf_model=PerfModelConfig(prefill_c1=1, decode_d1=5, decode_d2=0),
    )


@pytest.mark.parametrize("source", ["synthetic", "trace"])
@pytest.mark.parametrize("mode", ["closed_loop", "open_loop"])
def test_all_client_modes_use_send_events_with_time_and_request_id(source: str, mode: str) -> None:
    engine_config = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2)
    workload = (
        synthetic_workload(mode)
        if source == "synthetic"
        else WorkloadConfig.model_validate(
            {
                "mode": "trace",
                "arrival_mode": mode,
                "session_rate": 10,
                "seed": 42,
                "trace": {},
            }
        )
    )
    client = Client(workload, engine_config, trace_plan=trace_plan() if source == "trace" else None)
    initial = client.initial_wakeups()
    engine = Engine(
        runtime_config(engine_config, PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=0)),
    )
    with (
        patch.object(
            EventFactory, "client_send", autospec=True, side_effect=EventFactory.client_send
        ) as sends,
        patch.object(engine, "submit", wraps=engine.submit) as submissions,
    ):
        coordinator = Coordinator((engine,), client, Router(1))
        assert {event.kind for event in coordinator._heap} == {EventKind.CLIENT_SEND}
        assert sorted((event.time, event.ref) for event in coordinator._heap) == sorted(
            (wakeup.send_time_ms, wakeup.turn_id) for wakeup in initial
        )
        if source == "synthetic" and mode == "open_loop":
            assert client.plan.turns == {}
        run = coordinator.run()[0]
    count = 4 if source == "synthetic" else 3
    assert sends.call_count == len(run.records) == count
    assert sorted((call.args[1], call.args[2]) for call in sends.call_args_list) == sorted(
        (call.args[1], call.args[0].request_id) for call in submissions.call_args_list
    )
    assert sorted(call.args[0].request_id for call in submissions.call_args_list) == list(
        range(count)
    )
    assert not run.aborted_requests and not run.skipped_requests


def test_single_session_children_arrive_only_after_completion() -> None:
    cfg = config(think=12)
    first = simulate(cfg)
    second = simulate(cfg)
    exclude = {"meta": {"wall_clock_ms"}}
    assert first.model_dump_json(exclude=exclude) == second.model_dump_json(exclude=exclude)
    records = first.requests
    workload = cfg.workload.synthetic
    assert workload is not None
    plan = build_synthetic_closed_loop_plan(cfg.workload, prepare_runtime(cfg).engine)
    assert len(records) == 3
    for parent, child in pairwise(records):
        delay = plan.turns[child.id].think_time_ms
        assert delay is not None
        assert child.arrival == pytest.approx(parent.finish + delay)
    assert first.meta.num_requests == 3


def test_zero_think_done_callback_send_step_order() -> None:
    cfg = config()
    workload = cfg.workload.synthetic
    assert workload is not None
    plan = build_synthetic_closed_loop_plan(cfg.workload, prepare_runtime(cfg).engine)
    client = planned_client(plan, prepare_runtime(cfg).engine)
    from simcore.router import Router

    run = Coordinator(
        (Engine(prepare_runtime(cfg)),),
        client,
        Router(1),
    ).run()[0]
    records = run.records
    assert records[0].finish == records[1].arrival
    assert records[1].prefill_start == records[1].arrival
    assert run.num_engine_steps == 6
    assert [record.id for record in records] == [0, 1, 2]
