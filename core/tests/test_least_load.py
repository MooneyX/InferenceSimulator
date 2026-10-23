"""最低 token 负载路由、事件快照与逐轮重新选路。"""

from dataclasses import replace

import pytest

from simcore import simulate
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.router import Router
from simcore.schemas import EngineConfig, PerfModelConfig, TurnPlan, WorkloadPlan
from test_client import planned_client
from test_f001_engine import make_engine, request
from test_f001_scheduler import make
from test_f001_scheduler import request as queued_request
from test_multi_instance import config
from test_schemas import runtime_config
from test_trace_replay import config as trace_config
from test_trace_replay import plan as trace_plan


def test_router_selects_global_minimum_and_rotates_ties() -> None:
    router = Router(4, policy="least_load")
    assert router.select(request(0), (10, 30, 5, 20)) == 2
    assert [router.select(request(key), (5, 30, 5, 20)) for key in range(1, 5)] == [0, 2, 0, 2]
    assert router.instance_of(4) == 2
    single = Router(1, policy="least_load")
    assert single.select(request(0), (100,)) == 0


@pytest.mark.parametrize("session_affinity", [False, True])
def test_router_ignores_parent_and_session_affinity(session_affinity: bool) -> None:
    router = Router(2, policy="least_load", session_affinity=session_affinity)
    assert router.select(request(0), (0, 0)) == 0
    child = replace(request(1), parent_request_id=0)
    assert router.select(child, (10, 0)) == 1
    assert router.instance_of(0) == 0 and router.instance_of(1) == 1


@pytest.mark.parametrize("exclusive", [True, False])
def test_simultaneous_sends_observe_previous_enqueue(exclusive: bool) -> None:
    cfg = EngineConfig(
        max_running_requests=4, max_total_tokens=128, page_size=4, pd_mutually_exclusive=exclusive
    )
    runtime = runtime_config(cfg, PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=0))
    turns = {
        key: TurnPlan(key, key, None, 0, None, prompt, 1, 1, 0)
        for key, prompt in enumerate((20, 4, 4, 4))
    }
    client = planned_client(WorkloadPlan(turns, tuple(turns), {}), cfg)
    engines = tuple(Engine(runtime, instance_id=index) for index in range(2))
    router = Router(2, policy="least_load")
    runs = Coordinator(engines, client, router).run()
    assert [router.instance_of(key) for key in turns] == [0, 1, 1, 1]
    assert sum(len(run.records) for run in runs) == 4
    assert [engine.load_tokens for engine in engines] == [0, 0]


def test_load_counts_blocked_inputs_but_not_output_predictions_or_evictable_cache() -> None:
    engine = make_engine()
    engine.submit(request(0, prompt=4, output=1), 0)
    assert engine.start_step(0, 0) == (4, (0,))
    engine.finish_step(4)
    engine.complete(4, 0)
    assert engine._prefix_cache.evictable_pages == 1
    assert engine.load_tokens == 0
    engine.submit(request(1, prompt=5, maximum=100, output=90), 4)
    engine._prefix_scheduler.set_ready(1, False)
    assert engine.load_tokens == 5
    engine.submit(request(2, prompt=3, maximum=1, output=1), 4)
    assert engine.load_tokens == 8


def test_waiting_load_removes_aborts_and_counts_original_prompt_on_requeue() -> None:
    scheduler, cache = make(capacity=128, page_size=4, max_running=1)
    scheduler.enqueue(queued_request(0, prompt=129), 0)
    item = queued_request(1, prompt=5, maximum=10)
    scheduler.enqueue(item, 0)
    assert scheduler.waiting_prompt_tokens == 134
    batch = scheduler.select_prefill({})
    assert [entry.request.request_id for entry in batch.aborted] == [0]
    assert scheduler.waiting_prompt_tokens == 0
    cache.cancel_pin(1)
    scheduler.requeue(item, 1, generated_tokens=3)
    assert scheduler.waiting_prompt_tokens == 5
    assert scheduler.select_prefill({}).admitted[0].waiting.target_tokens == 7
    assert scheduler.waiting_prompt_tokens == 0


def test_load_is_zero_after_submission_or_admission_failure() -> None:
    engine = make_engine(capacity=8)
    assert not engine.submit(replace(request(0), is_trace=True, output_len=0), 0)
    assert engine.load_tokens == 0
    engine.submit(request(1, prompt=9), 0)
    assert engine.load_tokens == 9
    assert engine.start_step(0, 0) == (0, ())
    assert engine.load_tokens == 0


def test_synthetic_children_reroute_and_results_remain_deterministic() -> None:
    settings = config(requests=4, sessions=1).model_copy(update={"routing_policy": "least_load"})
    first = simulate(settings)
    second = simulate(settings)
    assert [item.instance_id for item in first.requests] == [0, 1, 0, 1]
    assert first.meta.config.routing_policy == "least_load"
    assert first.meta.num_completed == 4
    assert first.model_dump(exclude={"meta": {"wall_clock_ms"}}) == second.model_dump(
        exclude={"meta": {"wall_clock_ms"}}
    )


@pytest.mark.parametrize("mode", ["open_loop", "closed_loop"])
def test_trace_children_can_leave_original_instance(mode: str) -> None:
    base = trace_config()
    settings = base.model_copy(
        update={
            "routing_policy": "least_load",
            "workload": base.workload.model_copy(update={"arrival_mode": mode}),
        }
    )
    result = simulate(settings, trace_plan=trace_plan())
    assert [item.instance_id for item in result.requests] == [0, 1]
    assert result.requests[1].actual_hit_tokens == 0


def test_failed_descendants_remain_at_failure_instance_without_routing() -> None:
    settings = config(capacity=4, prompt=8).model_copy(update={"routing_policy": "least_load"})
    result = simulate(settings)
    assert (result.meta.num_aborted, result.meta.num_skipped) == (2, 2)
    owners = {item.id: item.instance_id for item in result.aborted_requests}
    assert all(
        item.instance_id == owners[item.parent_request_id] for item in result.skipped_requests
    )
