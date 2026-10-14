"""统一客户端：四种投放模式、请求身份与完成/失败生命周期。"""

from dataclasses import replace
from unittest.mock import patch

import pytest

from simcore.client import Client
from simcore.prefix_hash import encode
from simcore.schemas import (
    ClientWakeup,
    EngineConfig,
    FixedLength,
    RequestCompletion,
    SessionConfig,
    SyntheticWorkloadConfig,
    TracePlan,
    TraceTurn,
    WorkloadConfig,
    WorkloadPlan,
)
from simcore.workload import build_synthetic_closed_loop_plan


def planned_client(plan: WorkloadPlan, engine: EngineConfig) -> Client:
    workload = synthetic_workload("closed_loop")
    with patch("simcore.client.build_synthetic_closed_loop_plan", return_value=plan):
        return Client(workload, engine)


def make_client(think: float = 0.0) -> Client:
    workload = WorkloadConfig(
        session_rate=5,
        seed=2,
        synthetic=SyntheticWorkloadConfig(
            num_requests=3,
            prompt_len=FixedLength(value=8),
            turn_increment_len=FixedLength(value=1),
            output_len=FixedLength(value=4),
            session=SessionConfig(num_sessions=1, think_time_ms=think),
        ),
    )
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2)
    return Client(workload, engine)


def test_child_waits_for_actual_parent_output() -> None:
    base = make_client(think=5)
    parent_id = base.plan.root_request_ids[0]
    child_id = base.plan.children[parent_id][0]
    turns = dict(base.plan.turns)
    turns[child_id] = replace(turns[child_id], prompt_len=12, ideal_reuse_blocks=5)
    client = planned_client(
        WorkloadPlan(turns, base.plan.root_request_ids, base.plan.children),
        EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2),
    )
    wakeup = client.initial_wakeups()[0]
    parent, _ = client.on_wakeup(wakeup)
    child_id = client.plan.children[parent.request_id][0]
    with pytest.raises(ValueError, match="未就绪"):
        client.on_wakeup(wakeup)
    with pytest.raises(ValueError, match="未发送"):
        client.on_request_completed(RequestCompletion(child_id, 29, 4))
    (next_wakeup,) = client.on_request_completed(RequestCompletion(parent.request_id, 30, 4))
    assert next_wakeup.turn_id == child_id
    sampled_delay = client.plan.turns[child_id].think_time_ms
    assert sampled_delay is not None
    assert next_wakeup.send_time_ms == 30 + sampled_delay
    assert next_wakeup.send_time_ms > 30
    child, _ = client.on_wakeup(next_wakeup)
    inherited = client.plan.turns[child_id].ideal_reuse_blocks
    assert parent.inherited_prefix_ids == ()
    assert child.inherited_prefix_ids == tuple(
        encode(parent.request_id, i) for i in range(inherited)
    )
    assert not hasattr(child, "input_chunk_ids")
    assert not hasattr(child, "output_len")
    with pytest.raises(ValueError, match="未发送"):
        client.on_request_completed(RequestCompletion(parent.request_id, 31, 4))


def test_child_without_reuse_still_waits_for_parent_completion() -> None:
    base = make_client()
    parent_id = base.plan.root_request_ids[0]
    child_id = base.plan.children[parent_id][0]
    turns = dict(base.plan.turns)
    turns[child_id] = replace(turns[child_id], ideal_reuse_blocks=0)
    client = planned_client(
        WorkloadPlan(turns, base.plan.root_request_ids, base.plan.children),
        EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2),
    )
    client.on_wakeup(client.initial_wakeups()[0])
    child = turns[child_id]
    with pytest.raises(ValueError, match="父请求完成"):
        client.on_wakeup(ClientWakeup(child.session_id, child_id, 0))
    wakeup = client.on_request_completed(RequestCompletion(parent_id, 30, 4))[0]
    assert client.on_wakeup(wakeup)[0].parent_request_id == parent_id


def test_root_wakeup_uses_planned_time() -> None:
    client = make_client()
    wakeup = client.initial_wakeups()[0]
    with pytest.raises(ValueError, match="发送时间与计划不符"):
        client.on_wakeup(replace(wakeup, send_time_ms=wakeup.send_time_ms + 1))
    assert client.on_wakeup(wakeup)[0].request_id == wakeup.turn_id


def test_last_sampled_token_does_not_create_inheritable_page() -> None:
    base = make_client()
    parent_id = base.plan.root_request_ids[0]
    child_id = base.plan.children[parent_id][0]
    turns = dict(base.plan.turns)
    turns[child_id] = replace(turns[child_id], prompt_len=12, ideal_reuse_blocks=5)
    client = planned_client(
        WorkloadPlan(turns, base.plan.root_request_ids, base.plan.children),
        EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2),
    )
    client.on_wakeup(client.initial_wakeups()[0])
    child_wakeup = client.on_request_completed(RequestCompletion(parent_id, 10, 1))[0]
    with pytest.raises(ValueError, match="完整块上下文"):
        client.on_wakeup(child_wakeup)


def test_zero_think_time_and_reproducibility() -> None:
    def sequence() -> tuple[tuple[str, ...], tuple[str, ...]]:
        client = make_client()
        wakeup = client.initial_wakeups()[0]
        parent, _ = client.on_wakeup(wakeup)
        child_wakeup = client.on_request_completed(RequestCompletion(parent.request_id, 12, 4))[0]
        assert child_wakeup.send_time_ms == 12
        return parent.inherited_prefix_ids, client.on_wakeup(child_wakeup)[0].inherited_prefix_ids

    assert sequence() == sequence()


def test_partial_input_filled_by_output_and_ancestor_ids_are_copied() -> None:
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=4)
    cfg = WorkloadConfig(
        session_rate=5,
        seed=2,
        synthetic=SyntheticWorkloadConfig(
            num_requests=3,
            prompt_len=FixedLength(value=3),
            turn_increment_len=FixedLength(value=1),
            output_len=FixedLength(value=3),
            session=SessionConfig(num_sessions=1),
        ),
    )
    base = build_synthetic_closed_loop_plan(cfg, engine)
    root_id = base.root_request_ids[0]
    child_id = base.children[root_id][0]
    grandchild_id = base.children[child_id][0]
    turns = dict(base.turns)
    turns[child_id] = replace(
        turns[child_id], prompt_len=5, output_len=5, max_new_tokens=8, ideal_reuse_blocks=1,
    )
    turns[grandchild_id] = replace(turns[grandchild_id], prompt_len=13, ideal_reuse_blocks=2)
    client = planned_client(WorkloadPlan(turns, base.root_request_ids, base.children), engine)

    root, _ = client.on_wakeup(client.initial_wakeups()[0])
    assert root.inherited_prefix_ids == ()
    child_wakeup = client.on_request_completed(RequestCompletion(root_id, 10, 3))[0]
    child, _ = client.on_wakeup(child_wakeup)
    assert child.inherited_prefix_ids == (encode(root_id, 0),)
    grandchild_wakeup = client.on_request_completed(RequestCompletion(child_id, 20, 5))[0]
    grandchild, _ = client.on_wakeup(grandchild_wakeup)
    assert grandchild.inherited_prefix_ids == (encode(root_id, 0), encode(child_id, 1))
    assert not hasattr(grandchild, "input_chunk_ids")
    assert grandchild_id == grandchild.request_id


def synthetic_workload(mode: str) -> WorkloadConfig:
    return WorkloadConfig.model_validate({
        "arrival_mode": mode,
        "seed": 42,
        "target_qps" if mode == "open_loop" else "session_rate": 10,
        "synthetic": {
            "num_requests": 4,
            "prompt_len": {"dist": "fixed", "value": 8},
            "turn_increment_len": {"dist": "fixed", "value": 1},
            "output_len": {"dist": "fixed", "value": 2},
            "session": {} if mode == "open_loop" else {"num_sessions": 1},
        },
    })


def trace_plan(output: int = 2) -> TracePlan:
    return TracePlan(
        sessions=("one",),
        block_size=2,
        requests=(
            TraceTurn(
                request_id=0,
                session_id=0,
                arrival_offset_ms=0,
                prompt_len=5,
                output_len=output,
                hash_ids=(0, 1),
            ),
            TraceTurn(
                request_id=1,
                session_id=0,
                parent_request_id=0,
                arrival_offset_ms=20,
                think_time_ms=3,
                prompt_len=7,
                output_len=1,
                hash_ids=(0, 1, 2),
            ),
            TraceTurn(
                request_id=2,
                session_id=0,
                parent_request_id=0,
                arrival_offset_ms=30,
                think_time_ms=4,
                prompt_len=9,
                output_len=1,
                hash_ids=(0, 1, 3, 4),
            ),
        ),
    )


@pytest.mark.parametrize("source", ["synthetic", "trace"])
@pytest.mark.parametrize("mode", ["closed_loop", "open_loop"])
def test_four_modes_share_preparation_dispatch_and_lifecycle_contract(
    source: str, mode: str
) -> None:
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2)
    workload = (
        synthetic_workload(mode)
        if source == "synthetic"
        else WorkloadConfig.model_validate({
            "mode": "trace", "arrival_mode": mode, "session_rate": 10,
            "seed": 42, "trace": {},
        })
    )
    client = Client(workload, engine, trace_plan=trace_plan() if source == "trace" else None)
    assert client.workload_mode == source and client.arrival_mode == mode
    assert not hasattr(client, "_ready") or (source == "synthetic" and mode == "open_loop")
    assert hasattr(client, "_finished") == (source == "synthetic")
    assert hasattr(client, "_completed") == (source == "trace" and mode == "closed_loop")
    with pytest.raises(AttributeError):
        client.workload_mode = "trace"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        client.arrival_mode = "closed_loop"  # type: ignore[misc]
    assert not client._sent and not client._inflight
    initial_plan = client.plan
    dynamic = source == "synthetic" and mode == "open_loop"
    assert len(initial_plan.turns) == (0 if dynamic else 4 if source == "synthetic" else 3)
    if source == "synthetic" and mode == "closed_loop":
        assert workload.synthetic is not None
        assert initial_plan == build_synthetic_closed_loop_plan(workload, engine)
    wakeups = client.initial_wakeups()
    assert len(wakeups) == (1 if mode == "closed_loop" else 4 if dynamic else 3)
    assert client.initial_wakeups() == wakeups
    assert client.plan == initial_plan
    request, output = client.on_wakeup(wakeups[0])
    assert request.is_trace == (source == "trace")
    assert output == 2 and not hasattr(request, "output_len")
    assert request.trace_hash_ids == ((0, 1) if source == "trace" else ())
    assert client.on_request_completed(RequestCompletion(request.request_id, 100, output)) == (
        ()
        if mode == "open_loop"
        else tuple(
            ClientWakeup(turn.session_id, child_id, 100 + (turn.think_time_ms or 0))
            for child_id in client.plan.children[request.request_id]
            for turn in (client.plan.turns[child_id],)
        )
    )
    assert not client._inflight


def test_synthetic_open_loop_completion_enables_next_send_and_failure_keeps_clock() -> None:
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2)
    client = Client(synthetic_workload("open_loop"), engine)
    length_states = (
        client._roots_rng.getstate(),
        client._increments_rng.getstate(),
        client._outputs_rng.getstate(),
    )
    wakeups = client.initial_wakeups()
    assert [item.turn_id for item in wakeups] == list(range(4))
    assert (
        client._roots_rng.getstate(),
        client._increments_rng.getstate(),
        client._outputs_rng.getstate(),
    ) == length_states
    assert client.plan.turns == {}
    first, output = client.on_wakeup(wakeups[0])
    assert client.on_request_completed(RequestCompletion(first.request_id, 1, output)) == ()
    assert len(client.plan.turns) == 1
    assert len(client._ready) == 1
    child, _ = client.on_wakeup(wakeups[1])
    assert child.parent_request_id == first.request_id
    assert child.prompt_len == 11
    assert child.inherited_prefix_ids == tuple(
        encode(first.request_id, index) for index in range(4)
    )
    assert client.on_request_aborted(child.request_id) == ()
    later, _ = client.on_wakeup(wakeups[2])
    assert later.parent_request_id is None and later.request_id == 2
    assert len(client.plan.turns) == 3 and client.plan.root_request_ids == (0, 2)
    assert not hasattr(client, "_next_id")


def test_synthetic_open_loop_ready_queue_tie_breaks_by_session() -> None:
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2)
    client = Client(synthetic_workload("open_loop"), engine)
    wakeups = client.initial_wakeups()
    first, output0 = client.on_wakeup(wakeups[0])
    second, output1 = client.on_wakeup(wakeups[1])
    client.on_request_completed(RequestCompletion(second.request_id, 100, output1))
    client.on_request_completed(RequestCompletion(first.request_id, 100, output0))
    assert client.on_wakeup(wakeups[2])[0].parent_request_id == first.request_id
    assert client.on_wakeup(wakeups[3])[0].parent_request_id == second.request_id


@pytest.mark.parametrize("mode", ["closed_loop", "open_loop"])
def test_preparation_rejects_trace_plan_for_wrong_workload_source(mode: str) -> None:
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2)
    with pytest.raises(ValueError, match="合成仿真不能提供 trace_plan"):
        Client(synthetic_workload(mode), engine, trace_plan=trace_plan())
    workload = WorkloadConfig.model_validate({
        "mode": "trace", "arrival_mode": mode, "session_rate": 10, "trace": {},
    })
    with pytest.raises(ValueError, match="Trace 仿真须提供 trace_plan"):
        Client(workload, engine)


@pytest.mark.parametrize("mode", ["closed_loop", "open_loop"])
def test_trace_preparation_preserves_all_independent_roots(mode: str) -> None:
    base = trace_plan()
    source = TracePlan(
        block_size=base.block_size,
        sessions=base.sessions,
        requests=(
            *base.requests,
            TraceTurn(
                request_id=3,
                session_id=0,
                arrival_offset_ms=40,
                prompt_len=5,
                output_len=2,
                hash_ids=(0, 1),
            ),
        ),
    )
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2)
    workload = WorkloadConfig.model_validate({
        "mode": "trace", "arrival_mode": mode, "session_rate": 10, "trace": {},
    })
    client = Client(workload, engine, trace_plan=source)
    wakeups = client.initial_wakeups()
    assert [item.turn_id for item in wakeups] == (
        [0, 3] if mode == "closed_loop" else [0, 1, 2, 3]
    )
    assert wakeups[-1].send_time_ms - wakeups[0].send_time_ms == pytest.approx(40)
    assert client.plan.children == {0: (1, 2)}
    request, output = client.on_wakeup(wakeups[-1])
    assert request.parent_request_id is None
    assert request.trace_hash_ids == (0, 1) and output == 2


@pytest.mark.parametrize("mode", ["closed_loop", "open_loop"])
def test_trace_failure_skips_branches_only_when_dependency_gates_sending(mode: str) -> None:
    source = trace_plan(output=0)
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2)
    workload = WorkloadConfig.model_validate({
        "mode": "trace", "arrival_mode": mode, "session_rate": 10, "trace": {},
    })
    client = Client(workload, engine, trace_plan=source)
    initial = client.initial_wakeups()
    root, output = client.on_wakeup(initial[0])
    assert output == 0
    skipped = client.on_request_aborted(root.request_id)
    if mode == "closed_loop":
        assert [(item.id, item.parent_request_id) for item in skipped] == [(1, 0), (2, 0)]
        assert len(initial) == 1
    else:
        assert skipped == ()
        child, output = client.on_wakeup(initial[1])
        assert child.parent_request_id == root.request_id and output == 1
        assert child.trace_hash_ids == (0, 1, 2)


def test_synthetic_abort_returns_all_unsent_descendants_in_id_order() -> None:
    client = make_client()
    root, _ = client.on_wakeup(client.initial_wakeups()[0])
    assert [
        (item.id, item.parent_request_id) for item in client.on_request_aborted(root.request_id)
    ] == [
        (1, 0),
        (2, 1),
    ]
    assert not client._inflight
