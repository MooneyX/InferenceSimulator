"""闭环客户端：请求完成是唯一子轮次发送触发器。"""

from dataclasses import replace

import pytest

from simcore.client import ClientDriver
from simcore.prefix_hash import encode
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    RequestCompletion,
    SessionConfig,
    SyntheticWorkloadConfig,
    WorkloadPlan,
)
from simcore.workload import build_plan


def make_client(think: float = 0.0) -> ClientDriver:
    cfg = SyntheticWorkloadConfig(
        num_requests=3,
        arrival_rate=5,
        prompt_len=FixedLength(value=8),
        output_len=FixedLength(value=4),
        seed=2,
        session=SessionConfig(
            num_sessions=1, think_time_ms=think, target_ideal_prefix_hit_rate=0.25
        ),
    )
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2)
    return ClientDriver(build_plan(cfg, engine), engine)


def test_child_waits_for_actual_parent_output() -> None:
    base = make_client(think=5)
    parent_id = base.plan.root_request_ids[0]
    child_id = base.plan.children[parent_id][0]
    turns = dict(base.plan.turns)
    turns[child_id] = replace(turns[child_id], prompt_len=12, ideal_reuse_blocks=5)
    client = ClientDriver(
        WorkloadPlan(turns, base.plan.root_request_ids, base.plan.children, block_size_tokens=2),
        EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=2),
    )
    wakeup = client.initial_wakeups()[0]
    parent = client.on_wakeup(wakeup)
    child_id = client.plan.children[parent.request_id][0]
    assert client.initial_wakeups() == ()
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
    child = client.on_wakeup(next_wakeup)
    inherited = client.plan.turns[child_id].ideal_reuse_blocks
    assert parent.inherited_prefix_ids == ()
    assert child.inherited_prefix_ids == tuple(
        encode(parent.request_id, i) for i in range(inherited)
    )
    assert not hasattr(child, "input_chunk_ids")
    assert not hasattr(child, "output_len")
    with pytest.raises(ValueError, match="未发送"):
        client.on_request_completed(RequestCompletion(parent.request_id, 31, 4))


def test_last_sampled_token_does_not_create_inheritable_page() -> None:
    base = make_client()
    parent_id = base.plan.root_request_ids[0]
    child_id = base.plan.children[parent_id][0]
    turns = dict(base.plan.turns)
    turns[child_id] = replace(turns[child_id], prompt_len=12, ideal_reuse_blocks=5)
    client = ClientDriver(
        WorkloadPlan(turns, base.plan.root_request_ids, base.plan.children, block_size_tokens=2),
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
        parent = client.on_wakeup(wakeup)
        child_wakeup = client.on_request_completed(RequestCompletion(parent.request_id, 12, 4))[0]
        assert child_wakeup.send_time_ms == 12
        return parent.inherited_prefix_ids, client.on_wakeup(child_wakeup).inherited_prefix_ids

    assert sequence() == sequence()


def test_partial_input_filled_by_output_and_ancestor_ids_are_copied() -> None:
    engine = EngineConfig(max_running_requests=4, max_total_tokens=256, page_size=4)
    cfg = SyntheticWorkloadConfig(
        num_requests=3,
        arrival_rate=5,
        prompt_len=FixedLength(value=3),
        output_len=FixedLength(value=3),
        seed=2,
        session=SessionConfig(num_sessions=1),
    )
    base = build_plan(cfg, engine)
    root_id = base.root_request_ids[0]
    child_id = base.children[root_id][0]
    grandchild_id = base.children[child_id][0]
    turns = dict(base.turns)
    turns[child_id] = replace(turns[child_id], prompt_len=5, output_len=5, ideal_reuse_blocks=1)
    turns[grandchild_id] = replace(turns[grandchild_id], prompt_len=13, ideal_reuse_blocks=2)
    client = ClientDriver(
        WorkloadPlan(turns, base.root_request_ids, base.children, block_size_tokens=4), engine
    )

    root = client.on_wakeup(client.initial_wakeups()[0])
    assert root.inherited_prefix_ids == ()
    child_wakeup = client.on_request_completed(RequestCompletion(root_id, 10, 3))[0]
    child = client.on_wakeup(child_wakeup)
    assert child.inherited_prefix_ids == (encode(root_id, 0),)
    grandchild_wakeup = client.on_request_completed(RequestCompletion(child_id, 20, 5))[0]
    grandchild = client.on_wakeup(grandchild_wakeup)
    assert grandchild.inherited_prefix_ids == (encode(root_id, 0), encode(child_id, 1))
    assert not hasattr(grandchild, "input_chunk_ids")
    assert grandchild_id == grandchild.request_id
