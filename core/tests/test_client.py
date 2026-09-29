"""闭环客户端：请求完成是唯一子轮次发送触发器。"""

from dataclasses import replace

import pytest

from simcore.client import ClientDriver
from simcore.schemas import (
    FixedLength,
    PrefixCacheConfig,
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
    cache = PrefixCacheConfig(block_size_tokens=2)
    return ClientDriver(build_plan(cfg, cache), cache)


def test_child_waits_for_actual_parent_output() -> None:
    base = make_client(think=5)
    parent_id = base.plan.root_request_ids[0]
    child_id = base.plan.children[parent_id][0]
    turns = dict(base.plan.turns)
    turns[child_id] = replace(turns[child_id], prompt_len=12, ideal_reuse_blocks=5)
    client = ClientDriver(
        WorkloadPlan(turns, base.plan.root_request_ids, base.plan.children, block_size_tokens=2),
        PrefixCacheConfig(block_size_tokens=2),
    )
    wakeup = client.initial_wakeups()[0]
    parent = client.on_wakeup(wakeup)
    child_id = client.plan.children[parent.request_id][0]
    assert client.initial_wakeups() == ()
    with pytest.raises(ValueError, match="未就绪"):
        client.on_wakeup(wakeup)
    with pytest.raises(ValueError, match="输出"):
        client.on_request_completed(RequestCompletion(parent.request_id, (), 30))
    actual_output = tuple(range(900, 904))
    (next_wakeup,) = client.on_request_completed(
        RequestCompletion(parent.request_id, actual_output, 30)
    )
    assert next_wakeup.turn_id == child_id
    assert next_wakeup.send_time_ms == 35
    child = client.on_wakeup(next_wakeup)
    prefix = client.plan.turns[child_id].ideal_reuse_blocks * 2
    assert child.input_token_ids[:prefix] == (parent.input_token_ids + actual_output)[:prefix]
    with pytest.raises(ValueError, match="未发送"):
        client.on_request_completed(RequestCompletion(parent.request_id, actual_output, 31))


def test_zero_think_time_and_reproducibility() -> None:
    def sequence() -> tuple[tuple[int, ...], tuple[int, ...]]:
        client = make_client()
        wakeup = client.initial_wakeups()[0]
        parent = client.on_wakeup(wakeup)
        child_wakeup = client.on_request_completed(
            RequestCompletion(parent.request_id, parent.output_token_plan, 12)
        )[0]
        assert child_wakeup.send_time_ms == 12
        return parent.input_token_ids, client.on_wakeup(child_wakeup).input_token_ids

    assert sequence() == sequence()
