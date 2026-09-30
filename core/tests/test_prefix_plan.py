"""PC1-A：会话长度与拓扑计划、块身份继承及预算量化。"""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from simcore.client import ClientDriver
from simcore.prefix_hash import encode, stable_derive
from simcore.schemas import (
    Arrival,
    EngineConfig,
    FixedLength,
    PrefixCacheConfig,
    RequestCompletion,
    SessionConfig,
    SyntheticWorkloadConfig,
)
from simcore.workload import build_plan, plan_from_arrivals


def config(
    n: int = 10, s: int = 3, rate: float = 25.0, target: float = 0.0
) -> SyntheticWorkloadConfig:
    return SyntheticWorkloadConfig(
        num_requests=n,
        arrival_rate=rate,
        prompt_len=FixedLength(value=10),
        output_len=FixedLength(value=6),
        seed=17,
        session=SessionConfig(
            num_sessions=s, think_time_ms=30, target_ideal_prefix_hit_rate=target
        ),
    )


def test_defaults_and_capacity() -> None:
    old = config(n=4).model_copy(update={"session": SessionConfig()})
    parsed = SyntheticWorkloadConfig.model_validate(old.model_dump())
    assert parsed.session.num_sessions == 4
    engine = EngineConfig(max_batch_size=2, kv_token_budget=11)
    assert engine.prefix_cache.l1 is not None
    assert engine.prefix_cache.l1.capacity_blocks == 11
    assert (
        EngineConfig(
            max_batch_size=2,
            kv_token_budget=11,
            prefix_cache=PrefixCacheConfig(block_size_tokens=4),
        ).prefix_cache.l1.capacity_blocks
        == 2
    )
    with pytest.raises(ValidationError, match="不足"):
        EngineConfig(
            max_batch_size=1,
            kv_token_budget=3,
            prefix_cache=PrefixCacheConfig(block_size_tokens=4),
        )
    with pytest.raises(ValidationError):
        config(n=2, s=3)
    with pytest.raises(ValidationError):
        config(rate=float("inf"))


def test_plan_topology_and_root_poisson_rate() -> None:
    cfg = config(n=100, s=20)
    plan = build_plan(cfg, PrefixCacheConfig(block_size_tokens=4))
    assert len(plan.turns) == 100
    assert len(plan.root_request_ids) == 20
    assert {sum(turn.session_id == sid for turn in plan.turns.values()) for sid in range(20)} == {5}
    for root_id in plan.root_request_ids:
        root = plan.turns[root_id]
        assert root.parent_request_id is None and root.think_time_ms is None
        assert root.root_arrival_time_ms is not None
    for turn in plan.turns.values():
        if turn.parent_request_id is not None:
            assert turn.root_arrival_time_ms is None
            assert turn.think_time_ms == 30
            assert plan.children[turn.parent_request_id] == (turn.request_id,)
    assert (plan.turns[plan.root_request_ids[0]].root_arrival_time_ms or 0.0) > 0
    roots = [(plan.turns[key].root_arrival_time_ms or 0.0) for key in plan.root_request_ids]
    assert roots == sorted(roots)
    assert build_plan(config(n=100, s=20, rate=50)).root_request_ids == plan.root_request_ids
    fast = build_plan(config(n=100, s=20, rate=50))
    for key in plan.root_request_ids:
        assert plan.turns[key].root_arrival_time_ms == pytest.approx(
            (fast.turns[key].root_arrival_time_ms or 0.0) * 2
        )


def test_quantization_and_infeasibility() -> None:
    cache = PrefixCacheConfig(block_size_tokens=4)
    plan = build_plan(config(n=6, s=2, target=0.2), cache)
    assert sum(t.ideal_reuse_blocks for t in plan.turns.values()) == 3
    assert plan.constructed_ideal_prefix_hit_rate == pytest.approx(12 / 60)
    assert plan.quantization_error == pytest.approx(0)
    with pytest.raises(ValueError, match="r_max"):
        build_plan(config(n=6, s=2, target=0.9), cache)
    with pytest.raises(ValueError, match="r_max=0"):
        build_plan(config(n=2, s=2, target=0.1), cache)


def test_half_block_quantization_prefers_smaller() -> None:
    plan = build_plan(config(n=6, s=2, target=0.1), PrefixCacheConfig(block_size_tokens=4))
    assert sum(t.ideal_reuse_blocks for t in plan.turns.values()) == 1
    assert plan.constructed_ideal_prefix_hit_rate == pytest.approx(4 / 60)


def test_plan_reproduction_and_distinct_root_chunk_ids() -> None:
    cache = PrefixCacheConfig(block_size_tokens=4)
    plan = build_plan(config(n=6, s=2, target=0.2), cache)
    assert plan == build_plan(config(n=6, s=2, target=0.2), cache)
    client = ClientDriver(plan, cache)
    roots = [client.on_wakeup(wakeup) for wakeup in client.initial_wakeups()]
    assert all(len(request.input_chunk_ids) == request.prompt_len // 4 for request in roots)
    assert roots[0].input_chunk_ids[0] == encode(roots[0].request_id, 0)
    assert roots[0].input_chunk_ids[0] != roots[1].input_chunk_ids[0]
    assert stable_derive(17, "length") != stable_derive(17, "root-arrival")
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        plan.turns[roots[0].request_id] = plan.turns[roots[1].request_id]  # type: ignore[index]


def test_plan_does_not_pregenerate_output_chunk_ids() -> None:
    cache = PrefixCacheConfig(block_size_tokens=4)
    plan = build_plan(config(n=3, s=1, target=0.2), cache)
    root_id = plan.root_request_ids[0]
    root = plan.turns[root_id]
    assert not hasattr(root, "output_token_plan")
    assert not hasattr(root, "unique_suffix_seed")
    client = ClientDriver(plan, cache)
    request = client.on_wakeup(client.initial_wakeups()[0])
    assert request.input_chunk_ids == (encode(root_id, 0), encode(root_id, 1))
    assert not hasattr(request, "output_token_plan")
    child_wakeup = client.on_request_completed(RequestCompletion(root_id, 10))[0]
    child = client.on_wakeup(child_wakeup)
    assert (
        child.input_chunk_ids[: child.ideal_reuse_blocks]
        == (request.input_chunk_ids[: child.ideal_reuse_blocks])
    )


def test_cross_session_shared_prefix_is_rejected_before_allocation() -> None:
    cfg = config(n=2, s=2, target=0.2)
    shared_cfg = cfg.model_copy(
        update={
            "session": cfg.session.model_copy(update={"allow_cross_session_shared_prefix": True})
        }
    )
    with pytest.raises(ValueError, match="不支持 allow_cross_session_shared_prefix=true"):
        build_plan(shared_cfg, PrefixCacheConfig(block_size_tokens=4))


def test_length_arrival_adapter() -> None:
    arrival = Arrival(request_id=8, sim_time=1, prompt_len=3, output_len=2)
    plan = plan_from_arrivals((arrival,), PrefixCacheConfig(block_size_tokens=4))
    assert plan.root_request_ids == (8,)
    client = ClientDriver(plan, PrefixCacheConfig(block_size_tokens=4))
    request = client.on_wakeup(client.initial_wakeups()[0])
    assert (request.prompt_len, request.output_len, request.input_chunk_ids) == (3, 2, ())
    assert client.on_request_completed(RequestCompletion(8, 5)) == ()
    with pytest.raises(ValueError, match="重复"):
        plan_from_arrivals((arrival, arrival))
