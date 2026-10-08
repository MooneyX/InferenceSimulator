"""PC1-A：会话长度与拓扑计划、块身份继承及预算量化。"""

import random
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from simcore.client import ClientDriver
from simcore.engine import run_engine
from simcore.prefix_hash import encode, stable_derive
from simcore.schemas import (
    Arrival,
    EngineConfig,
    FixedLength,
    PerfModelConfig,
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


def engine(page_size: int = 4, capacity: int = 4096) -> EngineConfig:
    return EngineConfig(max_running_requests=8, max_total_tokens=capacity, page_size=page_size)


def test_defaults_and_capacity() -> None:
    old = config(n=4).model_copy(update={"session": SessionConfig()})
    parsed = SyntheticWorkloadConfig.model_validate(old.model_dump())
    assert parsed.session.num_sessions == 4
    assert parsed.max_new_tokens == 128
    assert engine(page_size=4, capacity=12).capacity_pages == 3
    with pytest.raises(ValidationError, match="整数倍"):
        engine(page_size=4, capacity=11)
    with pytest.raises(ValidationError):
        config(n=2, s=3)
    with pytest.raises(ValidationError):
        config(rate=float("inf"))


def test_plan_topology_and_root_poisson_rate() -> None:
    cfg = config(n=100, s=20)
    plan = build_plan(cfg, engine())
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
            assert turn.think_time_ms is not None and turn.think_time_ms >= 0
            assert plan.children[turn.parent_request_id] == (turn.request_id,)
    assert (plan.turns[plan.root_request_ids[0]].root_arrival_time_ms or 0.0) > 0
    roots = [(plan.turns[key].root_arrival_time_ms or 0.0) for key in plan.root_request_ids]
    assert roots == sorted(roots)
    assert (
        build_plan(config(n=100, s=20, rate=50), engine()).root_request_ids == plan.root_request_ids
    )
    fast = build_plan(config(n=100, s=20, rate=50), engine())
    for key in plan.root_request_ids:
        assert plan.turns[key].root_arrival_time_ms == pytest.approx(
            (fast.turns[key].root_arrival_time_ms or 0.0) * 2
        )


def test_think_time_is_sampled_per_child_with_independent_seeded_stream() -> None:
    cfg = config(n=100, s=20)
    plan = build_plan(cfg, engine())
    delays = [
        turn.think_time_ms for turn in plan.turns.values() if turn.parent_request_id is not None
    ]
    rng = random.Random(stable_derive(cfg.seed, "think-time"))
    assert delays == [rng.expovariate(1.0 / 30) for _ in range(80)]
    assert len(set(delays)) == 80
    assert build_plan(cfg, engine()) == plan

    changed = cfg.model_copy(
        update={"session": cfg.session.model_copy(update={"think_time_ms": 60.0})}
    )
    changed_plan = build_plan(changed, engine())
    for request_id, turn in plan.turns.items():
        other = changed_plan.turns[request_id]
        assert turn.root_arrival_time_ms == other.root_arrival_time_ms
        assert (turn.prompt_len, turn.output_len, turn.ideal_reuse_blocks) == (
            other.prompt_len, other.output_len, other.ideal_reuse_blocks
        )
        if turn.think_time_ms is not None:
            assert other.think_time_ms == pytest.approx(2 * turn.think_time_ms)

    zero = cfg.model_copy(
        update={"session": cfg.session.model_copy(update={"think_time_ms": 0.0})}
    )
    zero_plan = build_plan(zero, engine())
    assert all(
        turn.think_time_ms == 0.0
        for turn in zero_plan.turns.values()
        if turn.parent_request_id is not None
    )


def test_quantization_and_infeasibility() -> None:
    cache = engine()
    plan = build_plan(config(n=6, s=2, target=0.2), cache)
    assert sum(t.ideal_reuse_blocks for t in plan.turns.values()) == 3
    assert plan.constructed_ideal_prefix_hit_rate == pytest.approx(12 / 60)
    assert not hasattr(plan, "quantization_error")
    assert not hasattr(plan, "target_ideal_prefix_hit_rate")
    assert not hasattr(plan, "block_size_tokens")
    with pytest.raises(ValueError, match="r_max"):
        build_plan(config(n=6, s=2, target=0.9), cache)
    with pytest.raises(ValueError, match="r_max=0"):
        build_plan(config(n=2, s=2, target=0.1), cache)


def test_half_block_quantization_prefers_smaller() -> None:
    plan = build_plan(config(n=6, s=2, target=0.1), engine())
    assert sum(t.ideal_reuse_blocks for t in plan.turns.values()) == 1
    assert plan.constructed_ideal_prefix_hit_rate == pytest.approx(4 / 60)


def test_plan_reproduction_and_distinct_root_chunk_ids() -> None:
    cache = engine()
    plan = build_plan(config(n=6, s=2, target=0.2), cache)
    assert plan == build_plan(config(n=6, s=2, target=0.2), cache)
    client = ClientDriver(plan, cache)
    roots = [client.on_wakeup(wakeup) for wakeup in client.initial_wakeups()]
    assert all(request.inherited_prefix_ids == () for request in roots)
    assert encode(roots[0].request_id, 0) != encode(roots[1].request_id, 0)
    assert stable_derive(17, "length") != stable_derive(17, "root-arrival")
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        plan.turns[roots[0].request_id] = plan.turns[roots[1].request_id]  # type: ignore[index]


def test_plan_does_not_pregenerate_output_chunk_ids() -> None:
    cache = engine()
    plan = build_plan(config(n=3, s=1, target=0.2), cache)
    root_id = plan.root_request_ids[0]
    root = plan.turns[root_id]
    assert not hasattr(root, "output_token_plan")
    assert not hasattr(root, "unique_suffix_seed")
    client = ClientDriver(plan, cache)
    request = client.on_wakeup(client.initial_wakeups()[0])
    assert request.inherited_prefix_ids == ()
    assert request.max_new_tokens == 128
    assert not hasattr(request, "output_len")
    assert not hasattr(request, "input_chunk_ids")
    child_wakeup = client.on_request_completed(RequestCompletion(root_id, 10, root.output_len))[0]
    child = client.on_wakeup(child_wakeup)
    assert child.inherited_prefix_ids == tuple(
        encode(root_id, index) for index in range(child.ideal_reuse_blocks)
    )


def test_distinct_roots_have_isolated_chunk_ids() -> None:
    cache = engine()
    plan = build_plan(config(n=2, s=2), cache)
    client = ClientDriver(plan, cache)
    roots = [client.on_wakeup(wakeup) for wakeup in client.initial_wakeups()]
    assert len(roots) == 2
    assert all(root.ideal_reuse_blocks == 0 for root in roots)
    assert roots[0].inherited_prefix_ids == roots[1].inherited_prefix_ids == ()
    assert {encode(roots[0].request_id, i) for i in range(2)}.isdisjoint(
        encode(roots[1].request_id, i) for i in range(2)
    )


def test_output_is_clamped_without_exposing_sample_to_client() -> None:
    cfg = SyntheticWorkloadConfig(
        num_requests=1,
        arrival_rate=1,
        prompt_len=FixedLength(value=65),
        output_len=FixedLength(value=100),
        max_new_tokens=200,
        seed=7,
    )
    runtime = engine(page_size=64, capacity=256)
    plan = build_plan(cfg, runtime)
    turn = plan.turns[0]
    assert (turn.max_new_tokens, turn.output_len) == (64, 64)
    client = ClientDriver(plan, runtime)
    request = client.on_wakeup(client.initial_wakeups()[0])
    assert request.max_new_tokens == 64
    assert not hasattr(request, "output_len")
    short_plan = build_plan(cfg.model_copy(update={"output_len": FixedLength(value=2)}), runtime)
    assert short_plan.turns[0].output_len == 2
    assert short_plan.turns[0].max_new_tokens == request.max_new_tokens
    assert build_plan(cfg, runtime) == plan


def test_manual_limit_and_unexecutable_request_are_planned() -> None:
    runtime = engine(page_size=64, capacity=192)
    arrivals = (
        Arrival(request_id=0, sim_time=0, prompt_len=64, output_len=30, max_new_tokens=3),
        Arrival(request_id=1, sim_time=1, prompt_len=65, output_len=5),
    )
    plan = plan_from_arrivals(arrivals, runtime)
    assert (plan.turns[0].max_new_tokens, plan.turns[0].output_len) == (3, 3)
    assert (plan.turns[1].max_new_tokens, plan.turns[1].output_len) == (0, 0)
    client = ClientDriver(plan, runtime)
    assert client.on_wakeup(client.initial_wakeups()[0]).max_new_tokens == 3


def test_engine_completes_at_private_target_not_visible_limit() -> None:
    runtime = engine(page_size=4, capacity=64)
    arrival = Arrival(request_id=3, sim_time=0, prompt_len=5, output_len=2, max_new_tokens=10)
    plan = plan_from_arrivals((arrival,), runtime)
    client = ClientDriver(plan, runtime)
    request = client.on_wakeup(client.initial_wakeups()[0])
    assert request.max_new_tokens == 10
    assert plan.turns[3].output_len == 2
    run = run_engine(
        runtime,
        PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=0),
        (arrival,),
    )
    assert run.records[0].output_len == 2
    assert run.num_engine_steps == 2


def test_length_arrival_adapter() -> None:
    arrival = Arrival(request_id=8, sim_time=1, prompt_len=3, output_len=2)
    plan = plan_from_arrivals((arrival,), engine())
    assert plan.root_request_ids == (8,)
    client = ClientDriver(plan, engine())
    request = client.on_wakeup(client.initial_wakeups()[0])
    assert (request.prompt_len, request.max_new_tokens, request.inherited_prefix_ids) == (
        3,
        128,
        (),
    )
    assert plan.turns[8].output_len == 2
    assert client.on_request_completed(RequestCompletion(8, 5, 2)) == ()
    with pytest.raises(ValueError, match="重复"):
        plan_from_arrivals((arrival, arrival), engine())
