"""PC1-A：累积会话长度与拓扑计划、完整页身份继承。"""

import random
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from simcore import simulate
from simcore.prefix_hash import encode, stable_derive
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    PerfModelConfig,
    SessionConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)
from simcore.workload import build_synthetic_closed_loop_plan
from test_client import planned_client
from test_schemas import resolved_config


def config(n: int = 10, s: int = 3, rate: float = 25.0) -> WorkloadConfig:
    return WorkloadConfig(
        session_rate=rate,
        seed=17,
        synthetic=SyntheticWorkloadConfig(
            num_requests=n,
            prompt_len=FixedLength(value=10),
            turn_increment_len=FixedLength(value=2),
            output_len=FixedLength(value=6),
            session=SessionConfig(num_sessions=s, think_time_ms=30),
        ),
    )


def engine(page_size: int = 4, capacity: int = 4096) -> EngineConfig:
    return EngineConfig(max_running_requests=8, max_total_tokens=capacity, page_size=page_size)


def test_defaults_and_capacity() -> None:
    payload = config(n=4).model_dump()
    payload["synthetic"]["session"] = {}
    parsed = WorkloadConfig.model_validate(payload)
    assert parsed.synthetic is not None and parsed.synthetic.session.num_sessions == 4
    assert engine().max_output_tokens == 65_536
    assert engine(page_size=4, capacity=12).capacity_pages == 3
    with pytest.raises(ValidationError, match="整数倍"):
        engine(page_size=4, capacity=11)
    with pytest.raises(ValidationError):
        config(n=2, s=3)
    with pytest.raises(ValidationError):
        config(rate=float("inf"))


def test_plan_topology_and_root_poisson_rate() -> None:
    cfg = config(n=100, s=20)
    plan = build_synthetic_closed_loop_plan(cfg, engine())
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
        build_synthetic_closed_loop_plan(config(n=100, s=20, rate=50), engine()).root_request_ids
        == plan.root_request_ids
    )
    fast = build_synthetic_closed_loop_plan(config(n=100, s=20, rate=50), engine())
    for key in plan.root_request_ids:
        assert plan.turns[key].root_arrival_time_ms == pytest.approx(
            (fast.turns[key].root_arrival_time_ms or 0.0) * 2
        )


def test_think_time_is_sampled_per_child_with_independent_seeded_stream() -> None:
    cfg = config(n=100, s=20)
    plan = build_synthetic_closed_loop_plan(cfg, engine())
    delays = [
        turn.think_time_ms for turn in plan.turns.values() if turn.parent_request_id is not None
    ]
    rng = random.Random(stable_derive(cfg.seed, "think-time"))
    assert delays == [rng.expovariate(1.0 / 30) for _ in range(80)]
    assert len(set(delays)) == 80
    assert build_synthetic_closed_loop_plan(cfg, engine()) == plan

    assert cfg.synthetic is not None
    changed = cfg.model_copy(update={
        "synthetic": cfg.synthetic.model_copy(update={
            "session": cfg.synthetic.session.model_copy(update={"think_time_ms": 60.0}),
        }),
    })
    changed_plan = build_synthetic_closed_loop_plan(changed, engine())
    for request_id, turn in plan.turns.items():
        other = changed_plan.turns[request_id]
        assert turn.root_arrival_time_ms == other.root_arrival_time_ms
        assert (turn.prompt_len, turn.output_len, turn.ideal_reuse_blocks) == (
            other.prompt_len,
            other.output_len,
            other.ideal_reuse_blocks,
        )
        if turn.think_time_ms is not None:
            assert other.think_time_ms == pytest.approx(2 * turn.think_time_ms)

    zero = cfg.model_copy(update={
        "synthetic": cfg.synthetic.model_copy(update={
            "session": cfg.synthetic.session.model_copy(update={"think_time_ms": 0.0}),
        }),
    })
    zero_plan = build_synthetic_closed_loop_plan(zero, engine())
    assert all(
        turn.think_time_ms == 0.0
        for turn in zero_plan.turns.values()
        if turn.parent_request_id is not None
    )


def test_context_grows_and_inherits_every_completed_parent_page() -> None:
    plan = build_synthetic_closed_loop_plan(config(n=6, s=2), engine())
    assert [plan.turns[key].prompt_len for key in range(3)] == [10, 18, 26]
    assert [plan.turns[key].ideal_reuse_blocks for key in range(3)] == [0, 3, 5]
    total = sum(turn.prompt_len for turn in plan.turns.values())
    ideal = 4 * sum(turn.ideal_reuse_blocks for turn in plan.turns.values())
    assert plan.constructed_ideal_prefix_hit_rate == pytest.approx(ideal / total)
    roots_only = build_synthetic_closed_loop_plan(config(n=2, s=2), engine())
    assert roots_only.constructed_ideal_prefix_hit_rate == 0


def test_plan_reproduction_and_distinct_root_chunk_ids() -> None:
    cache = engine()
    plan = build_synthetic_closed_loop_plan(config(n=6, s=2), cache)
    assert plan == build_synthetic_closed_loop_plan(config(n=6, s=2), cache)
    client = planned_client(plan, cache)
    roots = [client.on_wakeup(wakeup) for wakeup in client.initial_wakeups()]
    assert all(request.inherited_prefix_ids == () for request in roots)
    assert encode(roots[0].request_id, 0) != encode(roots[1].request_id, 0)
    assert stable_derive(17, "length") != stable_derive(17, "root-arrival")
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        plan.turns[roots[0].request_id] = plan.turns[roots[1].request_id]  # type: ignore[index]


def test_plan_does_not_pregenerate_output_chunk_ids() -> None:
    cache = engine()
    plan = build_synthetic_closed_loop_plan(config(n=3, s=1), cache)
    root_id = plan.root_request_ids[0]
    root = plan.turns[root_id]
    assert not hasattr(root, "output_token_plan")
    assert not hasattr(root, "unique_suffix_seed")
    client = planned_client(plan, cache)
    request = client.on_wakeup(client.initial_wakeups()[0])
    assert request.output_len == client.plan.turns[request.request_id].output_len
    assert request.inherited_prefix_ids == ()
    assert request.max_new_tokens == 8
    assert not hasattr(request, "input_chunk_ids")
    child_wakeup = client.on_request_completed(root_id, 10)[0]
    child = client.on_wakeup(child_wakeup)
    assert child.inherited_prefix_ids == tuple(
        encode(root_id, index) for index in range(child.ideal_reuse_blocks)
    )


def test_distinct_roots_have_isolated_chunk_ids() -> None:
    cache = engine()
    plan = build_synthetic_closed_loop_plan(config(n=2, s=2), cache)
    client = planned_client(plan, cache)
    roots = [client.on_wakeup(wakeup) for wakeup in client.initial_wakeups()]
    assert len(roots) == 2
    assert all(root.ideal_reuse_blocks == 0 for root in roots)
    assert roots[0].inherited_prefix_ids == roots[1].inherited_prefix_ids == ()
    assert {encode(roots[0].request_id, i) for i in range(2)}.isdisjoint(
        encode(roots[1].request_id, i) for i in range(2)
    )


def test_output_is_not_clamped_and_is_bound_to_request() -> None:
    cfg = WorkloadConfig(
        session_rate=1,
        seed=7,
        synthetic=SyntheticWorkloadConfig(
            num_requests=1,
            prompt_len=FixedLength(value=65),
            turn_increment_len=FixedLength(value=2),
            output_len=FixedLength(value=100),
        ),
    )
    runtime = engine(page_size=64, capacity=256)
    plan = build_synthetic_closed_loop_plan(cfg, runtime)
    turn = plan.turns[0]
    assert (turn.max_new_tokens, turn.output_len) == (128, 100)
    client = planned_client(plan, runtime)
    request = client.on_wakeup(client.initial_wakeups()[0])
    assert request.output_len == 100 and request.max_new_tokens == 128
    assert cfg.synthetic is not None
    short_plan = build_synthetic_closed_loop_plan(
        cfg.model_copy(update={
            "synthetic": cfg.synthetic.model_copy(update={"output_len": FixedLength(value=2)}),
        }), runtime,
    )
    assert (short_plan.turns[0].output_len, short_plan.turns[0].max_new_tokens) == (2, 2)
    small = build_synthetic_closed_loop_plan(cfg, engine(page_size=64, capacity=192))
    assert small.turns == plan.turns
    assert build_synthetic_closed_loop_plan(cfg, runtime) == plan
    result = simulate(resolved_config(
        workload=cfg, engine=runtime,
        perf_model=PerfModelConfig(prefill_c1=0, decode_d1=0, decode_d2=0),
    ))
    assert result.requests[0].output_len == 100


def test_full_parent_output_drives_child_context_even_beyond_kv_capacity() -> None:
    cfg = WorkloadConfig(
        session_rate=1, seed=7,
        synthetic=SyntheticWorkloadConfig(
            num_requests=3,
            prompt_len=FixedLength(value=65), turn_increment_len=FixedLength(value=1),
            output_len=FixedLength(value=100), session=SessionConfig(num_sessions=1),
        ),
    )
    runtime = engine(page_size=64, capacity=256)
    plan = build_synthetic_closed_loop_plan(cfg, runtime)
    parent, child, descendant = (plan.turns[key] for key in range(3))
    assert parent.output_len == 100
    assert (child.prompt_len, child.max_new_tokens, child.output_len) == (166, 128, 100)
    assert child.ideal_reuse_blocks == 2
    assert descendant.prompt_len == 267 and descendant.ideal_reuse_blocks == 4
    result = simulate(resolved_config(
        workload=cfg, engine=runtime,
        perf_model=PerfModelConfig(prefill_c1=0, decode_d1=0, decode_d2=0),
    ))
    assert result.requests[0].output_len == 100
    assert result.aborted_requests[0].id == child.request_id
    assert result.aborted_requests[0].reason == "decode_kv_exhausted"
    assert result.aborted_requests[0].generated_tokens == 91
    assert [item.id for item in result.skipped_requests] == [descendant.request_id]


def test_engine_completes_at_actual_output_not_rounded_request_limit() -> None:
    runtime = engine(page_size=4, capacity=64)
    cfg = WorkloadConfig(
        session_rate=1,
        synthetic=SyntheticWorkloadConfig(
            num_requests=1, prompt_len=FixedLength(value=5),
            turn_increment_len=FixedLength(value=1), output_len=FixedLength(value=3),
        ),
    )
    plan = build_synthetic_closed_loop_plan(cfg, runtime)
    client = planned_client(plan, runtime)
    request = client.on_wakeup(client.initial_wakeups()[0])
    assert request.output_len == 3 and request.max_new_tokens == 4
    result = simulate(resolved_config(
        workload=cfg, engine=runtime,
        perf_model=PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=0),
    ))
    assert result.requests[0].output_len == 3
    assert result.meta.num_engine_steps == 3


@pytest.mark.parametrize("mode", ["closed_loop", "open_loop"])
@pytest.mark.parametrize("cap", [64, 100])
def test_synthetic_output_over_engine_cap_aborts_without_truncation(mode: str, cap: int) -> None:
    workload = WorkloadConfig.model_validate({
        "arrival_mode": mode, "session_rate" if mode == "closed_loop" else "target_qps": 1,
        "synthetic": {
            "num_requests": 2, "prompt_len": {"dist": "fixed", "value": 1},
            "turn_increment_len": {"dist": "fixed", "value": 1},
            "output_len": {"dist": "fixed", "value": cap + 1},
            "session": {"num_sessions": 1} if mode == "closed_loop" else {},
        },
    })
    runtime = engine().model_copy(update={"max_output_tokens": cap})
    result = simulate(resolved_config(
        workload=workload, engine=runtime,
        perf_model=PerfModelConfig(prefill_c1=0, decode_d1=0, decode_d2=0),
    ))
    assert result.meta.num_completed == 0
    assert result.meta.num_aborted == (1 if mode == "closed_loop" else 2)
    assert result.meta.num_skipped == (1 if mode == "closed_loop" else 0)
    assert all(item.reason == "output_exceeds_engine_limit" and item.generated_tokens == 0
               and item.failed_at == item.arrival for item in result.aborted_requests)
    if mode == "closed_loop":
        plan = build_synthetic_closed_loop_plan(workload, runtime)
        assert all(turn.output_len == cap + 1 and turn.max_new_tokens == cap
                   for turn in plan.turns.values())
        assert plan.turns[1].prompt_len == cap + 3


@pytest.mark.parametrize("cap", [64, 100])
def test_synthetic_output_at_engine_cap_completes_uncut(cap: int) -> None:
    workload = WorkloadConfig(
        session_rate=1,
        synthetic=SyntheticWorkloadConfig(
            num_requests=1, prompt_len=FixedLength(value=1),
            turn_increment_len=FixedLength(value=1), output_len=FixedLength(value=cap),
        ),
    )
    runtime = engine(page_size=64, capacity=128).model_copy(update={"max_output_tokens": cap})
    plan = build_synthetic_closed_loop_plan(workload, runtime)
    assert (plan.turns[0].output_len, plan.turns[0].max_new_tokens) == (cap, cap)
    result = simulate(resolved_config(
        workload=workload, engine=runtime,
        perf_model=PerfModelConfig(prefill_c1=0, decode_d1=0, decode_d2=0),
    ))
    assert result.meta.num_completed == 1 and result.requests[0].output_len == cap
