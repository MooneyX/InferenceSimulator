"""PC1-A：会话计划、内容继承、哈希身份与预算量化。"""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from simcore.prefix_hash import cache_namespace, chunk_hashes, stable_derive
from simcore.schemas import (
    Arrival,
    EngineConfig,
    FixedLength,
    PrefixCacheConfig,
    SessionConfig,
    SyntheticWorkloadConfig,
)
from simcore.workload import build_input, build_plan, make_request, plan_from_arrivals


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


def test_content_and_hash_reproduction_and_isolation() -> None:
    cache = PrefixCacheConfig(block_size_tokens=4)
    plan = build_plan(config(n=6, s=2, target=0.2), cache)
    assert plan == build_plan(config(n=6, s=2, target=0.2), cache)
    root = plan.turns[plan.root_request_ids[0]]
    parent_request = make_request(root, cache)
    child = plan.turns[plan.children[root.request_id][0]]
    with pytest.raises(ValueError, match="父请求"):
        make_request(child, cache)
    parent_context = parent_request.input_token_ids + root.output_token_plan
    child_request = make_request(child, cache, parent_context)
    copied = child.ideal_reuse_blocks * 4
    assert child_request.input_token_ids[:copied] == parent_context[:copied]
    assert len(child_request.input_token_ids) == child.prompt_len
    assert len(child_request.output_token_plan) == child.output_len
    assert len(child_request.input_chunk_hashes) == child.prompt_len // 4
    if copied < child.prompt_len:
        assert child_request.input_token_ids[copied] != parent_context[copied]
    assert build_input(root, 4) != build_input(plan.turns[plan.root_request_ids[1]], 4)
    assert parent_request.input_chunk_hashes == make_request(root, cache).input_chunk_hashes
    assert stable_derive(17, "length") != stable_derive(17, "root-arrival")
    assert cache_namespace(cache) != cache_namespace(cache.model_copy(update={"cache_salt": "x"}))
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        plan.turns[root.request_id] = child  # type: ignore[index]


def test_hash_chain_and_partial_tail() -> None:
    namespace = b"test"
    assert chunk_hashes((1, 2, 3), 2, namespace) == chunk_hashes((1, 2), 2, namespace)
    assert (
        chunk_hashes((1, 2, 3, 4), 2, namespace)[1] != chunk_hashes((7, 8, 3, 4), 2, namespace)[1]
    )
    assert chunk_hashes((1, 2), 2, namespace) != chunk_hashes((1, 2), 2, b"other")
    assert chunk_hashes((1, 2, 3, 4), 2, namespace, hasher=lambda _: "collision") == (
        "collision",
        "collision",
    )


def test_domains_and_explicit_public_prefix() -> None:
    cache = PrefixCacheConfig(block_size_tokens=4)
    cfg = config(n=4, s=2, target=0.1)
    changed_content = cfg.model_copy(
        update={"session": cfg.session.model_copy(update={"content_seed": 123})}
    )
    a = build_plan(cfg, cache)
    b = build_plan(changed_content, cache)
    assert a.root_request_ids == b.root_request_ids
    assert [t.root_arrival_time_ms for t in a.turns.values()] == [
        t.root_arrival_time_ms for t in b.turns.values()
    ]
    assert [t.prompt_len for t in a.turns.values()] == [t.prompt_len for t in b.turns.values()]
    assert (
        a.turns[a.root_request_ids[0]].output_token_plan
        != b.turns[b.root_request_ids[0]].output_token_plan
    )

    shared_cfg = config(n=2, s=2, target=0.2)
    shared_cfg = shared_cfg.model_copy(
        update={
            "session": shared_cfg.session.model_copy(
                update={"allow_cross_session_shared_prefix": True}
            )
        }
    )
    shared = build_plan(shared_cfg, cache)
    first, second = (shared.turns[key] for key in shared.root_request_ids)
    assert second.ideal_reuse_blocks == 1
    public = build_input(first, 4)
    other = build_input(second, 4, public_prefix=public)
    assert other[:4] == public[:4]
    assert other[4] != public[4]
    assert shared.constructed_ideal_prefix_hit_rate == pytest.approx(4 / 20)


def test_length_arrival_adapter() -> None:
    arrival = Arrival(request_id=8, sim_time=1, prompt_len=3, output_len=2)
    plan = plan_from_arrivals((arrival,))
    assert plan.root_request_ids == (8,)
    request = make_request(plan.turns[8], PrefixCacheConfig())
    assert len(request.input_token_ids) == 3
    assert request == make_request(plan.turns[8], PrefixCacheConfig())
