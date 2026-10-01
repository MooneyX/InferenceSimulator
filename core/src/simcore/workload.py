"""合成负载规划及手写 Arrival 到达序列的适配。

``build_plan`` 只生成 session 拓扑、长度、理想继承块数和根发送计划，不物化 token
或预生成块 ID；根到达、长度、拓扑、分配分别使用稳定派生的独立随机流。
``plan_from_arrivals`` 将手写到达序列转换为各自独立的根 session。

依赖层级：``schemas``、``prefix_hash.stable_derive``。
"""

import math
import random
from collections.abc import Iterable
from dataclasses import replace

from simcore.prefix_hash import stable_derive
from simcore.schemas import (
    Arrival,
    FixedLength,
    LengthDist,
    LognormalLength,
    PoissonLength,
    PrefixCacheConfig,
    SyntheticWorkloadConfig,
    TurnPlan,
    WorkloadPlan,
)

__all__ = ["build_plan", "plan_from_arrivals", "sample_length"]

_MS_PER_SECOND = 1000.0


def sample_length(dist: LengthDist, rng: random.Random) -> int:
    """按分布采样一个 token 长度。

    结果向上取整并钳制到 >= 1（见 ``docs/design/workload.md``）：长度为 0 的请求无意义，
    且会让性能模型与 KV 容量计算失去定义。

    - ``fixed``：直接返回配置值，不消耗随机数。
    - ``poisson``：以 Knuth 乘积法采样，均值即 ``mean``。
    - ``lognormal``：以 ``mean`` 为**期望值**（非对数均值）反解 mu，使
      ``E[X] = exp(mu + sigma^2/2) = mean``，从而调大 ``sigma`` 只加重尾部、
      不改变平均负载——这正是用它对比长尾冲击的前提。
    """
    match dist:
        case FixedLength():
            return dist.value
        case PoissonLength():
            return max(1, _sample_poisson(dist.mean, rng))
        case LognormalLength():
            mu = math.log(dist.mean) - dist.sigma * dist.sigma / 2.0
            return max(1, math.ceil(rng.lognormvariate(mu, dist.sigma)))


def _sample_poisson(mean: float, rng: random.Random) -> int:
    """泊松采样（Knuth 乘积法）。

    循环次数期望为 ``mean + 1``，在 MVP 的长度量级（数十到数千 token）下可接受；
    选它是因为只依赖 ``rng.random()``，随机流消耗方式稳定、跨 Python 版本可复现。
    """
    target = math.exp(-mean)
    count = 0
    product = rng.random()
    while product > target:
        count += 1
        product *= rng.random()
    return count


def build_plan(
    cfg: SyntheticWorkloadConfig, cache: PrefixCacheConfig | None = None
) -> WorkloadPlan:
    """独立随机流一次规划拓扑、根到达、长度和理想命中。"""
    cache = cache or PrefixCacheConfig()
    block_size = cache.block_size_tokens
    sessions = cfg.session.num_sessions
    assert sessions is not None  # schema 解析后必有 session 数
    n = cfg.num_requests
    topology_rng = random.Random(stable_derive(cfg.seed, "topology"))
    arrival_rng = random.Random(stable_derive(cfg.seed, "root-arrival"))
    length_rng = random.Random(stable_derive(cfg.seed, "length"))
    allocation_rng = random.Random(stable_derive(cfg.seed, "allocation"))
    counts = [n // sessions] * sessions
    shuffled = list(range(sessions))
    topology_rng.shuffle(shuffled)
    for session_id in shuffled[: n % sessions]:
        counts[session_id] += 1
    turns: dict[int, TurnPlan] = {}
    roots: list[int] = []
    children: dict[int, tuple[int, ...]] = {}
    clock = 0.0
    rate_per_ms = cfg.arrival_rate * sessions / n / _MS_PER_SECOND
    request_id = 0
    for session_id, count in enumerate(counts):
        clock += arrival_rng.expovariate(rate_per_ms)
        parent_id: int | None = None
        for index in range(count):
            prompt_len = sample_length(cfg.prompt_len, length_rng)
            output_len = sample_length(cfg.output_len, length_rng)
            turn = TurnPlan(
                request_id=request_id,
                session_id=session_id,
                parent_request_id=parent_id,
                root_arrival_time_ms=clock if index == 0 else None,
                think_time_ms=None if index == 0 else cfg.session.think_time_ms,
                prompt_len=prompt_len,
                output_len=output_len,
                ideal_reuse_blocks=0,
            )
            turns[request_id] = turn
            if parent_id is None:
                roots.append(request_id)
            else:
                children[parent_id] = (request_id,)
            parent_id = request_id
            request_id += 1

    maxima: dict[int, int] = {}
    for turn in turns.values():
        if turn.parent_request_id is not None:
            parent = turns[turn.parent_request_id]
            maxima[turn.request_id] = min(
                (parent.prompt_len + parent.output_len) // block_size,
                (turn.prompt_len - 1) // block_size,
            )
        else:
            maxima[turn.request_id] = 0
    total_input = sum(turn.prompt_len for turn in turns.values())
    maximum = sum(maxima.values())
    r_max = block_size * maximum / total_input
    target = cfg.session.target_ideal_prefix_hit_rate
    if target > r_max + 1e-12:
        raise ValueError(f"理想前缀命中率不可行: target={target}, r_max={r_max}")
    raw = target * total_input / block_size
    lower = math.floor(raw)
    desired = min(maximum, lower + int(raw - lower > 0.5))
    if maximum:
        shares = {key: desired * cap / maximum for key, cap in maxima.items()}
        assigned = {key: math.floor(value) for key, value in shares.items()}
        ties = list(turns)
        allocation_rng.shuffle(ties)
        tie_order = {key: index for index, key in enumerate(ties)}
        ranked = sorted(turns, key=lambda key: (-(shares[key] - assigned[key]), tie_order[key]))
        for key in ranked[: desired - sum(assigned.values())]:
            assigned[key] += 1
    else:
        assigned = dict.fromkeys(turns, 0)
    for key, count in assigned.items():
        turns[key] = replace(turns[key], ideal_reuse_blocks=count)
    achieved = block_size * desired / total_input
    return WorkloadPlan(
        turns=turns,
        root_request_ids=tuple(roots),
        children=children,
        target_ideal_prefix_hit_rate=target,
        constructed_ideal_prefix_hit_rate=achieved,
        quantization_error=abs(achieved - target),
        block_size_tokens=block_size,
    )


def plan_from_arrivals(
    arrivals: Iterable[Arrival], cache: PrefixCacheConfig | None = None
) -> WorkloadPlan:
    """将手写 Arrival 到达序列适配为各自独立的根 session。"""
    cache = cache or PrefixCacheConfig()
    turns: dict[int, TurnPlan] = {}
    roots: list[int] = []
    for arrival in arrivals:
        key = arrival.request_id
        if key in turns:
            raise ValueError(f"重复的 request_id: {key}")
        turns[key] = TurnPlan(
            request_id=key,
            session_id=key,
            parent_request_id=None,
            root_arrival_time_ms=arrival.sim_time,
            think_time_ms=None,
            prompt_len=arrival.prompt_len,
            output_len=arrival.output_len,
            ideal_reuse_blocks=0,
        )
        roots.append(key)
    roots.sort(key=lambda key: (turns[key].root_arrival_time_ms or 0, key))
    return WorkloadPlan(
        turns=turns,
        root_request_ids=tuple(roots),
        children={},
        block_size_tokens=cache.block_size_tokens,
    )
