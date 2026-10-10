"""合成负载规划及手写 Arrival 到达序列的适配。

``build_plan`` 只生成 session 拓扑、累积上下文长度、理想继承块数和根发送计划，不物化 token
或预生成块 ID；根到达、根/增量长度、拓扑分别使用稳定派生的独立随机流。
``plan_from_arrivals`` 将手写到达序列转换为各自独立的根 session。

依赖层级：``schemas``、``prefix_hash.stable_derive``。
"""

import math
import random
from collections.abc import Iterable
from typing import cast

from simcore.prefix_hash import stable_derive
from simcore.schemas import (
    Arrival,
    EngineConfig,
    FixedLength,
    LengthDist,
    LognormalLength,
    PoissonLength,
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
    - ``poisson``：小均值用 Knuth 乘积法，大均值用 PTRS 拒绝采样；均值即 ``mean``。
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
    """小均值保留 Knuth 乘积法及其随机流，大均值使用 PTRS 避免下溢。"""
    if mean >= 30.0:
        return _sample_poisson_ptrs(mean, rng)
    target = math.exp(-mean)
    count = 0
    product = rng.random()
    while product > target:
        count += 1
        product *= rng.random()
    return count


def _sample_poisson_ptrs(mean: float, rng: random.Random) -> int:
    """Hörmann PTRS 变换拒绝采样；大均值下期望尝试次数不随均值增长。"""
    root = math.sqrt(mean)
    b = 0.931 + 2.53 * root
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2)
    log_mean = math.log(mean)
    log_inv_alpha = math.log(inv_alpha)
    while True:
        u = rng.random() - 0.5
        v = 1.0 - rng.random()  # 避免对 0 取对数
        us = 0.5 - abs(u)
        if us == 0.0:
            continue
        k = math.floor((2 * a / us + b) * u + mean + 0.43)
        if us >= 0.07 and v <= vr:
            return k
        if k < 0 or (us < 0.013 and v > us):
            continue
        if math.log(v) + log_inv_alpha - math.log(
            a / (us * us) + b
        ) <= -mean + k * log_mean - math.lgamma(k + 1):
            return k


def _visible_limit(prompt_len: int, configured_limit: int, engine: EngineConfig) -> int:
    # 输入向上取整占页后再留一页，避免输出挤占最后的 KV 空间。
    prompt_pages = (prompt_len + engine.page_size - 1) // engine.page_size
    return max(
        0, min(configured_limit, engine.max_total_tokens - (prompt_pages + 1) * engine.page_size)
    )


def build_plan(cfg: SyntheticWorkloadConfig, engine: EngineConfig) -> WorkloadPlan:
    """预采样根输入、子轮增量与截断输出；父轮完成后才实际发送子轮。"""
    block_size = engine.page_size
    sessions = cast(int, cfg.session.num_sessions)
    n = cfg.num_requests
    topology_rng = random.Random(stable_derive(cfg.seed, "topology"))
    arrival_rng = random.Random(stable_derive(cfg.seed, "root-arrival"))
    roots_rng = random.Random(stable_derive(cfg.seed, "root-length"))
    increments_rng = random.Random(stable_derive(cfg.seed, "turn-increment"))
    output_rng = random.Random(stable_derive(cfg.seed, "output-length"))
    increment_dist = cast(LengthDist, cfg.turn_increment_len)
    think_time_rng = random.Random(stable_derive(cfg.seed, "think-time"))
    counts = [n // sessions] * sessions
    shuffled = list(range(sessions))
    topology_rng.shuffle(shuffled)
    for session_id in shuffled[: n % sessions]:
        counts[session_id] += 1
    turns: dict[int, TurnPlan] = {}
    roots: list[int] = []
    children: dict[int, tuple[int, ...]] = {}
    clock = 0.0
    # 闭环根会话按独立的会话到达率采样，与子轮完成时刻解耦。
    rate_per_ms = cast(float, cfg.session_rate) / _MS_PER_SECOND
    request_id = 0
    for session_id, count in enumerate(counts):
        clock += arrival_rng.expovariate(rate_per_ms)
        parent_id: int | None = None
        for index in range(count):
            if parent_id is None:
                prompt_len = sample_length(cfg.prompt_len, roots_rng)
                inherited = 0
            else:
                parent = turns[parent_id]
                prompt_len = parent.prompt_len + parent.output_len + sample_length(
                    increment_dist, increments_rng
                )
                inherited = (
                    min(
                        (parent.prompt_len + parent.output_len - 1) // block_size,
                        (prompt_len - 1) // block_size,
                    )
                    if parent.output_len > 0 else 0
                )
            sampled_output_len = sample_length(cfg.output_len, output_rng)
            max_new_tokens = _visible_limit(prompt_len, cfg.max_new_tokens, engine)
            think_time_ms = None
            if parent_id is not None:
                think_time_ms = (
                    think_time_rng.expovariate(1.0 / cfg.session.think_time_ms)
                    if cfg.session.think_time_ms > 0
                    else 0.0
                )
            turn = TurnPlan(
                request_id=request_id,
                session_id=session_id,
                parent_request_id=parent_id,
                root_arrival_time_ms=clock if index == 0 else None,
                think_time_ms=think_time_ms,
                prompt_len=prompt_len,
                output_len=min(sampled_output_len, max_new_tokens),
                max_new_tokens=max_new_tokens,
                ideal_reuse_blocks=inherited,
            )
            turns[request_id] = turn
            if parent_id is None:
                roots.append(request_id)
            else:
                children[parent_id] = (request_id,)
            parent_id = request_id
            request_id += 1

    total_input = sum(turn.prompt_len for turn in turns.values())
    ideal_tokens = block_size * sum(turn.ideal_reuse_blocks for turn in turns.values())
    achieved = ideal_tokens / total_input
    return WorkloadPlan(
        turns=turns,
        root_request_ids=tuple(roots),
        children=children,
        constructed_ideal_prefix_hit_rate=achieved,
    )


def plan_from_arrivals(arrivals: Iterable[Arrival], engine: EngineConfig) -> WorkloadPlan:
    """将手写 Arrival 到达序列适配为各自独立的根 session。"""
    turns: dict[int, TurnPlan] = {}
    roots: list[int] = []
    for arrival in arrivals:
        key = arrival.request_id
        if key in turns:
            raise ValueError(f"重复的 request_id: {key}")
        max_new_tokens = _visible_limit(arrival.prompt_len, arrival.max_new_tokens, engine)
        turns[key] = TurnPlan(
            request_id=key,
            session_id=key,
            parent_request_id=None,
            root_arrival_time_ms=arrival.sim_time,
            think_time_ms=None,
            prompt_len=arrival.prompt_len,
            output_len=min(arrival.output_len, max_new_tokens),
            max_new_tokens=max_new_tokens,
            ideal_reuse_blocks=0,
        )
        roots.append(key)
    roots.sort(key=lambda key: (turns[key].root_arrival_time_ms or 0, key))
    return WorkloadPlan(
        turns=turns,
        root_request_ids=tuple(roots),
        children={},
    )
