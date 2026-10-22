"""合成负载长度采样与闭环规划。

``build_synthetic_closed_loop_plan`` 只为合成闭环生成 session 拓扑、累积上下文长度、
理想继承块数和根发送计划，不物化 token 或预生成块 ID；根到达、根/增量长度、拓扑
分别使用稳定派生的独立随机流。

依赖层级：``schemas``、``prefix_hash.stable_derive``。
"""

import math
import random
from typing import cast

from simcore.prefix_hash import stable_derive
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    LengthDist,
    LognormalLength,
    PoissonLength,
    SyntheticWorkloadConfig,
    TurnPlan,
    WorkloadConfig,
    WorkloadPlan,
)

__all__ = ["build_synthetic_closed_loop_plan", "request_output_limit", "sample_length"]

_MS_PER_SECOND = 1000.0


def sample_length(
    dist: LengthDist, rng: random.Random
) -> int:  # 长度分布与随机流 → 至少 1 的 token 长度
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


def _sample_poisson(mean: float, rng: random.Random) -> int:  # 泊松均值与随机流 → 非负整数采样值
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


def _sample_poisson_ptrs(
    mean: float, rng: random.Random
) -> int:  # 大泊松均值与随机流 → PTRS 采样的非负整数
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


def request_output_limit(output_len: int, engine_limit: int) -> int:
    """按已知输出向上取二次幂并受引擎硬上限约束；超限目标由引擎报告失败。"""
    return min(1 << (max(1, output_len) - 1).bit_length(), engine_limit)


def build_synthetic_closed_loop_plan(
    workload: WorkloadConfig, engine: EngineConfig
) -> WorkloadPlan:
    """预采样合成闭环输入、增量与原始输出；父轮完成后才实际发送子轮。"""
    synthetic_cfg = cast(SyntheticWorkloadConfig, workload.synthetic)
    block_size = engine.page_size
    sessions = cast(int, synthetic_cfg.session.num_sessions)
    n = synthetic_cfg.num_requests
    topology_rng = random.Random(stable_derive(workload.seed, "topology"))
    arrival_rng = random.Random(stable_derive(workload.seed, "root-arrival"))
    roots_rng = random.Random(stable_derive(workload.seed, "root-length"))
    increments_rng = random.Random(stable_derive(workload.seed, "turn-increment"))
    output_rng = random.Random(stable_derive(workload.seed, "output-length"))
    increment_dist = synthetic_cfg.turn_increment_len
    think_time_rng = random.Random(stable_derive(workload.seed, "think-time"))
    counts = [n // sessions] * sessions  # 各会话轮次数；余数随机分配以免偏向前面的会话
    shuffled = list(range(sessions))
    topology_rng.shuffle(shuffled)
    for session_id in shuffled[: n % sessions]:
        counts[session_id] += 1
    turns: dict[int, TurnPlan] = {}
    roots: list[int] = []
    children: dict[int, tuple[int, ...]] = {}
    clock = 0.0
    # 闭环根会话按独立的会话到达率采样，与子轮完成时刻解耦。
    rate_per_ms = cast(float, workload.session_rate) / _MS_PER_SECOND
    request_id = 0
    for session_id, count in enumerate(counts):
        clock += arrival_rng.expovariate(rate_per_ms)
        parent_id: int | None = None
        for index in range(count):
            if parent_id is None:
                prompt_len = sample_length(synthetic_cfg.prompt_len, roots_rng)
                inherited = 0
            else:
                parent = turns[parent_id]
                prompt_len = (
                    parent.prompt_len
                    + parent.output_len
                    + sample_length(increment_dist, increments_rng)
                )
                inherited = (
                    min(
                        (parent.prompt_len + parent.output_len - 1) // block_size,
                        (prompt_len - 1) // block_size,
                    )
                    if parent.output_len > 0
                    else 0
                )
            sampled_output_len = sample_length(synthetic_cfg.output_len, output_rng)
            max_new_tokens = request_output_limit(sampled_output_len, engine.max_output_tokens)
            think_time_ms = None
            if parent_id is not None:
                think_time_ms = (
                    think_time_rng.expovariate(1.0 / synthetic_cfg.session.think_time_ms)
                    if synthetic_cfg.session.think_time_ms > 0
                    else 0.0
                )
            turn = TurnPlan(
                request_id=request_id,
                session_id=session_id,
                parent_request_id=parent_id,
                root_arrival_time_ms=clock if index == 0 else None,
                think_time_ms=think_time_ms,
                prompt_len=prompt_len,
                output_len=sampled_output_len,
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
