"""工作负载生成：向引擎供给 ``Arrival`` 序列。

生成器只产出到达事实，永不接触引擎——这是 ``docs/design/kernel.md`` 定下的边界，
使未来的 trace replay 可以替换实现而不触碰内核。

可复现性：所有随机性来自 ``random.Random(seed)`` 这一个实例，采样顺序固定，
因此同种子两次运行必然产出逐字节相同的序列。不使用 ``numpy``，避免其跨版本
随机流变更破坏可复现性。

依赖层级：仅依赖 ``schemas``。
"""

import math
import random
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from simcore.schemas import (
    Arrival,
    FixedLength,
    LengthDist,
    LognormalLength,
    PoissonLength,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)

__all__ = ["SyntheticSource", "WorkloadSource", "build_source", "sample_length"]

_MS_PER_SECOND = 1000.0


@runtime_checkable
class WorkloadSource(Protocol):
    """负载源协议：产出按 ``sim_time`` 升序排列的 ``Arrival`` 序列。

    实现方须保证：
    - 升序：``sim_time`` 单调不减（引擎依赖此性质批量入队）。
    - 可复现：同配置多次迭代产出完全相同的序列。
    - 可重复迭代：每次 ``__iter__`` 都从头产出同一序列，而非耗尽即空。
    """

    def __iter__(self) -> Iterator[Arrival]: ...


def sample_length(dist: LengthDist, rng: random.Random) -> int:
    """按分布采样一个 token 长度。

    结果向上取整并钳制到 >= 1（见 ``docs/design/api.md``）：长度为 0 的请求无意义，
    且会让性能模型与 KV 账本失去定义。

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


class SyntheticSource:
    """合成负载：泊松到达 + 长度分布采样 + 固定种子。

    到达过程为泊松过程，等价于间隔服从指数分布 ``Exp(arrival_rate)``。配置中的
    ``arrival_rate`` 单位为 请求/秒，内部换算为 ms 时间轴。

    首个请求的到达时刻同样采样一个间隔，而非固定为 0——避免 t=0 处出现人为的
    突发，使统计量不受首请求特殊化的干扰。

    采样顺序固定为「间隔 → prompt → output」逐请求推进，任何顺序改动都会改变
    随机流，因此视为对外契约的一部分。
    """

    def __init__(self, cfg: SyntheticWorkloadConfig) -> None:
        self._cfg = cfg
        self._arrivals: tuple[Arrival, ...] = self._materialize(cfg)

    @staticmethod
    def _materialize(cfg: SyntheticWorkloadConfig) -> tuple[Arrival, ...]:
        rng = random.Random(cfg.seed)
        rate_per_ms = cfg.arrival_rate / _MS_PER_SECOND
        arrivals: list[Arrival] = []
        clock = 0.0
        for request_id in range(cfg.num_requests):
            clock += rng.expovariate(rate_per_ms)
            arrivals.append(
                Arrival(
                    request_id=request_id,
                    sim_time=clock,
                    prompt_len=sample_length(cfg.prompt_len, rng),
                    output_len=sample_length(cfg.output_len, rng),
                )
            )
        return tuple(arrivals)

    def __iter__(self) -> Iterator[Arrival]:
        return iter(self._arrivals)

    def __len__(self) -> int:
        return len(self._arrivals)


def build_source(cfg: WorkloadConfig) -> WorkloadSource:
    """按配置构造负载源。

    Raises:
        NotImplementedError: ``mode='trace'``。trace replay 是预留接缝，MVP 不实现。
    """
    if cfg.mode == "trace":
        raise NotImplementedError("trace replay 尚未实现，MVP 仅支持 workload.mode='synthetic'")
    if cfg.synthetic is None:  # pragma: no cover - schema 层已保证
        raise ValueError("workload.mode='synthetic' 时必须提供 workload.synthetic")
    return SyntheticSource(cfg.synthetic)
