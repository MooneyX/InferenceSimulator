"""指标汇总：由区间记录计算全场 ``Metrics``。

计算口径见 ``docs/design/api.md``「Metrics 定义」：

- 分位数用线性插值（numpy ``percentile`` 默认 ``linear`` 方法）。
- TPOT 先逐请求算 ``decode 时长 / (output_len - 1)``，再对请求取平均；单 token 请求不计入。
- 吞吐单位 请求/秒；makespan 为 0 时无定义。

纯函数，无状态。依赖层级：仅依赖 ``schemas``。
"""

import math
from collections.abc import Sequence

from simcore.schemas import Metrics, RequestRecord

__all__ = ["compute_metrics", "percentile"]

_MS_PER_SECOND = 1000.0


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """线性插值分位数。

    对升序样本取位置 ``h = (n - 1) * q``，在 ``floor(h)`` 与 ``ceil(h)`` 两点间线性插值，
    与 ``numpy.percentile(..., method="linear")`` 一致。

    Args:
        sorted_values: 升序样本，非空。调用方负责排序（避免多个分位数重复排序）。
        q: 分位点，``[0, 1]``。

    Raises:
        ValueError: 样本为空或 ``q`` 越界。
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("样本为空")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q 须在 [0, 1] 内，得到 {q}")
    h = (n - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    frac = h - lo
    lo_value = sorted_values[lo]
    return lo_value + (sorted_values[hi] - lo_value) * frac


def compute_metrics(records: Sequence[RequestRecord]) -> Metrics:
    """由区间记录计算全场指标。

    Raises:
        ValueError: ``records`` 为空——配置层保证至少 1 个请求，空输入说明调用有误。
    """
    if not records:
        raise ValueError("records 为空，无法计算指标")

    ttfts = sorted(r.prefill_end - r.arrival for r in records)
    ttft_mean = math.fsum(ttfts) / len(ttfts)

    tpots = [(r.finish - r.prefill_end) / (r.output_len - 1) for r in records if r.output_len > 1]
    tpot_mean = math.fsum(tpots) / len(tpots) if tpots else None

    sim_start = min(r.arrival for r in records)
    sim_end = max(r.finish for r in records)
    makespan = sim_end - sim_start
    throughput = len(records) / (makespan / _MS_PER_SECOND) if makespan > 0 else None

    return Metrics(
        ttft_mean=ttft_mean,
        ttft_p50=percentile(ttfts, 0.50),
        ttft_p99=percentile(ttfts, 0.99),
        tpot_mean=tpot_mean,
        throughput=throughput,
        makespan=makespan,
    )
