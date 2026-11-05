"""指标汇总：由区间记录计算全场 ``Metrics``。

计算口径见 ``docs/design/metrics.md``：

- 分位数用线性插值（numpy ``percentile`` 默认 ``linear`` 方法）。
- TPOT 先逐请求算 ``decode 时长 / (output_len - 1)``，再对请求取平均；单 token 请求不计入。
- ITL 以每个相邻输出 token 间隔为一个样本（与 SGLang ``bench_serving`` 拼接全部间隔一致）；
  引擎按（间隔, 样本数）分组提供，分位数与展开后的线性插值严格相等。
- 吞吐单位 请求/秒；makespan 为 0 时无定义。

纯函数，无状态。依赖层级：仅依赖 ``schemas``。
"""

import math
from bisect import bisect_right
from collections.abc import Sequence
from itertools import accumulate
from operator import mul
from typing import Any

from simcore.kv_cache import L1Snapshot, TierSnapshot
from simcore.schemas import Metrics, RequestRecord, WorkloadMode, WorkloadPlan

__all__ = ["compute_metrics", "percentile", "weighted_percentile"]

_MS_PER_SECOND = 1000.0


def percentile(
    sorted_values: Sequence[float], q: float
) -> float:  # 升序样本与 [0,1] 分位点 → 插值后的数值
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


def weighted_percentile(
    sorted_values: Sequence[float], cumulative: Sequence[int], q: float
) -> float:  # 升序组值、组内样本数累计和与 [0,1] 分位点 → 展开样本的插值分位数
    """各组值重复对应次数展开后，按 ``percentile`` 同一口径插值；不实际展开。

    ``cumulative[i]`` 为前 ``i + 1`` 组样本数之和；展开序号 ``k`` 落在首个
    ``cumulative[i] > k`` 的组。
    """
    n = cumulative[-1]
    h = (n - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    lo_value = sorted_values[bisect_right(cumulative, lo)]
    return lo_value + (sorted_values[bisect_right(cumulative, hi)] - lo_value) * (h - lo)


def _itl_fields(
    values_ms: Sequence[float], counts: Sequence[int]
) -> dict[str, Any]:  # ITL 分组间隔与样本数 → ITL 均值/P50/P99/最大值；无样本时为空
    if not values_ms:
        return {}
    # 合并相同间隔后只对浮点键排序（远快于元组排序）；间隔多为互异浮点，重复时才逐项合并。
    weights = dict(zip(values_ms, counts, strict=True))
    if len(weights) < len(values_ms):
        weights = {}
        for value, count in zip(values_ms, counts, strict=True):
            weights[value] = weights.get(value, 0) + count
    ordered = sorted(weights)
    cumulative = list(accumulate(map(weights.__getitem__, ordered)))
    return {
        "itl_mean": math.fsum(map(mul, values_ms, counts)) / cumulative[-1],
        "itl_p50": weighted_percentile(ordered, cumulative, 0.50),
        "itl_p99": weighted_percentile(ordered, cumulative, 0.99),
        "itl_max": ordered[-1],
    }


def _stall_fields(
    stalls_ms: Sequence[float], step_time_ms: float
) -> dict[str, Any]:  # 逐步 L3 读取停顿与步长之和 → 停顿步数、总量、分位数及占比
    ordered = sorted(stalls_ms)
    total = math.fsum(ordered)
    fields: dict[str, Any] = {
        "l3_read_steps": len(ordered),
        "l3_stall_ms": total,
        "l3_stall_fraction": total / step_time_ms if step_time_ms > 0 else None,
    }
    if ordered:
        fields |= {
            "l3_stall_p50": percentile(ordered, 0.50),
            "l3_stall_p99": percentile(ordered, 0.99),
            "l3_stall_max": ordered[-1],
        }
    return fields


def compute_metrics(
    records: Sequence[RequestRecord],
    *,
    plan: WorkloadPlan,
    page_size: int,
    workload_mode: WorkloadMode,
    cache_snapshot: L1Snapshot,
    l2_snapshot: TierSnapshot | None,
    l3_snapshot: TierSnapshot | None,
    l1_evictions: int,
    l1_peak_resident_blocks: int,
    promotions: int,
    demotions: int,
    restore_bytes: int,
    restore_time_ms: float,
    earliest_sent_arrival: float | None,
    latest_sent_arrival: float | None,
    num_sent: int,
    replayed_prefill_tokens: int,
    step_time_ms: float,
    l3_stalls_ms: Sequence[float],
    itl_values_ms: Sequence[float],
    itl_counts: Sequence[int],
) -> Metrics:  # 记录、计划/配置、缓存、发送窗口/计数及逐步采样 → 全场指标（无成功请求时部分未定义）
    """仅对完成请求计算时延；全失败时返回无定义指标和实际容量。"""
    constructed = plan.constructed_ideal_prefix_hit_rate
    num_sessions = len(plan.root_request_ids)
    root_count = (
        sum(turn.parent_request_id is None for turn in plan.turns.values())
        if workload_mode == "trace"
        else num_sessions
    )
    root_ratio = root_count / len(plan.turns) if plan.turns else 0.0
    arrival_qps = (
        (num_sent - 1) * _MS_PER_SECOND / (latest_sent_arrival - earliest_sent_arrival)
        if num_sent > 1
        and earliest_sent_arrival is not None
        and latest_sent_arrival is not None
        and latest_sent_arrival > earliest_sent_arrival
        else None
    )
    peak_resident = max(l1_peak_resident_blocks, cache_snapshot.resident_blocks)
    tier_fields: dict[str, Any] = {
        f"{tier}_{name}": getattr(snapshot, name) if snapshot is not None else None
        for tier, snapshot in (("l2", l2_snapshot), ("l3", l3_snapshot))
        for name in (
            "evictions",
            "peak_resident_blocks",
            "capacity_blocks",
            "resident_blocks",
            "active_blocks",
            "evictable_blocks",
            "free_blocks",
        )
    }
    stall_fields = _stall_fields(l3_stalls_ms, step_time_ms)  # 物理停顿与请求是否成功无关
    if not records:
        return Metrics(
            ttft_mean=None,
            ttft_p50=None,
            ttft_p99=None,
            tpot_mean=None,
            throughput=None,
            makespan=0.0,
            constructed_ideal_prefix_hit_rate=constructed,
            arrival_qps=arrival_qps,
            num_sessions=num_sessions,
            root_request_ratio=root_ratio,
            replayed_prefill_tokens=replayed_prefill_tokens,
            restore_bytes=restore_bytes,
            restore_time_ms=restore_time_ms,
            promotions=promotions,
            demotions=demotions,
            l1_evictions=l1_evictions,
            l1_peak_resident_blocks=peak_resident,
            l1_capacity_blocks=cache_snapshot.capacity_blocks,
            l1_resident_blocks=cache_snapshot.resident_blocks,
            l1_active_blocks=cache_snapshot.active_blocks,
            l1_evictable_blocks=cache_snapshot.evictable_blocks,
            l1_free_blocks=cache_snapshot.free_blocks,
            l1_reserved_future_blocks=cache_snapshot.reserved_future_blocks,
            l1_internal_fragmentation_tokens=cache_snapshot.internal_fragmentation_tokens,
            **tier_fields,
            **stall_fields,
        )

    ttfts = sorted(r.prefill_end - r.arrival for r in records)
    ttft_mean = math.fsum(ttfts) / len(ttfts)

    # 首 token 已在 Prefill 生成，单 token 请求不参与 TPOT。
    tpots = sorted(
        (r.finish - r.prefill_end) / (r.output_len - 1) for r in records if r.output_len > 1
    )
    tpot_mean = math.fsum(tpots) / len(tpots) if tpots else None

    sim_start = min(r.arrival for r in records)
    if earliest_sent_arrival is not None:
        sim_start = min(sim_start, earliest_sent_arrival)  # 失败请求也可能更早到达
    sim_end = max(r.finish for r in records)
    makespan = sim_end - sim_start
    throughput = len(records) / (makespan / _MS_PER_SECOND) if makespan > 0 else None
    input_tokens = sum(record.prompt_len for record in records)
    ideal = sum(record.ideal_reuse_tokens for record in records)
    matched = sum(record.matched_cache_tokens for record in records)
    actual = sum(record.actual_hit_tokens for record in records)
    l1_hits = sum(record.l1_hit_tokens for record in records)
    l2_hits = sum(record.l2_hit_tokens for record in records)
    l3_hits = sum(record.l3_hit_tokens for record in records)
    children = [record for record in records if record.parent_request_id is not None]
    child_input = sum(record.prompt_len for record in children)
    prefill = sum(record.actual_prefill_tokens for record in records)
    block_size = page_size
    fully_cached = sum(
        0 < (record.prompt_len // block_size) * block_size == record.matched_cache_tokens
        for record in records
    )  # 完全缓存只看输入中的完整页，末尾不足一页不算
    if actual > ideal or prefill + actual != input_tokens:
        raise ValueError("理想/实际命中和 Prefill token 守恒不成立")

    return Metrics(
        ttft_mean=ttft_mean,
        ttft_p50=percentile(ttfts, 0.50),
        ttft_p99=percentile(ttfts, 0.99),
        tpot_mean=tpot_mean,
        tpot_p99=percentile(tpots, 0.99) if tpots else None,
        throughput=throughput,
        makespan=makespan,
        constructed_ideal_prefix_hit_rate=constructed,
        arrival_qps=arrival_qps,
        num_sessions=num_sessions,
        root_request_ratio=root_ratio,
        mean_prompt_tokens=input_tokens / len(records),
        mean_output_tokens=sum(record.output_len for record in records) / len(records),
        child_ideal_prefix_hit_rate=(
            sum(record.ideal_reuse_tokens for record in children) / child_input
            if child_input
            else None
        ),
        child_actual_prefix_hit_rate=(
            sum(record.actual_hit_tokens for record in children) / child_input
            if child_input
            else None
        ),
        actual_prefix_hit_rate=actual / input_tokens,
        l1_prefix_hit_rate=l1_hits / input_tokens,
        l2_prefix_hit_rate=l2_hits / input_tokens,
        l3_prefix_hit_rate=l3_hits / input_tokens,
        requests_with_hit_ratio=sum(record.actual_hit_tokens > 0 for record in records)
        / len(records),
        requests_fully_cached_ratio=fully_cached / len(records),
        total_input_tokens=input_tokens,
        ideal_hit_tokens=ideal,
        matched_cache_tokens=matched,
        actual_hit_tokens=actual,
        l1_hit_tokens=l1_hits,
        l2_hit_tokens=l2_hits,
        l3_hit_tokens=l3_hits,
        actual_prefill_tokens=prefill,
        replayed_prefill_tokens=replayed_prefill_tokens,
        saved_prefill_tokens=input_tokens - prefill,
        restore_bytes=restore_bytes,
        restore_time_ms=restore_time_ms,
        promotions=promotions,
        demotions=demotions,
        l1_evictions=l1_evictions,
        l1_peak_resident_blocks=peak_resident,
        l1_capacity_blocks=cache_snapshot.capacity_blocks,
        l1_resident_blocks=cache_snapshot.resident_blocks,
        l1_active_blocks=cache_snapshot.active_blocks,
        l1_evictable_blocks=cache_snapshot.evictable_blocks,
        l1_free_blocks=cache_snapshot.free_blocks,
        l1_reserved_future_blocks=cache_snapshot.reserved_future_blocks,
        l1_internal_fragmentation_tokens=cache_snapshot.internal_fragmentation_tokens,
        **tier_fields,
        **stall_fields,
        **_itl_fields(itl_values_ms, itl_counts),
    )
