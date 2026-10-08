"""指标汇总：由区间记录计算全场 ``Metrics``。

计算口径见 ``docs/design/metrics.md``：

- 分位数用线性插值（numpy ``percentile`` 默认 ``linear`` 方法）。
- TPOT 先逐请求算 ``decode 时长 / (output_len - 1)``，再对请求取平均；单 token 请求不计入。
- 吞吐单位 请求/秒；makespan 为 0 时无定义。

纯函数，无状态。依赖层级：仅依赖 ``schemas``。
"""

import math
from collections.abc import Sequence

from simcore.kv_cache import L1Snapshot, TierSnapshot
from simcore.schemas import Metrics, RequestRecord, SimConfig, WorkloadPlan

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


def compute_metrics(
    records: Sequence[RequestRecord],
    *,
    plan: WorkloadPlan | None = None,
    config: SimConfig | None = None,
    cache_snapshot: L1Snapshot | None = None,
    l2_snapshot: TierSnapshot | None = None,
    l3_snapshot: TierSnapshot | None = None,
    l1_evictions: int = 0,
    l1_peak_resident_blocks: int = 0,
    promotions: int = 0,
    demotions: int = 0,
    restore_bytes: int = 0,
    restore_time_ms: float = 0.0,
    earliest_sent_arrival: float | None = None,
    replayed_prefill_tokens: int = 0,
) -> Metrics:
    """仅对完成请求计算时延；全失败时返回无定义指标和实际容量。"""
    synthetic = config.workload.synthetic if config else None
    target = synthetic.session.target_ideal_prefix_hit_rate if synthetic else 0.0
    constructed = plan.constructed_ideal_prefix_hit_rate if plan else 0.0
    quantization_error = abs(constructed - target) if plan and synthetic else 0.0
    peak_resident = max(
        l1_peak_resident_blocks, cache_snapshot.resident_blocks if cache_snapshot else 0
    )
    if not records:
        return Metrics(
            ttft_mean=None,
            ttft_p50=None,
            ttft_p99=None,
            tpot_mean=None,
            throughput=None,
            makespan=0.0,
            target_ideal_prefix_hit_rate=target,
            constructed_ideal_prefix_hit_rate=constructed,
            ideal_hit_quantization_error=quantization_error,
            replayed_prefill_tokens=replayed_prefill_tokens,
            restore_bytes=restore_bytes,
            restore_time_ms=restore_time_ms,
            promotions=promotions,
            demotions=demotions,
            l1_evictions=l1_evictions,
            l1_peak_resident_blocks=peak_resident,
            l1_capacity_blocks=cache_snapshot.capacity_blocks if cache_snapshot else 0,
            l1_resident_blocks=cache_snapshot.resident_blocks if cache_snapshot else 0,
            l1_active_blocks=cache_snapshot.active_blocks if cache_snapshot else 0,
            l1_evictable_blocks=cache_snapshot.evictable_blocks if cache_snapshot else 0,
            l1_free_blocks=cache_snapshot.free_blocks if cache_snapshot else 0,
            l1_reserved_future_blocks=(
                cache_snapshot.reserved_future_blocks if cache_snapshot else 0
            ),
            l1_internal_fragmentation_tokens=(
                cache_snapshot.internal_fragmentation_tokens if cache_snapshot else 0
            ),
            l2_evictions=l2_snapshot.evictions if l2_snapshot else 0,
            l2_peak_resident_blocks=l2_snapshot.peak_resident_blocks if l2_snapshot else 0,
            l2_capacity_blocks=l2_snapshot.capacity_blocks if l2_snapshot else 0,
            l2_resident_blocks=l2_snapshot.resident_blocks if l2_snapshot else 0,
            l2_active_blocks=l2_snapshot.active_blocks if l2_snapshot else 0,
            l2_evictable_blocks=l2_snapshot.evictable_blocks if l2_snapshot else 0,
            l2_free_blocks=l2_snapshot.free_blocks if l2_snapshot else 0,
            l3_evictions=l3_snapshot.evictions if l3_snapshot else 0,
            l3_peak_resident_blocks=l3_snapshot.peak_resident_blocks if l3_snapshot else 0,
            l3_capacity_blocks=l3_snapshot.capacity_blocks if l3_snapshot else 0,
            l3_resident_blocks=l3_snapshot.resident_blocks if l3_snapshot else 0,
            l3_active_blocks=l3_snapshot.active_blocks if l3_snapshot else 0,
            l3_evictable_blocks=l3_snapshot.evictable_blocks if l3_snapshot else 0,
            l3_free_blocks=l3_snapshot.free_blocks if l3_snapshot else 0,
        )

    ttfts = sorted(r.prefill_end - r.arrival for r in records)
    ttft_mean = math.fsum(ttfts) / len(ttfts)

    # 首 token 已在 Prefill 生成，单 token 请求不参与 TPOT。
    tpots = [(r.finish - r.prefill_end) / (r.output_len - 1) for r in records if r.output_len > 1]
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
    prefill = sum(
        record.actual_prefill_tokens
        if record.actual_prefill_tokens is not None
        else record.prompt_len - record.actual_hit_tokens
        for record in records
    )
    block_size = config.engine.page_size if config else 1
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
        throughput=throughput,
        makespan=makespan,
        target_ideal_prefix_hit_rate=target,
        constructed_ideal_prefix_hit_rate=constructed if plan else ideal / input_tokens,
        ideal_hit_quantization_error=quantization_error,
        actual_prefix_hit_rate=actual / input_tokens,
        requests_with_hit_ratio=sum(record.actual_hit_tokens > 0 for record in records)
        / len(records),
        requests_fully_cached_ratio=fully_cached / len(records),
        total_input_tokens=input_tokens,
        ideal_hit_tokens=ideal,
        matched_cache_tokens=matched,
        actual_hit_tokens=actual,
        l1_hit_tokens=sum(record.l1_hit_tokens for record in records),
        l2_hit_tokens=sum(record.l2_hit_tokens for record in records),
        l3_hit_tokens=sum(record.l3_hit_tokens for record in records),
        actual_prefill_tokens=prefill,
        replayed_prefill_tokens=replayed_prefill_tokens,
        saved_prefill_tokens=input_tokens - prefill,
        restore_bytes=restore_bytes,
        restore_time_ms=restore_time_ms,
        promotions=promotions,
        demotions=demotions,
        l1_evictions=l1_evictions,
        l1_peak_resident_blocks=peak_resident,
        l1_capacity_blocks=cache_snapshot.capacity_blocks if cache_snapshot else 0,
        l1_resident_blocks=cache_snapshot.resident_blocks if cache_snapshot else 0,
        l1_active_blocks=cache_snapshot.active_blocks if cache_snapshot else 0,
        l1_evictable_blocks=cache_snapshot.evictable_blocks if cache_snapshot else 0,
        l1_free_blocks=cache_snapshot.free_blocks if cache_snapshot else 0,
        l1_reserved_future_blocks=(cache_snapshot.reserved_future_blocks if cache_snapshot else 0),
        l1_internal_fragmentation_tokens=(
            cache_snapshot.internal_fragmentation_tokens if cache_snapshot else 0
        ),
        l2_evictions=l2_snapshot.evictions if l2_snapshot else 0,
        l2_peak_resident_blocks=l2_snapshot.peak_resident_blocks if l2_snapshot else 0,
        l2_capacity_blocks=l2_snapshot.capacity_blocks if l2_snapshot else 0,
        l2_resident_blocks=l2_snapshot.resident_blocks if l2_snapshot else 0,
        l2_active_blocks=l2_snapshot.active_blocks if l2_snapshot else 0,
        l2_evictable_blocks=l2_snapshot.evictable_blocks if l2_snapshot else 0,
        l2_free_blocks=l2_snapshot.free_blocks if l2_snapshot else 0,
        l3_evictions=l3_snapshot.evictions if l3_snapshot else 0,
        l3_peak_resident_blocks=l3_snapshot.peak_resident_blocks if l3_snapshot else 0,
        l3_capacity_blocks=l3_snapshot.capacity_blocks if l3_snapshot else 0,
        l3_resident_blocks=l3_snapshot.resident_blocks if l3_snapshot else 0,
        l3_active_blocks=l3_snapshot.active_blocks if l3_snapshot else 0,
        l3_evictable_blocks=l3_snapshot.evictable_blocks if l3_snapshot else 0,
        l3_free_blocks=l3_snapshot.free_blocks if l3_snapshot else 0,
    )
