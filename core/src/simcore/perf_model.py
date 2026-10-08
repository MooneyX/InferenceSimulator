"""性能模型：把 token 规模映射为耗时（ms）。

本模块全部为**纯函数**——输出仅取决于显式入参，无全局状态、无缓存、无随机性。
单步计费使用 F001 ``StepWorkload``。
只要系数一致，任何语言的等价实现都必须给出逐位相同的结果。

依赖层级：仅依赖 ``schemas``。
"""

from dataclasses import dataclass
from typing import Literal

from simcore.schemas import PerfModelConfig

__all__ = [
    "DecodeWork",
    "PrefillWork",
    "StepWorkload",
    "decode_step_time",
    "prefill_cost",
    "restore_bytes",
    "restore_time",
    "workload_time",
]


@dataclass(frozen=True, slots=True)
class PrefillWork:
    request_id: int
    reused_tokens: int
    new_tokens: int
    context_after: int

    def __post_init__(self) -> None:
        if self.request_id < 0 or self.reused_tokens < 0 or self.new_tokens < 1:
            raise ValueError("Prefill 请求 ID、复用量或新计算量不合法")
        if self.context_after != self.reused_tokens + self.new_tokens:
            raise ValueError("Prefill 上下文长度必须等于复用量与新计算量之和")


@dataclass(frozen=True, slots=True)
class DecodeWork:
    request_id: int
    context_after_decode: int

    def __post_init__(self) -> None:
        if self.request_id < 0 or self.context_after_decode < 1:
            raise ValueError("Decode 请求 ID 或上下文长度不合法")


@dataclass(frozen=True, slots=True)
class StepWorkload:
    phase: Literal["PREFILL", "DECODE", "MIXED", "RESTORE"]
    prefill: tuple[PrefillWork, ...]
    decode: tuple[DecodeWork, ...]
    allocated_pages: int
    batch_requests: int
    restore_sources: tuple[tuple[str, str], ...] = ()
    restore_page_size: int = 0
    restore_only: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        prefill_ids = tuple(item.request_id for item in self.prefill)
        decode_ids = tuple(item.request_id for item in self.decode)
        phase = (
            "MIXED" if prefill_ids and decode_ids else
            "PREFILL" if prefill_ids else
            "DECODE" if decode_ids else "RESTORE"
        )
        if (not prefill_ids and not decode_ids and not self.restore_only) or self.phase != phase:
            raise ValueError("单步阶段须与非空 Prefill/Decode/Restore 集合一致")
        if (
            prefill_ids != tuple(sorted(set(prefill_ids)))
            or decode_ids != tuple(sorted(set(decode_ids)))
            or set(prefill_ids) & set(decode_ids)
            or self.restore_only != tuple(sorted(set(self.restore_only)))
            or (set(self.restore_only) & (set(prefill_ids) | set(decode_ids)))
        ):
            raise ValueError("Prefill/Decode/Restore 请求须各自按 ID 排序且不可重叠")
        if self.allocated_pages < 0 or self.batch_requests != (
            len(prefill_ids) + len(decode_ids) + len(self.restore_only)
        ):
            raise ValueError("单步页申请量或批内请求数不合法")
        if (
            self.restore_sources != tuple(sorted(self.restore_sources))
            or len({chunk_id for chunk_id, _ in self.restore_sources}) != len(self.restore_sources)
            or any(source not in ("L2", "L3") for _, source in self.restore_sources)
            or self.restore_page_size < 0
            or (bool(self.restore_sources) != (self.restore_page_size > 0))
        ):
            raise ValueError("恢复来源须唯一有序、来自低层并具有正页大小")
        if self.phase == "RESTORE" and not self.restore_sources:
            raise ValueError("纯 Restore 步须实际传输缓存页")


def restore_bytes(workload: StepWorkload, cfg: PerfModelConfig) -> int:
    """上行链路字节数；L3 命中需要经过两条链路。"""
    if not workload.restore_sources:
        return 0
    if cfg.kv_bytes_per_token is None:
        raise ValueError("恢复计费须配置 kv_bytes_per_token")
    l3 = sum(source == "L3" for _, source in workload.restore_sources)
    return (
        (len(workload.restore_sources) + l3)
        * workload.restore_page_size
        * cfg.kv_bytes_per_token
    )


def restore_time(workload: StepWorkload, cfg: PerfModelConfig) -> float:
    """每步每条实际使用的链路只计一次固定延迟。"""
    if not workload.restore_sources:
        return 0.0
    if cfg.kv_bytes_per_token is None or cfg.l2_to_l1_bandwidth_bytes_per_ms is None:
        raise ValueError("恢复计费须配置 KV 字节大小与 L2 到 L1 带宽")
    l3 = sum(source == "L3" for _, source in workload.restore_sources)
    page_bytes = workload.restore_page_size * cfg.kv_bytes_per_token
    duration = cfg.l2_latency_ms + (
        len(workload.restore_sources) * page_bytes / cfg.l2_to_l1_bandwidth_bytes_per_ms
    )
    if l3:
        if cfg.l3_to_l2_bandwidth_bytes_per_ms is None:
            raise ValueError("L3 恢复计费须配置 L3 到 L2 带宽")
        duration += cfg.l3_latency_ms + l3 * page_bytes / cfg.l3_to_l2_bandwidth_bytes_per_ms
    return duration


def workload_time(workload: StepWorkload, cfg: PerfModelConfig) -> float:
    """纯步取各自成本；融合取较大成本，不重复统计同一步 Decode。"""
    prefill_total = 0.0
    for item in workload.prefill:
        prefill_total += prefill_cost(item.reused_tokens, item.new_tokens, cfg)
    decode_total = (
        decode_step_time(sum(item.context_after_decode for item in workload.decode), cfg)
        if workload.decode
        else 0.0
    )
    return max(prefill_total, decode_total)


def prefill_cost(context_before: int, new_prefill_tokens: int, cfg: PerfModelConfig) -> float:
    """前缀命中后仍需读取历史上下文：二次项按历史与新输入的面积差计算。"""
    if context_before < 0 or new_prefill_tokens < 1:
        raise ValueError("context_before 须 >= 0 且 new_prefill_tokens 须 >= 1")
    return cfg.prefill_c1 * new_prefill_tokens + cfg.prefill_c2 * new_prefill_tokens * (
        2 * context_before + new_prefill_tokens
    )


def decode_step_time(batch_total_tokens: int, cfg: PerfModelConfig) -> float:
    """一次 decode 迭代的耗时（ms）。

    ``decode_d1 + decode_d2 * T``。常数项为与 batch 无关的固定开销（kernel launch 等），
    线性项随 batch 内 KV token 总量增长（访存带宽受限）。

    注意：耗时按**整个 batch 一次迭代**计，不是每请求。batch 为空时无意义，应由调用方
    避免（引擎在 batch 为空时直接跳跃时钟，不执行 step）。

    Args:
        batch_total_tokens: 本步 batch 内所有请求的 KV token 总量，须 >= 1。
        cfg: 性能模型系数。

    Raises:
        ValueError: ``batch_total_tokens`` < 1。
    """
    if batch_total_tokens < 1:
        raise ValueError(f"batch_total_tokens 须 >= 1，得到 {batch_total_tokens}")
    return cfg.decode_d1 + cfg.decode_d2 * batch_total_tokens
