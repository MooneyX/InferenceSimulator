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
    "workload_time",
]


@dataclass(frozen=True, slots=True)
class PrefillWork:
    request_id: int
    reused_tokens: int
    new_tokens: int
    context_after: int

    def __post_init__(self) -> None:  # 校验 Prefill 请求身份、复用量和输入规模；无返回值
        if self.request_id < 0 or self.reused_tokens < 0 or self.new_tokens < 1:
            raise ValueError("Prefill 请求 ID、复用量或新计算量不合法")
        if self.context_after != self.reused_tokens + self.new_tokens:
            raise ValueError("Prefill 上下文长度必须等于复用量与新计算量之和")


@dataclass(frozen=True, slots=True)
class DecodeWork:
    request_id: int
    context_after_decode: int

    def __post_init__(self) -> None:  # 校验 Decode 请求 ID 与上下文长度；无返回值
        if self.request_id < 0 or self.context_after_decode < 1:
            raise ValueError("Decode 请求 ID 或上下文长度不合法")


@dataclass(frozen=True, slots=True)
class StepWorkload:
    phase: Literal["PREFILL", "DECODE", "MIXED"]
    prefill: tuple[PrefillWork, ...]
    decode: tuple[DecodeWork, ...]
    allocated_pages: int
    batch_requests: int
    restore_only: tuple[int, ...] = ()

    def __post_init__(self) -> None:  # 校验 GPU 步阶段、请求集合与页数；无返回值
        prefill_ids = tuple(item.request_id for item in self.prefill)
        decode_ids = tuple(item.request_id for item in self.decode)
        phase = (
            "MIXED"
            if prefill_ids and decode_ids
            else "PREFILL"
            if prefill_ids
            else "DECODE"
        )
        if (not prefill_ids and not decode_ids) or self.phase != phase:
            raise ValueError("单步阶段须与非空 Prefill/Decode 集合一致")
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


def workload_time(workload: StepWorkload, cfg: PerfModelConfig) -> float:  # Prefill/Decode 工作量与经验系数 → 融合后的设备耗时（ms）
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


def prefill_cost(context_before: int, new_prefill_tokens: int, cfg: PerfModelConfig) -> float:  # 历史上下文、新 token 数与系数 → Prefill 耗时（ms）
    """前缀命中后仍需读取历史上下文：二次项按历史与新输入的面积差计算。"""
    if context_before < 0 or new_prefill_tokens < 1:
        raise ValueError("context_before 须 >= 0 且 new_prefill_tokens 须 >= 1")
    return cfg.prefill_c1 * new_prefill_tokens + cfg.prefill_c2 * new_prefill_tokens * (
        2 * context_before + new_prefill_tokens  # (旧上下文 + 新输入)² − 旧上下文²
    )


def decode_step_time(batch_total_tokens: int, cfg: PerfModelConfig) -> float:  # 批内 KV token 总量与性能系数 → 一次 Decode 耗时（ms）
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
