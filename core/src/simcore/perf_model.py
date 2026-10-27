"""性能模型的单步工作量：记录 Prefill、Decode 与恢复请求的输入规模。"""

from dataclasses import dataclass
from typing import Literal

__all__ = ["DecodeWork", "PrefillWork", "StepWorkload"]


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
        phase = "MIXED" if prefill_ids and decode_ids else "PREFILL" if prefill_ids else "DECODE"
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
