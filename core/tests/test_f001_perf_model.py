"""F001 单步 P/D 工作量快照与四系数成本契约。"""

from dataclasses import FrozenInstanceError

import pytest

from simcore.perf_model import DecodeWork, PrefillWork, StepWorkload, workload_time
from simcore.schemas import PerfModelConfig

PERF = PerfModelConfig(prefill_c1=2, prefill_c2=0.25, decode_d1=5, decode_d2=0.5)


def test_prefill_cost_sums_by_request_id_and_reused_context() -> None:
    workload = StepWorkload(
        "PREFILL",
        (PrefillWork(0, 4, 2, 6), PrefillWork(2, 1, 3, 4)),
        (),
        allocated_pages=3,
        batch_requests=2,
    )
    expected = (2 * 2 + 0.25 * 2 * (2 * 4 + 2)) + (2 * 3 + 0.25 * 3 * (2 * 1 + 3))
    assert workload_time(workload, PERF) == expected
    with pytest.raises(FrozenInstanceError):
        workload.phase = "DECODE"  # type: ignore[misc]


def test_decode_counts_only_previously_running_contexts() -> None:
    workload = StepWorkload("DECODE", (), (DecodeWork(1, 65), DecodeWork(3, 20)), 1, 2)
    assert workload_time(workload, PERF) == 5 + 0.5 * 85


def test_mixed_takes_max_of_two_costs_not_sum() -> None:
    mixed = StepWorkload("MIXED", (PrefillWork(1, 0, 8, 8),), (DecodeWork(0, 12),), 3, 2)
    assert workload_time(mixed, PERF) == max(2 * 8 + 0.25 * 64, 5 + 0.5 * 12)
    assert workload_time(mixed, PERF) != (2 * 8 + 0.25 * 64) + (5 + 0.5 * 12)


@pytest.mark.parametrize(
    ("phase", "prefill", "decode"),
    [
        ("PREFILL", (), ()),
        ("PREFILL", (), (DecodeWork(0, 1),)),
        ("DECODE", (PrefillWork(0, 0, 1, 1),), ()),
        ("MIXED", (PrefillWork(0, 0, 1, 1),), ()),
    ],
)
def test_invalid_phase_or_empty_step_rejected(
    phase: str, prefill: tuple[PrefillWork, ...], decode: tuple[DecodeWork, ...]
) -> None:
    with pytest.raises(ValueError, match="阶段"):
        StepWorkload(phase, prefill, decode, 0, len(prefill) + len(decode))  # type: ignore[arg-type]


def test_invalid_workload_detail_rejected() -> None:
    with pytest.raises(ValueError, match="上下文"):
        PrefillWork(0, 2, 1, 2)
    with pytest.raises(ValueError, match="排序"):
        StepWorkload("PREFILL", (PrefillWork(2, 0, 1, 1), PrefillWork(1, 0, 1, 1)), (), 0, 2)
    with pytest.raises(ValueError, match="不可重叠"):
        StepWorkload("MIXED", (PrefillWork(0, 0, 1, 1),), (DecodeWork(0, 2),), 0, 2)
    with pytest.raises(ValueError, match="申请量"):
        StepWorkload("DECODE", (), (DecodeWork(0, 2),), -1, 1)


def test_zero_coefficients_have_zero_duration_but_valid_workload() -> None:
    zero = PerfModelConfig(prefill_c1=0, prefill_c2=0, decode_d1=0, decode_d2=0)
    mixed = StepWorkload("MIXED", (PrefillWork(1, 0, 1, 1),), (DecodeWork(0, 20),), 2, 2)
    assert workload_time(mixed, zero) == 0
