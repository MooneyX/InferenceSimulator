"""F001 单步 P/D 工作量快照与四系数成本契约。"""

from dataclasses import FrozenInstanceError

import pytest

from simcore.perf_model import (
    DecodeWork,
    PrefillWork,
    StepWorkload,
    restore_bytes,
    restore_time,
    workload_time,
)
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


def test_restore_only_l3_uses_two_links_and_charges_each_fixed_latency_once() -> None:
    perf = PerfModelConfig(
        prefill_c1=1, decode_d1=1, decode_d2=0,
        kv_bytes_per_token=2,
        l2_to_l1_bandwidth_bytes_per_ms=4,
        l3_to_l2_bandwidth_bytes_per_ms=2,
        l2_latency_ms=3,
        l3_latency_ms=5,
    )
    transfer = StepWorkload(
        "RESTORE", (), (), 0, 1,
        (("a", "L2"), ("b", "L3")), 4, (1,),
    )
    assert workload_time(transfer, perf) == 0
    assert restore_bytes(transfer, perf) == 24
    assert restore_time(transfer, perf) == 3 + 2 * 8 / 4 + 5 + 8 / 2
    mixed = StepWorkload(
        "MIXED", (PrefillWork(1, 0, 2, 2),), (DecodeWork(2, 8),), 1, 2,
        transfer.restore_sources, 4,
    )
    assert restore_time(mixed, perf) == restore_time(transfer, perf)
    assert workload_time(mixed, perf) == max(2, 1)


def test_restore_sources_must_be_unique_and_have_valid_tier_and_page_size() -> None:
    for sources, page_size in (
        ((("a", "L2"), ("a", "L3")), 4),
        ((("a", "L1"),), 4),
        ((("b", "L2"), ("a", "L3")), 4),
        ((("a", "L2"),), 0),
        ((), -1),
    ):
        with pytest.raises(ValueError, match="恢复来源"):
            StepWorkload("RESTORE", (), (), 0, 1, sources, page_size, (1,))
