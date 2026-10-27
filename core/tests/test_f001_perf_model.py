"""F001 单步 P/D 工作量快照与分析成本契约。"""

from dataclasses import FrozenInstanceError

import pytest

from simcore.analytical import step_cost
from simcore.perf_model import DecodeWork, PrefillWork, StepWorkload
from test_schemas import TOY_HARDWARE, TOY_MODEL


def test_prefill_snapshot_preserves_request_id_and_reused_context() -> None:
    workload = StepWorkload(
        "PREFILL",
        (PrefillWork(0, 4, 2, 6), PrefillWork(2, 1, 3, 4)),
        (),
        allocated_pages=3,
        batch_requests=2,
    )
    assert workload.prefill == (PrefillWork(0, 4, 2, 6), PrefillWork(2, 1, 3, 4))
    assert workload.decode == ()
    assert (workload.allocated_pages, workload.batch_requests) == (3, 2)
    with pytest.raises(FrozenInstanceError):
        workload.phase = "DECODE"  # type: ignore[misc]


def test_decode_counts_only_previously_running_contexts() -> None:
    workload = StepWorkload("DECODE", (), (DecodeWork(1, 65), DecodeWork(3, 20)), 1, 2)
    assert workload.prefill == ()
    assert sum(item.context_after_decode for item in workload.decode) == 85
    assert workload.batch_requests == 2


def test_mixed_combines_device_work_and_reuses_weight_reads() -> None:
    mixed = StepWorkload("MIXED", (PrefillWork(1, 0, 8, 8),), (DecodeWork(0, 12),), 3, 2)
    prefill = StepWorkload("PREFILL", mixed.prefill, (), 2, 1)
    decode = StepWorkload("DECODE", (), mixed.decode, 1, 1)
    combined = step_cost(mixed, TOY_MODEL, TOY_HARDWARE)
    prefill_cost = step_cost(prefill, TOY_MODEL, TOY_HARDWARE)
    decode_cost = step_cost(decode, TOY_MODEL, TOY_HARDWARE)
    assert combined.flops == prefill_cost.flops + decode_cost.flops
    assert combined.hbm_bytes < prefill_cost.hbm_bytes + decode_cost.hbm_bytes


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


def test_restored_requests_do_not_add_device_work_or_generate_tokens() -> None:
    decode = (DecodeWork(2, 8),)
    workload = StepWorkload("DECODE", (), decode, 1, 1)
    with_restore = StepWorkload("DECODE", (), decode, 1, 2, restore_only=(1,))
    assert step_cost(with_restore, TOY_MODEL, TOY_HARDWARE) == step_cost(
        workload, TOY_MODEL, TOY_HARDWARE
    )
    assert with_restore.decode == decode
    with pytest.raises(ValueError, match="阶段"):
        StepWorkload("DECODE", (), (), 0, 1, restore_only=(1,))


@pytest.mark.parametrize("restored", [(2,), (3, 1), (1, 1)])
def test_restored_requests_must_be_sorted_unique_and_disjoint(restored: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="不可重叠"):
        StepWorkload("DECODE", (), (DecodeWork(2, 8),), 0, 1 + len(restored), restored)
