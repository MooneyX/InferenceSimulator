"""F001 单步 P/D 工作量快照与分析成本契约。"""

from dataclasses import FrozenInstanceError

import pytest

from simcore.perf_model import DecodeWork, PrefillWork, StepWorkload, build_layer_plan, step_cost
from simcore.schemas import LayerSegment
from test_schemas import TOY_HARDWARE, TOY_PERFORMANCE


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
    combined = step_cost(mixed, TOY_PERFORMANCE.plan, TOY_HARDWARE)
    prefill_cost = step_cost(prefill, TOY_PERFORMANCE.plan, TOY_HARDWARE)
    decode_cost = step_cost(decode, TOY_PERFORMANCE.plan, TOY_HARDWARE)
    assert combined.flops == prefill_cost.flops + decode_cost.flops
    assert combined.hbm_bytes < prefill_cost.hbm_bytes + decode_cost.hbm_bytes


def test_restored_requests_do_not_add_device_work_or_generate_tokens() -> None:
    decode = (DecodeWork(2, 8),)
    workload = StepWorkload("DECODE", (), decode, 1, 1)
    with_restore = StepWorkload("DECODE", (), decode, 1, 2, restore_only=(1,))
    assert step_cost(with_restore, TOY_PERFORMANCE.plan, TOY_HARDWARE) == step_cost(
        workload, TOY_PERFORMANCE.plan, TOY_HARDWARE
    )
    assert with_restore.decode == decode


def test_layer_plan_merges_adjacent_kinds_and_costs_follow_layer_order() -> None:
    model = TOY_PERFORMANCE.plan.model.model_copy(
        update={
            "layers": [
                LayerSegment(count=2),
                LayerSegment(count=1),
                LayerSegment(count=1, ffn="moe"),
                LayerSegment(count=2),
            ],
            "num_experts": 4,
            "experts_per_token": 2,
            "expert_intermediate_size": 8,
        }
    )
    plan = build_layer_plan(model)
    assert plan.segments == ((3, 0), (1, 1), (2, 0))
    assert plan.kind_counts == (5, 1)
    cost = step_cost(StepWorkload("DECODE", (), (DecodeWork(0, 12),), 1, 1), plan, TOY_HARDWARE)
    assert [count for count, _ in cost.layer_ms] == [3, 1, 2]
    assert cost.layer_ms[0][1] == cost.layer_ms[2][1] != cost.layer_ms[1][1]
    assert sum(count * layer for count, layer in cost.layer_ms) == pytest.approx(cost.duration_ms)
