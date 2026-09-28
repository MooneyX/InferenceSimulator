"""scheduler 模块单测：FCFS 顺序、双重约束、强制准入、进展保证与步长计算。"""

import pytest

from simcore.kv_cache import KVCache
from simcore.perf_model import decode_step_time, prefill_time
from simcore.scheduler import FCFSScheduler, StepPlan
from simcore.schemas import Arrival, EngineConfig, PerfModelConfig

PERF = PerfModelConfig(prefill_c1=0.5, decode_d1=10.0, decode_d2=0.001)


def make(max_batch_size: int = 4, budget: int = 1000) -> tuple[FCFSScheduler, KVCache]:
    kv = KVCache(budget)
    engine = EngineConfig(max_batch_size=max_batch_size, kv_token_budget=budget)
    return FCFSScheduler(engine, PERF, kv), kv


def arrival(request_id: int, prompt_len: int = 100, t: float | None = None) -> Arrival:
    return Arrival(
        request_id=request_id,
        sim_time=float(request_id) if t is None else t,
        prompt_len=prompt_len,
        output_len=8,
    )


def must_plan(scheduler: FCFSScheduler) -> StepPlan:
    plan = scheduler.plan_step()
    assert plan is not None
    return plan


def ids(plan: StepPlan) -> list[int]:
    return [a.request_id for a in plan.admitted]


class TestConstruction:
    def test_budget_mismatch_rejected(self) -> None:
        engine = EngineConfig(max_batch_size=4, kv_token_budget=1000)
        with pytest.raises(ValueError, match="不一致"):
            FCFSScheduler(engine, PERF, KVCache(999))


class TestIdle:
    def test_idle_returns_none(self) -> None:
        scheduler, _ = make()
        assert scheduler.plan_step() is None

    def test_decode_only_step_when_queue_empty(self) -> None:
        scheduler, kv = make()
        scheduler.enqueue(arrival(0))
        must_plan(scheduler)
        plan = must_plan(scheduler)
        assert plan.admitted == () and plan.num_decoding == 1 and plan.has_decode
        assert len(kv) == 1


class TestEnqueue:
    def test_out_of_order_rejected(self) -> None:
        scheduler, _ = make()
        scheduler.enqueue(arrival(0, t=5.0))
        with pytest.raises(ValueError, match="到达升序"):
            scheduler.enqueue(arrival(1, t=4.0))

    def test_equal_time_accepted(self) -> None:
        scheduler, _ = make()
        scheduler.enqueue(arrival(0, t=5.0))
        scheduler.enqueue(arrival(1, t=5.0))
        assert scheduler.num_waiting == 2


class TestBatchSizeConstraint:
    def test_admits_up_to_max_batch_size(self) -> None:
        scheduler, _ = make(max_batch_size=3)
        for i in range(5):
            scheduler.enqueue(arrival(i, prompt_len=10))
        plan = must_plan(scheduler)
        assert ids(plan) == [0, 1, 2]
        assert scheduler.num_waiting == 2

    def test_running_requests_consume_capacity(self) -> None:
        scheduler, _ = make(max_batch_size=3)
        scheduler.enqueue(arrival(0, prompt_len=10))
        scheduler.enqueue(arrival(1, prompt_len=10))
        must_plan(scheduler)
        for i in range(2, 6):
            scheduler.enqueue(arrival(i, prompt_len=10))
        plan = must_plan(scheduler)
        assert ids(plan) == [2]
        assert plan.batch_size == 3

    def test_full_batch_admits_nothing(self) -> None:
        scheduler, _ = make(max_batch_size=2)
        for i in range(4):
            scheduler.enqueue(arrival(i, prompt_len=10))
        must_plan(scheduler)
        plan = must_plan(scheduler)
        assert plan.admitted == () and plan.num_decoding == 2


class TestKVConstraint:
    def test_stops_when_budget_exhausted(self) -> None:
        scheduler, kv = make(budget=250)
        for i in range(3):
            scheduler.enqueue(arrival(i, prompt_len=100))
        plan = must_plan(scheduler)
        assert ids(plan) == [0, 1]
        assert kv.used == 200

    def test_strict_fcfs_no_skip(self) -> None:
        """队头放不下时不跳过它去准入后面更小的请求。"""
        scheduler, _ = make(budget=250)
        scheduler.enqueue(arrival(0, prompt_len=200))
        scheduler.enqueue(arrival(1, prompt_len=100))  # 放不下，队头
        scheduler.enqueue(arrival(2, prompt_len=10))  # 放得下，但不得越过 1
        plan = must_plan(scheduler)
        assert ids(plan) == [0]
        assert scheduler.num_waiting == 2

    def test_release_unblocks_head(self) -> None:
        scheduler, kv = make(budget=250)
        scheduler.enqueue(arrival(0, prompt_len=200))
        scheduler.enqueue(arrival(1, prompt_len=100))
        must_plan(scheduler)
        kv.release(0)
        assert ids(must_plan(scheduler)) == [1]

    def test_no_admission_while_over_budget(self) -> None:
        """在跑请求增长超预算后，任何新请求都不准入（即使 batch 有空位）。"""
        scheduler, kv = make(budget=100)
        scheduler.enqueue(arrival(0, prompt_len=100))
        must_plan(scheduler)
        kv.advance([0])
        scheduler.enqueue(arrival(1, prompt_len=1))
        plan = must_plan(scheduler)
        assert plan.admitted == ()


class TestForcedAdmission:
    def test_oversized_head_forced_when_batch_empty(self) -> None:
        scheduler, kv = make(budget=100)
        scheduler.enqueue(arrival(0, prompt_len=150))
        plan = must_plan(scheduler)
        assert ids(plan) == [0] and plan.forced == (0,)
        assert kv.over_budget and kv.stats().forced_admissions == 1

    def test_forced_request_runs_alone(self) -> None:
        """强制准入后预算已超，同步不再准入其他请求。"""
        scheduler, _ = make(budget=100)
        scheduler.enqueue(arrival(0, prompt_len=150))
        scheduler.enqueue(arrival(1, prompt_len=1))
        plan = must_plan(scheduler)
        assert ids(plan) == [0]
        assert scheduler.num_waiting == 1

    def test_oversized_head_waits_while_batch_non_empty(self) -> None:
        scheduler, kv = make(budget=100)
        scheduler.enqueue(arrival(0, prompt_len=50))
        must_plan(scheduler)
        scheduler.enqueue(arrival(1, prompt_len=150))
        plan = must_plan(scheduler)
        assert plan.admitted == () and plan.forced == ()
        kv.release(0)
        plan = must_plan(scheduler)
        assert plan.forced == (1,)

    def test_oversized_head_waits_for_same_step_admissions(self) -> None:
        """同一步内已有请求准入时，batch 不再为空，超大请求不得强制准入。"""
        scheduler, _ = make(budget=100)
        scheduler.enqueue(arrival(0, prompt_len=10))
        scheduler.enqueue(arrival(1, prompt_len=150))
        plan = must_plan(scheduler)
        assert ids(plan) == [0] and plan.forced == ()

    def test_normal_request_not_counted_as_forced(self) -> None:
        scheduler, _ = make(budget=100)
        scheduler.enqueue(arrival(0, prompt_len=100))
        assert must_plan(scheduler).forced == ()


class TestProgressGuarantee:
    @pytest.mark.parametrize("prompt_len", [1, 100, 1000, 10_000])
    def test_empty_batch_always_admits_head(self, prompt_len: int) -> None:
        scheduler, _ = make(budget=100)
        scheduler.enqueue(arrival(0, prompt_len=prompt_len))
        assert ids(must_plan(scheduler)) == [0]


class TestStepDuration:
    def test_pure_prefill_step(self) -> None:
        scheduler, _ = make()
        scheduler.enqueue(arrival(0, prompt_len=100))
        scheduler.enqueue(arrival(1, prompt_len=60))
        plan = must_plan(scheduler)
        assert not plan.has_decode
        assert plan.duration == pytest.approx(prefill_time(100, PERF) + prefill_time(60, PERF))

    def test_batch_total_tokens_includes_admitted_prompts(self) -> None:
        scheduler, kv = make()
        scheduler.enqueue(arrival(0, prompt_len=100))
        must_plan(scheduler)
        kv.advance([0])
        scheduler.enqueue(arrival(1, prompt_len=40))
        plan = must_plan(scheduler)
        assert plan.batch_total_tokens == 101 + 40 == kv.used

    def test_fused_step_takes_max(self) -> None:
        scheduler, _ = make()
        scheduler.enqueue(arrival(0, prompt_len=100))
        must_plan(scheduler)
        scheduler.enqueue(arrival(1, prompt_len=4))
        plan = must_plan(scheduler)
        # prefill(4)=2.0 < decode(104)=10.104
        assert plan.duration == pytest.approx(decode_step_time(104, PERF))

    def test_decode_only_step_duration(self) -> None:
        scheduler, _ = make()
        scheduler.enqueue(arrival(0, prompt_len=100))
        must_plan(scheduler)
        assert must_plan(scheduler).duration == pytest.approx(decode_step_time(100, PERF))
