"""engine 模块单测：分析模型时序、同刻顺序、告警、全局不变式与可复现性。"""

from typing import Literal

import pytest

from simcore.coordinator import Coordinator
from simcore.engine import Engine, EngineRun
from simcore.perf_model import DecodeWork, PrefillWork, StepCost, StepWorkload, step_cost
from simcore.router import Router
from simcore.schemas import (
    Arrival,
    EngineConfig,
    InferenceRequest,
    LognormalLength,
    PerfModelConfig,
    PoissonLength,
    RequestRecord,
    SyntheticWorkloadConfig,
    TurnPlan,
    WorkloadConfig,
    WorkloadPlan,
)
from simcore.workload import build_synthetic_closed_loop_plan, request_output_limit
from test_client import planned_client
from test_f001_engine import start_step_end
from test_schemas import TOY_HARDWARE, TOY_MODEL, runtime_config

DEFAULT_PERF = PerfModelConfig()


def duration(*, prefill: tuple[int, ...] = (), decode: tuple[int, ...] = ()) -> float:
    phase: Literal["PREFILL", "DECODE", "MIXED"] = (
        "MIXED" if prefill and decode else "PREFILL" if prefill else "DECODE"
    )
    workload = StepWorkload(
        phase,
        tuple(PrefillWork(key, 0, tokens, tokens) for key, tokens in enumerate(prefill)),
        tuple(DecodeWork(key + len(prefill), context) for key, context in enumerate(decode)),
        0,
        len(prefill) + len(decode),
    )
    return step_cost(workload, TOY_MODEL, TOY_HARDWARE).duration_ms


def req(request_id: int, t: float, prompt_len: int, output_len: int) -> Arrival:
    return Arrival(
        request_id=request_id,
        sim_time=t,
        prompt_len=prompt_len,
        output_len=output_len,
        max_new_tokens=128,
    )


def run(
    arrivals: list[Arrival],
    *,
    max_batch_size: int = 4,
    budget: int = 1000,
    perf: PerfModelConfig = DEFAULT_PERF,
) -> EngineRun:
    config = EngineConfig(
        max_running_requests=max_batch_size,
        max_total_tokens=budget,
        page_size=1,
        pd_mutually_exclusive=False,
    )
    turns = {}
    for arrival in arrivals:
        if arrival.request_id in turns:
            raise ValueError("重复的请求 ID")
        maximum = request_output_limit(arrival.output_len, config.max_output_tokens)
        turns[arrival.request_id] = TurnPlan(
            arrival.request_id,
            arrival.request_id,
            None,
            arrival.sim_time,
            None,
            arrival.prompt_len,
            arrival.output_len,
            maximum,
            0,
        )
    plan = WorkloadPlan(turns, tuple(turns), {})
    return Coordinator(
        (Engine(runtime_config(config, perf)),),
        planned_client(plan, config),
        Router(1),
    ).run()[0]


def interval(r: RequestRecord) -> tuple[float, float, float, float]:
    return (r.arrival, r.prefill_start, r.prefill_end, r.finish)


class TestImmediateEngine:
    def test_submit_step_and_complete_updates_record(self) -> None:
        config = EngineConfig(max_running_requests=1, max_total_tokens=100, page_size=1)
        request = InferenceRequest(7, 7, None, 5, 1, 1, (), 0)
        engine = Engine(runtime_config(config, PerfModelConfig()))
        assert not engine.has_work
        assert not hasattr(engine, "_heap")
        assert not hasattr(engine, "step")
        engine.submit(request, 3.0)
        assert engine.has_work
        end = start_step_end(engine, 3.0, 0, (7,))
        assert not engine.has_work
        assert engine._prefix_cache.page("chunk:7:0") is None
        assert engine.finish_step(end) == (7,)
        engine.complete(end, 7)
        assert interval(engine.finalize().records[0]) == (3.0, 3.0, end, end)

    @pytest.mark.parametrize("maximum", [4, 8])
    def test_direct_submit_uses_actual_output_length_without_registration(
        self, maximum: int
    ) -> None:
        config = EngineConfig(max_running_requests=1, max_total_tokens=100, page_size=1)
        request = InferenceRequest(7, 7, None, 5, maximum, 3, (), 0)
        engine = Engine(runtime_config(config, PerfModelConfig()))
        engine.submit(request, 3.0)
        prefill_end = start_step_end(engine, 3.0, 0)
        assert engine.finish_step(prefill_end) == ()
        now = start_step_end(engine, prefill_end, 1)
        assert engine.finish_step(now) == ()
        now = start_step_end(engine, now, 2, (7,))
        assert engine.finish_step(now) == (7,)
        engine.complete(now, 7)
        result = engine.finalize()
        assert interval(result.records[0]) == (3.0, 3.0, prefill_end, now)
        assert result.records[0].output_len == request.output_len == 3
        assert result.num_engine_steps == 3


class TestAnalyticalTimeline:
    def test_single_request(self) -> None:
        prefill_end = duration(prefill=(5,))
        finish = prefill_end + duration(decode=(6,)) + duration(decode=(7,))
        result = run([req(0, 0.0, 5, 3)])
        assert interval(result.records[0]) == pytest.approx((0.0, 0.0, prefill_end, finish))
        assert result.num_engine_steps == 3
        assert (result.sim_start, result.sim_end) == pytest.approx((0.0, finish))
        assert result.warnings == ()

    def test_single_token_output_finishes_at_prefill_end(self) -> None:
        end = 3.0 + duration(prefill=(5,))
        result = run([req(0, 3.0, 5, 1)])
        assert interval(result.records[0]) == pytest.approx((3.0, 3.0, end, end))
        assert result.num_engine_steps == 1

    def test_queueing_under_batch_limit(self) -> None:
        prefill = duration(prefill=(5,))
        finish = prefill + duration(decode=(6,))
        arrival = prefill / 2
        result = run([req(0, 0.0, 5, 2), req(1, arrival, 5, 2)], max_batch_size=1)
        assert interval(result.records[0]) == pytest.approx((0.0, 0.0, prefill, finish))
        assert interval(result.records[1]) == pytest.approx(
            (arrival, finish, finish + prefill, 2 * finish)
        )
        assert result.num_engine_steps == 4

    def test_idle_gap_jumps_clock(self) -> None:
        """batch 为空时不压步，时钟直接跳到下一个到达。"""
        result = run([req(0, 0.0, 5, 1), req(1, 100.0, 5, 1)])
        cost = duration(prefill=(5,))
        assert cost < 100
        assert interval(result.records[0]) == pytest.approx((0.0, 0.0, cost, cost))
        assert interval(result.records[1]) == pytest.approx((100.0, 100.0, 100 + cost, 100 + cost))
        assert result.num_engine_steps == 2

    def test_fused_step_uses_combined_analytical_workload(self) -> None:
        first_end = duration(prefill=(5,))
        arrival = first_end / 2
        mixed_end = first_end + duration(prefill=(20,), decode=(6,))
        finish = mixed_end + duration(decode=(7, 21))
        result = run([req(0, 0.0, 5, 3), req(1, arrival, 20, 2)])
        assert interval(result.records[0]) == pytest.approx((0.0, 0.0, first_end, finish))
        assert interval(result.records[1]) == pytest.approx((arrival, first_end, mixed_end, finish))
        assert result.num_engine_steps == 3

    def test_same_time_arrivals_batched_together(self) -> None:
        result = run([req(0, 0.0, 5, 1), req(1, 0.0, 7, 1)])
        end = duration(prefill=(5, 7))
        assert all(interval(r) == pytest.approx((0.0, 0.0, end, end)) for r in result.records)
        assert result.num_engine_steps == 1

    def test_arrival_during_step_waits_for_step_end(self) -> None:
        first_end = duration(prefill=(50,))
        arrival = first_end / 2
        finish = first_end + duration(prefill=(5,))
        result = run([req(0, 0.0, 50, 1), req(1, arrival, 5, 1)])
        assert interval(result.records[1]) == pytest.approx((arrival, first_end, finish, finish))

    def test_unsorted_input_is_ordered_by_heap(self) -> None:
        ordered = run([req(0, 0.0, 5, 2), req(1, 1.0, 5, 2)], max_batch_size=1)
        shuffled = run([req(1, 1.0, 5, 2), req(0, 0.0, 5, 2)], max_batch_size=1)
        assert ordered.records == shuffled.records


class TestSameTimeOrdering:
    def test_release_visible_to_same_time_step(self) -> None:
        """同刻 RequestDone 先释放额度，EngineStep 才能准入排队请求。"""
        end = duration(prefill=(10,))
        result = run([req(0, 0.0, 10, 1), req(1, 0.0, 10, 1)], budget=14)
        assert interval(result.records[0]) == pytest.approx((0.0, 0.0, end, end))
        assert interval(result.records[1]) == pytest.approx((0.0, end, 2 * end, 2 * end))
        assert result.warnings == ()

    def test_arrival_visible_to_same_time_step(self) -> None:
        """同刻 Arrival 先于 EngineStep，到达请求立即参与下一步。"""
        arrival = duration(prefill=(5,))
        end = arrival + duration(prefill=(3,), decode=(6,))
        result = run([req(0, 0.0, 5, 2), req(1, arrival, 3, 1)])
        assert interval(result.records[1]) == pytest.approx((arrival, arrival, end, end))


class TestKVBehaviour:
    def test_budget_blocks_admission(self) -> None:
        result = run([req(0, 0.0, 8, 2), req(1, 0.0, 5, 1)], budget=11)
        first_end = duration(prefill=(8,)) + duration(decode=(9,))
        finish = first_end + duration(prefill=(5,))
        assert interval(result.records[1]) == pytest.approx((0.0, first_end, finish, finish))
        assert result.warnings == ()

    def test_future_growth_is_estimated_not_reserved_before_admission(self) -> None:
        result = run([req(0, 0.0, 5, 10), req(1, 0.0, 5, 10)], budget=15)
        assert len(result.records) == 2
        assert result.records[1].prefill_start >= result.records[0].finish
        assert result.cache_snapshot.resident_blocks <= result.cache_snapshot.capacity_blocks
        assert result.cache_snapshot.reserved_future_blocks == 0

    @pytest.mark.parametrize(
        "arrivals",
        [[req(0, 0.0, 15, 3)], [req(0, 0.0, 10, 1)]],
    )
    def test_request_exceeding_l1_is_aborted_without_forcing(self, arrivals: list[Arrival]) -> None:
        result = run(arrivals, budget=10)
        assert result.num_engine_steps == 0 and result.records == ()
        assert result.aborted_requests[0].reason == "input_exceeds_kv_capacity"


class TestEdgeCases:
    def test_empty_workload(self) -> None:
        result = run([])
        assert result.records == () and result.num_engine_steps == 0
        assert (result.sim_start, result.sim_end) == (0.0, 0.0)

    def test_zero_cost_model_terminates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """零步长不推进时钟，但 token 仍逐步产出并最终终止。"""
        monkeypatch.setattr("simcore.engine.step_cost", lambda *args: StepCost(0, 0, 0, 0, 0))
        result = run([req(0, 1.0, 5, 4), req(1, 1.0, 5, 4)])
        assert all(interval(r) == (1.0, 1.0, 1.0, 1.0) for r in result.records)
        assert result.num_engine_steps == 4


def synthetic_run(seed: int = 7, **engine: int) -> EngineRun:
    workload = WorkloadConfig(
        session_rate=40.0,
        seed=seed,
        synthetic=SyntheticWorkloadConfig(
            num_requests=300,
            prompt_len=LognormalLength(mean=200.0, sigma=1.0),
            turn_increment_len=LognormalLength(mean=16.0, sigma=1.0),
            output_len=PoissonLength(mean=30.0),
        ),
    )
    cfg = EngineConfig.model_validate(
        {
            "max_running_requests": 16,
            "max_total_tokens": 8000,
            "page_size": 1,
            "pd_mutually_exclusive": False,
            **engine,
        }
    )
    perf = PerfModelConfig()
    plan = build_synthetic_closed_loop_plan(workload, cfg)
    client = planned_client(plan, cfg)
    from simcore.router import Router

    return Coordinator(
        (Engine(runtime_config(cfg, perf)),),
        client,
        Router(1),
    ).run()[0]


@pytest.fixture(scope="module")
def result() -> EngineRun:
    return synthetic_run()


class TestInvariants:
    def test_all_requests_complete(self, result: EngineRun) -> None:
        assert [r.id for r in result.records] == list(range(300))
        assert result.cache_snapshot.active_blocks == 0
        assert result.cache_snapshot.reserved_future_blocks == 0
        assert (
            result.cache_snapshot.resident_blocks + result.cache_snapshot.free_blocks
            == result.cache_snapshot.capacity_blocks
        )

    def test_completed_request_intervals_are_ordered(self, result: EngineRun) -> None:
        assert all(
            r.arrival <= r.prefill_start <= r.prefill_end <= r.finish for r in result.records
        )

    def test_fcfs_admission_order(self, result: EngineRun) -> None:
        starts = [r.prefill_start for r in result.records]
        assert starts == sorted(starts)

    def test_meta_window(self, result: EngineRun) -> None:
        assert result.sim_start == min(r.arrival for r in result.records)
        assert result.sim_end == max(r.finish for r in result.records)
        assert result.num_engine_steps > 0


class TestReproducibility:
    def test_same_seed_bit_identical(self, result: EngineRun) -> None:
        rerun = synthetic_run()
        assert [r.model_dump() for r in result.records] == [r.model_dump() for r in rerun.records]
        assert result == rerun
