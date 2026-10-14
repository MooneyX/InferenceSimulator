"""engine 模块单测：手算场景逐时刻核对、同刻顺序、告警、全局不变式与可复现性。

手算场景统一使用 SIMPLE 性能模型：prefill 耗时 = prompt_len ms，decode 每步固定 10 ms。
"""

import pytest

from simcore.coordinator import Coordinator
from simcore.engine import Engine, EngineRun
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
from test_schemas import runtime_config

SIMPLE = PerfModelConfig(prefill_c1=1.0, decode_d1=10.0, decode_d2=0.0)


def req(request_id: int, t: float, prompt_len: int, output_len: int) -> Arrival:
    return Arrival(
        request_id=request_id, sim_time=t, prompt_len=prompt_len,
        output_len=output_len, max_new_tokens=128,
    )


def run(
    arrivals: list[Arrival],
    *,
    max_batch_size: int = 4,
    budget: int = 1000,
    perf: PerfModelConfig = SIMPLE,
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
            arrival.request_id, arrival.request_id, None, arrival.sim_time, None,
            arrival.prompt_len, arrival.output_len, maximum, 0,
        )
    plan = WorkloadPlan(turns, tuple(turns), {})
    return Coordinator(
        (Engine(runtime_config(config, perf)),),
        planned_client(plan, config), Router(1),
    ).run()[0]


def interval(r: RequestRecord) -> tuple[float, float, float, float]:
    return (r.arrival, r.prefill_start, r.prefill_end, r.finish)


class TestImmediateEngine:
    def test_submit_step_and_complete_updates_record(self) -> None:
        config = EngineConfig(max_running_requests=1, max_total_tokens=100, page_size=1)
        request = InferenceRequest(7, 7, None, 5, 1, 1, (), 0)
        engine = Engine(runtime_config(config, SIMPLE))
        assert not engine.has_work
        assert not hasattr(engine, "_heap")
        assert not hasattr(engine, "step")
        engine.submit(request, 3.0)
        assert engine.has_work
        with pytest.raises(ValueError, match="重复"):
            engine.submit(request, 3.0)
        started = engine.start_step(3.0, 0)
        assert started == (5.0, (7,))
        assert not engine.has_work
        assert engine._prefix_cache.page("chunk:7:0") is None
        assert engine.finish_step(8.0) == (7,)
        engine.complete(8.0, 7)
        with pytest.raises(ValueError, match="未待结算"):
            engine.complete(8.0, 7)
        assert interval(engine.finalize().records[0]) == (3.0, 3.0, 8.0, 8.0)

    @pytest.mark.parametrize("maximum", [4, 8])
    def test_direct_submit_uses_actual_output_length_without_registration(
        self, maximum: int
    ) -> None:
        config = EngineConfig(max_running_requests=1, max_total_tokens=100, page_size=1)
        request = InferenceRequest(7, 7, None, 5, maximum, 3, (), 0)
        engine = Engine(runtime_config(config, SIMPLE))
        engine.submit(request, 3.0)
        assert engine.start_step(3.0, 0) == (5.0, ())
        assert engine.finish_step(8.0) == ()
        assert engine.start_step(8.0, 1) == (10.0, ())
        assert engine.finish_step(18.0) == ()
        assert engine.start_step(18.0, 2) == (10.0, (7,))
        assert engine.finish_step(28.0) == (7,)
        engine.complete(28.0, 7)
        result = engine.finalize()
        assert interval(result.records[0]) == (3.0, 3.0, 8.0, 28.0)
        assert result.records[0].output_len == request.output_len == 3
        assert result.num_engine_steps == 3


class TestHandComputed:
    def test_single_request(self) -> None:
        # step0 @0: prefill 5 → 5；step1 @5: decode → 15；step2 @15: decode → 25
        result = run([req(0, 0.0, 5, 3)])
        assert [interval(r) for r in result.records] == [(0.0, 0.0, 5.0, 25.0)]
        assert result.num_engine_steps == 3
        assert (result.sim_start, result.sim_end) == (0.0, 25.0)
        assert result.warnings == ()

    def test_single_token_output_finishes_at_prefill_end(self) -> None:
        result = run([req(0, 3.0, 5, 1)])
        assert interval(result.records[0]) == (3.0, 3.0, 8.0, 8.0)
        assert result.num_engine_steps == 1

    def test_queueing_under_batch_limit(self) -> None:
        # r1 在 t=1 到达，batch 已满，等到 r0 于 15 完成后才准入
        result = run([req(0, 0.0, 5, 2), req(1, 1.0, 5, 2)], max_batch_size=1)
        assert [interval(r) for r in result.records] == [
            (0.0, 0.0, 5.0, 15.0),
            (1.0, 15.0, 20.0, 30.0),
        ]
        assert result.num_engine_steps == 4

    def test_idle_gap_jumps_clock(self) -> None:
        """batch 为空时不压步，时钟直接跳到下一个到达。"""
        result = run([req(0, 0.0, 5, 1), req(1, 100.0, 5, 1)])
        assert [interval(r) for r in result.records] == [
            (0.0, 0.0, 5.0, 5.0),
            (100.0, 100.0, 105.0, 105.0),
        ]
        assert result.num_engine_steps == 2

    def test_fused_step_takes_max(self) -> None:
        # step1 @5: r0 decode(10) 与 r1 prefill(20) 融合 → 20 → 25
        result = run([req(0, 0.0, 5, 3), req(1, 2.0, 20, 2)])
        assert [interval(r) for r in result.records] == [
            (0.0, 0.0, 5.0, 35.0),
            (2.0, 5.0, 25.0, 35.0),
        ]
        assert result.num_engine_steps == 3

    def test_same_time_arrivals_batched_together(self) -> None:
        result = run([req(0, 0.0, 5, 1), req(1, 0.0, 7, 1)])
        # 同一步准入，prefill 相加：5 + 7
        assert [interval(r) for r in result.records] == [
            (0.0, 0.0, 12.0, 12.0),
            (0.0, 0.0, 12.0, 12.0),
        ]
        assert result.num_engine_steps == 1

    def test_arrival_during_step_waits_for_step_end(self) -> None:
        result = run([req(0, 0.0, 50, 1), req(1, 10.0, 5, 1)])
        assert interval(result.records[1]) == (10.0, 50.0, 55.0, 55.0)

    def test_unsorted_input_is_ordered_by_heap(self) -> None:
        ordered = run([req(0, 0.0, 5, 2), req(1, 1.0, 5, 2)], max_batch_size=1)
        shuffled = run([req(1, 1.0, 5, 2), req(0, 0.0, 5, 2)], max_batch_size=1)
        assert ordered.records == shuffled.records


class TestSameTimeOrdering:
    def test_release_visible_to_same_time_step(self) -> None:
        """同刻 RequestDone 先于 EngineStep：r0 于 10 释放的额度，r1 在 10 即可用。

        若顺序颠倒，t=10 的步会看到 KV 仍被占满而无法准入，且 batch 已无 decode 工作。
        """
        result = run([req(0, 0.0, 10, 1), req(1, 0.0, 10, 1)], budget=14)
        assert [interval(r) for r in result.records] == [
            (0.0, 0.0, 10.0, 10.0),
            (0.0, 10.0, 20.0, 20.0),
        ]
        assert result.warnings == ()

    def test_arrival_visible_to_same_time_step(self) -> None:
        """同刻 Arrival 先于 EngineStep：t=5 到达的请求被 t=5 的步准入。"""
        result = run([req(0, 0.0, 5, 2), req(1, 5.0, 3, 1)])
        assert interval(result.records[1]) == (5.0, 5.0, 15.0, 15.0)


class TestKVBehaviour:
    def test_budget_blocks_admission(self) -> None:
        result = run([req(0, 0.0, 8, 2), req(1, 0.0, 5, 1)], budget=11)
        # r1 需等 r0 完成（@18）释放
        assert interval(result.records[1]) == (0.0, 18.0, 23.0, 23.0)
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

    def test_duplicate_request_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="重复"):
            run([req(0, 0.0, 5, 1), req(0, 1.0, 5, 1)])

    def test_zero_cost_model_terminates(self) -> None:
        """全零系数下步长为 0，时钟不前进但 token 仍逐步产出，必须终止。"""
        zero = PerfModelConfig(prefill_c1=0.0, decode_d1=0.0, decode_d2=0.0)
        result = run([req(0, 1.0, 5, 4), req(1, 1.0, 5, 4)], perf=zero)
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
    perf = PerfModelConfig(prefill_c1=0.05, prefill_c2=1e-5, decode_d1=8.0, decode_d2=0.002)
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
        assert all(r.arrival <= r.prefill_start <= r.prefill_end <= r.finish for r in result.records)

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
