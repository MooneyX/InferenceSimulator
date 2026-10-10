"""engine 模块单测：手算场景逐时刻核对、同刻顺序、告警、全局不变式与可复现性。

手算场景统一使用 SIMPLE 性能模型：prefill 耗时 = prompt_len ms，decode 每步固定 10 ms。
"""

import itertools

import pytest

from simcore.client import ClientDriver
from simcore.coordinator import Coordinator
from simcore.engine import Engine, EngineRun, run_engine
from simcore.events import StepTrace
from simcore.schemas import (
    Arrival,
    EngineConfig,
    LognormalLength,
    PerfModelConfig,
    PoissonLength,
    RequestRecord,
    SyntheticWorkloadConfig,
)
from simcore.workload import build_plan, plan_from_arrivals

SIMPLE = PerfModelConfig(prefill_c1=1.0, decode_d1=10.0, decode_d2=0.0)


def req(request_id: int, t: float, prompt_len: int, output_len: int) -> Arrival:
    return Arrival(request_id=request_id, sim_time=t, prompt_len=prompt_len, output_len=output_len)


def run(
    arrivals: list[Arrival],
    *,
    max_batch_size: int = 4,
    budget: int = 1000,
    perf: PerfModelConfig = SIMPLE,
) -> EngineRun:
    return run_engine(
        EngineConfig(
            max_running_requests=max_batch_size,
            max_total_tokens=budget,
            page_size=1,
            pd_mutually_exclusive=False,
        ),
        perf,
        arrivals,
    )


def interval(r: RequestRecord) -> tuple[float, float, float, float]:
    return (r.arrival, r.prefill_start, r.prefill_end, r.finish)


def steps(result: EngineRun) -> list[StepTrace]:
    return list(result.trace.steps())


class TestImmediateEngine:
    def test_submit_step_and_completion_return_finish_time(self) -> None:
        config = EngineConfig(max_running_requests=1, max_total_tokens=100, page_size=1)
        plan = plan_from_arrivals((req(7, 3.0, 5, 1),), config)
        client = ClientDriver(plan, config)
        request = client.on_wakeup(client.initial_wakeups()[0])
        engine = Engine(config, SIMPLE, output_targets={7: plan.turns[7].output_len})
        assert not engine.has_work
        assert not hasattr(engine, "_heap")
        engine.submit(request, 3.0)
        assert engine.has_work
        with pytest.raises(ValueError, match="重复"):
            engine.submit(request, 3.0)
        duration, finished = engine.step(3.0, 0)
        assert (duration, finished) == (5.0, (7,))
        assert not engine.has_work
        completion = engine.complete(8.0, 7)
        assert completion.request_id == request.request_id
        assert completion.finish_time_ms == 8.0
        with pytest.raises(ValueError, match="未待结算"):
            engine.complete(8.0, 7)
        assert interval(engine.finalize().records[0]) == (3.0, 3.0, 8.0, 8.0)


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
        assert [s.time for s in steps(result)] == [0.0, 100.0]

    def test_fused_step_takes_max(self) -> None:
        # step1 @5: r0 decode(10) 与 r1 prefill(20) 融合 → 20 → 25
        result = run([req(0, 0.0, 5, 3), req(1, 2.0, 20, 2)])
        assert [interval(r) for r in result.records] == [
            (0.0, 0.0, 5.0, 35.0),
            (2.0, 5.0, 25.0, 35.0),
        ]
        assert [s.duration for s in steps(result)] == [5.0, 20.0, 10.0]

    def test_same_time_arrivals_batched_together(self) -> None:
        result = run([req(0, 0.0, 5, 1), req(1, 0.0, 7, 1)])
        # 同一步准入，prefill 相加：5 + 7
        assert [interval(r) for r in result.records] == [
            (0.0, 0.0, 12.0, 12.0),
            (0.0, 0.0, 12.0, 12.0),
        ]
        assert steps(result)[0].admitted == (0, 1)

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
        assert steps(result)[1].admitted == (1,)


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
        assert all(step.kv_overflow == 0 and not step.forced for step in steps(result))

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
        assert len(result.trace) == 0

    def test_duplicate_request_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="重复"):
            run([req(0, 0.0, 5, 1), req(0, 1.0, 5, 1)])

    def test_zero_cost_model_terminates(self) -> None:
        """全零系数下步长为 0，时钟不前进但 token 仍逐步产出，必须终止。"""
        zero = PerfModelConfig(prefill_c1=0.0, decode_d1=0.0, decode_d2=0.0)
        result = run([req(0, 1.0, 5, 4), req(1, 1.0, 5, 4)], perf=zero)
        assert all(interval(r) == (1.0, 1.0, 1.0, 1.0) for r in result.records)
        assert result.num_engine_steps == 4


def synthetic_run(seed: int = 7, *, record_trace: bool = True, **engine: int) -> EngineRun:
    workload = SyntheticWorkloadConfig(
        num_requests=300,
        session_rate=40.0,
        prompt_len=LognormalLength(mean=200.0, sigma=1.0),
        output_len=PoissonLength(mean=30.0),
        seed=seed,
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
    plan = build_plan(workload, cfg)
    client = ClientDriver(plan, cfg)
    return Coordinator(
        Engine(
            cfg,
            perf,
            output_targets={key: turn.output_len for key, turn in plan.turns.items()},
            record_trace=record_trace,
        ),
        client,
    ).run()


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

    def test_batch_size_never_exceeded(self, result: EngineRun) -> None:
        assert all(s.num_decoding + len(s.admitted) <= 16 for s in steps(result))

    def test_each_request_spans_output_len_steps(self, result: EngineRun) -> None:
        """每请求恰好参与 output_len 个连续步：1 个准入步 + (output_len-1) 个 decode 步。"""
        step_list = steps(result)
        for r in result.records:
            spanned = [s for s in step_list if r.prefill_start <= s.time < r.finish]
            assert len(spanned) == r.output_len

    def test_fcfs_admission_order(self, result: EngineRun) -> None:
        starts = [r.prefill_start for r in result.records]
        assert starts == sorted(starts)

    def test_steps_are_contiguous_while_busy(self, result: EngineRun) -> None:
        """忙碌期内步首尾相接，只在空闲时出现跳跃，且跳跃终点是某个到达时刻。"""
        arrival_times = {r.arrival for r in result.records}
        step_list = steps(result)
        for prev, cur in itertools.pairwise(step_list):
            assert cur.time >= prev.end
            if cur.time > prev.end:
                assert cur.time in arrival_times

    def test_trace_time_monotonic(self, result: EngineRun) -> None:
        times = [record.time for record in result.trace]
        assert times == sorted(times)

    def test_meta_window(self, result: EngineRun) -> None:
        assert result.sim_start == min(r.arrival for r in result.records)
        assert result.sim_end == max(r.finish for r in result.records)
        assert result.num_engine_steps == len(steps(result))


class TestReproducibility:
    def test_same_seed_bit_identical(self, result: EngineRun) -> None:
        rerun = synthetic_run()
        assert [r.model_dump() for r in result.records] == [r.model_dump() for r in rerun.records]
        assert result.warnings == rerun.warnings and list(result.trace) == list(rerun.trace)

    def test_trace_toggle_does_not_change_results(self, result: EngineRun) -> None:
        without = synthetic_run(record_trace=False)
        assert result.records == without.records
        assert result.warnings == without.warnings
        assert len(without.trace) == 0 and len(result.trace) > 0
