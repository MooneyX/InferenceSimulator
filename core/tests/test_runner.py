"""runner 模块单测：端到端串联、元信息一致性、trace 接缝与可复现性。"""

import random
from itertools import pairwise
from unittest.mock import patch

import pytest

import simcore
from simcore import SimConfig, SimResult, simulate
from simcore.analytical import StepCost, step_cost
from simcore.capacity import prepare_runtime
from simcore.metrics import compute_metrics
from simcore.perf_model import DecodeWork, PrefillWork, StepWorkload
from simcore.prefix_hash import stable_derive
from simcore.schemas import (
    AnalyticalSummary,
    EngineConfig,
    FixedLength,
    LognormalLength,
    PerfModelConfig,
    PoissonLength,
    SessionConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)
from simcore.workload import build_synthetic_closed_loop_plan
from test_schemas import metric_inputs, resolved_config


def make_config(seed: int = 0, num_requests: int = 200, budget: int = 16_000) -> SimConfig:
    return resolved_config(
        num_instances=1,
        workload=WorkloadConfig(
            session_rate=30.0,
            seed=seed,
            synthetic=SyntheticWorkloadConfig(
                num_requests=num_requests,
                prompt_len=LognormalLength(mean=256.0, sigma=0.8),
                turn_increment_len=FixedLength(value=8),
                output_len=PoissonLength(mean=40.0),
            ),
        ),
        engine=EngineConfig(max_running_requests=16, max_total_tokens=budget, page_size=1),
        perf_model=PerfModelConfig(),
    )


@pytest.mark.parametrize("instances", [1, 2])
def test_request_progress_is_throttled_and_matches_result(instances: int) -> None:
    config = make_config(num_requests=8).model_copy(update={"num_instances": instances})
    snapshots = []
    clock = 0.0

    def tick() -> float:
        nonlocal clock
        clock += 0.125
        return clock

    with patch("simcore.coordinator.monotonic", side_effect=tick):
        result = simulate(config, on_progress=lambda item: snapshots.append((clock, item)))
    assert len(snapshots) > 2
    assert snapshots[0][1].model_dump() == {
        "total": 8, "completed": 0, "aborted": 0, "skipped": 0,
    }
    assert all(b[0] - a[0] >= 1 for a, b in pairwise(snapshots[:-1]))
    counts = [item.completed + item.aborted + item.skipped for _, item in snapshots]
    assert counts == sorted(counts)
    assert all(item.total == 8 for _, item in snapshots)
    assert snapshots[-1][1].model_dump() == {
        "total": result.meta.num_requests, "completed": result.meta.num_completed,
        "aborted": result.meta.num_aborted, "skipped": result.meta.num_skipped,
    }
    baseline = simulate(config)
    assert result.model_dump(exclude={"meta": {"wall_clock_ms"}}) == baseline.model_dump(
        exclude={"meta": {"wall_clock_ms"}},
    )


def test_fast_simulation_reports_initial_and_final_progress() -> None:
    snapshots = []
    with patch("simcore.coordinator.monotonic", return_value=0):
        simulate(make_config(num_requests=2), on_progress=snapshots.append)
    assert len(snapshots) == 2
    assert snapshots[0].completed == 0
    assert snapshots[-1].completed == 2


@pytest.fixture(scope="module")
def result() -> SimResult:
    result = simulate(make_config())
    assert result.analysis is not None
    assert result.analysis.steps == result.meta.num_engine_steps
    assert result.analysis.flops > 0 and result.analysis.hbm_bytes > 0
    assert result.analysis.device_ms > 0
    assert result.resolution == prepare_runtime(make_config()).resolution
    return result


def expected_open_loop_arrivals(seed: int, qps: float, count: int) -> list[float]:
    rng = random.Random(stable_derive(seed, "open-loop-arrival"))
    times: list[float] = []
    clock = 0.0
    for _ in range(count):
        clock += rng.expovariate(qps / 1000)
        times.append(clock)
    return times


def test_open_loop_poisson_arrivals_reuse_completed_context() -> None:
    config = resolved_config(
        num_instances=1,
        workload=WorkloadConfig(
            arrival_mode="open_loop",
            target_qps=10,
            seed=42,
            synthetic=SyntheticWorkloadConfig(
                num_requests=5,
                prompt_len=FixedLength(value=32),
                turn_increment_len=FixedLength(value=5),
                output_len=FixedLength(value=2),
            ),
        ),
        engine=EngineConfig(max_running_requests=4, max_total_tokens=1024, page_size=16),
        perf_model=PerfModelConfig(),
    )
    result = simulate(config)
    assert result.meta.num_completed == 5
    assert result.meta.num_skipped == 0
    arrivals = expected_open_loop_arrivals(42, 10, 5)
    assert [record.arrival for record in result.requests] == pytest.approx(arrivals)
    assert [record.prompt_len for record in result.requests] == [32, 39, 46, 53, 60]
    assert [record.parent_request_id for record in result.requests] == [None, 0, 1, 2, 3]
    assert [record.ideal_reuse_tokens for record in result.requests] == [0, 32, 32, 32, 48]
    assert result.metrics.arrival_qps == pytest.approx(4000 / (arrivals[-1] - arrivals[0]))
    assert result.metrics.arrival_qps != pytest.approx(10)
    assert result.metrics.num_sessions == 1
    assert result.metrics.child_ideal_prefix_hit_rate == pytest.approx(144 / (39 + 46 + 53 + 60))
    assert result.metrics.mean_prompt_tokens == pytest.approx(46)
    repeated = simulate(config)
    assert repeated.requests == result.requests


def test_open_loop_completion_at_tick_is_ready() -> None:
    config = resolved_config(
        num_instances=1,
        workload=WorkloadConfig(
            arrival_mode="open_loop",
            target_qps=31.25,
            seed=42,
            synthetic=SyntheticWorkloadConfig(
                num_requests=2,
                prompt_len=FixedLength(value=32),
                turn_increment_len=FixedLength(value=5),
                output_len=FixedLength(value=1),
            ),
        ),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=1024, page_size=16),
        perf_model=PerfModelConfig(),
    )
    with (
        patch("simcore.client.random.Random.expovariate", side_effect=[7.0, 32.0]),
        patch("simcore.engine.step_cost", return_value=StepCost(32, 0, 0, 0, 0)),
    ):
        result = simulate(config)
    assert [record.arrival for record in result.requests] == [7.0, 39.0]
    assert result.requests[0].finish == 39.0
    assert result.requests[1].parent_request_id == 0
    assert result.metrics.num_sessions == 1


def test_open_loop_creates_new_sessions_when_no_child_ready() -> None:
    config = resolved_config(
        num_instances=1,
        workload=WorkloadConfig(
            arrival_mode="open_loop",
            target_qps=10,
            seed=42,
            synthetic=SyntheticWorkloadConfig(
                num_requests=4,
                prompt_len=FixedLength(value=32),
                turn_increment_len=FixedLength(value=5),
                output_len=FixedLength(value=2),
            ),
        ),
        engine=EngineConfig(max_running_requests=4, max_total_tokens=1024, page_size=16),
        perf_model=PerfModelConfig(),
    )
    config = config.model_copy(update={
        "inference": config.inference.model_copy(update={
            "hardware": config.inference.hardware.model_copy(update={
                "effective_fp16_tflops": 1e-6,
                "effective_hbm_gb_s": 1e-6,
            }),
        }),
    })
    result = simulate(config)
    assert result.meta.num_completed + result.meta.num_aborted == 4
    arrivals = expected_open_loop_arrivals(42, 10, 4)
    assert [record.arrival for record in result.requests] == pytest.approx(arrivals)
    assert result.metrics.arrival_qps == pytest.approx(3000 / (arrivals[-1] - arrivals[0]))
    assert result.metrics.num_sessions == 4
    assert result.metrics.root_request_ratio == 1
    assert result.metrics.constructed_ideal_prefix_hit_rate == 0


def test_open_loop_capacity_failure_still_consumes_tick() -> None:
    config = resolved_config(
        num_instances=1,
        workload=WorkloadConfig(
            arrival_mode="open_loop",
            target_qps=10,
            seed=1,
            synthetic=SyntheticWorkloadConfig(
                num_requests=3,
                prompt_len=FixedLength(value=32),
                turn_increment_len=FixedLength(value=40),
                output_len=FixedLength(value=2),
            ),
        ),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=64, page_size=16),
        perf_model=PerfModelConfig(),
    )
    result = simulate(config)
    assert result.meta.num_completed == 2
    assert result.meta.num_aborted == 1
    assert result.meta.num_skipped == 0
    arrivals = expected_open_loop_arrivals(1, 10, 3)
    assert [record.arrival for record in result.requests] == pytest.approx(
        [arrivals[0], arrivals[2]]
    )
    assert result.aborted_requests[0].arrival == pytest.approx(arrivals[1])
    assert result.metrics.num_sessions == 2
    assert result.metrics.arrival_qps == pytest.approx(2000 / (arrivals[2] - arrivals[0]))


def test_open_loop_single_arrival_has_no_observed_qps() -> None:
    config = resolved_config(
        num_instances=1,
        workload=WorkloadConfig(
            arrival_mode="open_loop",
            target_qps=10,
            seed=1,
            synthetic=SyntheticWorkloadConfig(
                num_requests=1,
                prompt_len=FixedLength(value=32),
                turn_increment_len=FixedLength(value=5),
                output_len=FixedLength(value=1),
            ),
        ),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=128, page_size=16),
        perf_model=PerfModelConfig(),
    )
    result = simulate(config)
    assert result.requests[0].arrival == pytest.approx(expected_open_loop_arrivals(1, 10, 1)[0])
    assert result.metrics.arrival_qps is None


def test_open_loop_reports_arrival_rate_even_if_all_requests_abort() -> None:
    config = resolved_config(
        num_instances=1,
        workload=WorkloadConfig(
            arrival_mode="open_loop",
            target_qps=10,
            seed=1,
            synthetic=SyntheticWorkloadConfig(
                num_requests=2,
                prompt_len=FixedLength(value=128),
                turn_increment_len=FixedLength(value=5),
                output_len=FixedLength(value=2),
            ),
        ),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=64, page_size=16),
        perf_model=PerfModelConfig(),
    )
    result = simulate(config)
    arrivals = expected_open_loop_arrivals(1, 10, 2)
    assert result.meta.num_completed == result.meta.num_engine_steps == 0
    assert result.analysis == AnalyticalSummary()
    assert [request.arrival for request in result.aborted_requests] == pytest.approx(arrivals)
    assert result.metrics.arrival_qps == pytest.approx(1000 / (arrivals[1] - arrivals[0]))
    assert result.metrics.throughput is None


class TestEndToEnd:
    def test_package_exports_simulate(self) -> None:
        assert simcore.simulate is simulate
        assert not hasattr(simcore, "simulate_resolved")
        assert not hasattr(simcore, "simulate_analytical")
        for removed in ("TraceReplay", "ResolvedRequest", "ResolvedConfig", "ResolvedRun"):
            assert not hasattr(simcore, removed)

    def test_all_requests_recorded(self, result: SimResult) -> None:
        assert [r.id for r in result.requests] == list(range(200))

    def test_meta_consistent(self, result: SimResult) -> None:
        meta = result.meta
        assert meta.config == make_config()
        assert meta.num_requests == 200
        assert meta.num_engine_steps > 0
        assert meta.sim_start == min(r.arrival for r in result.requests)
        assert meta.sim_end == max(r.finish for r in result.requests)

    def test_wall_clock_measures_kernel_run(self) -> None:
        with patch("simcore.runner.perf_counter", side_effect=[10.0, 10.125]) as clock:
            result = simulate(make_config(num_requests=1))
        assert clock.call_count == 2
        assert result.meta.wall_clock_ms == pytest.approx(125.0)
        assert result.metrics.makespan == pytest.approx(result.meta.sim_end - result.meta.sim_start)

    def test_metrics_consistent_with_records(self, result: SimResult) -> None:
        baseline = compute_metrics(
            result.requests,
            **(
                metric_inputs()
                | {
                    "page_size": result.meta.config.inference.page_size,
                    "workload_mode": result.meta.config.workload.mode,
                }
            ),
        )
        for name in ("ttft_mean", "ttft_p50", "ttft_p99", "tpot_mean", "throughput", "makespan"):
            assert getattr(result.metrics, name) == getattr(baseline, name)
        assert result.metrics.total_input_tokens == sum(r.prompt_len for r in result.requests)
        assert result.metrics.actual_hit_tokens == sum(r.actual_hit_tokens for r in result.requests)
        assert (
            result.metrics.l1_capacity_blocks
            == prepare_runtime(result.meta.config).engine.capacity_pages
        )
        assert result.metrics.makespan == pytest.approx(result.meta.sim_end - result.meta.sim_start)

    def test_ttft_percentiles_ordered(self, result: SimResult) -> None:
        m = result.metrics
        assert m.ttft_p50 is not None and m.ttft_p99 is not None
        assert 0.0 < m.ttft_p50 <= m.ttft_p99
        assert m.tpot_mean is not None and m.tpot_mean > 0
        assert m.throughput is not None and m.throughput > 0

    def test_prefix_metrics_are_weighted_and_capacity_conserved(self) -> None:
        config = resolved_config(
            num_instances=1,
            workload=WorkloadConfig(
                session_rate=10,
                seed=2,
                synthetic=SyntheticWorkloadConfig(
                    num_requests=2,
                    prompt_len=FixedLength(value=16),
                    turn_increment_len=FixedLength(value=1),
                    output_len=FixedLength(value=8),
                    session=SessionConfig(num_sessions=1),
                ),
            ),
            engine=EngineConfig(
                max_running_requests=2,
                max_total_tokens=80,
                page_size=2,
            ),
            perf_model=PerfModelConfig(),
        )
        result = simulate(config)
        m = result.metrics
        assert m.total_input_tokens == 41
        assert m.constructed_ideal_prefix_hit_rate == pytest.approx(22 / 41)
        assert m.ideal_hit_tokens == m.actual_hit_tokens == m.l1_hit_tokens == 22
        assert m.l2_hit_tokens == m.l3_hit_tokens == 0
        assert m.restore_bytes == m.restore_time_ms == m.promotions == m.demotions == 0
        assert m.actual_prefix_hit_rate == pytest.approx(22 / 41)
        assert m.actual_prefill_tokens == 19 and m.saved_prefill_tokens == 22
        assert m.requests_with_hit_ratio == 0.5
        assert m.l1_capacity_blocks == 40
        assert m.l1_resident_blocks + m.l1_free_blocks + m.l1_reserved_future_blocks == 40
        assert m.l1_active_blocks == 0
        assert result.requests[1].parent_request_id == result.requests[0].id
        assert result.requests[1].prompt_len == 25
        assert result.requests[1].actual_prefill_tokens == 3

    def test_constructed_ideal_hit_rate_is_plan_result(self) -> None:
        config = resolved_config(
            num_instances=1,
            workload=WorkloadConfig(
                session_rate=10,
                seed=2,
                synthetic=SyntheticWorkloadConfig(
                    num_requests=2,
                    prompt_len=FixedLength(value=16),
                    turn_increment_len=FixedLength(value=1),
                    output_len=FixedLength(value=8),
                    session=SessionConfig(num_sessions=1),
                ),
            ),
            engine=EngineConfig(max_running_requests=2, max_total_tokens=80, page_size=2),
            perf_model=PerfModelConfig(),
        )
        synthetic = config.workload.synthetic
        assert synthetic is not None
        plan = build_synthetic_closed_loop_plan(config.workload, prepare_runtime(config).engine)
        result = simulate(config)
        metrics = result.metrics
        assert metrics.constructed_ideal_prefix_hit_rate == pytest.approx(22 / 41)
        empty = compute_metrics(
            [],
            **(
                metric_inputs()
                | {
                    "plan": plan,
                    "page_size": config.inference.page_size,
                }
            ),
        )
        assert empty.constructed_ideal_prefix_hit_rate == metrics.constructed_ideal_prefix_hit_rate

    def test_infeasible_request_reports_abort_without_ending_simulation(self) -> None:
        cfg = resolved_config(
            num_instances=1,
            workload=WorkloadConfig(
                session_rate=10,
                synthetic=SyntheticWorkloadConfig(
                    num_requests=2,
                    prompt_len=FixedLength(value=193),
                    turn_increment_len=FixedLength(value=1),
                    output_len=FixedLength(value=1),
                ),
            ),
            engine=EngineConfig(max_running_requests=2, max_total_tokens=192, page_size=64),
            perf_model=PerfModelConfig(),
        )
        result = simulate(cfg)
        assert (result.meta.num_requests, result.meta.num_aborted) == (2, 2)
        assert result.requests == []
        assert all(item.reason == "input_exceeds_kv_capacity" for item in result.aborted_requests)
        assert result.metrics.ttft_mean is None

    def test_enabled_tiers_propagate_capacity_and_zero_transfer(self) -> None:
        cfg = resolved_config(
            num_instances=1,
            workload=WorkloadConfig(
                session_rate=1,
                synthetic=SyntheticWorkloadConfig(
                    num_requests=1,
                    prompt_len=FixedLength(value=3),
                    turn_increment_len=FixedLength(value=1),
                    output_len=FixedLength(value=1),
                ),
            ),
            engine=EngineConfig(
                max_running_requests=1,
                max_total_tokens=12,
                page_size=4,
                l2_capacity_tokens=4,
                l3_capacity_tokens=8,
            ),
            perf_model=PerfModelConfig(
                kv_bytes_per_token=16,
                l2_to_l1_bandwidth_bytes_per_ms=64,
                l3_to_l2_bandwidth_bytes_per_ms=32,
            ),
        )
        result = simulate(cfg)
        assert result.meta.num_completed == 1
        assert (result.metrics.l2_capacity_blocks, result.metrics.l3_capacity_blocks) == (1, 2)
        assert (result.metrics.l2_free_blocks, result.metrics.l3_free_blocks) == (1, 2)
        assert result.metrics.restore_bytes == result.metrics.restore_time_ms == 0
        assert result.metrics.promotions == result.metrics.demotions == 0
        assert SimResult.model_validate_json(result.model_dump_json()) == result

    def test_closed_loop_skipped_turns_do_not_count_toward_request_qps(self) -> None:
        cfg = resolved_config(
            num_instances=1,
            workload=WorkloadConfig(
                session_rate=2,
                synthetic=SyntheticWorkloadConfig(
                    num_requests=3,
                    prompt_len=FixedLength(value=128),
                    turn_increment_len=FixedLength(value=1),
                    output_len=FixedLength(value=1),
                    session=SessionConfig(num_sessions=1),
                ),
            ),
            engine=EngineConfig(max_running_requests=1, max_total_tokens=64, page_size=16),
            perf_model=PerfModelConfig(),
        )
        snapshots = []
        result = simulate(cfg, on_progress=snapshots.append)
        assert (result.meta.num_completed, result.meta.num_aborted, result.meta.num_skipped) == (
            0,
            1,
            2,
        )
        assert snapshots[-1].model_dump() == {
            "total": 3, "completed": 0, "aborted": 1, "skipped": 2,
        }
        assert result.meta.num_engine_steps == 0
        assert result.analysis == AnalyticalSummary()
        assert result.metrics.arrival_qps is None

    def test_json_roundtrip(self, result: SimResult) -> None:
        """服务层将直接序列化该对象，必须 JSON 往返无损。"""
        assert SimResult.model_validate_json(result.model_dump_json()) == result


class TestTraceSeam:
    def test_trace_workload_requires_explicit_plan(self) -> None:
        workload = WorkloadConfig.model_validate(
            {
                "mode": "trace",
                "arrival_mode": "open_loop",
                "session_rate": 1,
                "trace": {},
            }
        )
        cfg = make_config().model_copy(update={"workload": workload})
        with pytest.raises(ValueError, match=r"plan|计划"):
            simulate(cfg)


class TestReproducibility:
    def test_same_seed_json_bit_identical(self, result: SimResult) -> None:
        """同种子仿真结果（除运行耗时）逐字节一致。"""
        exclude = {"meta": {"wall_clock_ms"}}
        repeated = simulate(make_config())
        assert result.model_dump_json(exclude=exclude) == repeated.model_dump_json(exclude=exclude)

    def test_different_seed_differs(self) -> None:
        first = simulate(make_config(seed=1))
        second = simulate(make_config(seed=2))
        assert first.requests != second.requests


class TestSmallCases:
    @pytest.mark.parametrize("think_time_ms", [0.0, 30.0])
    def test_followup_arrives_only_after_parent_completion(self, think_time_ms: float) -> None:
        cfg = resolved_config(
            num_instances=1,
            workload=WorkloadConfig(
                session_rate=1.0,
                seed=7,
                synthetic=SyntheticWorkloadConfig(
                    num_requests=3,
                    prompt_len=FixedLength(value=10),
                    turn_increment_len=FixedLength(value=1),
                    output_len=FixedLength(value=2),
                    session=SessionConfig(num_sessions=1, think_time_ms=think_time_ms),
                ),
            ),
            engine=EngineConfig(max_running_requests=2, max_total_tokens=100, page_size=2),
            perf_model=PerfModelConfig(),
        )
        result = simulate(cfg)
        synthetic = cfg.workload.synthetic
        assert synthetic is not None
        plan = build_synthetic_closed_loop_plan(cfg.workload, prepare_runtime(cfg).engine)
        assert len(result.requests) == 3
        assert result.metrics.arrival_qps == pytest.approx(
            2000 / (result.requests[-1].arrival - result.requests[0].arrival)
        )
        assert result.metrics.arrival_qps != pytest.approx(1.0)
        for parent, child in zip(result.requests, result.requests[1:], strict=False):
            sampled_delay = plan.turns[child.id].think_time_ms
            assert sampled_delay is not None
            assert child.arrival == parent.finish + sampled_delay
            if think_time_ms == 0:
                assert child.arrival == parent.finish
            assert child.prefill_start >= child.arrival
        assert result.meta.num_engine_steps == 6

    def test_single_request_fixed_lengths(self) -> None:
        cfg = resolved_config(
            num_instances=1,
            workload=WorkloadConfig(
                session_rate=1.0,
                synthetic=SyntheticWorkloadConfig(
                    num_requests=1,
                    prompt_len=FixedLength(value=5),
                    turn_increment_len=FixedLength(value=1),
                    output_len=FixedLength(value=3),
                ),
            ),
            engine=EngineConfig(max_running_requests=1, max_total_tokens=100, page_size=1),
            perf_model=PerfModelConfig(),
        )
        r = simulate(cfg)
        record = r.requests[0]
        inference = cfg.inference
        prefill = step_cost(
            StepWorkload("PREFILL", (PrefillWork(0, 0, 5, 5),), (), 0, 1),
            inference.model, inference.hardware,
        ).duration_ms
        decode = sum(
            step_cost(
                StepWorkload("DECODE", (), (DecodeWork(0, context),), 0, 1),
                inference.model, inference.hardware,
            ).duration_ms
            for context in (6, 7)
        )
        assert record.prefill_end - record.arrival == pytest.approx(prefill)
        assert record.finish - record.prefill_end == pytest.approx(decode)
        assert r.metrics.ttft_mean == pytest.approx(prefill)
        assert r.metrics.tpot_mean == pytest.approx(decode / 2)
        assert r.meta.num_engine_steps == 3
