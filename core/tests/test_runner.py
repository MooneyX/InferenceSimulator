"""runner 模块单测：端到端串联、元信息一致性、trace 接缝与可复现性。"""

import random
from unittest.mock import patch

import pytest

import simcore
from simcore import SimConfig, SimResult, simulate
from simcore.metrics import compute_metrics
from simcore.prefix_hash import stable_derive
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    LognormalLength,
    PerfModelConfig,
    PoissonLength,
    SessionConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)
from simcore.workload import build_plan


def make_config(seed: int = 0, num_requests: int = 200, budget: int = 16_000) -> SimConfig:
    return SimConfig(
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=num_requests,
                session_rate=30.0,
                prompt_len=LognormalLength(mean=256.0, sigma=0.8),
                output_len=PoissonLength(mean=40.0),
                seed=seed,
            )
        ),
        engine=EngineConfig(max_running_requests=16, max_total_tokens=budget, page_size=1),
        perf_model=PerfModelConfig(
            prefill_c1=0.05, prefill_c2=1e-5, decode_d1=8.0, decode_d2=0.002
        ),
    )


@pytest.fixture(scope="module")
def result() -> SimResult:
    return simulate(make_config())


def expected_open_loop_arrivals(seed: int, qps: float, count: int) -> list[float]:
    rng = random.Random(stable_derive(seed, "open-loop-arrival"))
    times: list[float] = []
    clock = 0.0
    for _ in range(count):
        clock += rng.expovariate(qps / 1000)
        times.append(clock)
    return times


def test_open_loop_poisson_arrivals_reuse_completed_context() -> None:
    config = SimConfig(
        workload=WorkloadConfig(synthetic=SyntheticWorkloadConfig(
            arrival_mode="open_loop", num_requests=5, arrival_rate=10,
            prompt_len=FixedLength(value=32), turn_increment_len=FixedLength(value=5),
            output_len=FixedLength(value=2), seed=42,
        )),
        engine=EngineConfig(max_running_requests=4, max_total_tokens=1024, page_size=16),
        perf_model=PerfModelConfig(prefill_c1=0.001, prefill_c2=0, decode_d1=0.001, decode_d2=0),
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
    assert result.metrics.target_ideal_prefix_hit_rate is None
    assert simulate(config).requests == result.requests


def test_open_loop_completion_at_tick_is_ready() -> None:
    config = SimConfig(
        workload=WorkloadConfig(synthetic=SyntheticWorkloadConfig(
            arrival_mode="open_loop", num_requests=2, arrival_rate=31.25,
            prompt_len=FixedLength(value=32), turn_increment_len=FixedLength(value=5),
            output_len=FixedLength(value=1), seed=42,
        )),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=1024, page_size=16),
        perf_model=PerfModelConfig(prefill_c1=1, prefill_c2=0, decode_d1=0, decode_d2=0),
    )
    with patch("simcore.coordinator.random.Random.expovariate", side_effect=[7.0, 32.0]):
        result = simulate(config)
    assert [record.arrival for record in result.requests] == [7.0, 39.0]
    assert result.requests[0].finish == 39.0
    assert result.requests[1].parent_request_id == 0
    assert result.metrics.num_sessions == 1


def test_open_loop_creates_new_sessions_when_no_child_ready() -> None:
    config = SimConfig(
        workload=WorkloadConfig(synthetic=SyntheticWorkloadConfig(
            arrival_mode="open_loop", num_requests=4, arrival_rate=10,
            prompt_len=FixedLength(value=32), turn_increment_len=FixedLength(value=5),
            output_len=FixedLength(value=2), seed=42,
        )),
        engine=EngineConfig(max_running_requests=4, max_total_tokens=1024, page_size=16),
        perf_model=PerfModelConfig(prefill_c1=10, prefill_c2=0, decode_d1=10, decode_d2=0),
    )
    result = simulate(config)
    assert result.meta.num_completed + result.meta.num_aborted == 4
    arrivals = expected_open_loop_arrivals(42, 10, 4)
    assert [record.arrival for record in result.requests] == pytest.approx(arrivals)
    assert result.metrics.arrival_qps == pytest.approx(3000 / (arrivals[-1] - arrivals[0]))
    assert result.metrics.num_sessions == 4
    assert result.metrics.root_request_ratio == 1
    assert result.metrics.constructed_ideal_prefix_hit_rate == 0


def test_open_loop_capacity_failure_still_consumes_tick() -> None:
    config = SimConfig(
        workload=WorkloadConfig(synthetic=SyntheticWorkloadConfig(
            arrival_mode="open_loop", num_requests=3, arrival_rate=10,
            prompt_len=FixedLength(value=32), turn_increment_len=FixedLength(value=40),
            output_len=FixedLength(value=2), seed=1,
        )),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=64, page_size=16),
        perf_model=PerfModelConfig(prefill_c1=0, prefill_c2=0, decode_d1=0, decode_d2=0),
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
    config = SimConfig(
        workload=WorkloadConfig(synthetic=SyntheticWorkloadConfig(
            arrival_mode="open_loop", num_requests=1, arrival_rate=10,
            prompt_len=FixedLength(value=32), turn_increment_len=FixedLength(value=5),
            output_len=FixedLength(value=1), seed=1,
        )),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=128, page_size=16),
        perf_model=PerfModelConfig(prefill_c1=0, prefill_c2=0, decode_d1=0, decode_d2=0),
    )
    result = simulate(config)
    assert result.requests[0].arrival == pytest.approx(expected_open_loop_arrivals(1, 10, 1)[0])
    assert result.metrics.arrival_qps is None


def test_open_loop_reports_arrival_rate_even_if_all_requests_abort() -> None:
    config = SimConfig(
        workload=WorkloadConfig(synthetic=SyntheticWorkloadConfig(
            arrival_mode="open_loop", num_requests=2, arrival_rate=10,
            prompt_len=FixedLength(value=128), turn_increment_len=FixedLength(value=5),
            output_len=FixedLength(value=2), seed=1,
        )),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=64, page_size=16),
        perf_model=PerfModelConfig(prefill_c1=0, prefill_c2=0, decode_d1=0, decode_d2=0),
    )
    result = simulate(config)
    arrivals = expected_open_loop_arrivals(1, 10, 2)
    assert result.meta.num_completed == 0
    assert [request.arrival for request in result.aborted_requests] == pytest.approx(arrivals)
    assert result.metrics.arrival_qps == pytest.approx(1000 / (arrivals[1] - arrivals[0]))
    assert result.metrics.throughput is None


class TestEndToEnd:
    def test_package_exports_simulate(self) -> None:
        assert simcore.simulate is simulate

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
        baseline = compute_metrics(result.requests)
        for name in ("ttft_mean", "ttft_p50", "ttft_p99", "tpot_mean", "throughput", "makespan"):
            assert getattr(result.metrics, name) == getattr(baseline, name)
        assert result.metrics.total_input_tokens == sum(r.prompt_len for r in result.requests)
        assert result.metrics.actual_hit_tokens == sum(r.actual_hit_tokens for r in result.requests)
        assert result.metrics.l1_capacity_blocks == result.meta.config.engine.capacity_pages
        assert result.metrics.makespan == pytest.approx(result.meta.sim_end - result.meta.sim_start)

    def test_ttft_percentiles_ordered(self, result: SimResult) -> None:
        m = result.metrics
        assert m.ttft_p50 is not None and m.ttft_p99 is not None
        assert 0.0 < m.ttft_p50 <= m.ttft_p99
        assert m.tpot_mean is not None and m.tpot_mean > 0
        assert m.throughput is not None and m.throughput > 0

    def test_prefix_metrics_are_weighted_and_capacity_conserved(self) -> None:
        config = SimConfig(
            workload=WorkloadConfig(
                synthetic=SyntheticWorkloadConfig(
                    num_requests=2,
                    session_rate=10,
                    prompt_len=FixedLength(value=16),
                    output_len=FixedLength(value=8),
                    seed=2,
                    session=SessionConfig(num_sessions=1, target_ideal_prefix_hit_rate=0.25),
                )
            ),
            engine=EngineConfig(
                max_running_requests=2,
                max_total_tokens=80,
                page_size=2,
            ),
            perf_model=PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=0),
        )
        result = simulate(config)
        m = result.metrics
        assert m.total_input_tokens == 32
        assert m.target_ideal_prefix_hit_rate == m.constructed_ideal_prefix_hit_rate == 0.25
        assert m.ideal_hit_tokens == m.actual_hit_tokens == m.l1_hit_tokens == 8
        assert m.l2_hit_tokens == m.l3_hit_tokens == 0
        assert m.restore_bytes == m.restore_time_ms == m.promotions == m.demotions == 0
        assert m.actual_prefix_hit_rate == 0.25
        assert m.actual_prefill_tokens == 24 and m.saved_prefill_tokens == 8
        assert m.requests_with_hit_ratio == 0.5
        assert m.l1_capacity_blocks == 40
        assert m.l1_resident_blocks + m.l1_free_blocks + m.l1_reserved_future_blocks == 40
        assert m.l1_active_blocks == 0
        assert result.requests[1].parent_request_id == result.requests[0].id
        assert result.requests[1].actual_prefill_tokens == 8

    def test_quantization_error_comes_from_config_and_plan(self) -> None:
        config = SimConfig(
            workload=WorkloadConfig(
                synthetic=SyntheticWorkloadConfig(
                    num_requests=2,
                    session_rate=10,
                    prompt_len=FixedLength(value=16),
                    output_len=FixedLength(value=8),
                    seed=2,
                    session=SessionConfig(num_sessions=1, target_ideal_prefix_hit_rate=0.1),
                )
            ),
            engine=EngineConfig(max_running_requests=2, max_total_tokens=80, page_size=2),
            perf_model=PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=0),
        )
        synthetic = config.workload.synthetic
        assert synthetic is not None
        plan = build_plan(synthetic, config.engine)
        metrics = simulate(config).metrics
        assert metrics.target_ideal_prefix_hit_rate == pytest.approx(0.1)
        assert metrics.constructed_ideal_prefix_hit_rate == pytest.approx(0.125)
        assert metrics.ideal_hit_quantization_error == pytest.approx(0.025)
        empty = compute_metrics([], plan=plan, config=config)
        assert empty.target_ideal_prefix_hit_rate == metrics.target_ideal_prefix_hit_rate
        assert empty.constructed_ideal_prefix_hit_rate == metrics.constructed_ideal_prefix_hit_rate
        assert empty.ideal_hit_quantization_error == metrics.ideal_hit_quantization_error

    def test_infeasible_request_reports_abort_without_ending_simulation(self) -> None:
        cfg = SimConfig(
            workload=WorkloadConfig(
                synthetic=SyntheticWorkloadConfig(
                    num_requests=2,
                    session_rate=10,
                    prompt_len=FixedLength(value=128),
                    output_len=FixedLength(value=1),
                    seed=0,
                )
            ),
            engine=EngineConfig(max_running_requests=2, max_total_tokens=192, page_size=64),
            perf_model=PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=0),
        )
        result = simulate(cfg)
        assert (result.meta.num_requests, result.meta.num_aborted) == (2, 2)
        assert result.requests == []
        assert all(item.reason == "input_exceeds_kv_capacity" for item in result.aborted_requests)
        assert result.metrics.ttft_mean is None

    def test_enabled_tiers_propagate_capacity_and_zero_transfer(self) -> None:
        cfg = SimConfig(
            workload=WorkloadConfig(
                synthetic=SyntheticWorkloadConfig(
                    num_requests=1,
                    session_rate=1,
                    prompt_len=FixedLength(value=3),
                    output_len=FixedLength(value=1),
                    max_new_tokens=1,
                    seed=0,
                )
            ),
            engine=EngineConfig(
                max_running_requests=1,
                max_total_tokens=12,
                page_size=4,
                l2_capacity_tokens=4,
                l3_capacity_tokens=8,
            ),
            perf_model=PerfModelConfig(
                prefill_c1=1,
                decode_d1=0,
                decode_d2=0,
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
        cfg = SimConfig(
            workload=WorkloadConfig(synthetic=SyntheticWorkloadConfig(
                num_requests=3, session_rate=2,
                prompt_len=FixedLength(value=128), output_len=FixedLength(value=1), seed=0,
                session=SessionConfig(num_sessions=1),
            )),
            engine=EngineConfig(max_running_requests=1, max_total_tokens=64, page_size=16),
            perf_model=PerfModelConfig(prefill_c1=0, decode_d1=0, decode_d2=0),
        )
        result = simulate(cfg)
        assert (
            result.meta.num_completed, result.meta.num_aborted, result.meta.num_skipped
        ) == (0, 1, 2)
        assert result.metrics.arrival_qps is None

    def test_json_roundtrip(self, result: SimResult) -> None:
        """服务层将直接序列化该对象，必须 JSON 往返无损。"""
        assert SimResult.model_validate_json(result.model_dump_json()) == result


class TestTraceSeam:
    def test_trace_mode_not_implemented(self) -> None:
        cfg = make_config().model_copy(update={"workload": WorkloadConfig(mode="trace")})
        with pytest.raises(NotImplementedError, match="trace"):
            simulate(cfg)


class TestReproducibility:
    def test_same_seed_json_bit_identical(self, result: SimResult) -> None:
        """同种子仿真结果（除运行耗时）逐字节一致。"""
        exclude = {"meta": {"wall_clock_ms"}}
        assert result.model_dump_json(exclude=exclude) == simulate(make_config()).model_dump_json(
            exclude=exclude
        )

    def test_different_seed_differs(self) -> None:
        assert simulate(make_config(seed=1)).requests != simulate(make_config(seed=2)).requests


class TestSmallCases:
    @pytest.mark.parametrize("think_time_ms", [0.0, 30.0])
    def test_followup_arrives_only_after_parent_completion(self, think_time_ms: float) -> None:
        cfg = SimConfig(
            workload=WorkloadConfig(
                synthetic=SyntheticWorkloadConfig(
                    num_requests=3,
                    session_rate=1.0,
                    prompt_len=FixedLength(value=10),
                    output_len=FixedLength(value=2),
                    seed=7,
                    session=SessionConfig(
                        num_sessions=1,
                        think_time_ms=think_time_ms,
                        target_ideal_prefix_hit_rate=0.2,
                    ),
                )
            ),
            engine=EngineConfig(max_running_requests=2, max_total_tokens=100, page_size=2),
            perf_model=PerfModelConfig(prefill_c1=1.0, decode_d1=10.0, decode_d2=0.0),
        )
        result = simulate(cfg)
        synthetic = cfg.workload.synthetic
        assert synthetic is not None
        plan = build_plan(synthetic, cfg.engine)
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
        cfg = SimConfig(
            workload=WorkloadConfig(
                synthetic=SyntheticWorkloadConfig(
                    num_requests=1,
                    session_rate=1.0,
                    prompt_len=FixedLength(value=5),
                    output_len=FixedLength(value=3),
                    seed=0,
                )
            ),
            engine=EngineConfig(max_running_requests=1, max_total_tokens=100, page_size=1),
            perf_model=PerfModelConfig(prefill_c1=1.0, decode_d1=10.0, decode_d2=0.0),
        )
        r = simulate(cfg)
        record = r.requests[0]
        assert record.prefill_end - record.arrival == pytest.approx(5.0)
        assert record.finish - record.prefill_end == pytest.approx(20.0)
        assert r.metrics.ttft_mean == pytest.approx(5.0)
        assert r.metrics.tpot_mean == pytest.approx(10.0)
        assert r.meta.num_engine_steps == 3
