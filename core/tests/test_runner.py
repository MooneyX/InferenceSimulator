"""runner 模块单测：端到端串联、元信息一致性、trace 接缝与可复现性。"""

import pytest

import simcore
from simcore import SimConfig, SimResult, simulate
from simcore.metrics import compute_metrics
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


def make_config(seed: int = 0, num_requests: int = 200, budget: int = 16_000) -> SimConfig:
    return SimConfig(
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=num_requests,
                arrival_rate=30.0,
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
                    arrival_rate=10,
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

    def test_infeasible_request_reports_abort_without_ending_simulation(self) -> None:
        cfg = SimConfig(
            workload=WorkloadConfig(
                synthetic=SyntheticWorkloadConfig(
                    num_requests=2,
                    arrival_rate=10,
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

    def test_json_roundtrip(self, result: SimResult) -> None:
        """服务层将直接序列化该对象，必须 JSON 往返无损。"""
        assert SimResult.model_validate_json(result.model_dump_json()) == result


class TestTraceSeam:
    def test_trace_mode_not_implemented(self) -> None:
        cfg = make_config().model_copy(update={"workload": WorkloadConfig(mode="trace")})
        with pytest.raises(NotImplementedError, match="trace"):
            simulate(cfg)


class TestReproducibility:
    def test_same_seed_json_bit_identical(self) -> None:
        """M1 验收线：同种子两次运行结果逐字节一致。"""
        assert simulate(make_config(seed=11)).model_dump_json() == (
            simulate(make_config(seed=11)).model_dump_json()
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
                    arrival_rate=1.0,
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
        assert len(result.requests) == 3
        for parent, child in zip(result.requests, result.requests[1:], strict=False):
            assert child.arrival == parent.finish + think_time_ms
            assert child.prefill_start >= child.arrival
        assert result.meta.num_engine_steps == 6

    def test_single_request_fixed_lengths(self) -> None:
        cfg = SimConfig(
            workload=WorkloadConfig(
                synthetic=SyntheticWorkloadConfig(
                    num_requests=1,
                    arrival_rate=1.0,
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
