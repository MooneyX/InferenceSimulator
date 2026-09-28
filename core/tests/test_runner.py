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
        engine=EngineConfig(max_batch_size=16, kv_token_budget=budget),
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
        assert result.metrics == compute_metrics(result.requests)
        assert result.metrics.makespan == pytest.approx(result.meta.sim_end - result.meta.sim_start)

    def test_ttft_percentiles_ordered(self, result: SimResult) -> None:
        m = result.metrics
        assert 0.0 < m.ttft_p50 <= m.ttft_p99
        assert m.tpot_mean is not None and m.tpot_mean > 0
        assert m.throughput is not None and m.throughput > 0

    def test_warnings_propagated(self) -> None:
        tight = simulate(make_config(num_requests=50, budget=500))
        assert tight.warnings
        assert isinstance(tight.warnings, list)

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
            engine=EngineConfig(max_batch_size=1, kv_token_budget=100),
            perf_model=PerfModelConfig(prefill_c1=1.0, decode_d1=10.0, decode_d2=0.0),
        )
        r = simulate(cfg)
        record = r.requests[0]
        assert record.prefill_end - record.arrival == pytest.approx(5.0)
        assert record.finish - record.prefill_end == pytest.approx(20.0)
        assert r.metrics.ttft_mean == pytest.approx(5.0)
        assert r.metrics.tpot_mean == pytest.approx(10.0)
        assert r.meta.num_engine_steps == 3
