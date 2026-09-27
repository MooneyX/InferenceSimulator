"""workload 模块单测：可复现性、升序不变式、分布统计性质与 trace 接缝。"""

import itertools
import random
import statistics

import pytest

from simcore.schemas import (
    Arrival,
    FixedLength,
    LengthDist,
    LognormalLength,
    PoissonLength,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)
from simcore.workload import SyntheticSource, WorkloadSource, build_source, sample_length


def make_workload(
    *,
    num_requests: int = 64,
    arrival_rate: float = 20.0,
    prompt_len: LengthDist | None = None,
    output_len: LengthDist | None = None,
    seed: int = 42,
) -> SyntheticWorkloadConfig:
    return SyntheticWorkloadConfig(
        num_requests=num_requests,
        arrival_rate=arrival_rate,
        prompt_len=prompt_len or FixedLength(value=128),
        output_len=output_len or FixedLength(value=16),
        seed=seed,
    )


class TestSampleLength:
    def test_fixed_is_deterministic_and_consumes_no_randomness(self) -> None:
        """fixed 分支不得消耗随机数，否则换分布会扰动整条随机流。"""
        rng = random.Random(0)
        before = rng.getstate()
        assert sample_length(FixedLength(value=256), rng) == 256
        assert rng.getstate() == before

    @pytest.mark.parametrize(
        "dist",
        [
            FixedLength(value=1),
            PoissonLength(mean=0.001),
            LognormalLength(mean=0.001, sigma=0.5),
        ],
    )
    def test_clamped_to_at_least_one(self, dist: LengthDist) -> None:
        """极小均值下仍须 >= 1 token。"""
        rng = random.Random(7)
        assert all(sample_length(dist, rng) >= 1 for _ in range(200))

    def test_lognormal_mean_is_expectation(self) -> None:
        """mean 解释为期望值：调大 sigma 不应抬高平均值，只加重尾。"""
        narrow = LognormalLength(mean=500.0, sigma=0.25)
        wide = LognormalLength(mean=500.0, sigma=1.5)
        narrow_samples = [sample_length(narrow, random.Random(i)) for i in range(4000)]
        wide_samples = [sample_length(wide, random.Random(i)) for i in range(4000)]

        assert statistics.mean(narrow_samples) == pytest.approx(500.0, rel=0.1)
        assert statistics.mean(wide_samples) == pytest.approx(500.0, rel=0.2)
        # 尾部显著变重：P99 拉开数量级
        assert max(wide_samples) > 3 * max(narrow_samples)
        # 中位数下移（对数正态的固有形态）
        assert statistics.median(wide_samples) < statistics.median(narrow_samples)

    def test_poisson_mean_matches(self) -> None:
        dist = PoissonLength(mean=40.0)
        rng = random.Random(3)
        samples = [sample_length(dist, rng) for _ in range(4000)]
        assert statistics.mean(samples) == pytest.approx(40.0, rel=0.05)
        assert statistics.pvariance(samples) == pytest.approx(40.0, rel=0.15)

    def test_poisson_spreads_values(self) -> None:
        dist = PoissonLength(mean=20.0)
        rng = random.Random(11)
        assert len({sample_length(dist, rng) for _ in range(500)}) > 10


class TestSyntheticSourceReproducibility:
    def test_same_seed_bit_identical(self) -> None:
        cfg = make_workload(
            prompt_len=LognormalLength(mean=300.0, sigma=1.2),
            output_len=PoissonLength(mean=50.0),
        )
        first = [a.model_dump() for a in SyntheticSource(cfg)]
        second = [a.model_dump() for a in SyntheticSource(cfg)]
        assert first == second

    def test_different_seed_differs(self) -> None:
        a = list(SyntheticSource(make_workload(seed=1)))
        b = list(SyntheticSource(make_workload(seed=2)))
        assert [x.sim_time for x in a] != [x.sim_time for x in b]

    def test_unaffected_by_global_random_state(self) -> None:
        """内部使用独立 Random 实例，全局 random 被污染也不影响结果。"""
        cfg = make_workload()
        baseline = list(SyntheticSource(cfg))
        random.seed(999)
        random.random()
        assert list(SyntheticSource(cfg)) == baseline

    def test_reiterable(self) -> None:
        """协议要求可重复迭代，而非一次性生成器。"""
        source = SyntheticSource(make_workload(num_requests=8))
        assert list(source) == list(source)

    def test_satisfies_protocol(self) -> None:
        source: WorkloadSource = SyntheticSource(make_workload(num_requests=4))
        assert isinstance(source, WorkloadSource)


class TestSyntheticSourceInvariants:
    def test_count_and_ids(self) -> None:
        source = SyntheticSource(make_workload(num_requests=50))
        arrivals = list(source)
        assert len(arrivals) == len(source) == 50
        assert [a.request_id for a in arrivals] == list(range(50))

    def test_sim_time_ascending(self) -> None:
        arrivals = list(SyntheticSource(make_workload(num_requests=500, arrival_rate=200.0)))
        times = [a.sim_time for a in arrivals]
        assert times == sorted(times)

    def test_first_arrival_is_positive(self) -> None:
        """首请求不固定在 t=0，避免 t=0 处的人为突发。"""
        assert next(iter(SyntheticSource(make_workload(num_requests=1)))).sim_time > 0.0

    def test_single_request(self) -> None:
        arrivals = list(SyntheticSource(make_workload(num_requests=1)))
        assert len(arrivals) == 1
        assert isinstance(arrivals[0], Arrival)

    def test_lengths_respect_config(self) -> None:
        source = SyntheticSource(
            make_workload(
                num_requests=30,
                prompt_len=FixedLength(value=333),
                output_len=FixedLength(value=7),
            )
        )
        assert all(a.prompt_len == 333 and a.output_len == 7 for a in source)

    def test_arrival_rate_controls_interval(self) -> None:
        """到达率单位为 请求/秒，时间轴为 ms：平均间隔 ≈ 1000 / rate。"""
        arrivals = list(SyntheticSource(make_workload(num_requests=3000, arrival_rate=50.0)))
        gaps = [b.sim_time - a.sim_time for a, b in itertools.pairwise(arrivals)]
        assert statistics.mean(gaps) == pytest.approx(20.0, rel=0.08)

    def test_higher_rate_compresses_timeline(self) -> None:
        slow = list(SyntheticSource(make_workload(num_requests=200, arrival_rate=10.0)))
        fast = list(SyntheticSource(make_workload(num_requests=200, arrival_rate=100.0)))
        assert fast[-1].sim_time < slow[-1].sim_time / 5


class TestBuildSource:
    def test_synthetic(self) -> None:
        cfg = WorkloadConfig(synthetic=make_workload(num_requests=5))
        assert len(list(build_source(cfg))) == 5

    def test_trace_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="trace"):
            build_source(WorkloadConfig(mode="trace"))
