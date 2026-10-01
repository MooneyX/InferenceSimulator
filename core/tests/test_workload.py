"""workload 模块长度分布采样单测。"""

import random
import statistics

import pytest

from simcore.schemas import FixedLength, LengthDist, LognormalLength, PoissonLength
from simcore.workload import sample_length


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
