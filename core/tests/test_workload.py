"""workload 模块长度分布采样单测。"""

import random
import statistics

import pytest

from simcore.schemas import (
    EngineConfig,
    FixedLength,
    LengthDist,
    LognormalLength,
    PoissonLength,
    SessionConfig,
    SyntheticWorkloadConfig,
)
from simcore.workload import build_synthetic_closed_loop_plan, sample_length


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

    def test_poisson_small_mean_keeps_seeded_sequence(self) -> None:
        rng = random.Random(11)
        dist = PoissonLength(mean=20.0)
        assert [sample_length(dist, rng) for _ in range(8)] == [21, 20, 16, 15, 19, 23, 18, 15]

    @pytest.mark.parametrize("mean", [30.0, 128.0, 32768.0])
    def test_poisson_ptrs_mean_and_variance(self, mean: float) -> None:
        dist = PoissonLength(mean=mean)
        rng = random.Random(42)
        samples = [sample_length(dist, rng) for _ in range(4000)]
        assert statistics.mean(samples) == pytest.approx(mean, rel=0.03)
        assert statistics.pvariance(samples) == pytest.approx(mean, rel=0.15)

    def test_poisson_ptrs_is_seeded_and_varies(self) -> None:
        dist = PoissonLength(mean=32768.0)
        samples = [sample_length(dist, random.Random(seed)) for seed in range(100)]
        again = [sample_length(dist, random.Random(seed)) for seed in range(100)]
        assert samples == again
        assert len(set(samples)) > 50

    def test_high_mean_prompt_plan_uses_actual_requested_length(self) -> None:
        config = SyntheticWorkloadConfig(
            num_requests=1000,
            session_rate=20.48,
            prompt_len=PoissonLength(mean=32768),
            turn_increment_len=PoissonLength(mean=256),
            output_len=PoissonLength(mean=128),
            seed=42,
            session=SessionConfig(num_sessions=20, think_time_ms=10000),
        )
        engine = EngineConfig(max_running_requests=32, max_total_tokens=4_000_000, page_size=64)
        plan = build_synthetic_closed_loop_plan(config, engine)
        assert len(plan.root_request_ids) == 20
        root_lengths = [plan.turns[key].prompt_len for key in plan.root_request_ids]
        assert statistics.mean(root_lengths) == pytest.approx(32768, rel=0.02)
        for turn in plan.turns.values():
            if turn.parent_request_id is not None:
                parent = plan.turns[turn.parent_request_id]
                assert turn.prompt_len > parent.prompt_len + parent.output_len
        assert plan.constructed_ideal_prefix_hit_rate > 0.9
        assert build_synthetic_closed_loop_plan(config, engine) == plan
