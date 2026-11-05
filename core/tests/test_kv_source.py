"""KV 来源：先来先服务的确定性计时、两状态降速（长尾降速段）与单次抖动的可复现波动。"""

import math
from unittest.mock import patch

import pytest

from simcore.kv_source import KvSource, KvSourceSpec
from simcore.schemas import L3Fluctuation


def _fluctuation(**update: float) -> L3Fluctuation:
    return L3Fluctuation().model_copy(update=update)


def test_deterministic_source_reads_groups_serially_and_queues_later_submissions() -> None:
    source = KvSource(KvSourceSpec(bytes_per_ms=4, latency_ms=1), "seed")
    assert source.submit(10, [8, 4]) == [13, 15]
    assert source.free_ms == 15
    # 链路忙时后续提交排队；空闲后从提交时刻开始。
    assert source.submit(12, [4]) == [17]
    assert source.submit(30, []) == []
    assert source.submit(30, [0]) == [31]


def test_fluctuating_source_is_reproducible_per_seed() -> None:
    spec = KvSourceSpec(bytes_per_ms=1000, latency_ms=0.1, fluctuation=L3Fluctuation())

    def run(seed: str) -> list[float]:
        source = KvSource(spec, seed)
        return [source.submit(start, [5000.0])[0] for start in range(0, 20_000, 400)]

    assert run("0:0:L3") == run("0:0:L3")
    assert run("0:0:L3") != run("0:1:L3")


def test_zero_fraction_and_jitter_match_deterministic_timing() -> None:
    off = _fluctuation(degraded_fraction=0, jitter_cv=0)
    fluctuating = KvSource(KvSourceSpec(4, 1, off), "s")
    deterministic = KvSource(KvSourceSpec(4, 1), "s")
    for start in (10, 12, 30, 1e9):
        assert fluctuating.submit(start, [8, 4]) == deterministic.submit(start, [8, 4])


def test_jitter_only_never_degrades_and_varies_around_nominal() -> None:
    source = KvSource(KvSourceSpec(1000, 0, _fluctuation(degraded_fraction=0, jitter_cv=0.1)), "j")
    durations = [source.submit(float(i), [100.0])[0] - i for i in range(20_000)]
    assert not source._degraded
    assert min(durations) < 0.1 < max(durations) < 0.2
    # 带宽系数均值为 1，耗时均值略高于标称值（Jensen）。
    assert sum(durations) / len(durations) == pytest.approx(0.1, rel=0.02)


@pytest.mark.parametrize("duration_cv", [0.0, 1.0])
def test_fluctuation_never_beats_nominal_without_jitter_and_degrades_expected_share(
    duration_cv: float,
) -> None:
    fluctuation = _fluctuation(
        degraded_fraction=0.2,
        mean_degraded_ms=10,
        degraded_duration_cv=duration_cv,
        degraded_bandwidth_factor=0.5,
        degraded_extra_latency_ms=0,
        jitter_cv=0,
    )
    source = KvSource(KvSourceSpec(bytes_per_ms=1000, latency_ms=0, fluctuation=fluctuation), "s")
    slow = 0
    reads = 100_000
    for start in range(reads):
        (end,) = source.submit(float(start), [1.0])
        duration = end - start
        assert duration >= 0.001 * (1 - 1e-6)
        slow += duration > 0.0015
    # 降速段均值固定，长期降速时间占比接近设定值。
    assert slow / reads == pytest.approx(0.2, abs=0.03)


def test_degraded_segments_follow_lognormal_with_configured_mean_and_tail() -> None:
    def segments(cv: float, count: int) -> list[float]:
        source = KvSource(
            KvSourceSpec(1, 0, _fluctuation(mean_degraded_ms=50, degraded_duration_cv=cv)), "d"
        )
        source._degraded = True
        return [source._segment_ms() for _ in range(count)]

    assert set(segments(0, 10)) == {50}
    samples = segments(3, 200_000)
    assert sum(samples) / len(samples) == pytest.approx(50, rel=0.05)
    # cv=3：中位数为均值 / sqrt(10)，约 1.1% 的段超过 10 倍均值。
    assert sorted(samples)[len(samples) // 2] == pytest.approx(50 / math.sqrt(10), rel=0.03)
    assert sum(sample > 500 for sample in samples) / len(samples) == pytest.approx(
        0.0114, abs=0.002
    )


def test_stationary_restart_uses_time_share_and_length_biased_residual() -> None:
    source = KvSource(
        KvSourceSpec(
            1, 0, _fluctuation(degraded_fraction=0.25, mean_degraded_ms=10, degraded_duration_cv=0)
        ),
        "r",
    )
    degraded: list[float] = []
    normal: list[float] = []
    for _ in range(100_000):
        source._restart(0.0)
        (degraded if source._degraded else normal).append(source._state_end)
    assert len(degraded) / 100_000 == pytest.approx(0.25, abs=0.01)
    # 固定 10 ms 降速段的剩余时长在 [0, 10) 均匀；正常段剩余时长均值为 30 ms。
    assert max(degraded) < 10
    assert sum(degraded) / len(degraded) == pytest.approx(5, rel=0.03)
    assert sum(normal) / len(normal) == pytest.approx(30, rel=0.03)


def test_read_crossing_state_switch_uses_each_segment_bandwidth() -> None:
    fluctuation = _fluctuation(
        degraded_fraction=0.5,
        mean_degraded_ms=1e12,
        degraded_duration_cv=0,
        degraded_bandwidth_factor=0.5,
        degraded_extra_latency_ms=2,
        jitter_cv=0,
    )
    source = KvSource(KvSourceSpec(bytes_per_ms=1, latency_ms=0, fluctuation=fluctuation), "s")
    source._degraded, source._state_end = False, 5.0
    # 正常段 5 ms 传 5 B，其余 5 B 在降速段以 0.5 B/ms 传输。
    assert source.submit(0, [10]) == [pytest.approx(15)]
    # 降速状态下发起的读取另加固定延迟。
    assert source.submit(15, [1]) == [pytest.approx(19)]


@pytest.mark.parametrize("latency", [0, 1e12])
def test_long_idle_gaps_restart_from_stationary_state(latency: float) -> None:
    source = KvSource(KvSourceSpec(1000, latency, L3Fluctuation()), "idle")
    with (
        patch.object(source._rng, "expovariate", wraps=source._rng.expovariate) as normal,
        patch.object(source._rng, "lognormvariate", wraps=source._rng.lognormvariate) as other,
    ):
        (end,) = source.submit(1e12, [1])
    assert end >= 1e12 + latency
    # 只做常数次稳态重抽与少量逐段推进，不随空档长度增长。
    assert normal.call_count + other.call_count <= 8
