"""KV 来源：先来先服务的确定性计时、两状态降速与单次抖动的可复现波动。"""

import pytest

from simcore.kv_source import L3_FLUCTUATION, Fluctuation, KvSource, KvSourceSpec


def test_deterministic_source_reads_groups_serially_and_queues_later_submissions() -> None:
    source = KvSource(KvSourceSpec(bytes_per_ms=4, latency_ms=1), "seed")
    assert source.submit(10, [8, 4]) == [13, 15]
    assert source.free_ms == 15
    # 链路忙时后续提交排队；空闲后从提交时刻开始。
    assert source.submit(12, [4]) == [17]
    assert source.submit(30, []) == []
    assert source.submit(30, [0]) == [31]


def test_fluctuating_source_is_reproducible_per_seed() -> None:
    spec = KvSourceSpec(bytes_per_ms=1000, latency_ms=0.1, fluctuation=L3_FLUCTUATION)

    def run(seed: str) -> list[float]:
        source = KvSource(spec, seed)
        return [source.submit(start, [5000.0])[0] for start in range(0, 20_000, 400)]

    assert run("0:0:L3") == run("0:0:L3")
    assert run("0:0:L3") != run("0:1:L3")


def test_fluctuation_never_beats_nominal_without_jitter_and_degrades_expected_share() -> None:
    fluctuation = Fluctuation(
        degraded_fraction=0.2,
        mean_degraded_ms=10,
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
    # 状态时长按指数分布切换，长期降速时间占比接近设定值。
    assert slow / reads == pytest.approx(0.2, abs=0.03)


def test_read_crossing_state_switch_uses_each_segment_bandwidth() -> None:
    fluctuation = Fluctuation(
        degraded_fraction=0.5,
        mean_degraded_ms=1e12,
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
def test_idle_gaps_and_latency_skip_unobserved_state_transitions(latency: float) -> None:
    from unittest.mock import patch

    source = KvSource(
        KvSourceSpec(bytes_per_ms=1000, latency_ms=latency, fluctuation=L3_FLUCTUATION), "idle"
    )
    with patch.object(source._rng, "expovariate", wraps=source._rng.expovariate) as draws:
        (end,) = source.submit(1e12, [1])
    assert end >= 1e12 + latency
    assert draws.call_count <= 3


@pytest.mark.parametrize("degraded", [False, True])
def test_idle_transition_probability_respects_known_first_switch(degraded: bool) -> None:
    import math
    from unittest.mock import patch

    fluctuation = Fluctuation(0.25, 10, 0.5, 0, 0)
    source = KvSource(KvSourceSpec(1000, 0, fluctuation), "transition")
    probability = 0.25 + (int(not degraded) - 0.25) * math.exp(-(1 / 30 + 1 / 10) * 5)
    for sample, expected in ((probability - 0.01, True), (probability + 0.01, False)):
        source._degraded, source._state_end = degraded, 10
        with patch.object(source._rng, "random", return_value=sample):
            source._advance(15, idle=True)
        assert source._degraded is expected
        assert source._state_end > 15
