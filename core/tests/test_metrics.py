"""metrics 模块单测：分位数插值、各指标口径与边界。"""

import pytest

from simcore.metrics import compute_metrics, percentile
from simcore.schemas import RequestRecord


def rec(
    request_id: int,
    arrival: float,
    prefill_end: float,
    finish: float,
    output_len: int,
    prefill_start: float | None = None,
) -> RequestRecord:
    return RequestRecord(
        id=request_id,
        arrival=arrival,
        prefill_start=arrival if prefill_start is None else prefill_start,
        prefill_end=prefill_end,
        finish=finish,
        prompt_len=10,
        output_len=output_len,
    )


class TestPercentile:
    def test_single_value(self) -> None:
        assert percentile([7.0], 0.0) == percentile([7.0], 0.99) == 7.0

    def test_endpoints(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert (percentile(values, 0.0), percentile(values, 1.0)) == (1.0, 4.0)

    @pytest.mark.parametrize(
        ("values", "q", "expected"),
        [
            ([1.0, 2.0, 3.0, 4.0], 0.5, 2.5),  # h=1.5
            ([1.0, 2.0, 3.0], 0.5, 2.0),  # h=1，恰落在样本上
            ([0.0, 10.0], 0.99, 9.9),  # h=0.99
            ([10.0, 20.0, 30.0, 40.0, 50.0], 0.25, 20.0),
            ([10.0, 20.0, 30.0, 40.0, 50.0], 0.9, 46.0),  # h=3.6
        ],
    )
    def test_linear_interpolation(self, values: list[float], q: float, expected: float) -> None:
        assert percentile(values, q) == pytest.approx(expected)

    def test_matches_numpy_linear_reference(self) -> None:
        """与 numpy.percentile(method='linear') 的已知结果比对（不引入 numpy 依赖）。"""
        values = [float(v) for v in range(1, 101)]
        # numpy.percentile(range(1,101), 99) == 99.01
        assert percentile(values, 0.99) == pytest.approx(99.01)
        # numpy.percentile(range(1,101), 50) == 50.5
        assert percentile(values, 0.50) == pytest.approx(50.5)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="为空"):
            percentile([], 0.5)

    @pytest.mark.parametrize("q", [-0.01, 1.01])
    def test_q_out_of_range_rejected(self, q: float) -> None:
        with pytest.raises(ValueError, match="q"):
            percentile([1.0, 2.0], q)


class TestTTFT:
    def test_ttft_is_prefill_end_minus_arrival(self) -> None:
        """TTFT 含排队时间：从到达算起，而非从 prefill_start。"""
        m = compute_metrics([rec(0, 10.0, 25.0, 25.0, 1, prefill_start=20.0)])
        assert (m.ttft_mean, m.ttft_p50, m.ttft_p99) == (15.0, 15.0, 15.0)

    def test_ttft_stats(self) -> None:
        records = [rec(i, 0.0, float(ttft), 100.0, 1) for i, ttft in enumerate([10, 20, 30, 40])]
        m = compute_metrics(records)
        assert m.ttft_mean == pytest.approx(25.0)
        assert m.ttft_p50 == pytest.approx(25.0)
        assert m.ttft_p99 == pytest.approx(39.7)  # h=2.97

    def test_order_independent(self) -> None:
        records = [rec(i, 0.0, float(t), 100.0, 1) for i, t in enumerate([40, 10, 30, 20])]
        assert compute_metrics(records) == compute_metrics(sorted(records, key=lambda r: r.id))


class TestTPOT:
    def test_per_request_then_averaged(self) -> None:
        """逐请求平均再对请求平均，而非按 token 加权。"""
        records = [
            rec(0, 0.0, 10.0, 20.0, 2),  # 10 ms / 1 token = 10
            rec(1, 0.0, 10.0, 40.0, 11),  # 30 ms / 10 token = 3
        ]
        m = compute_metrics(records)
        # 按请求平均 = 6.5；若按 token 加权则为 40 / 11 ≈ 3.64
        assert m.tpot_mean == pytest.approx(6.5)

    def test_single_token_requests_excluded(self) -> None:
        records = [rec(0, 0.0, 10.0, 10.0, 1), rec(1, 0.0, 10.0, 30.0, 3)]
        assert compute_metrics(records).tpot_mean == pytest.approx(10.0)

    def test_all_single_token_gives_none(self) -> None:
        records = [rec(0, 0.0, 10.0, 10.0, 1), rec(1, 5.0, 12.0, 12.0, 1)]
        assert compute_metrics(records).tpot_mean is None


class TestThroughputAndMakespan:
    def test_makespan_from_first_arrival_to_last_finish(self) -> None:
        records = [rec(0, 100.0, 110.0, 150.0, 2), rec(1, 120.0, 130.0, 600.0, 2)]
        assert compute_metrics(records).makespan == pytest.approx(500.0)

    def test_throughput_in_requests_per_second(self) -> None:
        """2 个请求 / 500 ms = 4 请求/秒。"""
        records = [rec(0, 100.0, 110.0, 150.0, 2), rec(1, 120.0, 130.0, 600.0, 2)]
        assert compute_metrics(records).throughput == pytest.approx(4.0)

    def test_zero_makespan_gives_none_throughput(self) -> None:
        records = [rec(0, 5.0, 5.0, 5.0, 1), rec(1, 5.0, 5.0, 5.0, 1)]
        m = compute_metrics(records)
        assert m.makespan == 0.0 and m.throughput is None


class TestEdgeCases:
    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="为空"):
            compute_metrics([])

    def test_accepts_tuple(self) -> None:
        m = compute_metrics((rec(0, 0.0, 10.0, 30.0, 3),))
        assert m.ttft_mean == 10.0
