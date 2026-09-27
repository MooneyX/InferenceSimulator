"""perf_model 模块单测：公式正确性、边界、纯函数性与融合批语义。"""

import math

import pytest

from simcore.perf_model import decode_step_time, prefill_time, step_time
from simcore.schemas import PerfModelConfig

LINEAR = PerfModelConfig(prefill_c1=0.5, decode_d1=10.0, decode_d2=0.001)
QUADRATIC = PerfModelConfig(prefill_c1=0.5, prefill_c2=0.002, decode_d1=10.0, decode_d2=0.001)
ZERO = PerfModelConfig(prefill_c1=0.0, prefill_c2=0.0, decode_d1=0.0, decode_d2=0.0)


class TestPrefillTime:
    @pytest.mark.parametrize(
        ("prompt_tokens", "expected"),
        [(1, 0.5), (128, 64.0), (1024, 512.0)],
    )
    def test_linear_term(self, prompt_tokens: int, expected: float) -> None:
        assert prefill_time(prompt_tokens, LINEAR) == pytest.approx(expected)

    def test_quadratic_term(self) -> None:
        # 0.5 * 100 + 0.002 * 100^2 = 50 + 20
        assert prefill_time(100, QUADRATIC) == pytest.approx(70.0)

    def test_c2_defaults_off(self) -> None:
        """prefill_c2 默认 0，模型退化为线性。"""
        assert prefill_time(256, LINEAR) == pytest.approx(prefill_time(256, LINEAR.model_copy()))
        assert LINEAR.prefill_c2 == 0.0

    def test_monotonic_in_prompt_len(self) -> None:
        values = [prefill_time(n, QUADRATIC) for n in (1, 10, 100, 1000)]
        assert values == sorted(values)

    def test_superlinear_when_c2_positive(self) -> None:
        """二次项使倍增 prompt 的耗时增长超过 2 倍。"""
        assert prefill_time(200, QUADRATIC) > 2 * prefill_time(100, QUADRATIC)

    def test_zero_coefficients_give_zero(self) -> None:
        assert prefill_time(512, ZERO) == 0.0

    @pytest.mark.parametrize("prompt_tokens", [0, -1])
    def test_rejects_non_positive(self, prompt_tokens: int) -> None:
        with pytest.raises(ValueError, match="prompt_tokens"):
            prefill_time(prompt_tokens, LINEAR)


class TestDecodeStepTime:
    @pytest.mark.parametrize(
        ("batch_total_tokens", "expected"),
        [(1, 10.001), (1000, 11.0), (100_000, 110.0)],
    )
    def test_affine_form(self, batch_total_tokens: int, expected: float) -> None:
        assert decode_step_time(batch_total_tokens, LINEAR) == pytest.approx(expected)

    def test_constant_term_dominates_small_batch(self) -> None:
        """小 batch 下固定开销占主导 —— decode 阶段 batching 收益的来源。"""
        assert decode_step_time(1, LINEAR) < 1.01 * LINEAR.decode_d1

    def test_monotonic_in_batch_tokens(self) -> None:
        values = [decode_step_time(n, LINEAR) for n in (1, 100, 10_000)]
        assert values == sorted(values)

    def test_zero_coefficients_give_zero(self) -> None:
        assert decode_step_time(4096, ZERO) == 0.0

    @pytest.mark.parametrize("batch_total_tokens", [0, -5])
    def test_rejects_non_positive(self, batch_total_tokens: int) -> None:
        with pytest.raises(ValueError, match="batch_total_tokens"):
            decode_step_time(batch_total_tokens, LINEAR)


class TestStepTime:
    def test_pure_decode_step(self) -> None:
        assert step_time([], 1000, LINEAR) == pytest.approx(decode_step_time(1000, LINEAR))

    def test_pure_prefill_step(self) -> None:
        """batch 内全是刚准入请求时无 decode 工作，步长仅由 prefill 决定。"""
        assert step_time([128], 128, LINEAR, has_decode=False) == pytest.approx(64.0)

    def test_takes_max_not_sum(self) -> None:
        """融合批语义的核心断言：prefill 与 decode 并行，取最大值。"""
        # prefill(128)=64.0 > decode(1000)=11.0
        assert step_time([128], 1000, LINEAR) == pytest.approx(64.0)

    def test_decode_dominates_when_prefill_small(self) -> None:
        # prefill(1)=0.5 < decode(1000)=11.0
        assert step_time([1], 1000, LINEAR) == pytest.approx(11.0)

    def test_multiple_admissions_accumulate(self) -> None:
        """同一步多个准入请求的 prefill 相加，再与 decode 取最大。"""
        expected = prefill_time(100, LINEAR) + prefill_time(200, LINEAR)
        assert step_time([100, 200], 1000, LINEAR) == pytest.approx(expected)
        assert expected > decode_step_time(1000, LINEAR)

    def test_result_never_below_either_component(self) -> None:
        for batch_tokens in (1, 500, 50_000):
            result = step_time([64, 64], batch_tokens, QUADRATIC)
            assert result >= decode_step_time(batch_tokens, QUADRATIC)
            assert result >= 2 * prefill_time(64, QUADRATIC) - 1e-9

    def test_accepts_iterator(self) -> None:
        """入参为一次性迭代器时也必须正确（内部需物化）。"""
        assert step_time(iter([100, 200]), 1000, LINEAR) == pytest.approx(
            step_time([100, 200], 1000, LINEAR)
        )

    def test_empty_step_rejected(self) -> None:
        with pytest.raises(ValueError, match="空步"):
            step_time([], 1000, LINEAR, has_decode=False)

    def test_zero_cost_prefill_only_step_is_not_empty(self) -> None:
        """系数全 0 时步长为 0，但有准入请求就不算空步。"""
        assert step_time([128], 128, ZERO, has_decode=False) == 0.0


class TestPurity:
    """纯函数性：同输入必同输出，且不改写入参 —— 可复现性与未来 Rust 内核替换的前提。"""

    def test_repeated_calls_bit_identical(self) -> None:
        first = [step_time([37, 91], n, QUADRATIC) for n in range(1, 200)]
        second = [step_time([37, 91], n, QUADRATIC) for n in range(1, 200)]
        assert first == second

    def test_config_not_mutated(self) -> None:
        snapshot = QUADRATIC.model_dump()
        step_time([128, 256], 4096, QUADRATIC)
        assert QUADRATIC.model_dump() == snapshot

    def test_no_nan_or_inf_on_large_inputs(self) -> None:
        result = step_time([100_000], 1_000_000, QUADRATIC)
        assert math.isfinite(result)
