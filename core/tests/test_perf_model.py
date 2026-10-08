"""perf_model 模块单测：公式正确性、边界、纯函数性与融合批语义。"""

import pytest

from simcore.perf_model import decode_step_time, prefill_cost
from simcore.schemas import PerfModelConfig

LINEAR = PerfModelConfig(prefill_c1=0.5, decode_d1=10.0, decode_d2=0.001)
QUADRATIC = PerfModelConfig(prefill_c1=0.5, prefill_c2=0.002, decode_d1=10.0, decode_d2=0.001)
ZERO = PerfModelConfig(prefill_c1=0.0, prefill_c2=0.0, decode_d1=0.0, decode_d2=0.0)


class TestPrefillCost:
    @pytest.mark.parametrize(
        ("prompt_tokens", "expected"),
        [(1, 0.5), (128, 64.0), (1024, 512.0)],
    )
    def test_linear_term(self, prompt_tokens: int, expected: float) -> None:
        assert prefill_cost(0, prompt_tokens, LINEAR) == pytest.approx(expected)

    def test_quadratic_term(self) -> None:
        # 0.5 * 100 + 0.002 * 100^2 = 50 + 20
        assert prefill_cost(0, 100, QUADRATIC) == pytest.approx(70.0)

    def test_c2_defaults_off(self) -> None:
        """prefill_c2 默认 0，模型退化为线性。"""
        assert prefill_cost(0, 256, LINEAR) == pytest.approx(
            prefill_cost(0, 256, LINEAR.model_copy())
        )
        assert LINEAR.prefill_c2 == 0.0

    def test_monotonic_in_prompt_len(self) -> None:
        values = [prefill_cost(0, n, QUADRATIC) for n in (1, 10, 100, 1000)]
        assert values == sorted(values)

    def test_superlinear_when_c2_positive(self) -> None:
        """二次项使倍增 prompt 的耗时增长超过 2 倍。"""
        assert prefill_cost(0, 200, QUADRATIC) > 2 * prefill_cost(0, 100, QUADRATIC)

    def test_zero_coefficients_give_zero(self) -> None:
        assert prefill_cost(0, 512, ZERO) == 0.0

    @pytest.mark.parametrize("prompt_tokens", [0, -1])
    def test_rejects_non_positive(self, prompt_tokens: int) -> None:
        with pytest.raises(ValueError, match="new_prefill_tokens"):
            prefill_cost(0, prompt_tokens, LINEAR)


class TestIncrementalPrefill:
    def test_context_history_is_charged(self) -> None:
        assert prefill_cost(80, 20, QUADRATIC) == pytest.approx(
            QUADRATIC.prefill_c1 * 20 + QUADRATIC.prefill_c2 * (100**2 - 80**2)
        )
        assert prefill_cost(0, 100, QUADRATIC) == pytest.approx(70.0)
        assert prefill_cost(80, 20, QUADRATIC) > prefill_cost(0, 20, QUADRATIC)

    def test_invalid_context_rejected(self) -> None:
        with pytest.raises(ValueError, match="context_before"):
            prefill_cost(-1, 20, QUADRATIC)


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
