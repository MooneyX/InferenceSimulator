"""schemas 模块单测：可判别联合、校验边界、不可变性、序列化往返。"""

from typing import ClassVar

import pytest
from pydantic import TypeAdapter, ValidationError

from simcore.schemas import (
    Arrival,
    EngineConfig,
    FixedLength,
    LengthDist,
    LognormalLength,
    Metrics,
    PerfModelConfig,
    PoissonLength,
    RequestRecord,
    SimConfig,
    SimMeta,
    SimResult,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)

LENGTH_DIST = TypeAdapter[LengthDist](LengthDist)


def make_config(**engine_overrides: int) -> SimConfig:
    """构造一个合法的最小配置，供各用例复用。"""
    return SimConfig(
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=4,
                arrival_rate=10.0,
                prompt_len=FixedLength(value=128),
                output_len=FixedLength(value=16),
                seed=0,
            )
        ),
        engine=EngineConfig(**{"max_batch_size": 8, "kv_token_budget": 4096, **engine_overrides}),
        perf_model=PerfModelConfig(prefill_c1=0.5, decode_d1=10.0, decode_d2=0.001),
    )


class TestLengthDist:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"dist": "fixed", "value": 32}, FixedLength),
            ({"dist": "poisson", "mean": 32.0}, PoissonLength),
            ({"dist": "lognormal", "mean": 32.0, "sigma": 1.5}, LognormalLength),
        ],
    )
    def test_discriminator_picks_variant(self, payload: dict[str, object], expected: type) -> None:
        assert isinstance(LENGTH_DIST.validate_python(payload), expected)

    def test_unknown_dist_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LENGTH_DIST.validate_python({"dist": "normal", "mean": 1.0})

    def test_wrong_variant_field_rejected(self) -> None:
        """fixed 分支不接受 mean —— 防止字段串味被静默忽略。"""
        with pytest.raises(ValidationError):
            LENGTH_DIST.validate_python({"dist": "fixed", "mean": 32.0})

    @pytest.mark.parametrize(
        "payload",
        [
            {"dist": "fixed", "value": 0},
            {"dist": "poisson", "mean": 0.0},
            {"dist": "lognormal", "mean": 8.0, "sigma": 0.0},
        ],
    )
    def test_non_positive_params_rejected(self, payload: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            LENGTH_DIST.validate_python(payload)

    def test_lognormal_requires_sigma(self) -> None:
        with pytest.raises(ValidationError):
            LENGTH_DIST.validate_python({"dist": "lognormal", "mean": 8.0})


class TestWorkloadConfig:
    def test_synthetic_payload_required(self) -> None:
        with pytest.raises(ValidationError, match=r"workload\.synthetic"):
            WorkloadConfig(mode="synthetic")

    def test_trace_mode_accepted_without_payload(self) -> None:
        """trace 是预留接缝：schema 层放行，由服务层返回未实现错误。"""
        assert WorkloadConfig(mode="trace").synthetic is None

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadConfig.model_validate({"mode": "replay"})


class TestConfigValidation:
    @pytest.mark.parametrize(
        "overrides",
        [{"max_batch_size": 0}, {"kv_token_budget": 0}],
    )
    def test_engine_bounds(self, overrides: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            make_config(**overrides)

    def test_prefill_c2_defaults_to_zero(self) -> None:
        assert make_config().perf_model.prefill_c2 == 0.0

    @pytest.mark.parametrize(
        "field",
        ["prefill_c1", "prefill_c2", "decode_d1", "decode_d2"],
    )
    def test_negative_coefficients_rejected(self, field: str) -> None:
        base = {"prefill_c1": 0.5, "decode_d1": 10.0, "decode_d2": 0.001}
        with pytest.raises(ValidationError):
            PerfModelConfig.model_validate({**base, field: -1.0})

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EngineConfig.model_validate(
                {"max_batch_size": 8, "kv_token_budget": 4096, "max_batch_tokens": 1024}
            )

    def test_config_is_frozen(self) -> None:
        cfg = make_config()
        with pytest.raises(ValidationError):
            cfg.engine.max_batch_size = 16

    def test_roundtrip_is_lossless(self) -> None:
        cfg = make_config()
        assert SimConfig.model_validate(cfg.model_dump()) == cfg


class TestArrival:
    def test_valid(self) -> None:
        a = Arrival(request_id=0, sim_time=0.0, prompt_len=10, output_len=5)
        assert (a.request_id, a.sim_time) == (0, 0.0)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"request_id": -1},
            {"sim_time": -0.1},
            {"prompt_len": 0},
            {"output_len": 0},
        ],
    )
    def test_bounds(self, overrides: dict[str, float]) -> None:
        base = {"request_id": 0, "sim_time": 1.0, "prompt_len": 10, "output_len": 5}
        with pytest.raises(ValidationError):
            Arrival.model_validate({**base, **overrides})


class TestRequestRecord:
    BASE: ClassVar[dict[str, float | int]] = {
        "id": 0,
        "arrival": 1.0,
        "prefill_start": 2.0,
        "prefill_end": 3.0,
        "finish": 4.0,
        "prompt_len": 10,
        "output_len": 5,
    }

    def test_monotonic_endpoints_accepted(self) -> None:
        assert RequestRecord.model_validate(self.BASE).prefill_end == 3.0

    def test_degenerate_intervals_accepted(self) -> None:
        """零长度区间合法：到达即被调度、单 token 输出等情形。"""
        flat = {**self.BASE, "prefill_start": 1.0, "prefill_end": 1.0, "finish": 1.0}
        assert RequestRecord.model_validate(flat).finish == 1.0

    @pytest.mark.parametrize(
        "overrides",
        [
            {"prefill_start": 0.5},  # 早于 arrival
            {"prefill_end": 1.5},  # 早于 prefill_start
            {"finish": 2.5},  # 早于 prefill_end
        ],
    )
    def test_out_of_order_endpoints_rejected(self, overrides: dict[str, float]) -> None:
        with pytest.raises(ValidationError, match="单调不减"):
            RequestRecord.model_validate({**self.BASE, **overrides})


class TestResult:
    def test_sim_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sim_end"):
            SimMeta(
                config=make_config(),
                num_requests=1,
                num_engine_steps=1,
                sim_start=5.0,
                sim_end=4.0,
            )

    def test_tpot_may_be_absent(self) -> None:
        m = Metrics(ttft_mean=1.0, ttft_p50=1.0, ttft_p99=1.0, throughput=2.0, makespan=500.0)
        assert m.tpot_mean is None

    def test_empty_result_roundtrip(self) -> None:
        result = SimResult(
            meta=SimMeta(
                config=make_config(),
                num_requests=0,
                num_engine_steps=0,
                sim_start=0.0,
                sim_end=0.0,
            ),
            metrics=Metrics(
                ttft_mean=0.0, ttft_p50=0.0, ttft_p99=0.0, throughput=0.0, makespan=0.0
            ),
        )
        assert SimResult.model_validate(result.model_dump()) == result
        assert result.requests == [] and result.warnings == []

    def test_json_schema_generates(self) -> None:
        """OpenAPI 导出依赖此能力（可判别联合易在此处出错）。"""
        schema = SimResult.model_json_schema()
        assert "$defs" in schema and "LognormalLength" in schema["$defs"]
