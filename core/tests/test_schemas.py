"""schemas 模块单测：可判别联合、校验边界、不可变性、序列化往返。"""

from typing import ClassVar

import pytest
from pydantic import TypeAdapter, ValidationError

from simcore.schemas import (
    AbortedRequest,
    Arrival,
    EngineConfig,
    FixedLength,
    LengthDist,
    LognormalLength,
    Metrics,
    PerfModelConfig,
    PoissonLength,
    RequestRecord,
    SessionConfig,
    SimConfig,
    SimMeta,
    SimResult,
    SkippedRequest,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)

LENGTH_DIST = TypeAdapter[LengthDist](LengthDist)


def make_config(**engine_overrides: object) -> SimConfig:
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
        engine=EngineConfig.model_validate(
            {"max_running_requests": 8, "max_total_tokens": 4096, **engine_overrides}
        ),
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
        [
            {"max_running_requests": 0},
            {"max_total_tokens": 0},
            {"max_total_tokens": 63},
            {"max_total_tokens": 130},
            {"max_prefill_tokens": 0},
            {"page_size": 0},
        ],
    )
    def test_engine_bounds(self, overrides: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            make_config(**overrides)

    def test_engine_defaults_and_page_capacity(self) -> None:
        engine = make_config().engine
        assert (engine.page_size, engine.capacity_pages) == (64, 64)
        assert engine.max_prefill_tokens == 16_384
        assert engine.pd_mutually_exclusive is True
        assert engine.prefix_cache.l1.capacity_blocks == engine.capacity_pages
        assert engine.model_dump() == {
            "max_running_requests": 8,
            "max_total_tokens": 4096,
            "max_prefill_tokens": 16_384,
            "page_size": 64,
            "pd_mutually_exclusive": True,
        }
        assert make_config(max_total_tokens=128).engine.capacity_pages == 2
        assert make_config(max_total_tokens=128, page_size=1).engine.capacity_pages == 128
        assert make_config(pd_mutually_exclusive=False).engine.pd_mutually_exclusive is False

    @pytest.mark.parametrize("legacy", ["max_batch_size", "kv_token_budget", "prefix_cache"])
    def test_legacy_capacity_inputs_rejected(self, legacy: str) -> None:
        with pytest.raises(ValidationError, match=legacy):
            make_config(**{legacy: 1})

    def test_synthetic_output_limit(self) -> None:
        synthetic = make_config().workload.synthetic
        assert synthetic is not None and synthetic.max_new_tokens == 128
        with pytest.raises(ValidationError):
            synthetic.model_validate({**synthetic.model_dump(), "max_new_tokens": 0})

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
                {"max_running_requests": 8, "max_total_tokens": 4096, "max_batch_tokens": 1024}
            )

    def test_session_without_cross_session_field_roundtrips(self) -> None:
        payload = make_config().model_dump()
        session = payload["workload"]["synthetic"]["session"]
        assert "allow_cross_session_shared_prefix" not in session
        parsed = SimConfig.model_validate(payload)
        assert (
            "allow_cross_session_shared_prefix"
            not in parsed.model_dump()["workload"]["synthetic"]["session"]
        )

    @pytest.mark.parametrize("value", [True, False])
    def test_cross_session_field_is_rejected_as_extra(self, value: bool) -> None:
        payload = make_config().model_dump()
        payload["workload"]["synthetic"]["session"]["allow_cross_session_shared_prefix"] = value
        with pytest.raises(ValidationError) as exc_info:
            SimConfig.model_validate(payload)
        assert any(
            error["type"] == "extra_forbidden"
            and error["loc"]
            == ("workload", "synthetic", "session", "allow_cross_session_shared_prefix")
            for error in exc_info.value.errors()
        )

    def test_session_content_seed_rejected(self) -> None:
        with pytest.raises(ValidationError, match="content_seed"):
            SessionConfig.model_validate({"content_seed": 123})

    def test_config_is_frozen(self) -> None:
        cfg = make_config()
        with pytest.raises(ValidationError):
            cfg.engine.max_running_requests = 16

    def test_roundtrip_is_lossless(self) -> None:
        cfg = make_config()
        assert SimConfig.model_validate(cfg.model_dump()) == cfg


class TestArrival:
    def test_valid(self) -> None:
        a = Arrival(request_id=0, sim_time=0.0, prompt_len=10, output_len=5)
        assert (a.request_id, a.sim_time, a.max_new_tokens) == (0, 0.0, 128)
        assert (
            Arrival(
                request_id=1, sim_time=0.0, prompt_len=10, output_len=5, max_new_tokens=2
            ).max_new_tokens
            == 2
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"request_id": -1},
            {"sim_time": -0.1},
            {"prompt_len": 0},
            {"output_len": 0},
            {"max_new_tokens": 0},
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
                num_completed=1,
                num_aborted=0,
                num_skipped=0,
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
                num_completed=0,
                num_aborted=0,
                num_skipped=0,
                num_engine_steps=0,
                sim_start=0.0,
                sim_end=0.0,
            ),
            metrics=Metrics(
                ttft_mean=None, ttft_p50=None, ttft_p99=None, throughput=None, makespan=0.0
            ),
        )
        assert SimResult.model_validate(result.model_dump()) == result
        assert result.requests == []
        assert result.aborted_requests == []
        assert result.skipped_requests == []
        assert result.warnings == []

    def test_failed_and_skipped_result_roundtrip(self) -> None:
        result = SimResult(
            meta=SimMeta(
                config=make_config(),
                num_requests=2,
                num_completed=0,
                num_aborted=1,
                num_skipped=1,
                num_engine_steps=0,
                sim_start=1.0,
                sim_end=1.0,
            ),
            metrics=Metrics(
                ttft_mean=None, ttft_p50=None, ttft_p99=None, throughput=None, makespan=0.0
            ),
            aborted_requests=[
                AbortedRequest(
                    id=0,
                    arrival=1.0,
                    failed_at=1.0,
                    reason="input_exceeds_kv_capacity",
                    generated_tokens=0,
                    retry_count=0,
                )
            ],
            skipped_requests=[SkippedRequest(id=1, parent_request_id=0)],
        )
        assert SimResult.model_validate_json(result.model_dump_json()) == result
        assert "arrival" not in result.skipped_requests[0].model_dump()

    def test_result_counts_must_match(self) -> None:
        with pytest.raises(ValidationError, match="数量之和"):
            SimMeta(
                config=make_config(),
                num_requests=1,
                num_completed=0,
                num_aborted=0,
                num_skipped=0,
                num_engine_steps=0,
                sim_start=0.0,
                sim_end=0.0,
            )

    def test_json_schema_generates(self) -> None:
        """OpenAPI 导出依赖此能力（可判别联合易在此处出错）。"""
        schema = SimResult.model_json_schema()
        assert "$defs" in schema and "LognormalLength" in schema["$defs"]
