"""schemas 模块单测：可判别联合、校验边界、不可变性、序列化往返。"""

from typing import ClassVar

import pytest
from pydantic import TypeAdapter, ValidationError

from simcore.schemas import (
    AbortedRequest,
    Arrival,
    AttentionGroup,
    EngineConfig,
    FixedLength,
    InstanceResult,
    LengthDist,
    LognormalLength,
    Metrics,
    ModelSpec,
    PerfModelConfig,
    PoissonLength,
    RequestRecord,
    SessionConfig,
    SimConfig,
    SimMeta,
    SimResult,
    SkippedRequest,
    StorageSpec,
    SyntheticWorkloadConfig,
    WorkloadConfig,
    validate_kv_config,
)

LENGTH_DIST = TypeAdapter[LengthDist](LengthDist)


def make_config(**engine_overrides: object) -> SimConfig:
    """构造一个合法的最小配置，供各用例复用。"""
    return SimConfig(
        num_instances=1,
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=4,
                session_rate=10.0,
                prompt_len=FixedLength(value=128),
                turn_increment_len=FixedLength(value=8),
                output_len=FixedLength(value=16),
                seed=0,
            )
        ),
        engine=EngineConfig.model_validate(
            {"max_running_requests": 8, "max_total_tokens": 4096, **engine_overrides}
        ),
        perf_model=PerfModelConfig(prefill_c1=0.5, decode_d1=10.0, decode_d2=0.001),
    )


def test_attention_group_validation_rejects_incomplete_and_misaligned_layouts() -> None:
    base = dict(name="mixed", num_layers=2, hidden_size=16, num_attention_heads=2,
                num_kv_heads=1, head_dim=8, ffn_intermediate_size=32, vocab_size=16)
    gqa = AttentionGroup(kind="gqa", layers=1, num_attention_heads=2, num_kv_heads=1, head_dim=8)
    with pytest.raises(ValidationError, match="注意力组层数之和"):
        ModelSpec(**base, attention_layout="grouped", attention_groups=[gqa])
    with pytest.raises(ValidationError, match="DSA 须配置"):
        AttentionGroup(kind="mla_dsa", layers=1, num_attention_heads=2,
                       q_rank=4, kv_rank=4, rope_dim=2, nope_dim=6, value_dim=8)
    with pytest.raises(ValidationError, match="GQA 只配置"):
        AttentionGroup(kind="gqa", layers=2, num_attention_heads=2, num_kv_heads=1,
                       head_dim=8, kv_rank=4)
    model = ModelSpec(**base, attention_layout="grouped", attention_groups=[gqa, gqa])
    assert model.attention_groups == [gqa, gqa]


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
            {"l2_capacity_tokens": -64},
            {"l2_capacity_tokens": 1},
            {"l3_capacity_tokens": 1},
            {"l3_capacity_tokens": 64},
            {"l3_prefetch_min_tokens": -1},
        ],
    )
    def test_engine_bounds(self, overrides: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            make_config(**overrides)

    def test_engine_defaults_and_page_capacity(self) -> None:
        engine = make_config().engine
        assert (engine.page_size, engine.capacity_pages) == (64, 64)
        assert engine.max_prefill_tokens == 16_384
        assert engine.l3_prefetch_min_tokens == 256
        assert engine.pd_mutually_exclusive is True
        assert engine.model_dump() == {
            "max_running_requests": 8,
            "max_total_tokens": 4096,
            "max_prefill_tokens": 16_384,
            "page_size": 64,
            "l2_capacity_tokens": 0,
            "l3_capacity_tokens": 0,
            "l3_prefetch_min_tokens": 256,
            "pd_mutually_exclusive": True,
        }
        assert make_config(l3_prefetch_min_tokens=0).engine.l3_prefetch_min_tokens == 0
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
        perf = make_config().perf_model
        assert (
            perf.prefill_c2
            == perf.l2_latency_ms
            == perf.l3_latency_ms
            == perf.l3_lookup_latency_ms
            == 0.0
        )
        assert StorageSpec(l1_available_gb=1).l3_lookup_latency_ms == 0.0
        assert perf.kv_bytes_per_token is None
        assert perf.l2_to_l1_bandwidth_bytes_per_ms is None
        assert perf.l3_to_l2_bandwidth_bytes_per_ms is None

    @pytest.mark.parametrize(
        ("field", "invalid"),
        [
            ("kv_bytes_per_token", 0),
            ("kv_bytes_per_token", -1),
            ("kv_bytes_per_token", 1.5),
            ("l2_to_l1_bandwidth_bytes_per_ms", 0),
            ("l3_to_l2_bandwidth_bytes_per_ms", -1),
            ("l2_to_l1_bandwidth_bytes_per_ms", float("inf")),
            ("l3_to_l2_bandwidth_bytes_per_ms", float("nan")),
            ("l2_latency_ms", -1),
            ("l3_latency_ms", -1),
            ("l2_latency_ms", float("inf")),
            ("l3_latency_ms", float("nan")),
            ("l3_lookup_latency_ms", -1),
            ("l3_lookup_latency_ms", float("inf")),
        ],
    )
    def test_invalid_tiered_perf_coefficients_rejected(self, field: str, invalid: float) -> None:
        payload = make_config().perf_model.model_dump()
        with pytest.raises(ValidationError):
            PerfModelConfig.model_validate({**payload, field: invalid})

    @pytest.mark.parametrize("invalid", [-1, float("inf"), float("nan")])
    def test_invalid_storage_lookup_latency_rejected(self, invalid: float) -> None:
        with pytest.raises(ValidationError):
            StorageSpec(l1_available_gb=1, l3_lookup_latency_ms=invalid)

    @pytest.mark.parametrize(
        ("tier", "perf_updates", "missing"),
        [
            ("l2", {}, "kv_bytes_per_token"),
            ("l2", {"kv_bytes_per_token": 16}, "l2_to_l1_bandwidth_bytes_per_ms"),
            (
                "l3",
                {"kv_bytes_per_token": 16, "l2_to_l1_bandwidth_bytes_per_ms": 128},
                "l3_to_l2_bandwidth_bytes_per_ms",
            ),
        ],
    )
    def test_enabled_tier_requires_transfer_params(
        self, tier: str, perf_updates: dict[str, float], missing: str
    ) -> None:
        payload = make_config().model_dump()
        payload["engine"]["l2_capacity_tokens"] = 64
        if tier == "l3":
            payload["engine"]["l3_capacity_tokens"] = 128
        payload["perf_model"].update(perf_updates)
        with pytest.raises(ValidationError, match=missing):
            SimConfig.model_validate(payload)

    def test_shared_kv_config_validator(self) -> None:
        engine = EngineConfig(max_running_requests=1, max_total_tokens=128, l2_capacity_tokens=64)
        perf = PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=0)
        with pytest.raises(ValueError, match="kv_bytes_per_token"):
            validate_kv_config(engine, perf)
        valid_perf = perf.model_copy(
            update={"kv_bytes_per_token": 16, "l2_to_l1_bandwidth_bytes_per_ms": 128.0}
        )
        validate_kv_config(engine, valid_perf)

    def test_enabled_tiers_roundtrip_and_disabled_tiers_need_no_bandwidth(self) -> None:
        payload = make_config().model_dump()
        payload["engine"].update(l2_capacity_tokens=64, l3_capacity_tokens=128)
        payload["perf_model"].update(
            kv_bytes_per_token=16,
            l2_to_l1_bandwidth_bytes_per_ms=128,
            l3_to_l2_bandwidth_bytes_per_ms=64,
            l2_latency_ms=0.5,
            l3_latency_ms=1.5,
        )
        config = SimConfig.model_validate(payload)
        assert SimConfig.model_validate_json(config.model_dump_json()) == config
        payload["engine"].update(l2_capacity_tokens=0, l3_capacity_tokens=0)
        payload["perf_model"].update(
            kv_bytes_per_token=None,
            l2_to_l1_bandwidth_bytes_per_ms=None,
            l3_to_l2_bandwidth_bytes_per_ms=None,
        )
        assert SimConfig.model_validate(payload).engine.l2_capacity_tokens == 0

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

    def test_session_rate_and_request_rate_are_mode_specific(self) -> None:
        workload = make_config().workload.synthetic
        assert workload is not None
        assert workload.arrival_rate is None and workload.session_rate == 10
        payload = workload.model_dump()
        with pytest.raises(ValidationError, match="session_rate"):
            SyntheticWorkloadConfig.model_validate({**payload, "session_rate": None})
        with pytest.raises(ValidationError, match="session_rate"):
            SyntheticWorkloadConfig.model_validate({**payload, "arrival_rate": 10})
        open_payload = {
            **payload,
            "arrival_mode": "open_loop",
            "session_rate": None,
            "turn_increment_len": {"dist": "fixed", "value": 8},
            "session": {"think_time_ms": 0},
        }
        with pytest.raises(ValidationError, match="arrival_rate"):
            SyntheticWorkloadConfig.model_validate(open_payload)
        assert (
            SyntheticWorkloadConfig.model_validate(
                {**open_payload, "arrival_rate": 10}
            ).arrival_rate
            == 10
        )

    def test_both_modes_require_turn_increment_and_reject_old_hit_target(self) -> None:
        workload = make_config().workload.synthetic
        assert workload is not None
        payload = workload.model_dump()
        with pytest.raises(ValidationError, match="turn_increment_len"):
            SyntheticWorkloadConfig.model_validate({**payload, "turn_increment_len": None})
        open_payload = {
            **payload,
            "arrival_mode": "open_loop",
            "session_rate": None,
            "arrival_rate": 10,
            "session": {},
        }
        with pytest.raises(ValidationError, match="turn_increment_len"):
            SyntheticWorkloadConfig.model_validate({**open_payload, "turn_increment_len": None})
        with pytest.raises(ValidationError, match="target_ideal_prefix_hit_rate"):
            SyntheticWorkloadConfig.model_validate(
                {
                    **payload,
                    "session": {
                        "num_sessions": 2,
                        "target_ideal_prefix_hit_rate": 0.5,
                    },
                }
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
                wall_clock_ms=0.0,
            )

    def test_negative_wall_clock_rejected(self) -> None:
        with pytest.raises(ValidationError, match="wall_clock_ms"):
            SimMeta(
                config=make_config(),
                num_requests=0,
                num_completed=0,
                num_aborted=0,
                num_skipped=0,
                num_engine_steps=0,
                sim_start=0,
                sim_end=0,
                wall_clock_ms=-1,
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
                wall_clock_ms=0.0,
            ),
            metrics=Metrics(
                ttft_mean=None, ttft_p50=None, ttft_p99=None, throughput=None, makespan=0.0
            ),
            instances=[
                InstanceResult(
                    instance_id=0,
                    num_completed=0,
                    num_aborted=0,
                    num_skipped=0,
                    num_engine_steps=0,
                    metrics=Metrics(
                        ttft_mean=None, ttft_p50=None, ttft_p99=None, throughput=None, makespan=0.0
                    ),
                )
            ],
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
                wall_clock_ms=0.0,
            ),
            metrics=Metrics(
                ttft_mean=None, ttft_p50=None, ttft_p99=None, throughput=None, makespan=0.0
            ),
            instances=[
                InstanceResult(
                    instance_id=0,
                    num_completed=0,
                    num_aborted=1,
                    num_skipped=1,
                    num_engine_steps=0,
                    metrics=Metrics(
                        ttft_mean=None, ttft_p50=None, ttft_p99=None, throughput=None, makespan=0.0
                    ),
                )
            ],
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
                wall_clock_ms=0.0,
            )

    def test_json_schema_generates(self) -> None:
        """OpenAPI 导出依赖此能力（可判别联合易在此处出错）。"""
        schema = SimResult.model_json_schema()
        assert "$defs" in schema and "LognormalLength" in schema["$defs"]
