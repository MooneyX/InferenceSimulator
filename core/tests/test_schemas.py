"""schemas 模块单测：可判别联合、校验边界、不可变性、序列化往返。"""

from typing import Any, ClassVar, Literal

import pytest
from pydantic import TypeAdapter, ValidationError

from simcore.runtime import RuntimeConfig, TransferLink, TransferLinks
from simcore.schemas import (
    AbortedRequest,
    Arrival,
    AttentionGroup,
    AuxWeightDtype,
    EmpiricalTimingConfig,
    EngineConfig,
    FixedLength,
    HardwareSpec,
    IHCSpec,
    InstanceResult,
    LengthDist,
    LognormalLength,
    Metrics,
    ModelSpec,
    NormSpec,
    PerfModelConfig,
    PhysicalInferenceConfig,
    PhysicalL3Config,
    PoissonLength,
    RequestRecord,
    RouterSpec,
    SchedulerConfig,
    SessionConfig,
    SimConfig,
    SimMeta,
    SimResult,
    SkippedRequest,
    StorageSpec,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)

LENGTH_DIST = TypeAdapter[LengthDist](LengthDist)

TOY_MODEL = ModelSpec(
    name="toy", num_layers=2, hidden_size=16, num_attention_heads=2,
    num_kv_heads=1, head_dim=8, ffn_intermediate_size=32, vocab_size=16,
)
TOY_HARDWARE = HardwareSpec(
    name="toy", memory_bytes_per_gpu=2_000_000,
    effective_fp16_tflops=1, effective_hbm_gb_s=1,
)


def physical_config(
    *,
    workload: WorkloadConfig,
    engine: EngineConfig,
    perf_model: PerfModelConfig,
    num_instances: int = 1,
) -> SimConfig:
    """用模型解析得到的每页 KV 字节数和权重字节数还原测试所需容量。"""
    from simcore.capacity import resolve_config

    scheduler = SchedulerConfig(
        max_running_requests=engine.max_running_requests,
        max_prefill_tokens=engine.max_prefill_tokens,
        pd_mutually_exclusive=engine.pd_mutually_exclusive,
        l3_prefetch_min_tokens=engine.l3_prefetch_min_tokens,
    )
    timing = EmpiricalTimingConfig(
        prefill_c1=perf_model.prefill_c1, prefill_c2=perf_model.prefill_c2,
        decode_d1=perf_model.decode_d1, decode_d2=perf_model.decode_d2,
    )
    storage = StorageSpec(l1_available_gb=0.001)
    base = SimConfig(
        workload=workload, scheduler=scheduler, num_instances=num_instances,
        inference=PhysicalInferenceConfig(
            page_size=engine.page_size, max_output_tokens=engine.max_output_tokens,
            model=TOY_MODEL, hardware=TOY_HARDWARE, storage=storage, timing=timing,
        ),
    )
    budget = resolve_config(base).capacity
    kv_bytes_per_token = budget.kv_bytes_per_token
    l1_bytes = budget.weights_resident_bytes + engine.max_total_tokens * kv_bytes_per_token
    l2_bytes = engine.l2_capacity_tokens * kv_bytes_per_token
    l3_bytes = engine.l3_capacity_tokens * kv_bytes_per_token
    physical_storage = StorageSpec(
        l1_available_gb=l1_bytes / 1_000_000_000,
        l2_available_gb=l2_bytes / 1_000_000_000,
        l2_to_l1_gb_s=(
            (perf_model.l2_to_l1_bandwidth_bytes_per_ms or 1_000_000) / 1_000_000
            if l2_bytes else None
        ),
        l3_to_l2_gb_s=(
            (perf_model.l3_to_l2_bandwidth_bytes_per_ms or 1_000_000) / 1_000_000
            if l3_bytes else None
        ),
        l2_latency_ms=perf_model.l2_latency_ms,
        l3_latency_ms=perf_model.l3_latency_ms,
    )
    return base.model_copy(update={
        "inference": base.inference.model_copy(update={"storage": physical_storage}),
        "shared_l3": PhysicalL3Config(
            available_gb=l3_bytes / 1_000_000_000,
            lookup_latency_ms=perf_model.l3_lookup_latency_ms,
        ),
    })


def runtime_config(engine: EngineConfig, perf: PerfModelConfig) -> RuntimeConfig:
    """直接 Engine 单测保留显式运行参数，不经过公开 SimConfig。"""
    l2 = (
        TransferLink(
            engine.page_size * perf.kv_bytes_per_token
            / perf.l2_to_l1_bandwidth_bytes_per_ms,
            perf.l2_latency_ms,
        )
        if engine.l2_capacity_tokens and perf.kv_bytes_per_token
        and perf.l2_to_l1_bandwidth_bytes_per_ms else None
    )
    l3 = (
        TransferLink(
            engine.page_size * perf.kv_bytes_per_token
            / perf.l3_to_l2_bandwidth_bytes_per_ms,
            perf.l3_latency_ms,
        )
        if engine.l3_capacity_tokens and perf.kv_bytes_per_token
        and perf.l3_to_l2_bandwidth_bytes_per_ms else None
    )
    return RuntimeConfig(engine=engine, perf_model=perf, links=TransferLinks(l2, l3))


def metric_inputs() -> dict[str, Any]:
    from simcore.kv_cache import L1Snapshot, TierSnapshot
    from simcore.schemas import WorkloadPlan

    return {
        "plan": WorkloadPlan({}, (), {}), "page_size": 1, "workload_mode": "synthetic",
        "cache_snapshot": L1Snapshot(0, 0, 0, 0, 0, 0),
        "l2_snapshot": TierSnapshot(0, 0, 0, 0, 0, 0, 0),
        "l3_snapshot": TierSnapshot(0, 0, 0, 0, 0, 0, 0),
        "l1_evictions": 0, "l1_peak_resident_blocks": 0, "promotions": 0, "demotions": 0,
        "restore_bytes": 0, "restore_time_ms": 0.0,
        "earliest_sent_arrival": None, "latest_sent_arrival": None, "num_sent": 0,
        "replayed_prefill_tokens": 0,
    }


def make_config(**engine_overrides: object) -> SimConfig:
    """构造一个合法的最小配置，供各用例复用。"""
    return physical_config(
        num_instances=1,
        workload=WorkloadConfig(
            session_rate=10.0,
            synthetic=SyntheticWorkloadConfig(
                num_requests=4,
                prompt_len=FixedLength(value=128),
                turn_increment_len=FixedLength(value=8),
                output_len=FixedLength(value=16),
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
    assert not gqa.output_gate
    with pytest.raises(ValidationError, match="输出门控当前仅支持 MLA"):
        AttentionGroup(
            kind="gqa",
            layers=1,
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=8,
            output_gate=True,
        )


def test_shared_expert_configuration_requires_moe_and_explicit_dimension() -> None:
    base = dict(
        name="shared",
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        ffn_intermediate_size=32,
        vocab_size=16,
    )
    model = ModelSpec(**base)
    assert model.num_shared_experts == 0
    assert model.shared_expert_intermediate_size is None
    with pytest.raises(ValidationError, match="共享专家需要 MoE 层"):
        ModelSpec(**base, num_shared_experts=1, shared_expert_intermediate_size=8)
    with pytest.raises(ValidationError, match="共享专家需要 MoE 层"):
        ModelSpec(**base, moe_layers=1, num_shared_experts=1)
    with pytest.raises(ValidationError, match="未启用共享专家"):
        ModelSpec(**base, moe_layers=1, shared_expert_intermediate_size=8)
    with pytest.raises(ValidationError):
        ModelSpec(**base, moe_layers=1, num_shared_experts=1, shared_expert_intermediate_size=0)
    shared = ModelSpec(
        **base,
        moe_layers=1,
        num_experts=4,
        experts_per_token=2,
        expert_intermediate_size=16,
        num_shared_experts=2,
        shared_expert_intermediate_size=8,
    )
    assert ModelSpec.model_validate_json(shared.model_dump_json()) == shared


def test_auxiliary_weight_defaults_and_explicit_roundtrip() -> None:
    model = ModelSpec(
        name="auxiliary", num_layers=2, hidden_size=16, num_attention_heads=2,
        num_kv_heads=1, head_dim=8, ffn_intermediate_size=32, vocab_size=16,
    )
    assert model.norms.model_dump() == {
        "kind": "rms", "block_per_layer": 2, "final_norm": True, "dtype": None,
    }
    assert model.router.model_dump() == {
        "dtype": None, "correction_bias": False, "correction_dtype": "fp32",
    }
    assert model.ihc is None
    group = AttentionGroup(
        kind="mla_dsa", layers=2, num_attention_heads=2, q_rank=4, kv_rank=4,
        rope_dim=2, nope_dim=6, value_dim=8, index_heads=2, index_dim=128,
        topk=4, full_indexer_layers=1,
    )
    assert group.qk_norm and group.norm_dtype is None
    assert group.index_norm == "layer" and group.index_norm_dtype is None
    assert group.sink_dtype is None
    payload = model.model_dump() | {
        "moe_layers": 1, "num_experts": 4, "experts_per_token": 2,
        "expert_intermediate_size": 8, "attention_layout": "grouped",
        "norms": {"kind": "layer", "block_per_layer": 0, "final_norm": False, "dtype": "fp32"},
        "router": {"dtype": "fp32", "correction_bias": True, "correction_dtype": "fp16"},
        "ihc": {"mult": 1},
        "attention_groups": [group.model_dump() | {
            "qk_norm": False, "norm_dtype": "fp16", "index_norm": "rms",
            "index_norm_dtype": "fp32", "sink_dtype": "bf16",
        }],
    }
    explicit = ModelSpec.model_validate(payload)
    assert explicit.ihc == IHCSpec(mult=1, dtype="fp32")
    assert ModelSpec.model_validate_json(explicit.model_dump_json()) == explicit
    assert ModelSpec.model_validate(explicit.model_dump() | {"ihc": None}).ihc is None


@pytest.mark.parametrize("spec,payload", [
    (NormSpec, {"kind": "batch"}), (NormSpec, {"block_per_layer": -1}),
    (NormSpec, {"block_per_layer": None}), (NormSpec, {"dtype": "fp8"}),
    (RouterSpec, {"dtype": "fp8"}), (RouterSpec, {"correction_dtype": "fp8"}),
    (RouterSpec, {"correction_dtype": None}), (IHCSpec, {}),
    (IHCSpec, {"mult": 0}), (IHCSpec, {"mult": -1}),
    (IHCSpec, {"mult": 1.5}), (IHCSpec, {"mult": None}),
    (IHCSpec, {"mult": 1, "dtype": "fp8"}), (IHCSpec, {"mult": 1, "dtype": None}),
])
def test_auxiliary_weight_spec_rejects_invalid_dtype_counts_and_null(
    spec: type[NormSpec] | type[RouterSpec] | type[IHCSpec], payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        spec.model_validate(payload)


@pytest.mark.parametrize("dtype", [None, "bf16", "fp16", "fp32"])
def test_auxiliary_dtype_accepts_supported_precision_and_nullable_inheritance(
    dtype: AuxWeightDtype | None,
) -> None:
    assert NormSpec(dtype=dtype).dtype == dtype
    assert RouterSpec(dtype=dtype).dtype == dtype
    group = AttentionGroup(
        kind="mla_dsa", layers=1, num_attention_heads=2, q_rank=4, kv_rank=4,
        rope_dim=2, nope_dim=6, value_dim=8, index_heads=2, index_dim=128,
        topk=4, full_indexer_layers=1,
        norm_dtype=dtype, index_norm_dtype=dtype, sink_dtype=dtype,
    )
    assert group.norm_dtype == group.index_norm_dtype == group.sink_dtype == dtype
    if dtype is not None:
        assert IHCSpec(mult=1, dtype=dtype).dtype == dtype
        assert RouterSpec(correction_dtype=dtype).correction_dtype == dtype


@pytest.mark.parametrize("field,invalid", [
    ("norm_dtype", "fp8"), ("index_norm_dtype", "fp8"), ("sink_dtype", "fp8"),
    ("index_norm", "batch"), ("index_norm", None),
])
def test_attention_auxiliary_spec_rejects_invalid_precision_or_norm(
    field: str, invalid: str | None,
) -> None:
    with pytest.raises(ValidationError, match=field):
        AttentionGroup.model_validate({
            "kind": "mla_dsa", "layers": 1, "num_attention_heads": 2,
            "q_rank": 4, "kv_rank": 4, "rope_dim": 2, "nope_dim": 6, "value_dim": 8,
            "index_heads": 2, "index_dim": 128, "topk": 4, "full_indexer_layers": 1,
            field: invalid,
        })


@pytest.mark.parametrize("kind", ["gqa", "mla"])
@pytest.mark.parametrize("update", [
    {"index_norm": "none"}, {"index_norm": "rms"}, {"index_norm_dtype": "fp32"},
])
def test_index_norm_overrides_require_dsa(
    kind: Literal["gqa", "mla"], update: dict[str, object],
) -> None:
    group = (
        AttentionGroup(kind="gqa", layers=1, num_attention_heads=2, num_kv_heads=1, head_dim=8)
        if kind == "gqa" else
        AttentionGroup(kind="mla", layers=1, num_attention_heads=2,
                       q_rank=4, kv_rank=4, rope_dim=2, nope_dim=6, value_dim=8)
    )
    assert group.index_norm == "layer" and group.index_norm_dtype is None
    with pytest.raises(ValidationError, match="仅 MLA\\+DSA 可以配置索引器归一化"):
        AttentionGroup.model_validate(group.model_dump() | update)


@pytest.mark.parametrize("router", [
    {"dtype": "fp32"}, {"correction_bias": True}, {"correction_dtype": "bf16"},
])
def test_non_default_router_requires_moe(router: dict[str, object]) -> None:
    payload = dict(
        name="router", num_layers=1, hidden_size=16, num_attention_heads=2,
        num_kv_heads=1, head_dim=8, ffn_intermediate_size=32, vocab_size=16,
    )
    assert ModelSpec.model_validate(payload | {"router": {}}).router == RouterSpec()
    with pytest.raises(ValidationError, match="Router 辅助配置需要 MoE 层"):
        ModelSpec.model_validate(payload | {"router": router})
    assert ModelSpec.model_validate(payload | {"moe_layers": 1, "router": router}).router == (
        RouterSpec.model_validate(router)
    )
    for field in ("norms", "router"):
        with pytest.raises(ValidationError, match=field):
            ModelSpec.model_validate(payload | {field: None})


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

    def test_trace_payload_required_and_parent_defaults(self) -> None:
        with pytest.raises(ValidationError, match="trace"):
            WorkloadConfig(mode="trace", session_rate=1)
        workload = WorkloadConfig.model_validate({
            "mode": "trace", "session_rate": 1, "trace": {},
        })
        assert workload.synthetic is None
        assert workload.arrival_mode == "closed_loop" and workload.seed == 0
        assert workload.target_qps is None
        assert workload.trace is not None and workload.trace.model_dump() == {"plan_id": None}
        assert WorkloadConfig.model_validate_json(workload.model_dump_json()) == workload

    @pytest.mark.parametrize("mode", ["synthetic", "trace"])
    def test_workload_payloads_are_mutually_exclusive(self, mode: str) -> None:
        payload = make_config().workload.model_dump()
        payload.update(mode=mode, trace={})
        with pytest.raises(ValidationError):
            WorkloadConfig.model_validate(payload)

    @pytest.mark.parametrize("field,invalid", [
        ("arrival_mode", "legacy"), ("arrival_mode", "session"),
        ("session_rate", 0), ("session_rate", float("inf")),
        ("seed", -1), ("target_qps", 0), ("target_qps", float("nan")),
    ])
    def test_parent_rejects_invalid_values(self, field: str, invalid: object) -> None:
        payload = {
            "mode": "trace", "arrival_mode": "open_loop", "session_rate": 1,
            "trace": {}, field: invalid,
        }
        with pytest.raises(ValidationError, match=field):
            WorkloadConfig.model_validate(payload)

    @pytest.mark.parametrize("source", ["synthetic", "trace"])
    @pytest.mark.parametrize("arrival_mode", ["closed_loop", "open_loop"])
    def test_four_modes_enforce_parent_rate_constraints(
        self, source: str, arrival_mode: str,
    ) -> None:
        synthetic = make_config().workload.synthetic
        assert synthetic is not None
        body = synthetic.model_dump() if source == "synthetic" else {}
        if source == "synthetic" and arrival_mode == "open_loop":
            body["session"] = {}
        payload = {"mode": source, "arrival_mode": arrival_mode, source: body}
        request_clock = source == "synthetic" and arrival_mode == "open_loop"
        required = "target_qps" if request_clock else "session_rate"
        with pytest.raises(ValidationError, match=required):
            WorkloadConfig.model_validate(payload)
        payload[required] = 10
        parsed = WorkloadConfig.model_validate(payload)
        assert parsed.seed == 0 and parsed.arrival_mode == arrival_mode
        assert parsed.session_rate == (None if request_clock else 10)
        assert parsed.target_qps == (10 if request_clock else None)
        assert WorkloadConfig.model_validate_json(parsed.model_dump_json()) == parsed
        forbidden = "session_rate" if request_clock else "target_qps"
        if source == "trace" and arrival_mode == "open_loop":
            assert WorkloadConfig.model_validate(payload | {"target_qps": 5}).target_qps == 5
        else:
            with pytest.raises(ValidationError):
                WorkloadConfig.model_validate(payload | {forbidden: 5})

    def test_synthetic_session_count_is_completed_only_by_closed_loop_parent(self) -> None:
        synthetic = make_config().workload.synthetic
        assert synthetic is not None
        body = synthetic.model_dump() | {"session": {}}
        assert SyntheticWorkloadConfig.model_validate(body).session.num_sessions is None
        closed = WorkloadConfig(
            session_rate=1, synthetic=SyntheticWorkloadConfig.model_validate(body),
        )
        assert closed.synthetic is not None and closed.synthetic.session.num_sessions == 4
        opened = WorkloadConfig(arrival_mode="open_loop", target_qps=1,
                                synthetic=SyntheticWorkloadConfig.model_validate(body))
        assert opened.synthetic is not None and opened.synthetic.session.num_sessions is None
        for sessions in (0, 5):
            with pytest.raises(ValidationError, match="num_sessions"):
                WorkloadConfig.model_validate({
                    "session_rate": 1, "synthetic": body | {"session": {"num_sessions": sessions}},
                })
        for session in ({"num_sessions": 1}, {"think_time_ms": 1}):
            with pytest.raises(ValidationError):
                WorkloadConfig.model_validate({
                    "arrival_mode": "open_loop", "target_qps": 1,
                    "synthetic": body | {"session": session},
                })

    @pytest.mark.parametrize("source", ["synthetic", "trace"])
    @pytest.mark.parametrize("field", ["arrival_mode", "seed", "session_rate", "target_qps",
                                       "arrival_rate", "max_new_tokens"])
    def test_removed_nested_workload_fields_are_rejected(self, source: str, field: str) -> None:
        synthetic = make_config().workload.synthetic
        assert synthetic is not None
        body = synthetic.model_dump() if source == "synthetic" else {}
        with pytest.raises(ValidationError, match=field):
            WorkloadConfig.model_validate({
                "mode": source, "session_rate": 1, source: body | {field: 1},
            })

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
        payload = EngineConfig(max_running_requests=8, max_total_tokens=4096).model_dump()
        payload.update(overrides)
        with pytest.raises(ValidationError):
            EngineConfig.model_validate(payload)

    def test_engine_defaults_and_page_capacity(self) -> None:
        from simcore.capacity import prepare_runtime

        config = make_config()
        assert set(config.model_dump()) == {
            "num_instances", "workload", "scheduler", "inference", "shared_l3",
        }
        assert not hasattr(config, "engine") and not hasattr(config, "perf_model")
        engine = prepare_runtime(config).engine
        assert (engine.page_size, engine.capacity_pages) == (64, 64)
        assert engine.max_prefill_tokens == 16_384
        assert engine.max_output_tokens == config.inference.max_output_tokens == 65_536
        assert engine.l3_prefetch_min_tokens == 256
        assert engine.pd_mutually_exclusive is True
        assert engine.model_dump() == {
            "max_running_requests": 8,
            "max_total_tokens": 4096,
            "max_prefill_tokens": 16_384,
            "max_output_tokens": 65_536,
            "page_size": 64,
            "l2_capacity_tokens": 0,
            "l3_capacity_tokens": 0,
            "l3_prefetch_min_tokens": 256,
            "pd_mutually_exclusive": True,
        }
        assert make_config(l3_prefetch_min_tokens=0).scheduler.l3_prefetch_min_tokens == 0
        assert prepare_runtime(make_config(max_total_tokens=128)).engine.capacity_pages == 2
        assert (
            prepare_runtime(make_config(max_total_tokens=128, page_size=1)).engine.capacity_pages
            == 128
        )
        assert make_config(pd_mutually_exclusive=False).scheduler.pd_mutually_exclusive is False

    @pytest.mark.parametrize("legacy", ["max_batch_size", "kv_token_budget", "prefix_cache"])
    def test_legacy_capacity_inputs_rejected(self, legacy: str) -> None:
        with pytest.raises(ValidationError, match=legacy):
            make_config(**{legacy: 1})

    @pytest.mark.parametrize("cap", [1, 64, 100, 65_536])
    def test_engine_output_limit_lives_on_physical_inference(self, cap: int) -> None:
        from simcore.capacity import prepare_runtime

        config = make_config(max_output_tokens=cap)
        assert config.inference.max_output_tokens == cap
        assert prepare_runtime(config).engine.max_output_tokens == cap
        assert "max_output_tokens" not in config.inference.storage.model_dump()

    @pytest.mark.parametrize("invalid", [0, -1, 1.5, None])
    def test_engine_output_limit_rejects_non_positive_or_non_integer_values(
        self, invalid: object,
    ) -> None:
        payload = make_config().model_dump()
        payload["inference"]["max_output_tokens"] = invalid
        with pytest.raises(ValidationError, match="max_output_tokens"):
            SimConfig.model_validate(payload)
        with pytest.raises(ValidationError, match="max_output_tokens"):
            EngineConfig.model_validate({
                "max_running_requests": 1, "max_total_tokens": 128,
                "max_output_tokens": invalid,
            })

    def test_prefill_c2_defaults_to_zero(self) -> None:
        from simcore.capacity import prepare_runtime

        config = make_config()
        inference = config.inference
        assert isinstance(inference.timing, EmpiricalTimingConfig)
        assert (inference.timing.prefill_c2 == inference.storage.l2_latency_ms
                == inference.storage.l3_latency_ms == 0.0)
        assert config.shared_l3 is not None and config.shared_l3.lookup_latency_ms == 0.0
        runtime = prepare_runtime(config)
        assert runtime.perf_model.kv_bytes_per_token == 64
        assert runtime.links.l2_to_l1 is None and runtime.links.l3_to_l2 is None

    @pytest.mark.parametrize("field,invalid", [
        ("l2_to_l1_gb_s", 0), ("l3_to_l2_gb_s", -1),
        ("l2_to_l1_gb_s", float("inf")), ("l3_to_l2_gb_s", float("nan")),
        ("l2_latency_ms", -1), ("l3_latency_ms", -1),
        ("l2_latency_ms", float("inf")), ("l3_latency_ms", float("nan")),
    ])
    def test_invalid_tiered_storage_fields_rejected(self, field: str, invalid: float) -> None:
        payload = make_config().model_dump()
        payload["inference"]["storage"][field] = invalid
        with pytest.raises(ValidationError, match=field):
            SimConfig.model_validate(payload)

    @pytest.mark.parametrize("invalid", [-1, float("inf"), float("nan")])
    def test_invalid_shared_lookup_latency_rejected(self, invalid: float) -> None:
        with pytest.raises(ValidationError, match="lookup_latency_ms"):
            PhysicalL3Config(lookup_latency_ms=invalid)

    @pytest.mark.parametrize("tier,missing", [
        ("l2", "L2.*L1"), ("l3", "L3.*L2"),
    ])
    def test_enabled_tier_requires_transfer_bandwidth(self, tier: str, missing: str) -> None:
        from simcore.capacity import resolve_config

        payload = make_config().model_dump()
        payload["inference"]["storage"].update(
            l2_available_gb=64 * 64 / 1_000_000_000, l2_to_l1_gb_s=1,
        )
        if tier == "l3":
            payload["shared_l3"]["available_gb"] = 128 * 64 / 1_000_000_000
            payload["inference"]["storage"]["l3_to_l2_gb_s"] = None
        else:
            payload["inference"]["storage"]["l2_to_l1_gb_s"] = None
        with pytest.raises(ValueError, match=missing):
            resolve_config(SimConfig.model_validate(payload))

    def test_enabled_tiers_roundtrip_and_disabled_tiers_need_no_bandwidth(self) -> None:
        from simcore.capacity import prepare_runtime

        payload = make_config().model_dump()
        payload["inference"]["storage"].update(
            l2_available_gb=64 * 64 / 1_000_000_000,
            l2_to_l1_gb_s=0.000128, l3_to_l2_gb_s=0.000064,
            l2_latency_ms=0.5, l3_latency_ms=1.5,
        )
        payload["shared_l3"]["available_gb"] = 128 * 64 / 1_000_000_000
        config = SimConfig.model_validate(payload)
        assert SimConfig.model_validate_json(config.model_dump_json()) == config
        runtime = prepare_runtime(config)
        assert runtime.engine.l2_capacity_tokens == 64
        assert runtime.engine.l3_capacity_tokens == 128
        assert runtime.links.l2_to_l1 is not None and runtime.links.l2_to_l1.transfer_ms == 32
        assert runtime.links.l3_to_l2 is not None and runtime.links.l3_to_l2.transfer_ms == 64
        payload["shared_l3"]["available_gb"] = 0
        payload["inference"]["storage"].update(
            l2_available_gb=0, l2_to_l1_gb_s=None, l3_to_l2_gb_s=None,
        )
        disabled = prepare_runtime(SimConfig.model_validate(payload))
        assert disabled.engine.l2_capacity_tokens == disabled.engine.l3_capacity_tokens == 0
        assert disabled.links.l2_to_l1 is None and disabled.links.l3_to_l2 is None

    @pytest.mark.parametrize(
        "field",
        ["prefill_c1", "prefill_c2", "decode_d1", "decode_d2"],
    )
    @pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("nan")])
    def test_negative_coefficients_rejected(self, field: str, invalid: float) -> None:
        payload = make_config().model_dump()
        payload["inference"]["timing"][field] = invalid
        with pytest.raises(ValidationError):
            SimConfig.model_validate(payload)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EngineConfig.model_validate(
                {"max_running_requests": 8, "max_total_tokens": 4096, "max_batch_tokens": 1024}
            )

    def test_closed_loop_is_default_and_common_fields_live_on_parent(self) -> None:
        workload = make_config().workload
        assert workload.arrival_mode == "closed_loop" and workload.seed == 0
        assert workload.session_rate == 10 and workload.target_qps is None
        assert workload.synthetic is not None
        assert set(workload.synthetic.model_dump()) == {
            "num_requests", "prompt_len", "turn_increment_len", "output_len", "session",
        }
        with pytest.raises(ValidationError, match="arrival_mode"):
            WorkloadConfig.model_validate(workload.model_dump() | {"arrival_mode": "session"})
        with pytest.raises(ValidationError, match="arrival_rate"):
            WorkloadConfig.model_validate(workload.model_dump() | {"arrival_rate": 10})

    @pytest.mark.parametrize("arrival_mode", ["closed_loop", "open_loop"])
    @pytest.mark.parametrize("missing", [True, False])
    def test_both_modes_require_turn_increment_and_reject_old_hit_target(
        self, arrival_mode: str, missing: bool,
    ) -> None:
        synthetic = make_config().workload.synthetic
        assert synthetic is not None
        body = synthetic.model_dump() | {"session": {}}
        if missing:
            del body["turn_increment_len"]
        else:
            body["turn_increment_len"] = None
        payload = {
            "arrival_mode": arrival_mode,
            "target_qps" if arrival_mode == "open_loop" else "session_rate": 10,
            "synthetic": body,
        }
        with pytest.raises(ValidationError, match="turn_increment_len"):
            WorkloadConfig.model_validate(payload)
        body["turn_increment_len"] = {"dist": "fixed", "value": 8}
        body["session"] = {"target_ideal_prefix_hit_rate": 0.5}
        with pytest.raises(ValidationError, match="target_ideal_prefix_hit_rate"):
            WorkloadConfig.model_validate(payload)

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
            cfg.scheduler.max_running_requests = 16
        with pytest.raises(ValidationError):
            cfg.inference.page_size = 1
        with pytest.raises(ValidationError):
            cfg.inference.storage.l1_available_gb = 0.001
        timing = cfg.inference.timing
        assert isinstance(timing, EmpiricalTimingConfig)
        with pytest.raises(ValidationError):
            timing.prefill_c1 = 1

    @pytest.mark.parametrize("field", ["engine", "perf_model", "model", "hardware", "storage"])
    def test_removed_top_level_configuration_fields_are_rejected(self, field: str) -> None:
        payload = make_config().model_dump() | {field: {}}
        with pytest.raises(ValidationError, match=field):
            SimConfig.model_validate(payload)

    @pytest.mark.parametrize("mode", ["manual", "unknown"])
    def test_removed_inference_modes_are_rejected(self, mode: str) -> None:
        payload = make_config().model_dump()
        payload["inference"]["mode"] = mode
        with pytest.raises(ValidationError, match="mode"):
            SimConfig.model_validate(payload)

    def test_removed_manual_l3_mode_is_rejected(self) -> None:
        payload = make_config().model_dump()
        payload["shared_l3"]["mode"] = "manual"
        with pytest.raises(ValidationError, match="mode"):
            SimConfig.model_validate(payload)

    @pytest.mark.parametrize("field", ["kv", "timing_coefficients"])
    def test_physical_inference_rejects_legacy_manual_fields(self, field: str) -> None:
        payload = make_config().model_dump()
        payload["inference"][field] = {}
        with pytest.raises(ValidationError, match=field):
            SimConfig.model_validate(payload)

    def test_physical_analytical_timing_requires_no_empirical_coefficients(self) -> None:
        payload = make_config().model_dump()
        payload["shared_l3"] = None
        payload["inference"] = {
            "mode": "physical", "page_size": 4,
            "model": {
                "name": "toy", "num_layers": 2, "hidden_size": 16,
                "num_attention_heads": 2, "num_kv_heads": 1,
                "head_dim": 8, "ffn_intermediate_size": 32, "vocab_size": 16,
            },
            "hardware": {
                "name": "toy", "memory_bytes_per_gpu": 2_000_000,
                "effective_fp16_tflops": 1, "effective_hbm_gb_s": 1,
            },
            "storage": {"l1_available_gb": 0.001},
            "timing": {"mode": "analytical"},
        }
        config = SimConfig.model_validate(payload)
        assert config.inference.timing.model_dump() == {"mode": "analytical"}
        assert config.inference.tp_size == config.inference.ep_size == 1
        assert SimConfig.model_validate_json(config.model_dump_json()) == config
        for field in ("kv", "prefill_c1"):
            inference = payload["inference"] | {field: {} if field == "kv" else 1}
            with pytest.raises(ValidationError, match=field):
                SimConfig.model_validate(payload | {"inference": inference})
        payload["inference"]["timing"]["prefill_c1"] = 1
        with pytest.raises(ValidationError, match="prefill_c1"):
            SimConfig.model_validate(payload)

    def test_inference_discriminator_rejects_unknown_mode(self) -> None:
        payload = make_config().model_dump()
        payload["inference"]["mode"] = "legacy"
        with pytest.raises(ValidationError):
            SimConfig.model_validate(payload)

    def test_runtime_is_frozen_and_preserves_resolved_physical_components(self) -> None:
        from dataclasses import FrozenInstanceError

        from simcore.capacity import prepare_runtime, resolve_config

        payload = make_config().model_dump()
        payload["shared_l3"]["available_gb"] = 128 * 64 / 1_000_000_000
        payload["inference"]["storage"].update(
            l2_available_gb=64 * 64 / 1_000_000_000,
            l2_to_l1_gb_s=0.000128, l3_to_l2_gb_s=0.000064,
            l2_latency_ms=0.5, l3_latency_ms=1.5,
        )
        config = SimConfig.model_validate(payload)
        original = config.model_dump()
        runtime = prepare_runtime(config)
        assert runtime.resolution == resolve_config(config) and runtime.analytical is None
        assert runtime.engine.page_size == 64
        timing = config.inference.timing
        assert isinstance(timing, EmpiricalTimingConfig)
        assert runtime.perf_model.prefill_c1 == timing.prefill_c1
        assert runtime.perf_model.kv_bytes_per_token == 64
        assert runtime.links.l2_to_l1 is not None and runtime.links.l2_to_l1.transfer_ms == 32
        assert runtime.links.l3_to_l2 is not None and runtime.links.l3_to_l2.transfer_ms == 64
        assert runtime.links.l2_to_l1.latency_ms == 0.5
        assert runtime.links.l3_to_l2.latency_ms == 1.5
        with pytest.raises(FrozenInstanceError):
            runtime.engine = runtime.engine  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            runtime.links.l2_to_l1.transfer_ms = 0  # type: ignore[misc]
        assert config.model_dump() == original

    def test_roundtrip_is_lossless(self) -> None:
        cfg = make_config()
        assert SimConfig.model_validate(cfg.model_dump()) == cfg
        assert SimConfig.model_validate_json(cfg.model_dump_json()) == cfg


class TestArrival:
    def test_valid(self) -> None:
        a = Arrival(request_id=0, sim_time=0.0, prompt_len=10, output_len=5, max_new_tokens=5)
        assert (a.request_id, a.sim_time, a.max_new_tokens) == (0, 0.0, 5)
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
        base = {
            "request_id": 0, "sim_time": 1.0, "prompt_len": 10,
            "output_len": 5, "max_new_tokens": 5,
        }
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
        "actual_prefill_tokens": 10,
        "output_len": 5,
    }

    def test_monotonic_endpoints_accepted(self) -> None:
        assert RequestRecord.model_validate(self.BASE).prefill_end == 3.0

    def test_actual_prefill_tokens_required_and_not_nullable(self) -> None:
        missing = {key: value for key, value in self.BASE.items() if key != "actual_prefill_tokens"}
        with pytest.raises(ValidationError, match="actual_prefill_tokens"):
            RequestRecord.model_validate(missing)
        with pytest.raises(ValidationError, match="actual_prefill_tokens"):
            RequestRecord.model_validate(self.BASE | {"actual_prefill_tokens": None})

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
