"""Pydantic 模型：仿真内核的全部对外数据类型。

本模块是 schema 的单一事实源（服务层与前端类型均由此派生），零依赖：
不 import 任何其他 simcore 模块，也不做任何 IO。

时间单位约定：
- 配置中的 ``arrival_rate`` 单位为 请求/秒（贴合用户直觉）。
- 其余一切时间量（仿真时钟、区间端点、性能模型系数与指标）单位均为 **毫秒 (ms)**。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AbortedRequest",
    "AnalyticalSummary",
    "Arrival",
    "AttentionGroup",
    "CapacityReport",
    "ClientWakeup",
    "EngineConfig",
    "FixedLength",
    "HardwareSpec",
    "IHCSpec",
    "InferenceRequest",
    "InstanceResult",
    "LengthDist",
    "LognormalLength",
    "Metrics",
    "ModelSpec",
    "NormSpec",
    "PerfModelConfig",
    "PoissonLength",
    "RankCapacityReport",
    "RequestCompletion",
    "RequestRecord",
    "AnalyticalTimingConfig",
    "EmpiricalTimingConfig",
    "ManualInferenceConfig",
    "ManualKVConfig",
    "PhysicalInferenceConfig",
    "ResolutionReport",
    "SchedulerConfig",
    "TraceWorkloadConfig",
    "RouterSpec",
    "SessionConfig",
    "SimConfig",
    "SimMeta",
    "SimResult",
    "SkippedRequest",
    "StorageSpec",
    "SyntheticWorkloadConfig",
    "TracePlan",
    "TraceTurn",
    "TurnPlan",
    "WorkloadConfig",
    "WorkloadMode",
    "WorkloadPlan",
    "validate_kv_config",
]


class _Frozen(BaseModel):
    """配置类基类：不可变 + 拒绝未知字段。

    不可变性服务于可复现性——配置对象在仿真过程中不可能被意外改写；
    ``extra="forbid"`` 让拼错的字段立即报错，而非被静默忽略。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class _Record(BaseModel):
    """结果类基类：拒绝未知字段，但允许构造后填充。"""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- 长度分布


class FixedLength(_Frozen):
    """固定长度。"""

    dist: Literal["fixed"] = "fixed"
    value: int = Field(ge=1, description="固定 token 数")


class PoissonLength(_Frozen):
    """泊松分布长度。"""

    dist: Literal["poisson"] = "poisson"
    mean: float = Field(gt=0, description="分布均值（token）")


class LognormalLength(_Frozen):
    """对数正态分布长度，用于表达长尾负载。"""

    dist: Literal["lognormal"] = "lognormal"
    mean: float = Field(gt=0, description="分布均值（token）")
    sigma: float = Field(gt=0, description="形状参数，越大尾越长")


LengthDist = Annotated[
    FixedLength | PoissonLength | LognormalLength,
    Field(discriminator="dist"),
]
"""长度分布的可判别联合，判别字段为 ``dist``。

采样语义（由 ``workload`` 模块实现）：结果向上取整并钳制到 >= 1 token。
"""


# ---------------------------------------------------------------- 配置

WorkloadMode = Literal["synthetic", "trace"]


class SessionConfig(_Frozen):
    """线性多轮会话；未指定会话数时每个请求各自成为一个根。"""

    num_sessions: int | None = Field(default=None, ge=1)
    topology: Literal["linear"] = "linear"
    think_time_ms: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description="子轮思考时间的指数分布均值（ms）；0 表示无等待",
    )


class SyntheticWorkloadConfig(_Frozen):
    """合成负载：开环按请求速率、闭环按新会话速率投放。"""

    num_requests: int = Field(ge=1, description="全部轮次请求数")
    arrival_rate: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="开环目标请求 QPS"
    )
    session_rate: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="闭环新会话投放频率（会话/秒）"
    )
    arrival_mode: Literal["closed_loop", "open_loop"] = "closed_loop"
    prompt_len: LengthDist
    turn_increment_len: LengthDist | None = None
    output_len: LengthDist
    max_new_tokens: int = Field(default=128, ge=1, description="每个请求可见的生成上限")
    seed: int = Field(ge=0, description="随机种子；同种子必须逐字节可复现")
    session: SessionConfig = Field(default_factory=SessionConfig)

    @model_validator(mode="after")
    def _resolve_session(self) -> Self:  # 当前负载配置 → 校验开/闭环并补全会话数，返回配置自身
        if self.arrival_mode == "open_loop":
            if self.arrival_rate is None or self.session_rate is not None:
                raise ValueError("开环负载必须仅提供请求到达率 arrival_rate")
            if self.turn_increment_len is None:
                raise ValueError("开环负载必须提供每轮新增长度分布 turn_increment_len")
            if self.session.num_sessions is not None or self.session.think_time_ms != 0:
                raise ValueError("开环负载不使用会话数或思考时间")
            return self
        if self.session_rate is None or self.arrival_rate is not None:
            raise ValueError("闭环负载必须仅提供新会话投放频率 session_rate")
        if self.turn_increment_len is None:
            raise ValueError("闭环负载必须提供每轮新增长度分布 turn_increment_len")
        count = self.session.num_sessions
        if count is None:
            object.__setattr__(
                self, "session", self.session.model_copy(update={"num_sessions": self.num_requests})
            )
        elif count > self.num_requests:
            raise ValueError("num_sessions 不得大于 num_requests")
        return self


class TraceWorkloadConfig(_Frozen):
    """计划资源引用与回放设置；计划内容由调用方独立提供。"""

    mode: Literal["open_loop", "closed_loop"]
    session_rate: float = Field(gt=0, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0)
    target_qps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    plan_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_target_qps(self) -> Self:
        if self.mode != "open_loop" and self.target_qps is not None:
            raise ValueError("目标 QPS 仅适用于开环 trace 回放")
        return self


class WorkloadConfig(_Frozen):
    """互斥的合成负载或 Trace 回放配置。"""

    mode: WorkloadMode = "synthetic"
    synthetic: SyntheticWorkloadConfig | None = None
    trace: TraceWorkloadConfig | None = None

    @model_validator(mode="after")
    def _check_mode_payload(self) -> Self:
        if self.mode == "synthetic":
            if self.synthetic is None:
                raise ValueError("workload.mode='synthetic' 时必须提供 workload.synthetic")
            if self.trace is not None:
                raise ValueError("合成负载不能同时配置 workload.trace")
        else:
            if self.trace is None:
                raise ValueError("workload.mode='trace' 时必须提供 workload.trace")
            if self.synthetic is not None:
                raise ValueError("Trace 负载不能同时配置 workload.synthetic")
        return self


class EngineConfig(_Frozen):
    """单实例 KV 池配置；页数仅由 token 容量和页大小推导。"""

    max_running_requests: int = Field(ge=1, description="同时驻留的运行请求数上限")
    max_total_tokens: int = Field(ge=1, description="KV 池的实际 token 容量")
    max_prefill_tokens: int = Field(default=16_384, ge=1, description="单次 Prefill 的新计算额度")
    page_size: int = Field(default=64, ge=1, description="每个 KV 页的 token 数")
    l2_capacity_tokens: int = Field(default=0, ge=0, description="L2 KV 的 token 容量；0 表示关闭")
    l3_capacity_tokens: int = Field(default=0, ge=0, description="L3 KV 的 token 容量；0 表示关闭")
    l3_prefetch_min_tokens: int = Field(
        default=256, ge=0, description="触发 L3 异步预取的最小可复用 token 数"
    )
    pd_mutually_exclusive: bool = Field(default=True, description="Prefill 与 Decode 分步执行")

    @model_validator(mode="after")
    def _check_capacity(self) -> Self:  # 当前引擎页容量设置 → 校验层级与页大小，返回自身
        if self.max_total_tokens < self.page_size or self.max_total_tokens % self.page_size:
            raise ValueError("max_total_tokens 必须至少一页且是 page_size 的整数倍")
        for tier in ("l2", "l3"):
            if getattr(self, f"{tier}_capacity_tokens") % self.page_size:
                raise ValueError(f"{tier}_capacity_tokens 必须是 page_size 的整数倍")
        if self.l3_capacity_tokens and not self.l2_capacity_tokens:
            raise ValueError("l3_capacity_tokens > 0 要求 l2_capacity_tokens > 0")
        return self

    @property
    def capacity_pages(self) -> int:  # L1 token 容量与页大小 → 物理页数
        return self.max_total_tokens // self.page_size


class PerfModelConfig(_Frozen):
    """性能模型系数，单位 ms。

    - prefill 耗时 = ``prefill_c1 * P + prefill_c2 * P**2``（P = prompt_tokens）
    - decode 单步耗时 = ``decode_d1 + decode_d2 * batch_total_tokens``
    """

    prefill_c1: float = Field(ge=0)
    prefill_c2: float = Field(default=0.0, ge=0)
    decode_d1: float = Field(ge=0)
    decode_d2: float = Field(ge=0)
    kv_bytes_per_token: int | None = Field(default=None, gt=0)
    l2_to_l1_bandwidth_bytes_per_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    l3_to_l2_bandwidth_bytes_per_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    l2_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    l3_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    l3_lookup_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)


def validate_kv_config(engine: EngineConfig, perf: PerfModelConfig) -> None:  # 引擎容量与传输系数 → 校验层间带宽配置；无返回值
    """为手动运行规格检查 KV 层与传输参数。"""
    if engine.l2_capacity_tokens and (
        perf.kv_bytes_per_token is None or perf.l2_to_l1_bandwidth_bytes_per_ms is None
    ):
        raise ValueError(
            "l2_capacity_tokens > 0 要求 kv_bytes_per_token 与 l2_to_l1_bandwidth_bytes_per_ms 为正"
        )
    if engine.l3_capacity_tokens and perf.l3_to_l2_bandwidth_bytes_per_ms is None:
        raise ValueError("l3_capacity_tokens > 0 要求 l3_to_l2_bandwidth_bytes_per_ms 为正")


class SchedulerConfig(_Frozen):
    """所有推理实例共用的调度规则，不包含容量与性能参数。"""

    max_running_requests: int = Field(ge=1)
    max_prefill_tokens: int = Field(default=16_384, ge=1)
    pd_mutually_exclusive: bool = True
    l3_prefetch_min_tokens: int = Field(default=256, ge=0)


class EmpiricalTimingConfig(_Frozen):
    """经验 GPU 计时的四系数，不包含 KV 链路参数。"""

    mode: Literal["empirical"] = "empirical"
    prefill_c1: float = Field(ge=0, allow_inf_nan=False)
    prefill_c2: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    decode_d1: float = Field(ge=0, allow_inf_nan=False)
    decode_d2: float = Field(ge=0, allow_inf_nan=False)


class AnalyticalTimingConfig(_Frozen):
    """复用物理推理规格的解析 GPU 计时。"""

    mode: Literal["analytical"] = "analytical"


class ManualKVConfig(_Frozen):
    """手动指定 KV 容量和每条上行链路的参数。"""

    max_total_tokens: int = Field(ge=1)
    l2_capacity_tokens: int = Field(default=0, ge=0)
    l3_capacity_tokens: int = Field(default=0, ge=0)
    kv_bytes_per_token: int | None = Field(default=None, gt=0)
    l2_to_l1_bandwidth_bytes_per_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    l3_to_l2_bandwidth_bytes_per_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    l2_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    l3_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    l3_lookup_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class ManualInferenceConfig(_Frozen):
    """抽象实例：直接填写容量、链路和经验计时。"""

    mode: Literal["manual"] = "manual"
    page_size: int = Field(default=64, ge=1)
    kv: ManualKVConfig
    timing: EmpiricalTimingConfig

    @model_validator(mode="after")
    def _check_kv(self) -> Self:
        kv = self.kv
        if kv.max_total_tokens < self.page_size or kv.max_total_tokens % self.page_size:
            raise ValueError("max_total_tokens 必须至少一页且是 page_size 的整数倍")
        for tier in ("l2", "l3"):
            if getattr(kv, f"{tier}_capacity_tokens") % self.page_size:
                raise ValueError(f"{tier}_capacity_tokens 必须是 page_size 的整数倍")
        if kv.l3_capacity_tokens and not kv.l2_capacity_tokens:
            raise ValueError("l3_capacity_tokens > 0 要求 l2_capacity_tokens > 0")
        if kv.l2_capacity_tokens and (
            kv.kv_bytes_per_token is None or kv.l2_to_l1_bandwidth_bytes_per_ms is None
        ):
            raise ValueError(
                "l2_capacity_tokens > 0 要求 kv_bytes_per_token 与 "
                "l2_to_l1_bandwidth_bytes_per_ms 为正"
            )
        if kv.l3_capacity_tokens and kv.l3_to_l2_bandwidth_bytes_per_ms is None:
            raise ValueError("l3_capacity_tokens > 0 要求 l3_to_l2_bandwidth_bytes_per_ms 为正")
        return self


AuxWeightDtype = Literal["bf16", "fp16", "fp32"]


class NormSpec(_Frozen):
    """Block 与最终归一化的容量规格；dtype 为 null 时跟随主权重。"""

    kind: Literal["none", "rms", "layer"] = "rms"
    block_per_layer: int = Field(default=2, ge=0)
    final_norm: bool = True
    dtype: AuxWeightDtype | None = None


class RouterSpec(_Frozen):
    """逐 rank 复制的专家打分矩阵及选位修正向量。"""

    dtype: AuxWeightDtype | None = None
    correction_bias: bool = False
    correction_dtype: AuxWeightDtype = "fp32"


class IHCSpec(_Frozen):
    """每层两处 iHC 与最终汇总 head 的容量规格，不包含运行耗时。"""

    mult: int = Field(ge=1)
    dtype: AuxWeightDtype = "fp32"


class AttentionGroup(_Frozen):
    """相同注意力实现的层数；层序不参与当前固定页与聚合性能估算。"""

    kind: Literal["gqa", "mla", "mla_dsa"]
    layers: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    num_kv_heads: int | None = Field(default=None, ge=1)
    head_dim: int | None = Field(default=None, ge=1)
    q_rank: int | None = Field(default=None, ge=1)
    kv_rank: int | None = Field(default=None, ge=1)
    rope_dim: int | None = Field(default=None, ge=1)
    nope_dim: int | None = Field(default=None, ge=1)
    value_dim: int | None = Field(default=None, ge=1)
    index_heads: int | None = Field(default=None, ge=1)
    index_dim: int | None = Field(default=None, ge=1)
    topk: int | None = Field(default=None, ge=1)
    full_indexer_layers: int | None = Field(default=None, ge=1)
    output_gate: bool = Field(
        default=False, description="MLA 输出门控投影，按头分片并与输出投影合并计时"
    )

    qk_norm: bool = True
    norm_dtype: AuxWeightDtype | None = None
    index_norm: Literal["none", "rms", "layer"] = "layer"
    index_norm_dtype: AuxWeightDtype | None = None
    sink_dtype: AuxWeightDtype | None = None

    @model_validator(mode="after")
    def _check_layout(self) -> Self:
        mla = (self.q_rank, self.kv_rank, self.rope_dim, self.nope_dim, self.value_dim)
        index = (self.index_heads, self.index_dim, self.topk, self.full_indexer_layers)
        if self.kind == "gqa":
            if self.output_gate:
                raise ValueError("输出门控当前仅支持 MLA 注意力组")
            if (self.num_kv_heads is None or self.head_dim is None
                    or any(v is not None for v in mla + index)):
                raise ValueError("GQA 只配置 KV 头数和头维度")
            if self.num_attention_heads % self.num_kv_heads:
                raise ValueError("Query 头数须是 KV 头数的整数倍")
        elif (self.num_kv_heads is not None or self.head_dim is not None
              or any(v is None for v in mla)):
            raise ValueError("MLA 须配置 Q/KV 低秩、位置/非位置及 V 维度")
        if self.kind == "mla_dsa":
            if any(v is None for v in index) or cast(int, self.full_indexer_layers) > self.layers:
                raise ValueError("DSA 须配置索引头/维度、top-k 及有效的重算层数")
            if cast(int, self.index_dim) % 128:
                raise ValueError("DSA 的 FP8 index-K 维度须为 128 的整数倍")
        elif any(v is not None for v in index):
            raise ValueError("仅 MLA+DSA 可以配置索引器")
        if self.kind != "mla_dsa" and (
            self.index_norm != "layer" or self.index_norm_dtype is not None
        ):
            raise ValueError("仅 MLA+DSA 可以配置索引器归一化")
        return self


class ModelSpec(_Frozen):
    """全模型规格；可选注意力组合组独立于 FFN/MoE 的层数。"""

    name: str = Field(min_length=1)
    revision: str | None = None
    num_layers: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    num_kv_heads: int = Field(ge=1)
    head_dim: int = Field(ge=1)
    ffn_intermediate_size: int = Field(ge=1)
    vocab_size: int = Field(ge=1)
    tie_word_embeddings: bool = False
    weight_dtype: Literal["bf16", "fp16", "fp8"] = "bf16"
    kv_dtype: Literal["bf16", "fp16", "fp8"] = "bf16"
    moe_layers: int = Field(default=0, ge=0)
    num_experts: int = Field(default=0, ge=0)
    experts_per_token: int = Field(default=0, ge=0)
    expert_intermediate_size: int | None = Field(default=None, gt=0)
    num_shared_experts: int = Field(
        default=0, ge=0, description="每个 MoE 层逐 rank 复制、始终参与计算的共享专家数"
    )
    shared_expert_intermediate_size: int | None = Field(default=None, gt=0)
    attention_layout: Literal["full_gqa", "grouped", "hy4_preview", "other"] = "full_gqa"
    attention_groups: list[AttentionGroup] | None = None
    norms: NormSpec = Field(default_factory=NormSpec)
    router: RouterSpec = Field(default_factory=RouterSpec)
    ihc: IHCSpec | None = None

    @model_validator(mode="after")
    def _check_heads(self) -> Self:  # 当前模型规格 → 校验注意力结构与 MoE 层数，返回自身
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("注意力头数须是 KV 头数的整数倍")
        if self.moe_layers > self.num_layers:
            raise ValueError("MoE 层数不得超过总层数")
        if not self.moe_layers and self.router != RouterSpec():
            raise ValueError("Router 辅助配置需要 MoE 层")
        if self.num_shared_experts:
            if not self.moe_layers or self.shared_expert_intermediate_size is None:
                raise ValueError("共享专家需要 MoE 层及共享专家中间维度")
        elif self.shared_expert_intermediate_size is not None:
            raise ValueError("未启用共享专家时不能配置其中间维度")
        if (
            self.attention_layout in ("grouped", "hy4_preview") and self.attention_groups is None
        ) or (
            self.attention_layout not in ("grouped", "hy4_preview")
            and self.attention_groups is not None
        ):
            raise ValueError("grouped 注意力布局必须提供注意力组，其他布局不能混用")
        if self.attention_groups is not None and (
            not self.attention_groups
            or sum(group.layers for group in self.attention_groups) != self.num_layers
        ):
            raise ValueError("注意力组层数之和必须等于模型总层数")
        return self


class HardwareSpec(_Frozen):
    """单卡容量及用户假设的有效性能，不使用标称峰值代入。"""

    name: str = Field(min_length=1)
    gpu_count: int = Field(default=1, ge=1)
    memory_bytes_per_gpu: int = Field(gt=0)
    effective_fp16_tflops: float = Field(gt=0, allow_inf_nan=False)
    effective_fp8_tflops: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    effective_hbm_gb_s: float = Field(gt=0, allow_inf_nan=False)
    effective_interconnect_gb_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    collective_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    ep_communication_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    kernel_overhead_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class StorageSpec(_Frozen):
    """GB 为十进制；L1 可用预算包括驻留权重，L2/L3 仅放 KV。"""

    l1_available_gb: float = Field(gt=0, allow_inf_nan=False)
    l2_available_gb: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    l3_available_gb: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    runtime_reserve_bytes: int = Field(default=0, ge=0)
    l2_to_l1_gb_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    l3_to_l2_gb_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    l2_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    l3_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    l3_lookup_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class PhysicalInferenceConfig(_Frozen):
    """物理实例：共用模型和硬件解析容量及 GPU 计时规格。"""

    mode: Literal["physical"] = "physical"
    page_size: int = Field(default=64, ge=1)
    model: ModelSpec
    hardware: HardwareSpec
    storage: StorageSpec
    timing: Annotated[
        EmpiricalTimingConfig | AnalyticalTimingConfig, Field(discriminator="mode")
    ] = Field(default_factory=AnalyticalTimingConfig)
    tp_size: int = Field(default=1, ge=1)
    ep_size: int = Field(default=1, ge=1)


class SimConfig(_Frozen):
    """唯一仿真输入；负载、调度和推理规格各自只有一个来源。"""

    workload: WorkloadConfig
    scheduler: SchedulerConfig
    inference: Annotated[
        ManualInferenceConfig | PhysicalInferenceConfig, Field(discriminator="mode")
    ]
    num_instances: int = Field(ge=1, description="独占相同单实例配置的推理副本数量")


class RankCapacityReport(_Record):
    rank: int
    sharded_weights_bytes: int
    replicated_weights_bytes: int
    kv_bytes_per_token: int
    l1_kv_budget_bytes: int
    l1_capacity_pages: int
    expert_weights_bytes: int = 0


class CapacityReport(_Record):
    kv_bytes_per_token: int
    kv_bytes_per_page: int
    weights_resident_bytes: int
    estimated_total_parameters: int
    estimated_active_parameters: int
    estimated_expert_parameters: int
    l1_budget_bytes: int
    l1_kv_budget_bytes: int
    l2_budget_bytes: int
    l3_budget_bytes: int
    l1_capacity_pages: int
    l2_capacity_pages: int
    l3_capacity_pages: int
    l2_to_l1_tokens_per_ms: float | None = None
    l3_to_l2_tokens_per_ms: float | None = None
    rank_reports: list[RankCapacityReport] = Field(default_factory=list)


class ResolutionReport(_Record):
    """容量与可运行性报告，不作为另一种仿真输入。"""

    capacity: CapacityReport
    max_total_tokens: int
    l2_capacity_tokens: int
    l3_capacity_tokens: int
    runnable: bool
    reason: str | None = None


class AnalyticalSummary(_Record):
    steps: int = 0
    flops: float = 0.0
    hbm_bytes: float = 0.0
    compute_ms: float = 0.0
    memory_ms: float = 0.0
    device_ms: float = 0.0
    restore_ms: float = 0.0
    communication_ms: float = 0.0
    communication_bytes: float = 0.0
    collectives: int = 0
    tp_communication_ms: float = 0.0
    tp_communication_bytes: float = 0.0
    ep_dispatch_ms: float = 0.0
    ep_combine_ms: float = 0.0
    ep_dispatch_bytes: float = 0.0
    ep_combine_bytes: float = 0.0
    ep_exchanges: int = 0


# ---------------------------------------------------------------- 外部 trace 请求计划


class TraceTurn(_Frozen):
    request_id: int = Field(ge=0)
    session_id: int = Field(ge=0)
    parent_request_id: int | None = Field(default=None, ge=0)
    arrival_offset_ms: float = Field(ge=0, allow_inf_nan=False)
    think_time_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    prompt_len: int = Field(ge=1)
    output_len: int = Field(ge=0)
    hash_ids: tuple[int, ...]


class TracePlan(_Frozen):
    version: Literal[1] = 1
    block_size: int = Field(ge=1)
    sessions: tuple[str, ...]
    requests: tuple[TraceTurn, ...]

    @model_validator(mode="after")
    def _check_plan(self) -> Self:
        if not self.sessions or len(set(self.sessions)) != len(self.sessions) or not self.requests:
            raise ValueError("计划须包含非空且唯一的会话及请求")
        ids = {turn.request_id for turn in self.requests}
        if len(ids) != len(self.requests):
            raise ValueError("计划中的 request_id 必须唯一")
        by_id = {turn.request_id: turn for turn in self.requests}
        if any(
            previous.request_id >= following.request_id
            for previous, following in zip(self.requests, self.requests[1:])
        ):
            raise ValueError("请求须按 ID 严格递增")
        contexts: dict[tuple[int, int], tuple[int, int | None]] = {}
        for turn in self.requests:
            if turn.session_id >= len(self.sessions):
                raise ValueError("请求引用了不存在的会话")
            if len(turn.hash_ids) != turn.prompt_len // self.block_size or any(
                not isinstance(item, int) or item < 0 for item in turn.hash_ids
            ):
                raise ValueError("输入块 ID 数量或取值不合法")
            predecessor: int | None = None
            for position, block in enumerate(turn.hash_ids):
                key = (turn.session_id, block)
                context = (position, predecessor)
                if key in contexts and contexts[key] != context:
                    raise ValueError("同一会话的块 ID 指向不同输入上下文")
                contexts[key] = context
                predecessor = block
            parent_id = turn.parent_request_id
            if parent_id is None:
                if turn.think_time_ms is not None:
                    raise ValueError("无父请求不得设置思考时间")
            elif (
                parent_id not in by_id
                or parent_id >= turn.request_id
                or by_id[parent_id].session_id != turn.session_id
                or turn.think_time_ms is None
            ):
                raise ValueError("父请求必须是同会话中更早的有效请求")
        return self


# ---------------------------------------------------------------- 内部不可变计划与消息


@dataclass(frozen=True, slots=True)
class TurnPlan:
    request_id: int
    session_id: int
    parent_request_id: int | None
    root_arrival_time_ms: float | None
    think_time_ms: float | None
    prompt_len: int
    output_len: int
    max_new_tokens: int
    ideal_reuse_blocks: int


@dataclass(frozen=True, slots=True)
class WorkloadPlan:
    turns: Mapping[int, TurnPlan]
    root_request_ids: tuple[int, ...]
    children: Mapping[int, tuple[int, ...]]
    constructed_ideal_prefix_hit_rate: float = 0.0

    def __post_init__(self) -> None:  # 轮次和后代映射 → 防御性拷贝并冻结；无返回值
        # 先拷贝再冻结，避免外部字典改写计划。
        object.__setattr__(self, "turns", MappingProxyType(dict(self.turns)))
        object.__setattr__(self, "children", MappingProxyType(dict(self.children)))


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: int
    session_id: int
    parent_request_id: int | None
    prompt_len: int
    max_new_tokens: int
    inherited_prefix_ids: tuple[str, ...]
    ideal_reuse_blocks: int
    trace_hash_ids: tuple[int, ...] = ()
    is_trace: bool = False


@dataclass(frozen=True, slots=True)
class RequestCompletion:
    request_id: int
    finish_time_ms: float
    generated_tokens: int


@dataclass(frozen=True, slots=True)
class ClientWakeup:
    session_id: int
    turn_id: int
    send_time_ms: float


# ---------------------------------------------------------------- 负载与结果


class Arrival(_Frozen):
    """引擎保存的请求 ID、实际提交时刻与长度记录，不传递 token 内容。"""

    request_id: int = Field(ge=0)
    sim_time: float = Field(ge=0, description="到达时刻（ms）")
    prompt_len: int = Field(ge=1, description="prompt token 数")
    output_len: int = Field(ge=1, description="待生成 token 数")
    max_new_tokens: int = Field(ge=1, description="请求的可见生成上限")


class RequestRecord(_Record):
    """单请求的生命周期区间记录，供甘特图与指标汇总消费。

    不变式：``arrival <= prefill_start <= prefill_end <= finish``；
    ``prefill_end`` 同时是 decode 阶段起点。
    """

    id: int = Field(ge=0)
    instance_id: int = Field(default=0, ge=0)
    arrival: float = Field(ge=0)
    prefill_start: float = Field(ge=0)
    prefill_end: float = Field(ge=0, description="= decode_start")
    finish: float = Field(ge=0)
    prompt_len: int = Field(ge=1)
    output_len: int = Field(ge=1)
    session_id: int = Field(default=0, ge=0)
    parent_request_id: int | None = Field(default=None, ge=0)
    ideal_reuse_tokens: int = Field(default=0, ge=0)
    matched_cache_tokens: int = Field(default=0, ge=0)
    actual_hit_tokens: int = Field(default=0, ge=0)
    actual_prefill_tokens: int = Field(ge=0)
    replayed_prefill_tokens: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    logical_context_tokens: int = Field(default=0, ge=0)
    l1_hit_tokens: int = Field(default=0, ge=0)
    l2_hit_tokens: int = Field(default=0, ge=0)
    l3_hit_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:  # 当前请求区间与分层命中记录 → 校验时序和 token 守恒，返回自身
        if not (self.arrival <= self.prefill_start <= self.prefill_end <= self.finish):
            raise ValueError(
                "区间端点必须单调不减: "
                f"arrival={self.arrival} prefill_start={self.prefill_start} "
                f"prefill_end={self.prefill_end} finish={self.finish}"
            )
        if self.l1_hit_tokens + self.l2_hit_tokens + self.l3_hit_tokens != self.actual_hit_tokens:
            raise ValueError("请求级分层复用与实际命中 token 不守恒")
        if not (  # 每个输入 token 要么命中，要么重算
            self.actual_hit_tokens <= self.ideal_reuse_tokens <= self.prompt_len
            and self.actual_prefill_tokens + self.actual_hit_tokens == self.prompt_len
        ):
            raise ValueError("请求级理想命中、分层复用与 Prefill token 不守恒")
        return self


class AbortedRequest(_Record):
    """已发送但执行失败的请求；失败不终止其他请求。"""

    id: int = Field(ge=0)
    instance_id: int = Field(default=0, ge=0)
    arrival: float = Field(ge=0)
    failed_at: float = Field(ge=0)
    reason: Literal[
        "input_exceeds_kv_capacity", "decode_kv_exhausted",
        "invalid_trace_output", "output_exceeds_kv_capacity",
    ]
    generated_tokens: int = Field(ge=0)
    retry_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_failure_time(self) -> Self:  # 当前失败请求记录 → 校验失败时刻不早于到达，返回自身
        if self.failed_at < self.arrival:
            raise ValueError("failed_at 不得早于 arrival")
        return self


class SkippedRequest(_Record):
    """父轮未成功，因而从未发送的子轮。"""

    id: int = Field(ge=0)
    instance_id: int = Field(default=0, ge=0)
    parent_request_id: int = Field(ge=0)
    reason: Literal["parent_not_completed"] = "parent_not_completed"


class Metrics(_Record):
    """全场指标。``None`` 表示该指标在本场仿真中无定义。"""

    ttft_mean: float | None = Field(ge=0, description="TTFT 均值（ms）")
    ttft_p50: float | None = Field(ge=0)
    ttft_p99: float | None = Field(ge=0)
    tpot_mean: float | None = Field(
        default=None,
        ge=0,
        description="成功且输出超过 1 token 的请求逐项平均；无样本时为 None（ms）",
    )
    throughput: float | None = Field(
        default=None,
        ge=0,
        description="请求数 / makespan（请求/秒）；makespan 为 0 时为 None",
    )
    makespan: float = Field(
        ge=0, description="末成功请求完成 − 最早已发送请求到达；无成功请求为 0（ms）"
    )
    constructed_ideal_prefix_hit_rate: float = Field(default=0.0, ge=0, le=1)
    arrival_qps: float | None = Field(
        default=None, ge=0, description="全部已发送请求首末时刻间按已发送数减一计算的实际请求 QPS"
    )
    num_sessions: int = Field(default=0, ge=0)
    root_request_ratio: float = Field(default=0.0, ge=0, le=1)
    mean_prompt_tokens: float | None = Field(default=None, ge=0)
    mean_output_tokens: float | None = Field(default=None, ge=0)
    child_ideal_prefix_hit_rate: float | None = Field(default=None, ge=0, le=1)
    child_actual_prefix_hit_rate: float | None = Field(default=None, ge=0, le=1)
    actual_prefix_hit_rate: float = Field(default=0.0, ge=0, le=1)
    requests_with_hit_ratio: float = Field(default=0.0, ge=0, le=1)
    requests_fully_cached_ratio: float = Field(default=0.0, ge=0, le=1)
    total_input_tokens: int = Field(default=0, ge=0)
    ideal_hit_tokens: int = Field(default=0, ge=0)
    matched_cache_tokens: int = Field(default=0, ge=0)
    actual_hit_tokens: int = Field(default=0, ge=0)
    l1_hit_tokens: int = Field(default=0, ge=0)
    l2_hit_tokens: int = Field(default=0, ge=0)
    l3_hit_tokens: int = Field(default=0, ge=0)
    restore_bytes: int = Field(default=0, ge=0)
    restore_time_ms: float = Field(default=0.0, ge=0)
    promotions: int = Field(default=0, ge=0)
    demotions: int = Field(default=0, ge=0)
    actual_prefill_tokens: int = Field(default=0, ge=0)
    replayed_prefill_tokens: int = Field(default=0, ge=0)
    saved_prefill_tokens: int = Field(default=0, ge=0)
    l1_evictions: int = Field(default=0, ge=0)
    l1_peak_resident_blocks: int = Field(default=0, ge=0)
    l1_capacity_blocks: int = Field(default=0, ge=0)
    l1_resident_blocks: int = Field(default=0, ge=0)
    l1_active_blocks: int = Field(default=0, ge=0)
    l1_evictable_blocks: int = Field(default=0, ge=0)
    l1_free_blocks: int = Field(default=0, ge=0)
    l1_reserved_future_blocks: int = Field(default=0, ge=0)
    l1_internal_fragmentation_tokens: int = Field(default=0, ge=0)
    l2_evictions: int = Field(default=0, ge=0)
    l2_peak_resident_blocks: int = Field(default=0, ge=0)
    l2_capacity_blocks: int = Field(default=0, ge=0)
    l2_resident_blocks: int = Field(default=0, ge=0)
    l2_active_blocks: int = Field(default=0, ge=0)
    l2_evictable_blocks: int = Field(default=0, ge=0)
    l2_free_blocks: int = Field(default=0, ge=0)
    l3_evictions: int = Field(default=0, ge=0)
    l3_peak_resident_blocks: int = Field(default=0, ge=0)
    l3_capacity_blocks: int = Field(default=0, ge=0)
    l3_resident_blocks: int = Field(default=0, ge=0)
    l3_active_blocks: int = Field(default=0, ge=0)
    l3_evictable_blocks: int = Field(default=0, ge=0)
    l3_free_blocks: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_cache_conservation(self) -> Self:  # 当前全场指标 → 校验各层物理页和命中 token 守恒，返回自身
        if not (  # 输入 token 分为命中与重算；物理页分为驻留与空闲
            self.actual_hit_tokens <= self.ideal_hit_tokens <= self.total_input_tokens
            and self.actual_prefill_tokens + self.actual_hit_tokens == self.total_input_tokens
            and self.saved_prefill_tokens == self.actual_hit_tokens
            and self.l1_hit_tokens + self.l2_hit_tokens + self.l3_hit_tokens
            == self.actual_hit_tokens
            and self.l1_reserved_future_blocks == 0
            and self.l1_free_blocks + self.l1_resident_blocks == self.l1_capacity_blocks
            and self.l1_active_blocks + self.l1_evictable_blocks == self.l1_resident_blocks
            and self.l1_resident_blocks <= self.l1_peak_resident_blocks <= self.l1_capacity_blocks
        ):
            raise ValueError("全场前缀命中、Prefill 或 L1 物理块容量不守恒")
        for tier in ("l2", "l3"):
            capacity = getattr(self, f"{tier}_capacity_blocks")
            resident = getattr(self, f"{tier}_resident_blocks")
            active = getattr(self, f"{tier}_active_blocks")
            evictable = getattr(self, f"{tier}_evictable_blocks")
            free = getattr(self, f"{tier}_free_blocks")
            peak = getattr(self, f"{tier}_peak_resident_blocks")
            if (
                free + resident != capacity
                or active + evictable != resident
                or not (resident <= peak <= capacity)
            ):
                raise ValueError(f"{tier.upper()} 物理块容量不守恒")
        return self


class SimMeta(_Record):
    """配置回显与仿真规模信息。"""

    config: SimConfig
    num_requests: int = Field(ge=0, description="计划中的请求总数")
    num_completed: int = Field(ge=0)
    num_aborted: int = Field(ge=0)
    num_skipped: int = Field(ge=0)
    num_engine_steps: int = Field(ge=0, description="引擎迭代步数")
    sim_start: float = Field(ge=0, description="首请求到达时刻（ms）")
    sim_end: float = Field(ge=0, description="末请求完成时刻（ms）")
    wall_clock_ms: float = Field(
        ge=0,
        description="从生成请求计划至汇总指标的内核实际运行耗时（ms），不含配置解析与响应序列化",
    )

    @model_validator(mode="after")
    def _check_window(self) -> Self:  # 当前仿真元数据 → 校验时间窗及请求计数，返回自身
        if self.sim_end < self.sim_start:
            raise ValueError(f"sim_end({self.sim_end}) 不得早于 sim_start({self.sim_start})")
        if self.num_completed + self.num_aborted + self.num_skipped != self.num_requests:
            raise ValueError("完成、失败与跳过数量之和必须等于计划请求总数")
        return self


class InstanceResult(_Record):
    """单个副本的统计；吞吐按本副本已发送请求到达至最后成功完成计算。"""

    instance_id: int = Field(ge=0)
    num_completed: int = Field(ge=0)
    num_aborted: int = Field(ge=0)
    num_skipped: int = Field(ge=0)
    num_engine_steps: int = Field(ge=0)
    metrics: Metrics


class SimResult(_Record):
    """一场仿真的完整输出，即 ``POST /api/simulate`` 的响应体。"""

    meta: SimMeta
    metrics: Metrics
    instances: list[InstanceResult]
    requests: list[RequestRecord] = Field(default_factory=list)
    aborted_requests: list[AbortedRequest] = Field(default_factory=list)
    skipped_requests: list[SkippedRequest] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolution: ResolutionReport | None = None
    analysis: AnalyticalSummary | None = None

    @model_validator(mode="after")
    def _check_result_counts(self) -> Self:  # 当前完整结果 → 校验实例计数、互斥状态和指标定义，返回自身
        if (
            [item.instance_id for item in self.instances]
            != list(range(self.meta.config.num_instances))
            or sum(item.num_completed for item in self.instances) != self.meta.num_completed
            or sum(item.num_aborted for item in self.instances) != self.meta.num_aborted
            or sum(item.num_skipped for item in self.instances) != self.meta.num_skipped
            or sum(item.num_engine_steps for item in self.instances) != self.meta.num_engine_steps
        ):
            raise ValueError("逐实例结果与全场规模不一致")
        if (
            len(self.requests) != self.meta.num_completed
            or len(self.aborted_requests) != self.meta.num_aborted
            or len(self.skipped_requests) != self.meta.num_skipped
        ):
            raise ValueError("结果列表长度与完成、失败、跳过计数不一致")
        groups = (
            tuple(item.id for item in self.requests),
            tuple(item.id for item in self.aborted_requests),
            tuple(item.id for item in self.skipped_requests),
        )
        if any(ids != tuple(sorted(set(ids))) for ids in groups):
            raise ValueError("完成、失败、跳过请求必须分别按 ID 严格升序排列")
        if len(set().union(*groups)) != self.meta.num_requests:
            raise ValueError("同一请求不可同时出现在多个结果状态")
        non_completed = set(groups[1]) | set(groups[2])
        if any(item.parent_request_id not in non_completed for item in self.skipped_requests):
            raise ValueError("跳过请求的父轮必须已失败或跳过")
        ttft_values = (self.metrics.ttft_mean, self.metrics.ttft_p50, self.metrics.ttft_p99)
        if self.meta.num_completed == 0:
            if (
                any(value is not None for value in ttft_values)
                or self.metrics.tpot_mean is not None
                or self.metrics.throughput is not None
                or self.metrics.makespan != 0
            ):
                raise ValueError("无成功请求时 TTFT、TPOT、吞吐无定义且 makespan 为 0")
        elif any(value is None for value in ttft_values):
            raise ValueError("有成功请求时 TTFT 统计必须有定义")
        elif (self.metrics.makespan == 0) != (self.metrics.throughput is None):
            raise ValueError("有成功请求时吞吐仅在 makespan 为 0 时无定义")
        if self.meta.num_completed and (
            (self.metrics.tpot_mean is None)
            == any(request.output_len > 1 for request in self.requests)
        ):
            raise ValueError("TPOT 仅对成功且输出超过 1 token 的请求有定义")
        return self
