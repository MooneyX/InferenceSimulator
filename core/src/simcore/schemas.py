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
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AbortedRequest",
    "AnalyticalSummary",
    "Arrival",
    "CapacityReport",
    "ClientWakeup",
    "EngineConfig",
    "FixedLength",
    "HardwareSpec",
    "InferenceRequest",
    "LengthDist",
    "LognormalLength",
    "Metrics",
    "ModelSpec",
    "PerfModelConfig",
    "PoissonLength",
    "RankCapacityReport",
    "RequestCompletion",
    "RequestRecord",
    "ResolvedConfig",
    "ResolvedRequest",
    "ResolvedRun",
    "SessionConfig",
    "SimConfig",
    "SimMeta",
    "SimResult",
    "SkippedRequest",
    "StorageSpec",
    "SyntheticWorkloadConfig",
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
    target_ideal_prefix_hit_rate: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)


class SyntheticWorkloadConfig(_Frozen):
    """合成负载参数：目标总轮次 QPS、长度分布和 session 拓扑。"""

    num_requests: int = Field(ge=1, description="全部轮次请求数")
    arrival_rate: float = Field(gt=0, allow_inf_nan=False, description="总请求目标 QPS")
    prompt_len: LengthDist
    output_len: LengthDist
    max_new_tokens: int = Field(default=128, ge=1, description="每个请求可见的生成上限")
    seed: int = Field(ge=0, description="随机种子；同种子必须逐字节可复现")
    session: SessionConfig = Field(default_factory=SessionConfig)

    @model_validator(mode="after")
    def _resolve_session(self) -> Self:
        count = self.session.num_sessions
        if count is None:
            object.__setattr__(
                self, "session", self.session.model_copy(update={"num_sessions": self.num_requests})
            )
        elif count > self.num_requests:
            raise ValueError("num_sessions 不得大于 num_requests")
        return self


class WorkloadConfig(_Frozen):
    """负载配置。``trace`` 分支仅预留字段，MVP 不实现。"""

    mode: WorkloadMode = "synthetic"
    synthetic: SyntheticWorkloadConfig | None = None

    @model_validator(mode="after")
    def _check_mode_payload(self) -> Self:
        if self.mode == "synthetic" and self.synthetic is None:
            raise ValueError("workload.mode='synthetic' 时必须提供 workload.synthetic")
        return self


class EngineConfig(_Frozen):
    """单实例 KV 池配置；页数仅由 token 容量和页大小推导。"""

    max_running_requests: int = Field(ge=1, description="同时驻留的运行请求数上限")
    max_total_tokens: int = Field(ge=1, description="KV 池的实际 token 容量")
    max_prefill_tokens: int = Field(default=16_384, ge=1, description="单次 Prefill 的新计算额度")
    page_size: int = Field(default=64, ge=1, description="每个 KV 页的 token 数")
    l2_capacity_tokens: int = Field(default=0, ge=0, description="L2 KV 的 token 容量；0 表示关闭")
    l3_capacity_tokens: int = Field(default=0, ge=0, description="L3 KV 的 token 容量；0 表示关闭")
    pd_mutually_exclusive: bool = Field(default=True, description="Prefill 与 Decode 分步执行")

    @model_validator(mode="after")
    def _check_capacity(self) -> Self:
        if self.max_total_tokens < self.page_size or self.max_total_tokens % self.page_size:
            raise ValueError("max_total_tokens 必须至少一页且是 page_size 的整数倍")
        for tier in ("l2", "l3"):
            if getattr(self, f"{tier}_capacity_tokens") % self.page_size:
                raise ValueError(f"{tier}_capacity_tokens 必须是 page_size 的整数倍")
        if self.l3_capacity_tokens and not self.l2_capacity_tokens:
            raise ValueError("l3_capacity_tokens > 0 要求 l2_capacity_tokens > 0")
        return self

    @property
    def capacity_pages(self) -> int:
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


def validate_kv_config(engine: EngineConfig, perf: PerfModelConfig) -> None:
    """同时供 SimConfig、Engine 与 run_engine 检查 KV 层与传输参数。"""
    if engine.l2_capacity_tokens and (
        perf.kv_bytes_per_token is None or perf.l2_to_l1_bandwidth_bytes_per_ms is None
    ):
        raise ValueError(
            "l2_capacity_tokens > 0 要求 kv_bytes_per_token 与 l2_to_l1_bandwidth_bytes_per_ms 为正"
        )
    if engine.l3_capacity_tokens and perf.l3_to_l2_bandwidth_bytes_per_ms is None:
        raise ValueError("l3_capacity_tokens > 0 要求 l3_to_l2_bandwidth_bytes_per_ms 为正")


class SimConfig(_Frozen):
    """一场仿真的完整输入。"""

    workload: WorkloadConfig
    engine: EngineConfig
    perf_model: PerfModelConfig

    @model_validator(mode="after")
    def _check_tiered_config(self) -> Self:
        validate_kv_config(self.engine, self.perf_model)
        return self


class ModelSpec(_Frozen):
    """单 GPU 全注意力模型的显式规格；MoE 字段为后续阶段保留。"""

    name: str = Field(min_length=1)
    revision: str | None = None
    num_layers: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    num_kv_heads: int = Field(ge=1)
    head_dim: int = Field(ge=1)
    ffn_intermediate_size: int = Field(ge=1)
    total_parameters: int = Field(ge=1)
    vocab_size: int | None = Field(default=None, ge=1)
    tie_word_embeddings: bool = False
    weight_dtype: Literal["bf16", "fp16", "fp8"] = "bf16"
    kv_dtype: Literal["bf16", "fp16", "fp8"] = "bf16"
    resident_weights_bytes: int | None = Field(default=None, gt=0)
    moe_layers: int = Field(default=0, ge=0)
    expert_parameters_total: int | None = Field(default=None, gt=0)
    active_parameters: int | None = Field(default=None, ge=1)
    num_experts: int = Field(default=0, ge=0)
    experts_per_token: int = Field(default=0, ge=0)
    expert_intermediate_size: int | None = Field(default=None, gt=0)
    attention_layout: Literal["full_gqa", "other"] = "full_gqa"

    @model_validator(mode="after")
    def _check_heads(self) -> Self:
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("注意力头数须是 KV 头数的整数倍")
        if self.moe_layers > self.num_layers:
            raise ValueError("MoE 层数不得超过总层数")
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


class ResolvedRequest(_Frozen):
    """物理规格与现有负载/调度输入；engine 容量在解析时被替换。"""

    config: SimConfig
    model: ModelSpec
    hardware: HardwareSpec
    storage: StorageSpec
    timing: Literal["empirical", "analytical"] = "analytical"
    tp_size: int = Field(default=1, ge=1)


class RankCapacityReport(_Record):
    rank: int
    sharded_weights_bytes: int
    replicated_weights_bytes: int
    kv_bytes_per_token: int
    l1_kv_budget_bytes: int
    l1_capacity_pages: int


class CapacityReport(_Record):
    kv_bytes_per_token: int
    kv_bytes_per_page: int
    weights_resident_bytes: int
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
    unassigned_parameters: int = 0


class ResolvedConfig(_Record):
    config: SimConfig
    model: ModelSpec
    hardware: HardwareSpec
    storage: StorageSpec
    timing: Literal["empirical", "analytical"]
    capacity: CapacityReport
    tp_size: int = 1
    runnable: bool = True


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


class ResolvedRun(_Record):
    result: "SimResult"
    resolution: ResolvedConfig
    analysis: AnalyticalSummary | None = None


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

    def __post_init__(self) -> None:
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
    """请求 ID、到达时刻与长度记录；手写序列由 ``plan_from_arrivals`` 转为独立根 session。

    客户端向引擎提交 ``InferenceRequest``；引擎仍用此类型记录实际提交时刻，
    不经此类型传递 token 内容。
    """

    request_id: int = Field(ge=0)
    sim_time: float = Field(ge=0, description="到达时刻（ms）")
    prompt_len: int = Field(ge=1, description="prompt token 数")
    output_len: int = Field(ge=1, description="待生成 token 数")
    max_new_tokens: int = Field(default=128, ge=1, description="手写到达请求的可见生成上限")


class RequestRecord(_Record):
    """单请求的生命周期区间记录，供甘特图与指标汇总消费。

    不变式：``arrival <= prefill_start <= prefill_end <= finish``；
    ``prefill_end`` 同时是 decode 阶段起点。
    """

    id: int = Field(ge=0)
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
    actual_prefill_tokens: int | None = Field(default=None, ge=0)
    replayed_prefill_tokens: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    logical_context_tokens: int = Field(default=0, ge=0)
    l1_hit_tokens: int = Field(default=0, ge=0)
    l2_hit_tokens: int = Field(default=0, ge=0)
    l3_hit_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:
        if not (self.arrival <= self.prefill_start <= self.prefill_end <= self.finish):
            raise ValueError(
                "区间端点必须单调不减: "
                f"arrival={self.arrival} prefill_start={self.prefill_start} "
                f"prefill_end={self.prefill_end} finish={self.finish}"
            )
        if self.l1_hit_tokens + self.l2_hit_tokens + self.l3_hit_tokens != self.actual_hit_tokens:
            raise ValueError("请求级分层复用与实际命中 token 不守恒")
        if self.actual_prefill_tokens is not None and not (  # 每个输入 token 要么命中，要么重算
            self.actual_hit_tokens <= self.ideal_reuse_tokens <= self.prompt_len
            and self.actual_prefill_tokens + self.actual_hit_tokens == self.prompt_len
        ):
            raise ValueError("请求级理想命中、分层复用与 Prefill token 不守恒")
        return self


class AbortedRequest(_Record):
    """已发送但执行失败的请求；失败不终止其他请求。"""

    id: int = Field(ge=0)
    arrival: float = Field(ge=0)
    failed_at: float = Field(ge=0)
    reason: Literal["input_exceeds_kv_capacity", "decode_kv_exhausted"]
    generated_tokens: int = Field(ge=0)
    retry_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_failure_time(self) -> Self:
        if self.failed_at < self.arrival:
            raise ValueError("failed_at 不得早于 arrival")
        return self


class SkippedRequest(_Record):
    """父轮未成功，因而从未发送的子轮。"""

    id: int = Field(ge=0)
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
    target_ideal_prefix_hit_rate: float = Field(default=0.0, ge=0, le=1)
    constructed_ideal_prefix_hit_rate: float = Field(default=0.0, ge=0, le=1)
    ideal_hit_quantization_error: float = Field(default=0.0, ge=0)
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
    def _check_cache_conservation(self) -> Self:
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

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.sim_end < self.sim_start:
            raise ValueError(f"sim_end({self.sim_end}) 不得早于 sim_start({self.sim_start})")
        if self.num_completed + self.num_aborted + self.num_skipped != self.num_requests:
            raise ValueError("完成、失败与跳过数量之和必须等于计划请求总数")
        return self


class SimResult(_Record):
    """一场仿真的完整输出，即 ``POST /api/simulate`` 的响应体。"""

    meta: SimMeta
    metrics: Metrics
    requests: list[RequestRecord] = Field(default_factory=list)
    aborted_requests: list[AbortedRequest] = Field(default_factory=list)
    skipped_requests: list[SkippedRequest] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_result_counts(self) -> Self:
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
