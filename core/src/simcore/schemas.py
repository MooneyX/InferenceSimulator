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
    "Arrival",
    "ClientWakeup",
    "EngineConfig",
    "FixedLength",
    "InferenceRequest",
    "L1CacheConfig",
    "LengthDist",
    "LognormalLength",
    "LowerTierConfig",
    "Metrics",
    "PerfModelConfig",
    "PoissonLength",
    "PrefixCacheConfig",
    "RequestCompletion",
    "RequestRecord",
    "SessionConfig",
    "SimConfig",
    "SimMeta",
    "SimResult",
    "SyntheticWorkloadConfig",
    "TurnPlan",
    "WorkloadConfig",
    "WorkloadMode",
    "WorkloadPlan",
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
    think_time_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    target_ideal_prefix_hit_rate: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    allow_cross_session_shared_prefix: bool = False


class SyntheticWorkloadConfig(_Frozen):
    """合成负载参数：目标总轮次 QPS、长度分布和 session 拓扑。"""

    num_requests: int = Field(ge=1, description="全部轮次请求数")
    arrival_rate: float = Field(gt=0, allow_inf_nan=False, description="总请求目标 QPS")
    prompt_len: LengthDist
    output_len: LengthDist
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


class L1CacheConfig(_Frozen):
    capacity_blocks: int = Field(ge=1)


class LowerTierConfig(_Frozen):
    capacity_blocks: int = Field(default=0, ge=0)
    read_bandwidth_bytes_per_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    fixed_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class PrefixCacheConfig(_Frozen):
    block_size_tokens: int = Field(default=1, ge=1)
    cache_salt: str = ""
    eviction_policy: Literal["lru"] = "lru"
    hierarchy_policy: Literal["exclusive_write_back"] = "exclusive_write_back"
    l1: L1CacheConfig = Field(default_factory=lambda: L1CacheConfig(capacity_blocks=1))
    l2: LowerTierConfig = Field(default_factory=LowerTierConfig)
    l3: LowerTierConfig = Field(default_factory=LowerTierConfig)
    kv_bytes_per_token: int = Field(default=1, ge=1)
    model_id: str = "default-model"
    tokenizer_id: str = "default-tokenizer"
    kv_dtype: str = "default-dtype"
    adapter_id: str = "default-adapter"
    multimodal_transform_id: str = "default-transform"


class EngineConfig(_Frozen):
    """KV token 预算是 L1 容量的唯一输入；解析后统一以 block 记账。"""

    max_batch_size: int = Field(ge=1, description="单步 batch 内最大请求数")
    kv_token_budget: int = Field(ge=1, description="KV cache token 预算总额")
    prefix_cache: PrefixCacheConfig = Field(default_factory=PrefixCacheConfig)

    @model_validator(mode="after")
    def _resolve_cache(self) -> Self:
        capacity = self.kv_token_budget // self.prefix_cache.block_size_tokens
        if capacity < 1:
            raise ValueError("kv_token_budget 不足以容纳一个 prefix cache block")
        l1 = self.prefix_cache.l1
        if "l1" in self.prefix_cache.model_fields_set and l1.capacity_blocks != capacity:
            raise ValueError(
                "l1.capacity_blocks 必须等于 floor(kv_token_budget / block_size_tokens)"
            )
        object.__setattr__(
            self,
            "prefix_cache",
            self.prefix_cache.model_copy(update={"l1": L1CacheConfig(capacity_blocks=capacity)}),
        )
        return self


class PerfModelConfig(_Frozen):
    """性能模型系数，单位 ms。

    - prefill 耗时 = ``prefill_c1 * P + prefill_c2 * P**2``（P = prompt_tokens）
    - decode 单步耗时 = ``decode_d1 + decode_d2 * batch_total_tokens``
    """

    prefill_c1: float = Field(ge=0)
    prefill_c2: float = Field(default=0.0, ge=0)
    decode_d1: float = Field(ge=0)
    decode_d2: float = Field(ge=0)


class SimConfig(_Frozen):
    """一场仿真的完整输入。"""

    workload: WorkloadConfig
    engine: EngineConfig
    perf_model: PerfModelConfig


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
    ideal_reuse_blocks: int
    unique_suffix_seed: int
    output_token_plan: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WorkloadPlan:
    turns: Mapping[int, TurnPlan]
    root_request_ids: tuple[int, ...]
    children: Mapping[int, tuple[int, ...]]
    target_ideal_prefix_hit_rate: float = 0.0
    constructed_ideal_prefix_hit_rate: float = 0.0
    quantization_error: float = 0.0
    block_size_tokens: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", MappingProxyType(dict(self.turns)))
        object.__setattr__(self, "children", MappingProxyType(dict(self.children)))


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: int
    session_id: int
    parent_request_id: int | None
    input_token_ids: tuple[int, ...]
    output_token_plan: tuple[int, ...]
    input_chunk_hashes: tuple[str, ...]
    ideal_reuse_blocks: int


@dataclass(frozen=True, slots=True)
class RequestCompletion:
    request_id: int
    output_token_ids: tuple[int, ...]
    finish_time_ms: float


@dataclass(frozen=True, slots=True)
class ClientWakeup:
    session_id: int
    turn_id: int
    send_time_ms: float


# ---------------------------------------------------------------- 负载与结果


class Arrival(_Frozen):
    """一个请求的到达事实，由 ``WorkloadSource`` 产出。

    这是负载生成器与引擎之间唯一的数据交换类型。
    """

    request_id: int = Field(ge=0)
    sim_time: float = Field(ge=0, description="到达时刻（ms）")
    prompt_len: int = Field(ge=1, description="prompt token 数")
    output_len: int = Field(ge=1, description="待生成 token 数")


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
        if self.actual_prefill_tokens is not None and not (
            self.actual_hit_tokens <= self.ideal_reuse_tokens <= self.prompt_len
            and self.actual_prefill_tokens + self.actual_hit_tokens == self.prompt_len
            and self.l1_hit_tokens + self.l2_hit_tokens + self.l3_hit_tokens
            == self.actual_hit_tokens
        ):
            raise ValueError("请求级理想命中、分层复用与 Prefill token 不守恒")
        return self


class Metrics(_Record):
    """全场指标。``None`` 表示该指标在本场仿真中无定义。"""

    ttft_mean: float = Field(ge=0, description="TTFT 均值（ms）")
    ttft_p50: float = Field(ge=0)
    ttft_p99: float = Field(ge=0)
    tpot_mean: float | None = Field(
        default=None,
        ge=0,
        description="decode 阶段平均每 token 耗时（ms）；所有请求 output_len 均为 1 时为 None",
    )
    throughput: float | None = Field(
        default=None,
        ge=0,
        description="请求数 / makespan（请求/秒）；makespan 为 0 时为 None",
    )
    makespan: float = Field(ge=0, description="末请求完成时刻 − 首请求到达时刻（ms）")
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

    @model_validator(mode="after")
    def _check_cache_conservation(self) -> Self:
        if not (
            self.actual_hit_tokens <= self.ideal_hit_tokens <= self.total_input_tokens
            and self.actual_prefill_tokens + self.actual_hit_tokens == self.total_input_tokens
            and self.saved_prefill_tokens == self.actual_hit_tokens
            and self.l1_hit_tokens + self.l2_hit_tokens + self.l3_hit_tokens
            == self.actual_hit_tokens
            and self.l1_free_blocks + self.l1_resident_blocks + self.l1_reserved_future_blocks
            == self.l1_capacity_blocks
            and self.l1_active_blocks <= self.l1_resident_blocks
        ):
            raise ValueError("全场前缀命中、Prefill 或 L1 物理块容量不守恒")
        return self


class SimMeta(_Record):
    """配置回显与仿真规模信息。"""

    config: SimConfig
    num_requests: int = Field(ge=0)
    num_engine_steps: int = Field(ge=0, description="引擎迭代步数")
    sim_start: float = Field(ge=0, description="首请求到达时刻（ms）")
    sim_end: float = Field(ge=0, description="末请求完成时刻（ms）")

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.sim_end < self.sim_start:
            raise ValueError(f"sim_end({self.sim_end}) 不得早于 sim_start({self.sim_start})")
        return self


class SimResult(_Record):
    """一场仿真的完整输出，即 ``POST /api/simulate`` 的响应体。"""

    meta: SimMeta
    metrics: Metrics
    requests: list[RequestRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
