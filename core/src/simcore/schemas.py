"""Pydantic 模型：仿真内核的全部对外数据类型。

本模块是 schema 的单一事实源（服务层与前端类型均由此派生），零依赖：
不 import 任何其他 simcore 模块，也不做任何 IO。

时间单位约定：
- 配置中的 ``arrival_rate`` 单位为 请求/秒（贴合用户直觉）。
- 其余一切时间量（仿真时钟、区间端点、性能模型系数与指标）单位均为 **毫秒 (ms)**。
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Arrival",
    "EngineConfig",
    "FixedLength",
    "LengthDist",
    "LognormalLength",
    "Metrics",
    "PerfModelConfig",
    "PoissonLength",
    "RequestRecord",
    "SimConfig",
    "SimMeta",
    "SimResult",
    "SyntheticWorkloadConfig",
    "WorkloadConfig",
    "WorkloadMode",
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


class SyntheticWorkloadConfig(_Frozen):
    """合成负载参数：泊松到达 + 长度分布采样 + 固定种子。"""

    num_requests: int = Field(ge=1, description="请求总数")
    arrival_rate: float = Field(gt=0, description="泊松到达率（请求/秒）")
    prompt_len: LengthDist
    output_len: LengthDist
    seed: int = Field(ge=0, description="随机种子；同种子必须逐字节可复现")


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
    """引擎配置：batch 规模与 KV cache 预算双重约束。"""

    max_batch_size: int = Field(ge=1, description="单步 batch 内最大请求数")
    kv_token_budget: int = Field(ge=1, description="KV cache token 预算总额")


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

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:
        if not (self.arrival <= self.prefill_start <= self.prefill_end <= self.finish):
            raise ValueError(
                "区间端点必须单调不减: "
                f"arrival={self.arrival} prefill_start={self.prefill_start} "
                f"prefill_end={self.prefill_end} finish={self.finish}"
            )
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
