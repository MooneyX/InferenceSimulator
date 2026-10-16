"""运行前准备的不可变实例规格；不持有请求、缓存或事件状态。"""

from dataclasses import dataclass

from simcore.schemas import (
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    PerfModelConfig,
    ResolutionReport,
)


@dataclass(frozen=True, slots=True)
class AnalyticalSpec:
    model: ModelSpec
    hardware: HardwareSpec
    tp_size: int
    ep_size: int


@dataclass(frozen=True, slots=True)
class TransferLink:
    transfer_ms: float
    latency_ms: float


@dataclass(frozen=True, slots=True)
class TransferLinks:
    l2_to_l1: TransferLink | None = None
    l3_to_l2: TransferLink | None = None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    engine: EngineConfig
    perf_model: PerfModelConfig
    links: TransferLinks
    analytical: AnalyticalSpec | None = None
    resolution: ResolutionReport | None = None
