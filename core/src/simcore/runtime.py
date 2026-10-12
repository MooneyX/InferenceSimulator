"""运行前准备的不可变实例规格；不持有请求、缓存或事件状态。"""

from dataclasses import dataclass

from simcore.schemas import (
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    PerfModelConfig,
    ResolutionReport,
    validate_kv_config,
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


def runtime_from_components(engine: EngineConfig, perf: PerfModelConfig) -> RuntimeConfig:
    """将手动配置的容量与经验系数准备为实例规格。"""
    validate_kv_config(engine, perf)
    page_bytes = engine.page_size * (perf.kv_bytes_per_token or 0)
    l2 = (
        TransferLink(
            page_bytes / perf.l2_to_l1_bandwidth_bytes_per_ms,
            perf.l2_latency_ms,
        )
        if engine.l2_capacity_tokens and perf.l2_to_l1_bandwidth_bytes_per_ms is not None
        else None
    )
    l3 = (
        TransferLink(
            page_bytes / perf.l3_to_l2_bandwidth_bytes_per_ms,
            perf.l3_latency_ms,
        )
        if engine.l3_capacity_tokens and perf.l3_to_l2_bandwidth_bytes_per_ms is not None
        else None
    )
    return RuntimeConfig(engine, perf, TransferLinks(l2, l3))
