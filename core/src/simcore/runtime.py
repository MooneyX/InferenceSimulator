"""运行前准备的不可变实例规格；不持有请求、缓存或事件状态。"""

from dataclasses import dataclass

from simcore.kv_source import KvSourceSpec
from simcore.perf_model import LayerPlan
from simcore.schemas import (
    EngineConfig,
    HardwareSpec,
    PerfModelConfig,
    ResolutionReport,
)


@dataclass(frozen=True, slots=True)
class AnalyticalSpec:
    plan: LayerPlan
    hardware: HardwareSpec
    tp_size: int
    ep_size: int


@dataclass(frozen=True, slots=True)
class TransferSpec:
    """KV 来源规格及按层序的每层每页字节数；各引擎据此建立带状态的来源。"""

    l2_to_l1: KvSourceSpec | None = None
    l3_read: KvSourceSpec | None = None  # 三层结构读到 L2，两层结构直接读到 L1
    layer_page_bytes: tuple[float, ...] = ()  # 整组每页 KV 在各层的字节数，合计为每页字节数
    seed: int = 0  # 来源波动种子，按实例与来源再派生


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    engine: EngineConfig
    perf_model: PerfModelConfig
    transfers: TransferSpec
    analytical: AnalyticalSpec
    resolution: ResolutionReport | None = None
