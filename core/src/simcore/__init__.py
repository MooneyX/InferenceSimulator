"""simcore —— LLM Serving 系统级离散事件仿真内核。

纯计算包：无文件、无网络、无时钟读取。唯一对外入口为 :func:`simulate`，
输入 :class:`SimConfig`，输出 :class:`SimResult`。

模块间单向依赖关系见 ``docs/design/architecture.md``「内核内部模块」。
"""

from simcore.runner import simulate
from simcore.schemas import (
    Arrival,
    EngineConfig,
    FixedLength,
    LengthDist,
    LognormalLength,
    Metrics,
    PerfModelConfig,
    PoissonLength,
    PrefixCacheConfig,
    RequestRecord,
    SessionConfig,
    SimConfig,
    SimMeta,
    SimResult,
    SyntheticWorkloadConfig,
    WorkloadConfig,
    WorkloadMode,
)

__version__ = "0.1.0"

__all__ = [
    "Arrival",
    "EngineConfig",
    "FixedLength",
    "LengthDist",
    "LognormalLength",
    "Metrics",
    "PerfModelConfig",
    "PoissonLength",
    "PrefixCacheConfig",
    "RequestRecord",
    "SessionConfig",
    "SimConfig",
    "SimMeta",
    "SimResult",
    "SyntheticWorkloadConfig",
    "WorkloadConfig",
    "WorkloadMode",
    "__version__",
    "simulate",
]
