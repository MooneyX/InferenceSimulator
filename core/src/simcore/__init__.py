"""simcore —— LLM Serving 系统级离散事件仿真内核。

仿真逻辑无文件、无网络 IO；顶层入口仅用性能时钟统计实际运行耗时。主要对外入口为 :func:`simulate`，
输入 :class:`SimConfig`，输出 :class:`SimResult`。

模块间单向依赖关系见 ``docs/design/architecture.md``「内核内部模块」。
"""

from simcore.runner import simulate
from simcore.schemas import (
    AbortedRequest,
    Arrival,
    EngineConfig,
    FixedLength,
    InstanceResult,
    LengthDist,
    LognormalLength,
    Metrics,
    PerfModelConfig,
    PoissonLength,
    RequestRecord,
    SessionConfig,
    SimConfig,
    SimMeta,
    SimResult,
    SkippedRequest,
    SyntheticWorkloadConfig,
    WorkloadConfig,
    WorkloadMode,
)

__version__ = "0.1.0"

__all__ = [
    "AbortedRequest",
    "Arrival",
    "EngineConfig",
    "FixedLength",
    "InstanceResult",
    "LengthDist",
    "LognormalLength",
    "Metrics",
    "PerfModelConfig",
    "PoissonLength",
    "RequestRecord",
    "SessionConfig",
    "SimConfig",
    "SimMeta",
    "SimResult",
    "SkippedRequest",
    "SyntheticWorkloadConfig",
    "WorkloadConfig",
    "WorkloadMode",
    "__version__",
    "simulate",
]
