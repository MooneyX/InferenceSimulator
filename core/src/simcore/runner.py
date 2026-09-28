"""顶层入口：``simulate(SimConfig) -> SimResult``。

``core`` 唯一的对外调用入口，输入输出均为 ``schemas`` 类型。串联
workload（生成到达）→ engine（执行仿真）→ metrics（汇总指标）。

事件流不经对外接口暴露（见 ``docs/design/architecture.md``），故默认不记录，
省去逐事件的记录开销；需要事件流的内核级调用方直接使用 ``engine.run_engine``。

依赖层级：依赖 ``workload``、``engine``、``metrics``。
"""

from simcore.engine import run_engine
from simcore.metrics import compute_metrics
from simcore.schemas import SimConfig, SimMeta, SimResult
from simcore.workload import build_source

__all__ = ["simulate"]


def simulate(config: SimConfig) -> SimResult:
    """执行一场完整仿真。

    Raises:
        NotImplementedError: ``workload.mode='trace'``（预留接缝，MVP 不实现）。
    """
    source = build_source(config.workload)
    run = run_engine(config.engine, config.perf_model, source, record_trace=False)
    return SimResult(
        meta=SimMeta(
            config=config,
            num_requests=len(run.records),
            num_engine_steps=run.num_engine_steps,
            sim_start=run.sim_start,
            sim_end=run.sim_end,
        ),
        metrics=compute_metrics(run.records),
        requests=list(run.records),
        warnings=list(run.warnings),
    )
