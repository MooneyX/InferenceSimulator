"""顶层入口：计划 → 闭环客户端 → 统一事件时钟 → 引擎 → 指标。

只规划根请求到达，子轮次必须等父请求实际完成后才由客户端发送。
事件流不经 API 暴露，默认关闭记录；旧长度 Arrival 由 ``engine.run_engine`` 适配。
"""

from simcore.client import ClientDriver
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.metrics import compute_metrics
from simcore.schemas import SimConfig, SimMeta, SimResult
from simcore.workload import build_plan

__all__ = ["simulate"]


def simulate(config: SimConfig) -> SimResult:
    """执行一场完整仿真。

    Raises:
        NotImplementedError: ``workload.mode='trace'``（预留接缝，MVP 不实现）。
    """
    if config.workload.mode == "trace":
        raise NotImplementedError("trace replay 尚未实现，MVP 仅支持 workload.mode='synthetic'")
    synthetic = config.workload.synthetic
    if synthetic is None:  # pragma: no cover - schema 层已保证
        raise ValueError("workload.mode='synthetic' 时必须提供 workload.synthetic")
    plan = build_plan(synthetic, config.engine.prefix_cache)
    client = ClientDriver(plan, config.engine.prefix_cache)
    run = Coordinator(
        Engine(config.engine, config.perf_model, record_trace=False),
        client,
    ).run()
    return SimResult(
        meta=SimMeta(
            config=config,
            num_requests=len(plan.turns),
            num_completed=len(run.records),
            num_aborted=0,
            num_skipped=0,
            num_engine_steps=run.num_engine_steps,
            sim_start=run.sim_start,
            sim_end=run.sim_end,
        ),
        metrics=compute_metrics(
            run.records,
            plan=plan,
            cache_snapshot=run.cache_snapshot,
            l1_evictions=run.l1_evictions,
            l1_peak_resident_blocks=run.l1_peak_resident_blocks,
        ),
        requests=list(run.records),
        warnings=list(run.warnings),
    )
