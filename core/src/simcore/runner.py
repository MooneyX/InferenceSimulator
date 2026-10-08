"""顶层入口：计划 → 闭环客户端 → 统一事件时钟 → 引擎 → 指标。

只规划根请求到达，子轮次必须等父请求实际完成后才由客户端发送。
事件流不经 API 暴露，默认关闭记录；旧长度 Arrival 由 ``engine.run_engine`` 适配。
"""

from typing import cast

from simcore.client import ClientDriver
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.metrics import compute_metrics
from simcore.schemas import SimConfig, SimMeta, SimResult, SyntheticWorkloadConfig
from simcore.workload import build_plan

__all__ = ["simulate"]


def simulate(config: SimConfig) -> SimResult:
    """执行一场完整仿真。

    Raises:
        NotImplementedError: ``workload.mode='trace'``（预留接缝，MVP 不实现）。
    """
    if config.workload.mode == "trace":
        raise NotImplementedError("trace replay 尚未实现，MVP 仅支持 workload.mode='synthetic'")
    synthetic = cast(SyntheticWorkloadConfig, config.workload.synthetic)
    plan = build_plan(synthetic, config.engine)
    client = ClientDriver(plan, config.engine)
    run = Coordinator(
        Engine(
            config.engine,
            config.perf_model,
            output_targets={key: turn.output_len for key, turn in plan.turns.items()},
            record_trace=False,
        ),
        client,
    ).run()
    return SimResult(
        meta=SimMeta(
            config=config,
            num_requests=len(plan.turns),
            num_completed=len(run.records),
            num_aborted=len(run.aborted_requests),
            num_skipped=len(run.skipped_requests),
            num_engine_steps=run.num_engine_steps,
            sim_start=run.sim_start,
            sim_end=run.sim_end,
        ),
        metrics=compute_metrics(
            run.records,
            plan=plan,
            config=config,
            cache_snapshot=run.cache_snapshot,
            l2_snapshot=run.l2_snapshot,
            l3_snapshot=run.l3_snapshot,
            l1_evictions=run.l1_evictions,
            l1_peak_resident_blocks=run.l1_peak_resident_blocks,
            promotions=run.promotions,
            demotions=run.demotions,
            restore_bytes=run.restore_bytes,
            restore_time_ms=run.restore_time_ms,
            earliest_sent_arrival=run.sim_start,
            replayed_prefill_tokens=run.replayed_prefill_tokens,
        ),
        requests=list(run.records),
        aborted_requests=list(run.aborted_requests),
        skipped_requests=list(run.skipped_requests),
        warnings=list(run.warnings),
    )
