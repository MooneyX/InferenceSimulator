"""顶层入口：负载客户端 → 统一事件时钟 → 引擎 → 指标。

开环模式按独立泊松时钟投放请求；显式闭环模式只预先规划根到达。
事件流不经 API 暴露，默认关闭记录；手写 Arrival 由 ``engine.run_engine`` 适配。
"""

from time import perf_counter
from typing import cast

from simcore.client import ClientDriver, OpenLoopClient
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.metrics import compute_metrics
from simcore.schemas import (
    AnalyticalSummary,
    HardwareSpec,
    ModelSpec,
    ResolvedConfig,
    SimConfig,
    SimMeta,
    SimResult,
    SyntheticWorkloadConfig,
)
from simcore.workload import build_plan

__all__ = ["simulate"]


def simulate(config: SimConfig) -> SimResult:
    """沿用单实例经验模型的完整仿真入口。"""
    result, _ = _simulate(config, None)
    return result


def simulate_analytical(
    config: SimConfig, model: ModelSpec, hardware: HardwareSpec
) -> tuple[SimResult, AnalyticalSummary]:
    """单 GPU Roofline 模式使用相同的调度与 KV 生命周期。"""
    result, analysis = _simulate(config, (model, hardware, 1, 1))
    if analysis is None:
        raise RuntimeError("解析运行未生成成本汇总")
    return result, analysis


def simulate_resolved(resolution: ResolvedConfig) -> tuple[SimResult, AnalyticalSummary | None]:
    """自动模式使用解析后的 token 容量和 token/ms 链路带宽。"""
    if not resolution.runnable:
        if resolution.timing == "analytical":
            raise ValueError("多卡解析运行须提供互联带宽、TP 集合通信延迟及所需的 EP 单次通信开销")
        raise NotImplementedError("TP/EP>1 仅支持解析计时；四系数无法表达逐卡计算与通信")
    analytical = (
        (resolution.model, resolution.hardware, resolution.tp_size, resolution.ep_size)
        if resolution.timing == "analytical"
        else None
    )
    return _simulate(
        resolution.config,
        analytical,
        (resolution.capacity.l2_to_l1_tokens_per_ms, resolution.capacity.l3_to_l2_tokens_per_ms),
    )


def _simulate(
    config: SimConfig,
    analytical: tuple[ModelSpec, HardwareSpec, int, int] | None,
    token_bandwidths: tuple[float | None, float | None] | None = None,
) -> tuple[SimResult, AnalyticalSummary | None]:
    if config.workload.mode == "trace":
        raise NotImplementedError("trace replay 尚未实现，MVP 仅支持 workload.mode='synthetic'")
    started_at = perf_counter()
    synthetic = cast(SyntheticWorkloadConfig, config.workload.synthetic)
    if synthetic.arrival_mode == "open_loop":
        client: ClientDriver | OpenLoopClient = OpenLoopClient(synthetic, config.engine)
        output_targets: dict[int, int] = {}
    else:
        plan = build_plan(synthetic, config.engine)
        client = ClientDriver(plan, config.engine)
        output_targets = {key: turn.output_len for key, turn in plan.turns.items()}
    run = Coordinator(
        Engine(
            config.engine,
            config.perf_model,
            output_targets=output_targets,
            record_trace=False,
            analytical=analytical,
            token_bandwidths=token_bandwidths,
        ),
        client,
    ).run()
    plan = client.plan
    metrics = compute_metrics(
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
        latest_sent_arrival=max(
            [request.arrival for request in run.records]
            + [request.arrival for request in run.aborted_requests],
            default=None,
        ),
        num_sent=len(run.records) + len(run.aborted_requests),
        replayed_prefill_tokens=run.replayed_prefill_tokens,
    )
    wall_clock_ms = (perf_counter() - started_at) * 1000
    result = SimResult(
        meta=SimMeta(
            config=config,
            num_requests=len(plan.turns),
            num_completed=len(run.records),
            num_aborted=len(run.aborted_requests),
            num_skipped=len(run.skipped_requests),
            num_engine_steps=run.num_engine_steps,
            sim_start=run.sim_start,
            sim_end=run.sim_end,
            wall_clock_ms=wall_clock_ms,
        ),
        metrics=metrics,
        requests=list(run.records),
        aborted_requests=list(run.aborted_requests),
        skipped_requests=list(run.skipped_requests),
        warnings=list(run.warnings),
    )
    return result, run.analysis
