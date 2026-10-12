"""顶层入口：负载客户端 → 统一事件时钟 → 引擎 → 指标。

开环模式按独立泊松时钟投放请求；显式闭环模式只预先规划根到达。
事件流不经 API 暴露，默认关闭记录；手写 Arrival 由 ``engine.run_engine`` 适配。
"""

from dataclasses import fields
from time import perf_counter
from typing import cast

from simcore.capacity import prepare_runtime
from simcore.client import ClientDriver, OpenLoopClient
from simcore.coordinator import Coordinator
from simcore.engine import Engine, EngineRun
from simcore.kv_cache import L1Snapshot, TierSnapshot
from simcore.metrics import compute_metrics
from simcore.router import Router
from simcore.schemas import (
    AbortedRequest,
    AnalyticalSummary,
    InstanceResult,
    Metrics,
    RequestRecord,
    SimConfig,
    SimMeta,
    SimResult,
    SkippedRequest,
    SyntheticWorkloadConfig,
    TracePlan,
    TraceWorkloadConfig,
    WorkloadPlan,
)
from simcore.trace_client import TraceClient
from simcore.workload import build_plan

__all__ = ["simulate"]


def simulate(config: SimConfig, *, trace_plan: TracePlan | None = None) -> SimResult:
    """按唯一配置运行仿真；Trace 计划内容通过独立资源参数提供。"""
    if config.workload.mode == "trace" and trace_plan is None:
        raise ValueError("Trace 仿真须提供 trace_plan 请求计划")
    if config.workload.mode != "trace" and trace_plan is not None:
        raise ValueError("合成仿真不能提供 trace_plan 请求计划")
    runtime = prepare_runtime(config)
    engine_config = runtime.engine
    started_at = perf_counter()
    client: ClientDriver | OpenLoopClient | TraceClient
    if trace_plan is not None:
        replay = cast(TraceWorkloadConfig, config.workload.trace)
        client = TraceClient(
            trace_plan, engine_config, mode=replay.mode, session_rate=replay.session_rate,
            seed=replay.seed, target_qps=replay.target_qps,
        )
    else:
        synthetic = cast(SyntheticWorkloadConfig, config.workload.synthetic)
        if synthetic.arrival_mode == "open_loop":
            client = OpenLoopClient(synthetic, engine_config)
        else:
            client = ClientDriver(build_plan(synthetic, engine_config), engine_config)
    runs = Coordinator(
        tuple(
            Engine(runtime, output_targets={}, record_trace=False)
            for _ in range(config.num_instances)
        ),
        client,
        Router(config.num_instances, session_affinity=trace_plan is not None),
    ).run()
    plan = client.plan
    instances: list[InstanceResult] = []
    records: list[RequestRecord] = []
    aborted: list[AbortedRequest] = []
    skipped: list[SkippedRequest] = []
    for instance_id, run in enumerate(runs):
        local_ids = (  # 本副本完成、失败、跳过的全部请求 ID
            {item.id for item in run.records}
            | {item.id for item in run.aborted_requests}
            | {item.id for item in run.skipped_requests}
        )
        turns = {key: plan.turns[key] for key in sorted(local_ids)}
        input_tokens = sum(turn.prompt_len for turn in turns.values())
        local_plan = WorkloadPlan(
            turns=turns,
            root_request_ids=tuple(key for key in plan.root_request_ids if key in turns),
            children={},
            constructed_ideal_prefix_hit_rate=(
                engine_config.page_size
                * sum(turn.ideal_reuse_blocks for turn in turns.values())
                / input_tokens
                if input_tokens
                else 0.0
            ),
        )
        instances.append(
            InstanceResult(
                instance_id=instance_id,
                num_completed=len(run.records),
                num_aborted=len(run.aborted_requests),
                num_skipped=len(run.skipped_requests),
                num_engine_steps=run.num_engine_steps,
                metrics=_summarize((run,), local_plan, config),
            )
        )
        records.extend(item.model_copy(update={"instance_id": instance_id}) for item in run.records)
        aborted.extend(
            item.model_copy(update={"instance_id": instance_id}) for item in run.aborted_requests
        )
        skipped.extend(
            item.model_copy(update={"instance_id": instance_id}) for item in run.skipped_requests
        )
    records.sort(key=lambda item: item.id)
    aborted.sort(key=lambda item: item.id)
    skipped.sort(key=lambda item: item.id)
    sent_arrivals = [item.arrival for item in records] + [item.arrival for item in aborted]
    wall_clock_ms = (perf_counter() - started_at) * 1000
    result = SimResult(
        meta=SimMeta(
            config=config,
            num_requests=len(plan.turns),
            num_completed=len(records),
            num_aborted=len(aborted),
            num_skipped=len(skipped),
            num_engine_steps=sum(run.num_engine_steps for run in runs),
            sim_start=min(sent_arrivals, default=0.0),
            sim_end=max(
                (run.sim_end for run in runs if run.records or run.aborted_requests), default=0.0
            ),
            wall_clock_ms=wall_clock_ms,
        ),
        metrics=_summarize(runs, plan, config),
        instances=instances,
        requests=records,
        aborted_requests=aborted,
        skipped_requests=skipped,
        warnings=[warning for run in runs for warning in run.warnings],
    )
    analyses = [run.analysis for run in runs if run.analysis is not None]
    analysis = (
        AnalyticalSummary(
            **{
                name: sum(getattr(entry, name) for entry in analyses)
                for name in AnalyticalSummary.model_fields
            }
        )
        if analyses
        else None
    )
    result.resolution = runtime.resolution
    result.analysis = analysis
    return result


def _summarize(runs: tuple[EngineRun, ...], plan: WorkloadPlan, config: SimConfig) -> Metrics:  # 实例结果、负载计划与配置 → 合并后的指标
    """按全场实际到达窗口重新计算时延；末态缓存按独占实例求和。"""
    records = [record for run in runs for record in run.records]
    sent_arrivals = [item.arrival for run in runs for item in run.records] + [
        item.arrival for run in runs for item in run.aborted_requests
    ]
    l1 = L1Snapshot(
        **{
            item.name: sum(getattr(run.cache_snapshot, item.name) for run in runs)
            for item in fields(L1Snapshot)
        }
    )
    tiers = [
        TierSnapshot(
            **{
                item.name: sum(getattr(getattr(run, tier), item.name) for run in runs)
                for item in fields(TierSnapshot)
            }
        )
        for tier in ("l2_snapshot", "l3_snapshot")
    ]
    return compute_metrics(
        records,
        plan=plan,
        page_size=config.inference.page_size,
        workload_mode=config.workload.mode,
        cache_snapshot=l1,
        l2_snapshot=tiers[0],
        l3_snapshot=tiers[1],
        l1_evictions=sum(run.l1_evictions for run in runs),
        # 各实例峰值发生时刻可能不同；该字段是各实例峰值之和（容量规划上界）。
        l1_peak_resident_blocks=sum(run.l1_peak_resident_blocks for run in runs),
        promotions=sum(run.promotions for run in runs),
        demotions=sum(run.demotions for run in runs),
        restore_bytes=sum(run.restore_bytes for run in runs),
        restore_time_ms=sum(run.restore_time_ms for run in runs),
        earliest_sent_arrival=min(sent_arrivals, default=None),
        latest_sent_arrival=max(sent_arrivals, default=None),
        num_sent=len(sent_arrivals),
        replayed_prefill_tokens=sum(run.replayed_prefill_tokens for run in runs),
    )
