"""真实完成触发会话后续轮次与统一时钟同刻事件顺序。"""

import pytest

from simcore import simulate
from simcore.client import ClientDriver
from simcore.coordinator import Coordinator
from simcore.engine import Engine
from simcore.events import ArrivalTrace, DoneTrace, StepTrace
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    PerfModelConfig,
    SessionConfig,
    SimConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)
from simcore.workload import build_plan


def config(think: float = 0) -> SimConfig:
    return SimConfig(
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=3,
                arrival_rate=10,
                prompt_len=FixedLength(value=4),
                output_len=FixedLength(value=2),
                seed=21,
                session=SessionConfig(num_sessions=1, think_time_ms=think),
            )
        ),
        engine=EngineConfig(max_batch_size=1, kv_token_budget=200),
        perf_model=PerfModelConfig(prefill_c1=1, decode_d1=5, decode_d2=0),
    )


def test_single_session_children_arrive_only_after_completion() -> None:
    cfg = config(think=12)
    first = simulate(cfg)
    assert first.model_dump_json() == simulate(cfg).model_dump_json()
    records = first.requests
    assert len(records) == 3
    assert records[1].arrival == pytest.approx(records[0].finish + 12)
    assert records[2].arrival == pytest.approx(records[1].finish + 12)
    assert first.meta.num_requests == 3


def test_zero_think_done_callback_send_step_order() -> None:
    cfg = config()
    workload = cfg.workload.synthetic
    assert workload is not None
    client = ClientDriver(build_plan(workload, cfg.engine.prefix_cache), cfg.engine.prefix_cache)
    run = Coordinator(Engine(cfg.engine, cfg.perf_model), client).run()
    records = run.records
    assert records[0].finish == records[1].arrival
    assert records[1].prefill_start == records[1].arrival
    entries = list(run.trace)
    same_time = [entry for entry in entries if entry.time == records[0].finish]
    assert [type(item) for item in same_time] == [DoneTrace, ArrivalTrace, StepTrace]
    assert [record.id for record in records] == [0, 1, 2]
