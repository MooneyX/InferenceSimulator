"""多实例并行、亲和路由与全场统计回归。"""

import pytest

from simcore import simulate
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    PerfModelConfig,
    SessionConfig,
    SimConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)


def config(
    *, instances: int = 2, requests: int = 4, sessions: int = 2, capacity: int = 64, prompt: int = 4
) -> SimConfig:
    return SimConfig(
        num_instances=instances,
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=requests,
                session_rate=100,
                prompt_len=FixedLength(value=prompt),
                turn_increment_len=FixedLength(value=1),
                output_len=FixedLength(value=2),
                max_new_tokens=2,
                seed=5,
                session=SessionConfig(num_sessions=sessions),
            )
        ),
        engine=EngineConfig(max_running_requests=1, max_total_tokens=capacity, page_size=4),
        perf_model=PerfModelConfig(prefill_c1=1, decode_d1=1, decode_d2=0),
    )


def test_parallel_engines_and_parent_affinity() -> None:
    settings = config()
    result = simulate(settings)
    assert result.meta.num_completed == 4
    assert [item.instance_id for item in result.requests if item.parent_request_id is None] == [
        0,
        1,
    ]
    by_id = {item.id: item for item in result.requests}
    for item in result.requests:
        if item.parent_request_id is not None:
            assert item.instance_id == by_id[item.parent_request_id].instance_id
            assert item.actual_hit_tokens > 0
    assert [item.num_completed for item in result.instances] == [2, 2]
    assert result.metrics.l1_capacity_blocks == 2 * settings.engine.capacity_pages
    assert result.metrics.ttft_mean == pytest.approx(
        sum(item.prefill_end - item.arrival for item in result.requests) / 4
    )
    assert result.metrics.throughput == pytest.approx(4000 / result.metrics.makespan)
    assert result.model_dump_json(exclude={"meta": {"wall_clock_ms"}}) == simulate(
        settings
    ).model_dump_json(exclude={"meta": {"wall_clock_ms"}})


def test_idle_instance_keeps_capacity_without_requests() -> None:
    result = simulate(config(instances=3, requests=1, sessions=1))
    assert [item.num_completed for item in result.instances] == [1, 0, 0]
    assert result.instances[1].metrics.throughput is None
    assert result.metrics.l1_capacity_blocks == 3 * result.instances[0].metrics.l1_capacity_blocks


def test_failures_and_descendants_stay_with_original_instance() -> None:
    result = simulate(config(capacity=4, prompt=8))
    assert result.meta.num_completed == 0
    assert (result.meta.num_aborted, result.meta.num_skipped) == (2, 2)
    assert [item.instance_id for item in result.aborted_requests] == [0, 1]
    owners = {item.id: item.instance_id for item in result.aborted_requests}
    assert all(
        item.instance_id == owners[item.parent_request_id] for item in result.skipped_requests
    )
    assert result.metrics.throughput is None
