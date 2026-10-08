"""F001 单步引擎：P/D 互斥、融合、KV 增长、发布和确定性。"""

import pytest

from simcore.engine import Engine, run_engine
from simcore.events import DoneTrace
from simcore.prefix_hash import encode
from simcore.runner import simulate
from simcore.schemas import (
    Arrival,
    EngineConfig,
    FixedLength,
    InferenceRequest,
    PerfModelConfig,
    SessionConfig,
    SimConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)

PERF = PerfModelConfig(prefill_c1=1, prefill_c2=0, decode_d1=5, decode_d2=1)


def make_engine(
    targets: dict[int, int], *, exclusive: bool = True, page_size: int = 4, capacity: int = 128
) -> Engine:
    config = EngineConfig(
        max_running_requests=4,
        max_total_tokens=capacity,
        page_size=page_size,
        pd_mutually_exclusive=exclusive,
    )
    return Engine(config, PERF, output_targets=targets)


def request(
    key: int, *, prompt: int = 4, maximum: int = 2, inherited: tuple[str, ...] = ()
) -> InferenceRequest:
    return InferenceRequest(key, 0, None, prompt, maximum, inherited, len(inherited))


def test_prefill_samples_first_token_but_does_not_create_its_kv() -> None:
    engine = make_engine({0: 3}, page_size=64, capacity=512)
    engine.submit(request(0, prompt=65, maximum=100), 0)
    duration, finished = engine.step(0, 0)
    assert duration == 65 and finished == ()
    prefill_step = list(engine._log.steps())[-1]
    assert prefill_step.phase == "PREFILL"
    assert prefill_step.admitted == (0,) and prefill_step.num_decoding == 0
    assert prefill_step.batch_total_tokens == 65 and prefill_step.allocated_blocks == 2
    assert engine._prefix_cache.state(0).k == 65
    assert engine._prefix_cache.state(0).f == 64
    assert engine._prefix_cache.private_page_count(0) == 1
    assert engine._prefix_cache.snapshot().internal_fragmentation_tokens == 63
    assert engine._prefix_cache.snapshot().reserved_future_blocks == 0
    duration, finished = engine.step(65, 1)
    assert duration == 71 and finished == ()
    decode_step = list(engine._log.steps())[-1]
    assert decode_step.phase == "DECODE"
    assert decode_step.num_decoding == 1 and decode_step.batch_total_tokens == 66
    assert engine._prefix_cache.state(0).k == 66
    duration, finished = engine.step(136, 2)
    assert duration == 72 and finished == (0,)
    assert engine._prefix_cache.state(0).k == 67  # 最后采样输出不建立 KV
    engine.complete(208, 0)
    assert DoneTrace(208, 0, 67) in engine.finalize().trace
    run = engine.finalize()
    assert (run.records[0].prefill_end, run.records[0].finish) == (65, 208)
    assert run.cache_snapshot.resident_blocks == run.cache_snapshot.evictable_blocks == 1
    assert run.cache_snapshot.free_blocks == 7


def test_output_len_one_finishes_at_prefill_end_and_reclaims_partial_page() -> None:
    engine = make_engine({0: 1}, page_size=64, capacity=256)
    engine.submit(request(0, prompt=63), 0)
    assert engine.step(0, 0) == (63, (0,))
    assert engine._prefix_cache.state(0).k == 63
    engine.complete(63, 0)
    run = engine.finalize()
    assert run.cache_snapshot.resident_blocks == 0
    assert run.records[0].prefill_end == run.records[0].finish == 63


def test_decode_full_page_remains_private_until_request_completion() -> None:
    engine = make_engine({0: 2}, page_size=64, capacity=256)
    engine.submit(request(0, prompt=63), 0)
    engine.step(0, 0)
    assert engine._prefix_cache.page(encode(0, 0)) is None
    assert engine.step(63, 1) == (69, (0,))
    assert engine._prefix_cache.state(0).k == 64
    assert engine._prefix_cache.page(encode(0, 0)) is None
    engine.complete(132, 0)
    page = engine._prefix_cache.page(encode(0, 0))
    assert page is not None and page.ref_count == 0
    assert engine.finalize().cache_snapshot.resident_blocks == 1


def test_exclusive_mode_prefills_next_arrival_without_decoding_existing_request() -> None:
    engine = make_engine({0: 2, 1: 2})
    engine.submit(request(0), 0)
    engine.step(0, 0)
    engine.submit(request(1), 4)
    assert engine.step(4, 1) == (4, ())
    prefill_step = list(engine._log.steps())[-1]
    assert prefill_step.phase == "PREFILL"
    assert prefill_step.admitted == (1,) and prefill_step.num_decoding == 0
    assert engine._prefix_cache.state(0).k == 4
    duration, finished = engine.step(8, 2)
    assert duration == 15 and finished == (0, 1)
    decode_step = list(engine._log.steps())[-1]
    assert decode_step.phase == "DECODE"
    assert decode_step.allocated_blocks == 2 and decode_step.num_decoding == 2
    assert decode_step.batch_total_tokens == 10
    engine.complete(23, 0)
    engine.complete(23, 1)
    assert [item.phase for item in engine.finalize().trace.steps()] == [
        "PREFILL",
        "PREFILL",
        "DECODE",
    ]


def test_fused_mode_only_decodes_request_running_at_step_start() -> None:
    engine = make_engine({0: 2, 1: 2}, exclusive=False)
    engine.submit(request(0), 0)
    engine.step(0, 0)
    engine.submit(request(1), 4)
    duration, finished = engine.step(4, 1)
    assert duration == 10 and finished == (0,)
    mixed_step = list(engine._log.steps())[-1]
    assert mixed_step.phase == "MIXED"
    assert mixed_step.admitted == (1,) and mixed_step.num_decoding == 1
    assert mixed_step.batch_total_tokens == 9 and mixed_step.allocated_blocks == 2
    assert engine._prefix_cache.state(1).k == 4
    engine.complete(14, 0)
    assert engine.step(14, 2) == (10, (1,))
    engine.complete(24, 1)
    assert [item.phase for item in engine.finalize().trace.steps()] == [
        "PREFILL",
        "MIXED",
        "DECODE",
    ]


def test_same_batch_duplicate_private_pages_deduplicate_after_execution() -> None:
    engine = make_engine({1: 1, 2: 1}, page_size=64, capacity=512)
    shared = (encode(0, 0),)
    engine.submit(request(1, prompt=65, maximum=1, inherited=shared), 0)
    engine.submit(request(2, prompt=65, maximum=1, inherited=shared), 0)
    duration, finished = engine.step(0, 0)
    assert duration == 130 and finished == (1, 2)
    assert list(engine._log.steps())[-1].allocated_blocks == 4
    page = engine._prefix_cache.page(shared[0])
    assert page is not None and page.ref_count == 2
    assert engine._prefix_cache.resident_pages == 3
    assert engine._prefix_cache.peak_resident_pages == 4
    engine.complete(130, 1)
    engine.complete(130, 2)
    snapshot = engine.finalize().cache_snapshot
    assert snapshot.resident_blocks == 1 and snapshot.evictable_blocks == 1
    assert snapshot.free_blocks + snapshot.resident_blocks == snapshot.capacity_blocks


def test_reused_prefill_cost_uses_reused_context_not_full_prompt() -> None:
    cfg = EngineConfig(max_running_requests=2, max_total_tokens=512, page_size=64)
    perf = PerfModelConfig(prefill_c1=0, prefill_c2=1, decode_d1=0, decode_d2=0)
    engine = Engine(cfg, perf, output_targets={0: 1, 1: 1})
    engine.submit(request(0, prompt=64, maximum=1), 0)
    assert engine.step(0, 0) == (4096, (0,))
    engine.complete(4096, 0)
    engine.submit(request(1, prompt=65, maximum=1, inherited=(encode(0, 0),)), 4096)
    assert engine.step(4096, 1) == (129, (1,))
    prefill_step = list(engine._log.steps())[-1]
    assert prefill_step.admitted == (1,)
    assert prefill_step.matched_cache_tokens == 64 and prefill_step.batch_total_tokens == 65
    engine.complete(4225, 1)
    assert engine.finalize().records[1].actual_prefill_tokens == 1


def test_zero_cost_engine_progresses_and_is_deterministic() -> None:
    cfg = EngineConfig(max_running_requests=2, max_total_tokens=128, page_size=4)
    zero = PerfModelConfig(prefill_c1=0, prefill_c2=0, decode_d1=0, decode_d2=0)
    arrivals = (Arrival(request_id=0, sim_time=1, prompt_len=5, output_len=3),)
    first = run_engine(cfg, zero, arrivals)
    second = run_engine(cfg, zero, arrivals)
    assert first.num_engine_steps == 3
    assert first.records[0].finish == first.records[0].arrival == 1
    assert first.records == second.records
    assert tuple(first.trace.steps()) == tuple(second.trace.steps())
    assert first.cache_snapshot.reserved_future_blocks == 0


def test_multi_turn_simulation_uses_published_parent_prefix_and_is_reproducible() -> None:
    config = SimConfig(
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=3,
                arrival_rate=10,
                prompt_len=FixedLength(value=65),
                output_len=FixedLength(value=2),
                seed=7,
                session=SessionConfig(num_sessions=1, target_ideal_prefix_hit_rate=64 / (65 * 3)),
            )
        ),
        engine=EngineConfig(max_running_requests=2, max_total_tokens=512, page_size=64),
        perf_model=PERF,
    )
    first = simulate(config)
    second = simulate(config)
    assert first.model_dump_json() == second.model_dump_json()
    assert first.meta.num_completed == first.meta.num_requests == 3
    assert first.metrics.actual_hit_tokens == 64
    assert first.metrics.l1_reserved_future_blocks == 0
    assert all(request.output_len == 2 for request in first.requests)


def test_empty_step_is_rejected_without_advancing_clock() -> None:
    engine = make_engine({})
    with pytest.raises(RuntimeError, match="没有可执行"):
        engine.step(0, 0)
    assert engine.num_steps == 0 and tuple(engine._log.steps()) == ()
