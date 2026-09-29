"""缓存执行接入：步末输出发布、同一步不可互见及真实命中加速。"""

import pytest

from simcore.engine import Engine
from simcore.events import StepTrace
from simcore.prefix_hash import cache_namespace, chunk_hashes
from simcore.schemas import EngineConfig, InferenceRequest, PerfModelConfig, PrefixCacheConfig

CACHE = PrefixCacheConfig(block_size_tokens=2)
ENGINE = EngineConfig(max_batch_size=4, kv_token_budget=20, prefix_cache=CACHE)
PERF = PerfModelConfig(prefill_c1=1, decode_d1=0, decode_d2=0)


def request(
    key: int, inputs: tuple[int, ...], outputs: tuple[int, ...], ideal: int = 0
) -> InferenceRequest:
    return InferenceRequest(
        key,
        key,
        None,
        inputs,
        outputs,
        chunk_hashes(inputs, 2, cache_namespace(CACHE)),
        ideal,
    )


def test_parent_output_full_block_reused_after_done() -> None:
    engine = Engine(ENGINE, PERF)
    parent = request(0, (1, 2), (3, 4))
    child = request(1, (1, 2, 3, 4, 5), (6,), ideal=2)
    engine.submit(parent, 0)
    duration, done = engine.step(0, 0)
    assert duration == pytest.approx(2)
    assert not done
    duration, done = engine.step(2, 1)
    assert done == (0,)
    completion = engine.complete(2 + duration, 0)
    assert completion.output_token_ids == (3, 4)
    engine.submit(child, 2 + duration)
    child_duration, done = engine.step(2 + duration, 2)
    assert child_duration == pytest.approx(1)
    assert done == (1,)
    engine.complete(2 + duration + child_duration, 1)
    run = engine.finalize()
    records = run.records
    assert records[1].arrival == records[0].finish
    assert records[1].prefill_end - records[1].prefill_start == pytest.approx(1)
    assert (records[1].ideal_reuse_tokens, records[1].matched_cache_tokens) == (4, 4)
    assert records[1].actual_hit_tokens == records[1].l1_hit_tokens == 4
    assert records[1].actual_prefill_tokens == 1
    last_step = list(run.trace.steps())[-1]
    assert last_step.probes == 1 and last_step.actual_hit_tokens == 4
    assert last_step.allocated_blocks == 1
    assert last_step.l1_free_after + last_step.l1_active_after + last_step.l1_evictable_after <= 10
    assert run.cache_snapshot is not None and run.cache_snapshot.active_blocks == 0


def test_same_step_identical_inputs_both_miss() -> None:
    engine = Engine(ENGINE, PERF)
    engine.submit(request(0, (1, 2), (3,)), 0)
    engine.submit(request(1, (1, 2), (4,)), 0)
    duration, done = engine.step(0, 0)
    assert done == (0, 1)
    assert duration == pytest.approx(4)
    engine.complete(duration, 0)
    engine.complete(duration, 1)
    engine.submit(request(2, (1, 2), (5,), ideal=1), duration)
    later, done = engine.step(duration, 1)
    assert done == (2,)
    assert later == pytest.approx(1)
    engine.complete(duration + later, 2)
    assert [
        entry.admitted for entry in engine.finalize().trace if isinstance(entry, StepTrace)
    ] == [(0, 1), (2,)]
