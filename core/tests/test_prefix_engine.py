"""缓存执行接入：步末输出发布、同一步不可互见及真实命中加速。"""

import pytest

from simcore.engine import Engine
from simcore.events import StepTrace
from simcore.prefix_hash import encode
from simcore.schemas import EngineConfig, InferenceRequest, PerfModelConfig, PrefixCacheConfig

CACHE = PrefixCacheConfig(block_size_tokens=2)
ENGINE = EngineConfig(max_batch_size=4, kv_token_budget=20, prefix_cache=CACHE)
PERF = PerfModelConfig(prefill_c1=1, decode_d1=0, decode_d2=0)


def request(
    key: int, prompt: int, output: int, ids: tuple[str, ...] | None = None, ideal: int = 0
) -> InferenceRequest:
    return InferenceRequest(
        request_id=key,
        session_id=key,
        parent_request_id=None,
        prompt_len=prompt,
        output_len=output,
        input_chunk_ids=(
            ids if ids is not None else tuple(encode(key, i) for i in range(prompt // 2))
        ),
        ideal_reuse_blocks=ideal,
    )


def test_parent_output_full_block_reused_after_done() -> None:
    engine = Engine(ENGINE, PERF)
    parent = request(0, 2, 2)
    child = request(1, 5, 1, (encode(0, 0), encode(0, 1)), ideal=2)
    engine.submit(parent, 0)
    duration, done = engine.step(0, 0)
    assert duration == pytest.approx(2)
    assert not done
    duration, done = engine.step(2, 1)
    assert done == (0,)
    completion = engine.complete(2 + duration, 0)
    assert (completion.request_id, completion.finish_time_ms) == (0, 2 + duration)
    assert encode(0, 1) in engine._prefix_cache.hash_index
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
    shared = ("same-explicit-id",)
    engine.submit(request(0, 2, 1, shared), 0)
    engine.submit(request(1, 2, 1, shared), 0)
    duration, done = engine.step(0, 0)
    assert done == (0, 1)
    assert duration == pytest.approx(4)
    engine.complete(duration, 0)
    engine.complete(duration, 1)
    engine.submit(request(2, 2, 1, shared, ideal=1), duration)
    later, done = engine.step(duration, 1)
    assert done == (2,)
    assert later == pytest.approx(1)
    engine.complete(duration + later, 2)
    assert [
        entry.admitted for entry in engine.finalize().trace if isinstance(entry, StepTrace)
    ] == [(0, 1), (2,)]


def test_partial_input_fills_on_first_output_and_final_tail_is_not_published() -> None:
    cache_config = PrefixCacheConfig(block_size_tokens=4)
    config = EngineConfig(max_batch_size=1, kv_token_budget=8, prefix_cache=cache_config)
    engine = Engine(config, PERF)
    parent = InferenceRequest(
        request_id=7,
        session_id=7,
        parent_request_id=None,
        prompt_len=3,
        output_len=3,
        input_chunk_ids=(),
        ideal_reuse_blocks=0,
    )
    engine.submit(parent, 0)
    duration, done = engine.step(0, 0)
    assert done == ()
    assert duration == pytest.approx(3)
    snapshot = engine._prefix_cache.snapshot()
    assert snapshot.resident_blocks == 1
    assert snapshot.reserved_future_blocks == 1
    assert dict(snapshot.hash_index) == {encode(7, 0): (0,)}
    assert snapshot.blocks[0].valid_tokens == 4

    _, done = engine.step(duration, 1)
    assert done == ()
    snapshot = engine._prefix_cache.snapshot()
    assert snapshot.resident_blocks == 2
    assert snapshot.reserved_future_blocks == 0
    assert snapshot.blocks[1].valid_tokens == 1
    assert snapshot.blocks[1].content_hash is None
    _, done = engine.step(duration, 2)
    assert done == (7,)
    snapshot = engine._prefix_cache.snapshot()
    assert snapshot.blocks[1].valid_tokens == 2
    assert dict(snapshot.hash_index) == {encode(7, 0): (0,)}
    completion = engine.complete(duration, 7)
    assert (completion.request_id, completion.finish_time_ms) == (7, duration)
    snapshot = engine._prefix_cache.snapshot()
    assert snapshot.resident_blocks == 1
    assert snapshot.reserved_future_blocks == 0
    assert snapshot.free_blocks == 1
    assert snapshot.blocks[0].state == "EVICTABLE"
    assert dict(snapshot.hash_index) == {encode(7, 0): (0,)}
