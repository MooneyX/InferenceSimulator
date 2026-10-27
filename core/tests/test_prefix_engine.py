"""F001 引擎接入：完整 KV 页发布、同批去重和父子身份继承。"""

from dataclasses import replace

import pytest

from simcore.engine import Engine
from simcore.prefix_hash import encode
from simcore.schemas import (
    EngineConfig,
    InferenceRequest,
    PerfModelConfig,
    TurnPlan,
    WorkloadPlan,
)
from test_client import planned_client
from test_schemas import runtime_config

PERF = PerfModelConfig()


def request(
    key: int,
    prompt: int,
    maximum: int = 1,
    output: int = 1,
    inherited: tuple[str, ...] = (),
) -> InferenceRequest:
    return InferenceRequest(key, key, None, prompt, maximum, output, inherited, len(inherited))


def test_parent_output_full_page_reused_only_after_completion() -> None:
    config = EngineConfig(max_running_requests=2, max_total_tokens=32, page_size=2)
    engine = Engine(runtime_config(config, PERF))
    engine.submit(request(0, 2, maximum=2, output=2), 0)
    started = engine.start_step(0, 0)
    assert started is not None and started[1] == ()
    now = started[0]
    assert engine._prefix_cache.page(encode(0, 0)) is None
    assert engine.finish_step(now) == ()
    assert engine._prefix_cache.page(encode(0, 1)) is None
    started = engine.start_step(now, 1)
    assert started is not None and started[1] == (0,)
    now += started[0]
    assert engine.finish_step(now) == (0,)
    assert engine._prefix_cache.page(encode(0, 1)) is None
    engine.complete(now, 0)
    assert engine._prefix_cache.page(encode(0, 1)) is None  # K=P+O-1=3
    child = request(1, 5, inherited=(encode(0, 0),))
    engine.submit(child, now)
    started = engine.start_step(now, 2)
    assert started is not None and started[1] == (1,)
    now += started[0]
    assert engine.finish_step(now) == (1,)
    engine.complete(now, 1)
    assert engine.finalize().records[1].actual_hit_tokens == 2


def test_same_batch_identical_inherited_pages_deduplicate_on_publication() -> None:
    config = EngineConfig(max_running_requests=3, max_total_tokens=32, page_size=2)
    engine = Engine(runtime_config(config, PERF))
    identity = (encode(99, 0),)
    engine.submit(request(0, 3, inherited=identity), 0)
    engine.submit(request(1, 3, inherited=identity), 0)
    started = engine.start_step(0, 0)
    assert started is not None and started[1] == (0, 1)
    now = started[0]
    assert engine._prefix_cache.page(identity[0]) is None
    assert engine.finish_step(now) == (0, 1)
    page = engine._prefix_cache.page(identity[0])
    assert page is not None and page.ref_count == 2
    assert engine._prefix_cache.peak_resident_pages == 4
    engine.complete(now, 0)
    engine.complete(now, 1)
    engine.submit(request(2, 3, inherited=identity), now)
    started = engine.start_step(now, 1)
    assert started is not None and started[1] == (2,)
    now += started[0]
    assert engine.finish_step(now) == (2,)
    engine.complete(now, 2)
    assert [record.actual_hit_tokens for record in engine.finalize().records] == [0, 0, 2]


@pytest.mark.parametrize("prompt", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("output", [1, 2, 3, 4, 5])
def test_completed_parent_publishes_only_established_kv(prompt: int, output: int) -> None:
    config = EngineConfig(max_running_requests=1, max_total_tokens=128, page_size=4)
    root = TurnPlan(7, 7, None, 0, None, prompt, output, output, 0)
    blocks = (prompt + output - 1) // config.page_size
    child = replace(
        root,
        request_id=8,
        parent_request_id=7,
        root_arrival_time_ms=None,
        think_time_ms=0,
        prompt_len=(blocks + 1) * config.page_size,
        output_len=1,
        ideal_reuse_blocks=blocks,
    )
    plan = WorkloadPlan({7: root, 8: child}, (7,), {7: (8,)})
    client = planned_client(plan, config)
    engine = Engine(runtime_config(config, PERF))
    parent = client.on_wakeup(client.initial_wakeups()[0])
    assert parent.output_len == root.output_len == output
    assert parent.inherited_prefix_ids == ()
    engine.submit(parent, 0)
    now = 0.0
    for produced in range(1, output + 1):
        started = engine.start_step(now, produced - 1)
        assert started is not None
        duration, expected_finished = started
        now += duration
        assert engine.finish_step(now) == expected_finished
        assert expected_finished == ((7,) if produced == output else ())
        k = prompt + produced - 1
        state = engine._prefix_cache.state(7)
        assert state.k == k
        assert state.f == (prompt // 4) * 4
        assert engine._prefix_cache.free_pages + engine._prefix_cache.resident_pages == 32
        if produced < output:
            assert engine._prefix_cache.page(encode(7, k // 4)) is None or state.f > k // 4 * 4
    engine.complete(now, 7)
    successor = client.on_wakeup(client.on_request_completed(7, now)[0])
    assert successor.inherited_prefix_ids == tuple(encode(7, i) for i in range(blocks))
    for index in range(blocks):
        page = engine._prefix_cache.page(encode(7, index))
        assert page is not None and page.ref_count == 0


def test_distinct_roots_do_not_share_input_pages() -> None:
    config = EngineConfig(max_running_requests=2, max_total_tokens=32, page_size=2)
    engine = Engine(runtime_config(config, PERF))
    now = 0.0
    for key in range(2):
        engine.submit(request(key, 2), now)
        started = engine.start_step(now, key)
        assert started is not None and started[1] == (key,)
        now += started[0]
        assert engine.finish_step(now) == (key,)
        engine.complete(now, key)
    assert engine.finalize().records[1].matched_cache_tokens == 0
