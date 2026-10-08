"""F001 引擎接入：完整 KV 页发布、同批去重和父子身份继承。"""

from dataclasses import replace

import pytest

from simcore.client import ClientDriver
from simcore.engine import Engine
from simcore.prefix_hash import encode
from simcore.schemas import (
    Arrival,
    EngineConfig,
    InferenceRequest,
    PerfModelConfig,
    WorkloadPlan,
)
from simcore.workload import plan_from_arrivals

PERF = PerfModelConfig(prefill_c1=1, decode_d1=0, decode_d2=0)


def request(
    key: int, prompt: int, maximum: int = 1, inherited: tuple[str, ...] = ()
) -> InferenceRequest:
    return InferenceRequest(key, key, None, prompt, maximum, inherited, len(inherited))


def test_parent_output_full_page_reused_only_after_completion() -> None:
    config = EngineConfig(max_running_requests=2, max_total_tokens=32, page_size=2)
    engine = Engine(config, PERF, output_targets={0: 2, 1: 1})
    engine.submit(request(0, 2, maximum=2), 0)
    assert engine.step(0, 0) == (2, ())
    assert engine._prefix_cache.page(encode(0, 1)) is None
    assert engine.step(2, 1) == (0, (0,))
    assert engine._prefix_cache.page(encode(0, 1)) is None
    engine.complete(2, 0)
    assert engine._prefix_cache.page(encode(0, 1)) is None  # K=P+O-1=3
    child = request(1, 5, inherited=(encode(0, 0),))
    engine.submit(child, 2)
    assert engine.step(2, 2) == (3, (1,))
    engine.complete(5, 1)
    assert engine.finalize().records[1].actual_hit_tokens == 2


def test_same_batch_identical_inherited_pages_deduplicate_on_publication() -> None:
    config = EngineConfig(max_running_requests=3, max_total_tokens=32, page_size=2)
    engine = Engine(config, PERF, output_targets={0: 1, 1: 1, 2: 1})
    identity = (encode(99, 0),)
    engine.submit(request(0, 3, inherited=identity), 0)
    engine.submit(request(1, 3, inherited=identity), 0)
    assert engine.step(0, 0) == (6, (0, 1))
    page = engine._prefix_cache.page(identity[0])
    assert page is not None and page.ref_count == 2
    assert engine._prefix_cache.peak_resident_pages == 4
    engine.complete(6, 0)
    engine.complete(6, 1)
    engine.submit(request(2, 3, inherited=identity), 6)
    assert engine.step(6, 1) == (1, (2,))
    engine.complete(7, 2)
    assert [record.actual_hit_tokens for record in engine.finalize().records] == [0, 0, 2]


@pytest.mark.parametrize("prompt", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("output", [1, 2, 3, 4, 5])
def test_completed_parent_publishes_only_established_kv(prompt: int, output: int) -> None:
    config = EngineConfig(max_running_requests=1, max_total_tokens=128, page_size=4)
    root_plan = plan_from_arrivals(
        (Arrival(request_id=7, sim_time=0, prompt_len=prompt, output_len=output),), config
    )
    blocks = (prompt + output - 1) // config.page_size
    child = replace(
        root_plan.turns[7],
        request_id=8,
        parent_request_id=7,
        root_arrival_time_ms=None,
        think_time_ms=0,
        prompt_len=(blocks + 1) * config.page_size,
        output_len=1,
        ideal_reuse_blocks=blocks,
    )
    plan = WorkloadPlan({7: root_plan.turns[7], 8: child}, (7,), {7: (8,)}, block_size_tokens=4)
    client = ClientDriver(plan, config)
    engine = Engine(config, PERF, output_targets={7: output, 8: 1})
    parent = client.on_wakeup(client.initial_wakeups()[0])
    assert parent.inherited_prefix_ids == ()
    engine.submit(parent, 0)
    now = 0.0
    for produced in range(1, output + 1):
        duration, finished = engine.step(now, produced - 1)
        now += duration
        assert finished == ((7,) if produced == output else ())
        k = prompt + produced - 1
        state = engine._prefix_cache.state(7)
        assert state.k == k
        assert state.f == (prompt // 4) * 4
        assert engine._prefix_cache.free_pages + engine._prefix_cache.resident_pages == 32
        if produced < output:
            assert engine._prefix_cache.page(encode(7, k // 4)) is None or state.f > k // 4 * 4
    completion = engine.complete(now, 7)
    successor = client.on_wakeup(client.on_request_completed(completion)[0])
    assert successor.inherited_prefix_ids == tuple(encode(7, i) for i in range(blocks))
    for index in range(blocks):
        page = engine._prefix_cache.page(encode(7, index))
        assert page is not None and page.ref_count == 0


def test_distinct_roots_do_not_share_input_pages() -> None:
    config = EngineConfig(max_running_requests=2, max_total_tokens=32, page_size=2)
    engine = Engine(config, PERF, output_targets={0: 1, 1: 1})
    engine.submit(request(0, 2), 0)
    assert engine.step(0, 0) == (2, (0,))
    engine.complete(2, 0)
    engine.submit(request(1, 2), 2)
    assert engine.step(2, 1) == (2, (1,))
    engine.complete(4, 1)
    assert engine.finalize().records[1].matched_cache_tokens == 0
