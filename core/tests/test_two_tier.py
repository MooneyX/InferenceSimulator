"""两层结构（L2 关闭、L3 启用）：到达只查找并保护，准入时 L3 页直接读入 L1。"""

from dataclasses import replace

import pytest

from simcore import simulate
from simcore.engine import Engine
from simcore.events import EventKind
from simcore.kv_cache import CachePool, TieredKVCache
from simcore.prefix_hash import encode
from simcore.schemas import (
    EngineConfig,
    FixedLength,
    InferenceRequest,
    PerfModelConfig,
    SessionConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
)
from test_schemas import resolved_config, runtime_config


def request(key: int, ids: tuple[str, ...]) -> InferenceRequest:
    return InferenceRequest(key, 0, None, len(ids) * 4 + 1, 1, 1, ids, len(ids))


def engine(
    shared: CachePool | None = None,
    *,
    min_tokens: int = 0,
    l1_tokens: int = 32,
    layerwise: bool = True,
) -> Engine:
    runtime = runtime_config(
        EngineConfig(
            max_running_requests=2,
            max_total_tokens=l1_tokens,
            page_size=4,
            l3_capacity_tokens=64,
            l3_prefetch_min_tokens=min_tokens,
        ),
        PerfModelConfig(
            kv_bytes_per_token=16,
            l3_read_bandwidth_bytes_per_ms=64,
            l3_latency_ms=1,
            l3_lookup_latency_ms=2,
        ),
    )
    transfers = replace(runtime.transfers, l3_layerwise=layerwise)
    return Engine(replace(runtime, transfers=transfers), shared_l3=shared)


def admit_two_hits(layerwise: bool) -> tuple[Engine, CachePool, tuple[str, ...], float]:
    shared = CachePool("L3", 9, 16, 4)
    ids = (encode(90, 0), encode(90, 1))
    for chunk in ids:
        shared.insert(chunk)
    runner = engine(shared, layerwise=layerwise)
    assert runner.submit(request(0, ids), 0)
    assert runner.drain_events() == ((2, EventKind.L3_LOOKUP_DONE, 0),)
    runner.on_l3_lookup(0, 2)
    assert runner.drain_events() == ()  # 到达只查找，不搬数据
    assert [shared.lookup(chunk).ref_count for chunk in ids] == [1, 1]
    assert runner.can_step
    started = runner.start_step(2, 0)
    assert started is not None
    duration, finished = started
    assert finished == (0,)
    return runner, shared, ids, duration


def test_l1_eviction_writes_directly_to_l3_without_l2() -> None:
    cache = TieredKVCache(16, 4, 0, 16)
    assert cache.direct_l3
    ids = tuple(encode(90, i) for i in range(4))
    for chunk in ids:
        cache.pools["L1"].insert(chunk)
    assert cache.ensure_free(1) == 1
    assert cache.pools["L3"].lookup(ids[0]) is not None
    assert cache.pools["L2"].resident == 0
    assert cache.demotions == 1
    assert not TieredKVCache(16, 4, 8, 16).direct_l3


def test_layerwise_read_pipelines_with_compute_and_locks_until_last_layer() -> None:
    runner, shared, ids, duration = admit_two_hits(layerwise=True)
    # 2 层、每层读 2 页 × 32 B：每层固定延迟 1 ms + 64 B ÷ 64 B/ms，逐层到齐于 +2、+4 ms。
    # 首层计算与第二层读取重叠：F_1 = 2 + c，F_2 = max(F_1, 4) + c。
    c = runner._analysis.device_ms / 2
    assert c < 2
    assert duration == pytest.approx(4 + c)
    assert runner._analysis.restore_exposed_ms == pytest.approx(duration - 2 * c)
    assert list(runner._l3_stalls_ms) == pytest.approx([duration - 2 * c])  # 逐步停顿
    assert runner._restore_bytes == 128
    (end, kind, read_id), = runner.drain_events()  # 全部层一次读取，末层读完才释放源页
    assert (end, kind) == (6, EventKind.L3_L1_DONE)
    l1 = runner._prefix_cache.pools["L1"]
    assert all(l1.lookup(chunk) is not None for chunk in ids)
    assert not runner._lookup_locks
    assert [shared.lookup(chunk).ref_count for chunk in ids] == [1, 1]
    runner.on_direct_read_done(read_id)
    assert [shared.lookup(chunk).ref_count for chunk in ids] == [0, 0]


def test_blocking_read_waits_for_all_batches_before_compute() -> None:
    runner, shared, ids, duration = admit_two_hits(layerwise=False)
    # 一批 2 页：固定延迟 1 ms + 2 页 × 64 B ÷ 64 B/ms；本步计算等读取完成。
    assert runner._analysis.restore_exposed_ms == 3
    assert list(runner._l3_stalls_ms) == [3]
    assert duration == 3 + runner._analysis.device_ms
    assert runner._restore_bytes == 128
    (end, kind, read_id), = runner.drain_events()
    assert (end, kind) == (5, EventKind.L3_L1_DONE)
    l1 = runner._prefix_cache.pools["L1"]
    assert all(l1.lookup(chunk) is not None for chunk in ids)
    assert not runner._lookup_locks
    assert [shared.lookup(chunk).ref_count for chunk in ids] == [1, 1]  # 读完前保护源页
    runner.on_direct_read_done(read_id)
    assert [shared.lookup(chunk).ref_count for chunk in ids] == [0, 0]
    runner.finish_step(2 + duration)
    runner.complete(2 + duration, 0)
    record, = runner.finalize().records
    assert (record.l1_hit_tokens, record.l2_hit_tokens, record.l3_hit_tokens) == (0, 0, 8)


def test_short_l3_hit_is_not_locked_and_recomputed() -> None:
    shared = CachePool("L3", 9, 16, 4)
    ids = (encode(90, 0), encode(90, 1))
    for chunk in ids:
        shared.insert(chunk)
    runner = engine(shared, min_tokens=8)
    runner.submit(request(0, ids), 0)
    runner.drain_events()
    runner.on_l3_lookup(0, 2)
    assert not runner._lookup_locks
    assert runner.start_step(2, 0) is not None
    assert runner.drain_events() == ()
    assert runner._restore_bytes == 0
    assert [shared.lookup(chunk).ref_count for chunk in ids] == [0, 0]


def test_lookup_lock_protects_hits_from_other_instance_eviction() -> None:
    shared = CachePool("L3", 9, 2, 4)
    ids = (encode(90, 0), encode(90, 1))
    for chunk in ids:
        shared.insert(chunk)
    runner = engine(shared)
    runner.submit(request(0, ids), 0)
    runner.drain_events()
    runner.on_l3_lookup(0, 2)
    other = TieredKVCache(16, 4, 0, 8, shared_l3=shared, pool_id_base=2)
    for index in range(4):
        other.pools["L1"].insert(encode(91, index))
    other.ensure_free(1)  # L1 淘汰页写 L3 失败时直接丢弃，不挤掉被保护的命中页
    assert all(shared.lookup(chunk) is not None for chunk in ids)
    assert shared.lookup(encode(91, 0)) is None


def test_pages_evicted_while_waiting_are_read_from_l3_without_new_lookup() -> None:
    runner = engine(l1_tokens=16)
    cache = runner._prefix_cache
    ids = (encode(90, 0), encode(90, 1))
    for chunk in ids:
        cache.pools["L1"].insert(chunk)
    runner.submit(request(0, ids), 0)
    assert runner.drain_events() == ()  # 本地命中完整，不查 L3
    cache.ensure_free(cache.capacity_pages)
    assert all(cache.pools["L3"].lookup(chunk) is not None for chunk in ids)
    assert runner.start_step(0, 0) is not None
    (_, kind, _), = runner.drain_events()
    assert kind is EventKind.L3_L1_DONE
    assert runner._restore_bytes == 128


@pytest.mark.parametrize("layerwise", [True, False])
def test_two_tier_simulation_reuses_l3_without_l2(layerwise: bool) -> None:
    config = resolved_config(
        num_instances=1,
        workload=WorkloadConfig(
            session_rate=100,
            seed=3,
            synthetic=SyntheticWorkloadConfig(
                num_requests=12,
                prompt_len=FixedLength(value=16),
                turn_increment_len=FixedLength(value=4),
                output_len=FixedLength(value=2),
                session=SessionConfig(num_sessions=3),
            ),
        ),
        engine=EngineConfig(
            max_running_requests=2,
            max_total_tokens=48,
            page_size=4,
            l3_capacity_tokens=512,
            l3_prefetch_min_tokens=0,
        ),
        perf_model=PerfModelConfig(
            kv_bytes_per_token=16,
            l3_read_bandwidth_bytes_per_ms=64,
            l3_lookup_latency_ms=2,
        ),
    )
    storage = config.inference.storage.model_copy(update={"l3_layerwise_load": layerwise})
    config = config.model_copy(
        update={"inference": config.inference.model_copy(update={"storage": storage})}
    )
    assert config.inference.storage.l2_available_gb == 0
    result = simulate(config)
    metrics = result.metrics
    assert result.meta.num_completed == 12
    assert metrics.l2_hit_tokens == 0
    assert metrics.l3_hit_tokens > 0
    assert metrics.restore_bytes > 0
    # 两层结构的暴露回载全部来自 L3→L1 读取，逐步停顿之和与之相等。
    assert metrics.l3_read_steps > 0 and result.analysis is not None
    assert metrics.l3_stall_ms == pytest.approx(result.analysis.restore_exposed_ms)
    assert metrics.l3_stall_max is not None and metrics.l3_stall_p99 is not None
    assert 0 <= metrics.l3_stall_p99 <= metrics.l3_stall_max
    assert metrics.l3_stall_fraction is not None and 0 < metrics.l3_stall_fraction <= 1


def test_three_tier_l2_restore_is_not_counted_as_l3_stall() -> None:
    runtime = runtime_config(
        EngineConfig(
            max_running_requests=2, max_total_tokens=32, page_size=4, l2_capacity_tokens=32
        ),
        PerfModelConfig(kv_bytes_per_token=16, l2_to_l1_bandwidth_bytes_per_ms=64),
    )
    runner = Engine(runtime)
    cache = runner._prefix_cache
    ids = (encode(90, 0), encode(90, 1))
    for chunk in ids:
        cache.pools["L1"].insert(chunk)
    cache.ensure_free(cache.capacity_pages)  # 命中页降至 L2
    assert all(cache.pools["L2"].lookup(chunk) is not None for chunk in ids)
    runner.submit(request(0, ids), 0)
    assert runner.start_step(0, 0) is not None
    assert runner._restore_bytes == 128 and runner._analysis.restore_exposed_ms > 0
    assert len(runner._l3_stalls_ms) == 0
