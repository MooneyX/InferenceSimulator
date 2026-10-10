"""F001 FCFS 准入、物理页预算与动态未来比例的独立验收。"""

import pytest

from simcore.kv_cache import L1KVCache, TieredKVCache
from simcore.prefix_hash import encode
from simcore.scheduler import F001Scheduler, RunningRequest
from simcore.schemas import EngineConfig, InferenceRequest


def make(
    *,
    capacity: int = 1024,
    page_size: int = 64,
    max_running: int = 8,
    max_prefill: int = 16_384,
    exclusive: bool = True,
) -> tuple[F001Scheduler, L1KVCache]:
    config = EngineConfig(
        max_running_requests=max_running,
        max_total_tokens=capacity,
        page_size=page_size,
        max_prefill_tokens=max_prefill,
        pd_mutually_exclusive=exclusive,
    )
    cache = L1KVCache(config.max_total_tokens, config.page_size)
    return F001Scheduler(config, cache), cache


def request(
    key: int, *, prompt: int = 64, maximum: int = 1, inherited: tuple[str, ...] = ()
) -> InferenceRequest:
    return InferenceRequest(key, 0, None, prompt, maximum, inherited, len(inherited))


def hold(cache: TieredKVCache, item: InferenceRequest, length: int | None = None) -> None:
    sources = cache.probe(item, item.prompt_len)
    cache.pin(item, len(sources) * cache.page_size)
    cache.admit(item.request_id)
    cache.extend(item.request_id, item.prompt_len)
    cache.publish(item.request_id)
    if length is not None and length > item.prompt_len:
        cache.extend(item.request_id, length)


def test_first_candidate_can_exceed_compute_quota_but_second_cannot() -> None:
    scheduler, cache = make(capacity=256, page_size=4, max_prefill=4)
    assert scheduler.enqueue(request(0, prompt=40), 0)
    assert scheduler.enqueue(request(1, prompt=4), 0)
    batch = scheduler.select_prefill({})
    assert [candidate.waiting.request.request_id for candidate in batch.admitted] == [0]
    assert batch.admitted[0].required_pages == 10
    assert batch.blocked and scheduler.num_waiting == 1
    assert (batch.current_offset, batch.total_offset) == (44, 45)
    assert cache.resident_pages == 0  # 未来估算和新输入都未在规划时物理分配
    assert cache.free_pages + cache.resident_pages == cache.capacity_pages


def test_fcfs_same_time_sorted_by_id_and_failed_head_not_skipped() -> None:
    scheduler, cache = make(capacity=512)
    for key in (3, 1, 2):
        scheduler.enqueue(request(key, maximum=200), 0)
    batch = scheduler.select_prefill({})
    assert [item.waiting.request.request_id for item in batch.admitted] == [1]
    assert batch.blocked and scheduler.num_waiting == 2
    assert cache.resident_pages == 0
    cache.admit(1)
    cache.extend(1, 64)
    later = scheduler.select_prefill({1: RunningRequest(request(1, maximum=200), 1)})
    assert later.admitted == () and later.blocked and scheduler.num_waiting == 2


def test_total_budget_can_fail_while_immediate_fits_without_reserving_pages() -> None:
    scheduler, cache = make(capacity=512)
    active = request(0)
    hold(cache, active)
    initial = cache.resident_pages
    scheduler.enqueue(request(1, maximum=350), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 1)})
    assert batch.admitted == () and batch.blocked
    assert 128 <= cache.available_tokens < 128 + 350
    assert cache.resident_pages == initial and cache.free_pages == cache.capacity_pages - initial
    assert scheduler.num_waiting == 1


def test_failed_pin_cannot_count_cached_page_as_available_or_refresh_lru() -> None:
    scheduler, cache = make(capacity=320)
    old = request(0)
    hold(cache, old)
    cache.finish(0)
    active = request(9)
    hold(cache, active)
    before = cache.page(encode(0, 0))
    assert before is not None
    scheduler.enqueue(request(1, prompt=65, maximum=80, inherited=(encode(0, 0),)), 1)
    batch = scheduler.select_prefill({9: RunningRequest(active, 1)})
    after = cache.page(encode(0, 0))
    assert batch.admitted == () and batch.blocked and scheduler.num_waiting == 1
    assert after is not None and after.ref_count == 0
    assert after.last_access_seq == before.last_access_seq
    assert cache.available_tokens == 256 and cache.resident_pages == 2


def test_shared_hits_only_pin_one_physical_page() -> None:
    scheduler, cache = make(capacity=384)
    parent = request(0)
    hold(cache, parent)
    cache.finish(0)
    for key in (1, 2):
        scheduler.enqueue(request(key, prompt=65, maximum=1, inherited=(encode(0, 0),)), 1)
    batch = scheduler.select_prefill({})
    assert len(batch.admitted) == 2
    assert [item.reused_tokens for item in batch.admitted] == [64, 64]
    page = cache.page(encode(0, 0))
    assert page is not None and page.ref_count == 2
    assert cache.available_tokens == 320 and cache.resident_pages == 1


def test_running_future_uses_ratio_and_decode_new_is_included() -> None:
    scheduler, cache = make(capacity=512, exclusive=False)
    active = request(0, prompt=25, maximum=200)
    hold(cache, active, 64)
    scheduler.enqueue(request(1, maximum=100), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 40)}, (0,))
    assert batch.decode_new_tokens == 64
    assert batch.current_offset == 192
    assert batch.total_offset == pytest.approx(404)  # 新候选 100；运行者 (200-40)*0.7
    assert len(batch.admitted) == 1
    assert cache.resident_pages == 1  # D_new 仍只是本步待申请量
    assert cache.free_pages + cache.resident_pages == cache.capacity_pages


def test_future_estimates_cap_at_4096_without_physical_reservation() -> None:
    scheduler, cache = make(capacity=16_384)
    active = request(0, maximum=10_000)
    hold(cache, active)
    scheduler.enqueue(request(1, maximum=10_000), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 1)})
    assert batch.current_offset == 128
    assert batch.total_offset == pytest.approx(128 + 4096 + 4096 * 0.7)
    assert len(batch.admitted) == 1
    assert cache.resident_pages == 1


def test_compute_quota_strict_inequality_and_decode_request_credit() -> None:
    scheduler, cache = make(capacity=128, page_size=4, max_prefill=9, exclusive=False)
    active = request(0, prompt=4, maximum=2)
    hold(cache, active)
    for key in (1, 2):
        scheduler.enqueue(request(key, prompt=4), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 1)}, (0,))
    assert [item.waiting.request.request_id for item in batch.admitted] == [1]
    assert batch.blocked and batch.decode_new_tokens == 4 and scheduler.num_waiting == 1
    assert cache.resident_pages == 1


def test_compute_quota_allows_second_candidate_only_below_remaining_limit() -> None:
    scheduler, _ = make(capacity=128, page_size=4, max_prefill=9)
    scheduler.enqueue(request(0, prompt=4), 0)
    scheduler.enqueue(request(1, prompt=4), 0)
    batch = scheduler.select_prefill({})
    assert [entry.waiting.request.request_id for entry in batch.admitted] == [0, 1]
    assert batch.current_offset == 16


def test_head_in_empty_pool_aborts_only_if_budget_cannot_ever_fit() -> None:
    scheduler, cache = make(capacity=192)
    scheduler.enqueue(request(0, prompt=128), 0)
    scheduler.enqueue(request(1, prompt=1), 0)
    batch = scheduler.select_prefill({})
    assert [entry.request.request_id for entry in batch.aborted] == [0]
    assert [entry.waiting.request.request_id for entry in batch.admitted] == [1]
    assert not batch.blocked and scheduler.num_waiting == 0
    assert cache.resident_pages == 0


def test_infeasible_restore_with_pinned_hit_does_not_stall_empty_pool() -> None:
    scheduler, cache = make(capacity=192)
    item = request(0, maximum=10)
    hold(cache, item)
    cache.finish(0)
    scheduler.enqueue(item, 1, generated_tokens=2)
    scheduler.enqueue(request(1, prompt=1), 1)
    batch = scheduler.select_prefill({})
    assert [entry.request.request_id for entry in batch.aborted] == [0]
    assert [candidate.waiting.request.request_id for candidate in batch.admitted] == [1]
    page = cache.page(encode(0, 0))
    assert page is not None and page.ref_count == 0
    assert cache.free_pages + cache.resident_pages == cache.capacity_pages


def test_max_running_requests_is_residency_limit_not_prefill_batch_limit() -> None:
    scheduler, cache = make(capacity=1024, max_running=2)
    active = request(0)
    hold(cache, active)
    scheduler.enqueue(request(1), 1)
    scheduler.enqueue(request(2), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 1)})
    assert [item.waiting.request.request_id for item in batch.admitted] == [1]
    assert batch.blocked and scheduler.num_waiting == 1


def test_unexecutable_request_rejected_at_send_and_never_queued() -> None:
    scheduler, _ = make()
    assert not scheduler.enqueue(request(0, maximum=0), 7)
    assert scheduler.num_waiting == 0
    assert scheduler.select_prefill({}).admitted == ()


def test_restored_full_hit_requires_no_empty_prefill_step() -> None:
    scheduler, cache = make(capacity=192)
    item = request(0, maximum=10)
    hold(cache, item)
    cache.finish(0)
    scheduler.enqueue(item, 1, generated_tokens=1)
    batch = scheduler.select_prefill({})
    assert batch.admitted == () and [i.request.request_id for i in batch.restored] == [0]
    assert batch.current_offset == 0 and cache.state(0).k == cache.state(0).f == 64
    assert cache.private_pages == 0 and cache.resident_pages == 1


def test_ratio_decays_only_for_decode_reestimates_after_withdraw_and_resets_when_idle() -> None:
    scheduler, _ = make()
    active = request(0, maximum=200)
    state = {0: RunningRequest(active, 40)}
    assert scheduler.advance_ratio(state) == 0.7
    assert scheduler.advance_ratio(state, decoded=True) == pytest.approx(0.7 - (0.7 - 0.098) / 600)
    assert scheduler.advance_ratio(state, decoded=True, withdrew=True) == pytest.approx(60 / 201)
    for _ in range(700):
        scheduler.advance_ratio(state, decoded=True)
    assert scheduler.ratio == 0.098
    assert scheduler.advance_ratio({0: RunningRequest(request(0), 1)}, withdrew=True) == 1.0
    assert scheduler.advance_ratio({}, idle=True) == 0.7


def test_exclusive_mode_rejects_mixed_budget_in_same_step() -> None:
    scheduler, cache = make(capacity=256)
    active = request(0)
    hold(cache, active)
    scheduler.enqueue(request(1), 1)
    with pytest.raises(ValueError, match="互斥"):
        scheduler.select_prefill({0: RunningRequest(active, 1)}, (0,))
    assert cache.resident_pages == 1 and scheduler.num_waiting == 1


def test_shared_l2_hit_reserves_one_unique_promotion_for_two_candidates() -> None:
    config = EngineConfig(
        max_running_requests=3,
        max_total_tokens=28,
        page_size=4,
        l2_capacity_tokens=8,
    )
    cache = TieredKVCache(28, 4, 8)
    parent = request(0, prompt=4)
    hold(cache, parent)
    cache.finish(0)
    filler = request(99, prompt=28)
    hold(cache, filler)
    cache.finish(99)
    shared = encode(0, 0)
    initial_page = cache.page(shared)
    assert initial_page is not None and initial_page.tier == "L2"
    scheduler = F001Scheduler(config, cache)
    for key in (1, 2):
        scheduler.enqueue(request(key, prompt=5, inherited=(shared,)), 1)
    batch = scheduler.select_prefill({})
    assert [item.waiting.request.request_id for item in batch.admitted] == [1, 2]
    assert [item.hit_sources for item in batch.admitted] == [((shared, "L2"),)] * 2
    assert batch.promote_ids == (shared,)
    assert batch.current_offset == 20
    pinned_page = cache.page(shared)
    assert pinned_page is not None and pinned_page.ref_count == 2 and pinned_page.tier == "L2"
    assert cache.plan(batch.promote_ids, 2) == ((shared, "L2"),)
    assert cache.promotions == 0


def test_tiered_fcfs_failed_pin_does_not_refresh_lru_or_skip_head() -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=8,
        page_size=4,
        l2_capacity_tokens=4,
    )
    cache = TieredKVCache(8, 4, 4)
    parent = request(0, prompt=4)
    hold(cache, parent)
    cache.finish(0)
    filler = request(99, prompt=8)
    hold(cache, filler)
    shared = encode(0, 0)
    before = cache.page(shared)
    assert before is not None and before.tier == "L2"
    counters = (cache.promotions, cache.demotions)
    scheduler = F001Scheduler(config, cache)
    scheduler.enqueue(request(1, prompt=5, inherited=(shared,)), 1)
    scheduler.enqueue(request(2, prompt=1), 1)
    batch = scheduler.select_prefill({99: RunningRequest(filler, 1)})
    after = cache.page(shared)
    assert batch.admitted == () and batch.blocked and scheduler.num_waiting == 2
    assert after == before and (cache.promotions, cache.demotions) == counters
    assert cache.tier_snapshot("L2").resident_blocks == 1
