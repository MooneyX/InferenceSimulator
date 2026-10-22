"""F001 准入顺序、物理页预算与动态未来比例的独立验收。"""

from collections import deque
from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest

from simcore.kv_cache import TieredKVCache
from simcore.prefix_hash import encode
from simcore.scheduler import AdmissionBatch, AdmissionScheduler, RunningRequest, WaitingRequest
from simcore.schemas import EngineConfig, InferenceRequest


def make(
    *,
    capacity: int = 1024,
    page_size: int = 64,
    max_running: int = 8,
    max_prefill: int = 16_384,
    exclusive: bool = True,
) -> tuple[AdmissionScheduler, TieredKVCache]:
    config = EngineConfig(
        max_running_requests=max_running,
        max_total_tokens=capacity,
        page_size=page_size,
        max_prefill_tokens=max_prefill,
        pd_mutually_exclusive=exclusive,
    )
    cache = TieredKVCache(config.max_total_tokens, config.page_size)
    return AdmissionScheduler(config, cache), cache


def request(
    key: int,
    *,
    prompt: int = 64,
    maximum: int = 1,
    output: int = 1,
    inherited: tuple[str, ...] = (),
) -> InferenceRequest:
    return InferenceRequest(key, 0, None, prompt, maximum, output, inherited, len(inherited))


def hold(cache: TieredKVCache, item: InferenceRequest, length: int | None = None) -> None:
    sources = cache.probe(item, item.prompt_len)
    cache.pin(item, sources)
    cache.admit(item.request_id)
    cache.extend(item.request_id, item.prompt_len)
    cache.publish(item.request_id)
    if length is not None and length > item.prompt_len:
        cache.extend(item.request_id, length)


@pytest.mark.parametrize("running_output", [2, 32])
@pytest.mark.parametrize("waiting_output", [2, 32])
def test_output_length_does_not_affect_current_budget_or_batch(
    running_output: int, waiting_output: int
) -> None:
    scheduler, cache = make(capacity=72, page_size=4)
    active = request(0, prompt=8, maximum=32, output=running_output)
    hold(cache, active)
    running = {0: RunningRequest(active, 1)}
    assert scheduler.advance_ratio(running, withdrew=True) == pytest.approx(21 / 33)
    scheduler.enqueue(request(1, prompt=4, maximum=32, output=waiting_output), 1)
    scheduler.enqueue(request(2, prompt=4, maximum=8, output=2), 1)
    batch = scheduler.select_prefill(running)
    assert [item.waiting.request.request_id for item in batch.admitted] == [1]
    assert batch.admitted[0].required_pages == 1
    assert not batch.restored and not batch.aborted and not batch.restore_only
    assert scheduler.head_request_id == 2 and scheduler.num_waiting == 1
    assert cache.resident_pages == 2 and cache.free_pages == 16


def test_first_candidate_can_exceed_compute_quota_but_second_cannot() -> None:
    scheduler, cache = make(capacity=256, page_size=4, max_prefill=4)
    assert scheduler.enqueue(request(0, prompt=40), 0)
    assert scheduler.enqueue(request(1, prompt=4), 0)
    batch = scheduler.select_prefill({})
    assert [candidate.waiting.request.request_id for candidate in batch.admitted] == [0]
    assert batch.admitted[0].required_pages == 10
    assert scheduler.num_waiting == 1
    assert cache.resident_pages == 0  # 未来估算和新输入都未在规划时物理分配
    assert cache.free_pages + cache.resident_pages == cache.capacity_pages


def test_same_time_sorted_by_id_and_budget_blocked_candidate_stops_batch() -> None:
    scheduler, cache = make(capacity=512)
    for key in (3, 1, 2):
        scheduler.enqueue(request(key, maximum=200), 0)
    batch = scheduler.select_prefill({})
    assert [item.waiting.request.request_id for item in batch.admitted] == [1]
    assert scheduler.num_waiting == 2
    assert cache.resident_pages == 0
    cache.admit(1)
    cache.extend(1, 64)
    later = scheduler.select_prefill({1: RunningRequest(request(1, maximum=200), 1)})
    assert later.admitted == () and scheduler.num_waiting == 2


def test_total_budget_can_fail_while_immediate_fits_without_reserving_pages() -> None:
    scheduler, cache = make(capacity=512)
    active = request(0)
    hold(cache, active)
    initial = cache.resident_pages
    scheduler.enqueue(request(1, maximum=350), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 1)})
    assert batch.admitted == ()
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
    assert batch.admitted == () and scheduler.num_waiting == 1
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


def test_running_future_uses_ratio_without_physical_reservation() -> None:
    scheduler, cache = make(capacity=512, exclusive=False)
    active = request(0, prompt=25, maximum=200)
    hold(cache, active, 64)
    scheduler.enqueue(request(1, maximum=100), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 40)})
    assert len(batch.admitted) == 1
    assert cache.resident_pages == 1
    assert cache.free_pages + cache.resident_pages == cache.capacity_pages


def test_future_estimates_cap_at_4096_without_physical_reservation() -> None:
    scheduler, cache = make(capacity=16_384)
    active = request(0, maximum=10_000)
    hold(cache, active)
    scheduler.enqueue(request(1, maximum=10_000), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 1)})
    assert len(batch.admitted) == 1
    assert cache.resident_pages == 1


def test_compute_quota_strict_inequality_with_running_request() -> None:
    scheduler, cache = make(capacity=128, page_size=4, max_prefill=8, exclusive=False)
    active = request(0, prompt=4, maximum=2)
    hold(cache, active)
    for key in (1, 2):
        scheduler.enqueue(request(key, prompt=4), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 1)})
    assert [item.waiting.request.request_id for item in batch.admitted] == [1]
    assert scheduler.num_waiting == 1
    assert cache.resident_pages == 1


def test_compute_quota_allows_second_candidate_only_below_remaining_limit() -> None:
    scheduler, _ = make(capacity=128, page_size=4, max_prefill=9)
    scheduler.enqueue(request(0, prompt=4), 0)
    scheduler.enqueue(request(1, prompt=4), 0)
    batch = scheduler.select_prefill({})
    assert [entry.waiting.request.request_id for entry in batch.admitted] == [0, 1]


def test_head_in_empty_pool_aborts_only_if_budget_cannot_ever_fit() -> None:
    scheduler, cache = make(capacity=192)
    scheduler.enqueue(request(0, prompt=129), 0)
    scheduler.enqueue(request(1, prompt=1), 0)
    batch = scheduler.select_prefill({})
    assert [entry.request.request_id for entry in batch.aborted] == [0]
    assert [entry.waiting.request.request_id for entry in batch.admitted] == [1]
    assert scheduler.num_waiting == 0
    assert cache.resident_pages == 0


def test_infeasible_restore_with_pinned_hit_does_not_stall_empty_pool() -> None:
    scheduler, cache = make(capacity=192)
    item = request(0, maximum=100)
    hold(cache, item)
    cache.finish(0)
    scheduler.enqueue(item, 1, generated_tokens=66)
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
    assert scheduler.num_waiting == 1


@pytest.mark.parametrize("waiting_count", [0, 1024])
def test_full_running_pool_does_not_iterate_waiting_queue(waiting_count: int) -> None:
    class TrackedWaitingQueue(deque[WaitingRequest]):
        iterations = 0

        def __iter__(self) -> Iterator[WaitingRequest]:
            self.iterations += 1
            return super().__iter__()

    scheduler, cache = make(max_running=1)
    active = request(0, maximum=32)
    hold(cache, active)
    running = {0: RunningRequest(active, 1)}
    for key in range(1, waiting_count + 1):
        scheduler.enqueue(request(key), 1)
    expected_waiting = tuple(scheduler._waiting)
    expected_queued = scheduler._queued.copy()
    waiting = TrackedWaitingQueue(scheduler._waiting)
    scheduler._waiting = waiting
    is_ready = Mock(return_value=True)
    accept_probe = Mock(return_value=True)

    with (
        patch.object(cache, "state", wraps=cache.state) as states,
        patch.object(cache, "probe", wraps=cache.probe) as probes,
        patch.object(cache, "pin", wraps=cache.pin) as pins,
    ):
        for _ in range(3):
            batch = scheduler.select_prefill(
                running, is_ready=is_ready, accept_probe=accept_probe
            )
            assert batch == AdmissionBatch((), (), ())

    assert states.call_count == 3
    states.assert_called_with(0)
    assert waiting.iterations == 0
    assert tuple(waiting) == expected_waiting
    assert scheduler._queued == expected_queued
    is_ready.assert_not_called()
    accept_probe.assert_not_called()
    probes.assert_not_called()
    pins.assert_not_called()

    cache.finish(0)
    batch = scheduler.select_prefill({})
    assert [item.waiting.request.request_id for item in batch.admitted] == (
        [1] if waiting_count else []
    )
    assert scheduler.num_waiting == max(waiting_count - 1, 0)


@pytest.mark.parametrize("key, generated", [(1, 1), (0, 0), (0, 3)])
def test_full_running_pool_still_validates_running_requests(key: int, generated: int) -> None:
    scheduler, cache = make(max_running=1)
    active = request(0, maximum=2)
    hold(cache, active)

    with pytest.raises(ValueError, match="运行请求身份或生成进度不合法"):
        scheduler.select_prefill({key: RunningRequest(active, generated)})


@pytest.mark.parametrize("maximum", [100, 65_536])
def test_empty_pool_reduces_future_budget_instead_of_rejecting_feasible_input(
    maximum: int,
) -> None:
    scheduler, cache = make(capacity=192)
    assert scheduler.enqueue(request(0, prompt=65, maximum=maximum), 7)
    batch = scheduler.select_prefill({})
    assert [entry.waiting.request.request_id for entry in batch.admitted] == [0]
    assert not batch.aborted and scheduler.num_waiting == 0
    assert batch.admitted[0].waiting.request.max_new_tokens == maximum
    assert batch.admitted[0].required_pages == 2
    assert cache.resident_pages == 0 and cache.snapshot().reserved_future_blocks == 0


def test_prefetch_waiting_head_is_skipped_without_pinning_and_keeps_position() -> None:
    scheduler, cache = make()
    for key in range(3):
        scheduler.enqueue(request(key), key)
    assert scheduler.has_ready(lambda key: key != 0)
    batch = scheduler.select_prefill({}, is_ready=lambda key: key != 0)
    assert [item.waiting.request.request_id for item in batch.admitted] == [1, 2]
    assert scheduler.num_waiting == 1 and scheduler.head_request_id == 0
    assert 0 not in cache._pins and 1 in cache._pins and 2 in cache._pins
    assert not scheduler.has_ready(lambda key: key != 0)
    ready = scheduler.select_prefill({}, is_ready=lambda _: True)
    assert [item.waiting.request.request_id for item in ready.admitted] == [0]


@pytest.mark.parametrize("restoring", [False, True])
@pytest.mark.parametrize("remote_index", [0, 2])
def test_admission_stops_at_first_l3_only_page(
    restoring: bool,
    remote_index: int,
) -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=32,
        page_size=4,
        l2_capacity_tokens=8,
        l3_capacity_tokens=8,
    )
    cache = TieredKVCache(32, 4, 8, 8)
    ids = tuple(encode(0, index) for index in range(4))
    tiers = ("L1", "L2", "L3", "L1") if remote_index == 2 else ("L3", "L1", "L2", "L1")
    for chunk_id, tier in zip(ids, tiers, strict=True):
        cache.pools[tier].insert(chunk_id)
    scheduler = AdmissionScheduler(config, cache)
    item = request(1, prompt=17, maximum=2, output=2, inherited=ids)
    scheduler.enqueue(item, 0, generated_tokens=int(restoring))

    batch = scheduler.select_prefill({})

    assert len(batch.admitted) == 1
    candidate = batch.admitted[0]
    assert candidate.hit_sources == tuple(
        zip(ids[:remote_index], tiers[:remote_index], strict=True)
    )
    assert candidate.reused_tokens == remote_index * 4
    assert candidate.new_tokens == 17 - remote_index * 4
    assert candidate.required_pages == 5 - remote_index
    assert batch.promote_ids == ((ids[1],) if remote_index == 2 else ())
    assert not batch.restored and not batch.restore_only and not batch.aborted
    for index, (chunk_id, tier) in enumerate(zip(ids, tiers, strict=True)):
        page = cache.pools[tier].lookup(chunk_id)
        assert page is not None and page.ref_count == int(index < remote_index)


@pytest.mark.parametrize("capacity, restores", [(16, False), (20, True)])
def test_full_l1_restore_preserves_selected_capacity_and_queue_order(
    capacity: int,
    restores: bool,
) -> None:
    scheduler, cache = make(capacity=capacity, page_size=4)
    item = request(0, prompt=8, maximum=2, output=2)
    hold(cache, item)
    cache.withdraw(0)
    scheduler.enqueue(request(1, prompt=8), 1)
    scheduler.requeue(item, 2, generated_tokens=1)
    scheduler.enqueue(request(2, prompt=4), 3)

    batch = scheduler.select_prefill({})

    assert [candidate.waiting.request.request_id for candidate in batch.admitted] == [1]
    assert [entry.request.request_id for entry in batch.restored] == ([0] if restores else [])
    assert scheduler.head_request_id == (2 if restores else 0)
    assert scheduler.num_waiting == (1 if restores else 2)
    assert 0 not in cache._pins and 2 not in cache._pins
    assert cache.available_tokens == (12 if restores else 16)
    for index in range(2):
        page = cache.page(encode(0, index))
        assert page is not None and page.ref_count == int(restores)
    cache.ensure_free(batch.admitted[0].required_pages)

    if not restores:
        cache.admit(1)
        cache.extend(1, 8)
        cache.finish(1)
        retried = scheduler.select_prefill({}, is_ready=lambda key: key == 0)
        assert [entry.request.request_id for entry in retried.restored] == [0]
        assert scheduler.head_request_id == 2


def test_restored_full_hit_requires_no_empty_prefill_step() -> None:
    scheduler, cache = make(capacity=192)
    item = request(0, maximum=10)
    hold(cache, item)
    cache.finish(0)
    scheduler.enqueue(item, 1, generated_tokens=1)
    batch = scheduler.select_prefill({})
    assert batch.admitted == () and [i.request.request_id for i in batch.restored] == [0]
    assert cache.state(0).k == cache.state(0).f == 64
    assert cache.private_page_count(0) == 0 and cache.resident_pages == 1


@pytest.mark.parametrize("mode", ["l1", "l2", "recompute"])
def test_unified_admission_exempts_only_direct_restore_from_future_budget(mode: str) -> None:
    config = EngineConfig(
        max_running_requests=3,
        max_total_tokens=512,
        page_size=64,
        l2_capacity_tokens=128,
        max_prefill_tokens=128,
    )
    cache = TieredKVCache(512, 64, 128)
    item = request(0, prompt=128, maximum=65_536, output=3)
    hold(cache, item)
    cache.withdraw(0)
    if mode == "l2":
        for index in range(2):
            chunk_id = encode(0, index)
            cache.reserve_transfer(chunk_id, "L2", source="L1")
            cache.complete_transfer(chunk_id)
    scheduler = AdmissionScheduler(config, cache)
    scheduler.enqueue(request(1), 1)
    scheduler.requeue(item, 2, generated_tokens=2 if mode == "recompute" else 1)

    batch = scheduler.select_prefill({})

    assert [candidate.waiting.request.request_id for candidate in batch.admitted] == [1]
    assert not batch.restore_only and not batch.promote_ids and not batch.aborted
    assert [entry.request.request_id for entry in batch.restored] == ([0] if mode == "l1" else [])
    assert scheduler.num_waiting == (0 if mode == "l1" else 1)
    assert scheduler.head_request_id == (None if mode == "l1" else 0)
    assert 0 not in cache._pins
    for index in range(2):
        page = cache.page(encode(0, index))
        assert page is not None and page.ref_count == int(mode == "l1")


def test_direct_restore_does_not_add_growth_reservation_to_full_l1() -> None:
    scheduler, cache = make(capacity=64)
    item = request(0, maximum=65_536, output=2)
    hold(cache, item)
    cache.withdraw(0)
    scheduler.requeue(item, 1, generated_tokens=1)

    batch = scheduler.select_prefill({})

    assert [entry.request.request_id for entry in batch.restored] == [0]
    assert not batch.admitted and not batch.aborted and not batch.restore_only
    assert cache.available_tokens == 0 and cache.free_pages == 0
    assert scheduler.num_waiting == 0


def test_ratio_decays_only_for_decode_reestimates_after_withdraw_and_resets_when_idle() -> None:
    scheduler, _ = make()
    active = request(0, maximum=200)
    state = {0: RunningRequest(active, 40)}
    assert scheduler.advance_ratio(state) == 0.7
    assert scheduler.advance_ratio(state, decoded=True) == pytest.approx(0.7 - (0.7 - 0.098) / 600)
    assert scheduler.advance_ratio(state, decoded=True, withdrew=True) == pytest.approx(60 / 201)
    for _ in range(700):
        ratio = scheduler.advance_ratio(state, decoded=True)
    assert ratio == 0.098
    assert scheduler.advance_ratio({0: RunningRequest(request(0), 1)}, withdrew=True) == 1.0
    assert scheduler.advance_ratio({}, idle=True) == 0.7


@pytest.mark.parametrize("exclusive", [True, False])
def test_prefill_budget_uses_running_future_in_both_modes(exclusive: bool) -> None:
    scheduler, cache = make(capacity=256, exclusive=exclusive)
    active = request(0)
    hold(cache, active)
    scheduler.enqueue(request(1), 1)
    batch = scheduler.select_prefill({0: RunningRequest(active, 1)})
    assert len(batch.admitted) == 1
    assert cache.resident_pages == 1 and scheduler.num_waiting == 0


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
    scheduler = AdmissionScheduler(config, cache)
    for key in (1, 2):
        scheduler.enqueue(request(key, prompt=5, inherited=(shared,)), 1)
    batch = scheduler.select_prefill({})
    assert [item.waiting.request.request_id for item in batch.admitted] == [1, 2]
    assert [item.hit_sources for item in batch.admitted] == [((shared, "L2"),)] * 2
    assert batch.promote_ids == (shared,)
    pinned_page = cache.page(shared)
    assert pinned_page is not None and pinned_page.ref_count == 2 and pinned_page.tier == "L2"
    assert cache.promotions == 0
    assert cache.reserve_transfer(shared, "L1", source="L2") == "L2"
    in_flight = cache.page(shared)
    assert in_flight is not None and in_flight.tier == "L2" and in_flight.ref_count == 3
    assert cache.complete_transfer(shared) == "L2"
    cache.ensure_free(sum(item.required_pages for item in batch.admitted))
    for item in batch.admitted:
        key = item.waiting.request.request_id
        cache.admit(key)
        cache.extend(key, item.context_after)
        cache.publish(key)
    promoted_page = cache.page(shared)
    assert promoted_page is not None and promoted_page.tier == "L1" and promoted_page.ref_count == 2
    assert cache.promotions == 1
    for key in (1, 2):
        cache.finish(key)


@pytest.mark.parametrize("trace", [False, True])
def test_promotions_follow_context_and_candidate_order_with_shared_restore(trace: bool) -> None:
    config = EngineConfig(
        max_running_requests=3, max_total_tokens=64, page_size=4, l2_capacity_tokens=64
    )
    cache = TieredKVCache(64, 4, 64)
    scheduler = AdmissionScheduler(config, cache)
    hashes = (2, 10, 11, 3, 1)
    ids = (
        tuple(f"trace:0:{block}" for block in hashes)
        if trace
        else (encode(90, 2), encode(7, 10), encode(90, 11), encode(3, 0), encode(1, 0))
    )
    cache.pools["L1"].insert(ids[0])
    for chunk_id in ids[1:]:
        cache.pools["L2"].insert(chunk_id)
    # 到达顺序与请求 ID、块 ID 排序均不同；中间穿插全命中的恢复候选。
    for arrival, (key, positions, prompt, generated) in enumerate(
        (
            (9, (0, 1, 2), 13, 0),
            (1, (0, 1, 3), 12, 1),
            (2, (0, 1, 2, 4), 17, 0),
        )
    ):
        item = InferenceRequest(
            request_id=key,
            session_id=0,
            parent_request_id=None,
            prompt_len=prompt,
            max_new_tokens=1,
            output_len=1,
            inherited_prefix_ids=() if trace else tuple(ids[i] for i in positions),
            ideal_reuse_blocks=len(positions),
            trace_hash_ids=tuple(hashes[i] for i in positions) if trace else (),
            is_trace=trace,
        )
        scheduler.enqueue(item, arrival, generated_tokens=generated)

    batch = scheduler.select_prefill({})

    assert [candidate.waiting.request.request_id for candidate in batch.admitted] == [9, 2]
    assert [item.request.request_id for item in batch.restore_only] == [1]
    assert batch.promote_ids == (ids[1], ids[2], ids[3], ids[4])
    assert cache.pools["L1"].entries[ids[0]].ref_count == 3
    assert cache.pools["L2"].entries[ids[1]].ref_count == 3
    assert len(batch.promote_ids) == 4  # 共享页只消耗一份回载容量。


def test_rejected_candidate_does_not_contribute_promotions() -> None:
    config = EngineConfig(
        max_running_requests=2,
        max_total_tokens=64,
        page_size=4,
        l2_capacity_tokens=16,
        max_prefill_tokens=4,
    )
    cache = TieredKVCache(64, 4, 16)
    scheduler = AdmissionScheduler(config, cache)
    first, rejected = encode(9, 10), encode(1, 2)
    cache.pools["L2"].insert(first)
    cache.pools["L2"].insert(rejected)
    scheduler.enqueue(request(9, prompt=5, inherited=(first,)), 0)
    scheduler.enqueue(request(1, prompt=5, inherited=(rejected,)), 1)

    batch = scheduler.select_prefill({})

    assert [candidate.waiting.request.request_id for candidate in batch.admitted] == [9]
    assert batch.promote_ids == (first,)
    assert scheduler.head_request_id == 1
    assert 1 not in cache._pins
    assert cache.pools["L2"].entries[rejected].ref_count == 0


def test_probe_rejection_precedes_pin_and_reuses_accepted_sources() -> None:
    scheduler, cache = make(capacity=512, page_size=4)
    shared = encode(90, 0)
    cache.pools["L1"].insert(shared)
    for key in (1, 2):
        scheduler.enqueue(request(key, prompt=5, inherited=(shared,)), 0)
    checked: dict[int, tuple[tuple[str, str], ...]] = {}

    def accept(item: WaitingRequest, sources: tuple[tuple[str, str], ...]) -> bool:
        assert item.request.request_id not in cache._pins
        checked[item.request.request_id] = sources
        return item.request.request_id == 2

    with (
        patch.object(cache, "probe", wraps=cache.probe) as probes,
        patch.object(cache, "pin", wraps=cache.pin) as pins,
    ):
        batch = scheduler.select_prefill({}, accept_probe=accept)
    assert probes.call_count == 2
    assert pins.call_count == 1 and pins.call_args.args[0].request_id == 2
    assert batch.admitted[0].hit_sources is checked[2]
    assert pins.call_args.args[1] is checked[2]
    assert scheduler.head_request_id == 1 and 1 not in cache._pins
    assert cache.pools["L1"].entries[shared].ref_count == 1


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
    scheduler = AdmissionScheduler(config, cache)
    scheduler.enqueue(request(1, prompt=5, inherited=(shared,)), 1)
    scheduler.enqueue(request(2, prompt=1), 1)
    batch = scheduler.select_prefill({99: RunningRequest(filler, 1)})
    after = cache.page(shared)
    assert batch.admitted == () and scheduler.num_waiting == 2
    assert after == before and (cache.promotions, cache.demotions) == counters
    assert cache.tier_snapshot("L2").resident_blocks == 1
