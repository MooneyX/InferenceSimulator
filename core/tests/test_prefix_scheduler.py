"""F001 FCFS 准入接入：同批不可互见、队头不跳过和空池失败。"""

from simcore.kv_cache import TieredKVCache
from simcore.prefix_hash import encode
from simcore.scheduler import F001Scheduler, RunningRequest
from simcore.schemas import EngineConfig, InferenceRequest


def setup(capacity: int = 80) -> tuple[F001Scheduler, TieredKVCache]:
    engine = EngineConfig(max_running_requests=3, max_total_tokens=capacity, page_size=4)
    cache = TieredKVCache(engine.max_total_tokens, engine.page_size)
    return F001Scheduler(engine, cache), cache


def request(
    key: int, *, prompt: int = 5, maximum: int = 1, inherited: tuple[str, ...] = ()
) -> InferenceRequest:
    return InferenceRequest(key, key, None, prompt, maximum, 1, inherited, len(inherited))


def test_same_batch_new_pages_not_visible_until_publication() -> None:
    scheduler, cache = setup()
    inherited = (encode(0, 0),)
    scheduler.enqueue(request(1, inherited=inherited), 0)
    scheduler.enqueue(request(2, inherited=inherited), 0)
    batch = scheduler.select_prefill({})
    assert [
        (candidate.waiting.request.request_id, candidate.reused_tokens)
        for candidate in batch.admitted
    ] == [(1, 0), (2, 0)]
    for item in batch.admitted:
        cache.admit(item.waiting.request.request_id)
        cache.extend(item.waiting.request.request_id, item.context_after)
    assert cache.page(inherited[0]) is None
    assert sum(cache.private_page_count(key) for key in (1, 2)) == 4
    assert cache.publish_batch((2, 1)) == 1
    assert cache.page(inherited[0]) is not None and cache.page(inherited[0]).ref_count == 2  # type: ignore[union-attr]
    cache.finish(1)
    cache.finish(2)
    scheduler.enqueue(request(3, inherited=inherited), 1)
    later = scheduler.select_prefill({})
    assert later.admitted[0].reused_tokens == 4


def test_fcfs_failed_head_remains_waiting_while_running_decodes() -> None:
    scheduler, cache = setup(capacity=32)
    running = request(0, maximum=2)
    cache.pin(running, 0)
    cache.admit(0)
    cache.extend(0, 5)
    cache.publish(0)
    scheduler.enqueue(request(1, maximum=30), 1)
    scheduler.enqueue(request(2), 1)
    batch = scheduler.select_prefill({0: RunningRequest(running, 1)})
    assert batch.admitted == ()
    assert scheduler.num_waiting == 2
    assert cache.free_pages + cache.resident_pages == cache.capacity_pages


def test_permanently_infeasible_head_aborts_then_admits_next_request() -> None:
    scheduler, cache = setup(capacity=16)
    scheduler.enqueue(request(0, prompt=16), 0)
    scheduler.enqueue(request(1, prompt=1), 0)
    batch = scheduler.select_prefill({})
    assert [item.request.request_id for item in batch.aborted] == [0]
    assert [item.waiting.request.request_id for item in batch.admitted] == [1]
    assert cache.resident_pages == 0 and scheduler.num_waiting == 0
