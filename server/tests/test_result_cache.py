"""持久缓存、内容身份、并发合并和 HTTP 契约。"""

import json
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from conftest import run_simulation
from simcore import SimConfig, SimResult, simulate
from simcore.schemas import TracePlan
from simserver import create_app
from simserver.result_cache import ResultCache, engine_fingerprint
from test_routes import make_payload


@pytest.fixture
def config() -> SimConfig:
    return SimConfig.model_validate(make_payload(num_requests=2))


@pytest.fixture
def result(config: SimConfig) -> SimResult:
    return simulate(config)


def test_lazy_creation_and_persistence(
    tmp_path: Path, config: SimConfig, result: SimResult
) -> None:
    path = tmp_path / "cache" / "results.sqlite3"
    cache = ResultCache(path, fingerprint="engine-1")
    assert not path.exists()
    first = cache.run(config, None, lambda: result)
    assert first.status == "miss"
    restarted = ResultCache(path, fingerprint="engine-1")
    compute = Mock(return_value=result)
    second = restarted.run(config, None, compute)
    assert second.status == "hit"
    compute.assert_not_called()
    assert SimResult.model_validate_json(second.content) == result
    assert second.lookup_ms >= 0
    assert ResultCache(path, fingerprint="engine-2").run(config, None, compute).status == "miss"
    compute.assert_called_once()


def test_default_values_and_object_order_share_key(tmp_path: Path) -> None:
    payload = make_payload(num_requests=2)
    normalized = SimConfig.model_validate(payload)
    explicit = SimConfig.model_validate(normalized.model_dump(mode="json"))
    reordered = SimConfig.model_validate(dict(reversed(list(payload.items()))))
    cache = ResultCache(tmp_path / "cache.sqlite3", fingerprint="engine")
    assert cache.identity(normalized, None) == cache.identity(explicit, None)
    assert cache.identity(normalized, None) == cache.identity(reordered, None)


@pytest.mark.parametrize(
    "field", ["seed", "num_requests", "scheduler", "hardware", "timing", "routing_policy"]
)
def test_effective_inputs_change_key(tmp_path: Path, config: SimConfig, field: str) -> None:
    payload = config.model_dump(mode="json")
    if field == "seed":
        payload["workload"]["seed"] += 1
    elif field == "num_requests":
        payload["workload"]["synthetic"]["num_requests"] += 1
    elif field == "scheduler":
        payload["scheduler"]["max_running_requests"] += 1
    elif field == "hardware":
        payload["inference"]["hardware"]["effective_fp16_tflops"] += 1
    elif field == "routing_policy":
        payload["routing_policy"] = "cache_affinity"
    else:
        payload["inference"]["timing"]["decode_d1"] += 1
    cache = ResultCache(tmp_path / "cache.sqlite3", fingerprint="engine")
    assert (
        cache.identity(config, None)[0]
        != cache.identity(SimConfig.model_validate(payload), None)[0]
    )


def test_fingerprint_tracks_source_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    original = engine_fingerprint()
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: read_bytes(path) + b"\n# changed")
    assert engine_fingerprint() != original


def test_force_refresh_and_clear(tmp_path: Path, config: SimConfig, result: SimResult) -> None:
    cache = ResultCache(tmp_path / "cache.sqlite3", fingerprint="engine")
    updated = result.model_copy(
        update={"meta": result.meta.model_copy(update={"wall_clock_ms": 123})}
    )
    compute = Mock(side_effect=[result, updated, result])
    cache.run(config, None, compute)
    assert cache.run(config, None, compute, force=True).status == "miss"
    hit = cache.run(config, None, compute)
    assert hit.status == "hit"
    assert json.loads(hit.content)["meta"]["wall_clock_ms"] == 123
    cache.clear()
    assert cache.run(config, None, compute).status == "miss"
    assert compute.call_count == 3


def test_lru_and_oversized_result(tmp_path: Path, config: SimConfig, result: SimResult) -> None:
    path = tmp_path / "cache.sqlite3"
    cache = ResultCache(path, fingerprint="engine")
    cache.run(config, None, lambda: result)
    with sqlite3.connect(path) as connection:
        size = connection.execute("SELECT size_bytes FROM results").fetchone()[0]
    cache.max_bytes = size * 2 + 100
    other = [SimConfig.model_validate(make_payload(seed=seed, num_requests=2)) for seed in (1, 2)]
    cache.run(other[0], None, lambda: simulate(other[0]))
    assert cache.run(config, None, lambda: result).status == "hit"
    cache.run(other[1], None, lambda: simulate(other[1]))
    with sqlite3.connect(path) as connection:
        keys = {row[0] for row in connection.execute("SELECT key FROM results")}
        total = connection.execute("SELECT SUM(size_bytes) FROM results").fetchone()[0]
    assert cache.identity(config, None)[0] in keys
    assert cache.identity(other[0], None)[0] not in keys
    assert total <= cache.max_bytes
    tiny = ResultCache(tmp_path / "tiny.sqlite3", max_bytes=1, fingerprint="engine")
    compute = Mock(return_value=result)
    assert tiny.run(config, None, compute).status == "miss"
    assert tiny.run(config, None, compute).status == "miss"
    assert compute.call_count == 2


def test_concurrent_requests_share_computation(
    tmp_path: Path,
    config: SimConfig,
    result: SimResult,
) -> None:
    cache = ResultCache(tmp_path / "cache.sqlite3", fingerprint="engine")
    started, waiting, release = Event(), Event(), Event()

    def compute() -> SimResult:
        started.set()
        assert release.wait(5)
        return result

    original_result = Future.result

    def wait_for_result(future: Future[str], timeout: float | None = None) -> str:
        waiting.set()
        return original_result(future, timeout)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.run, config, None, compute)
        assert started.wait(5)
        with patch("simserver.result_cache.Future.result", wait_for_result):
            second = pool.submit(cache.run, config, None, compute, force=True)
            assert waiting.wait(5)
            release.set()
        assert first.result().status == "miss"
        shared = second.result()
        assert shared.status == "shared"
        assert SimResult.model_validate_json(shared.content) == result


def test_exceptions_are_not_cached(tmp_path: Path, config: SimConfig, result: SimResult) -> None:
    cache = ResultCache(tmp_path / "cache.sqlite3", fingerprint="engine")
    compute = Mock(side_effect=[ValueError("failed"), result])
    with pytest.raises(ValueError, match="failed"):
        cache.run(config, None, compute)
    assert cache.run(config, None, compute).status == "miss"
    assert compute.call_count == 2


def test_clear_does_not_repopulate_from_inflight(
    tmp_path: Path,
    config: SimConfig,
    result: SimResult,
) -> None:
    cache = ResultCache(tmp_path / "cache.sqlite3", fingerprint="engine")
    started, release = Event(), Event()

    def compute() -> SimResult:
        started.set()
        assert release.wait(5)
        return result

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(cache.run, config, None, compute)
        assert started.wait(5)
        cache.clear()
        release.set()
        assert pending.result().status == "miss"
    assert cache.run(config, None, lambda: result).status == "miss"


def test_clear_across_worker_cache_instances(
    tmp_path: Path,
    config: SimConfig,
    result: SimResult,
) -> None:
    path = tmp_path / "cache.sqlite3"
    worker_cache = ResultCache(path, fingerprint="engine")
    controller_cache = ResultCache(path, fingerprint="engine")
    started, release = Event(), Event()

    def compute() -> SimResult:
        started.set()
        assert release.wait(5)
        return result

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(worker_cache.run, config, None, compute)
        assert started.wait(5)
        controller_cache.clear()
        release.set()
        assert pending.result().status == "miss"
    assert controller_cache.run(config, None, lambda: result).status == "miss"


def test_http_hit_force_clear_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    payload = make_payload(num_requests=2)
    with TestClient(create_app(cache_path=path)) as client:
        first = run_simulation(client, payload)
        assert first.status_code == 200
        assert first.headers["X-Simulation-Cache"] == "miss"
        hit = run_simulation(client, payload)
        assert hit.headers["X-Simulation-Cache"] == "hit"
        assert hit.json() == first.json()
        assert float(hit.headers["X-Simulation-Cache-Lookup-Ms"]) >= 0
        assert run_simulation(client, payload, force=True).headers["X-Simulation-Cache"] == "miss"
    with TestClient(create_app(cache_path=path)) as restarted:
        assert run_simulation(restarted, payload).headers["X-Simulation-Cache"] == "hit"
        assert restarted.delete("/api/simulation-cache").status_code == 204
        assert run_simulation(restarted, payload).headers["X-Simulation-Cache"] == "miss"


def test_trace_content_identity_and_current_config_echo(tmp_path: Path) -> None:
    plan = {
        "version": 1,
        "block_size": 1,
        "sessions": ["s"],
        "requests": [
            {
                "request_id": 0,
                "session_id": 0,
                "arrival_offset_ms": 0,
                "prompt_len": 2,
                "output_len": 1,
                "hash_ids": [10, 20],
            }
        ],
    }
    with TestClient(create_app(cache_path=tmp_path / "cache.sqlite3")) as client:
        first_id = client.post("/api/replay-plans", content=json.dumps(plan)).json()["id"]
        second_id = client.post(
            "/api/replay-plans",
            content=json.dumps(plan, indent=2, sort_keys=True),
        ).json()["id"]
        assert first_id != second_id
        payload = make_payload(num_requests=2)
        payload["workload"] = {
            "mode": "trace",
            "session_rate": 1,
            "trace": {"plan_id": first_id},
        }
        first = run_simulation(client, payload)
        assert first.status_code == 200, first.text
        payload["workload"]["trace"]["plan_id"] = second_id
        hit = run_simulation(client, payload)
        assert hit.headers["X-Simulation-Cache"] == "hit"
        expected = first.json()
        expected["meta"]["config"]["workload"]["trace"]["plan_id"] = second_id
        assert hit.json() == expected
        changed = TracePlan.model_validate(plan).model_dump(mode="json")
        changed["requests"][0]["output_len"] = 2
        third_id = client.post("/api/replay-plans", content=json.dumps(changed)).json()["id"]
        payload["workload"]["trace"]["plan_id"] = third_id
        assert run_simulation(client, payload).headers["X-Simulation-Cache"] == "miss"
        payload["workload"]["trace"]["plan_id"] = "expired"
        assert client.post("/api/simulate", json=payload).status_code == 404


def test_app_openapi_has_no_database_side_effect(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    app = create_app(cache_path=path)
    app.openapi()
    assert not path.exists()
