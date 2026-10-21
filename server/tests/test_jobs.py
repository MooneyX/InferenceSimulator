"""后台调度接口与参数扫描的端到端测试。"""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from simserver import create_app
from test_routes import make_payload


def wait_job(client: TestClient, key: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{key}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"任务未完成：{client.get(f'/api/jobs/{key}').json()}")


def test_single_task_and_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMULATOR_WORKERS", "1")
    with TestClient(create_app(cache_path=tmp_path / "results.sqlite3")) as client:
        config = make_payload(num_requests=2)
        submitted = client.post("/api/simulate", json=config)
        assert submitted.status_code == 202
        key = submitted.json()["id"]
        assert wait_job(client, key)["status"] == "completed"
        first = client.get(f"/api/jobs/{key}/result")
        assert first.status_code == 200
        assert first.headers["X-Simulation-Cache"] == "miss"
        assert first.json()["meta"]["num_requests"] == 2
        second_key = client.post("/api/simulate", json=config).json()["id"]
        assert wait_job(client, second_key)["status"] == "completed"
        second = client.get(f"/api/jobs/{second_key}/result")
        assert second.headers["X-Simulation-Cache"] == "hit"
        assert second.json() == first.json()
        assert client.get("/api/jobs/not-found").status_code == 404


def test_scan_fifo_points_and_cancel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMULATOR_WORKERS", "1")
    with TestClient(create_app(cache_path=tmp_path / "results.sqlite3")) as client:
        config = make_payload(num_requests=1)
        request = {"base": config, "axes": [
            {"key": "scheduler.max_running_requests", "values": [1, 2]},
            {"key": "scheduler.pd_mutually_exclusive", "values": [True, False]},
            {"key": "num_instances", "values": [1]},
            {"key": "inference.max_output_tokens", "values": [65536]},
            {"key": "scheduler.max_prefill_tokens", "values": [16384, 32768]},
        ]}
        response = client.post("/api/scans", json=request)
        assert response.status_code == 202, response.text
        key = response.json()["id"]
        single = client.post("/api/simulate", json=config).json()["id"]
        status = wait_job(client, key)
        ordinary = wait_job(client, single)
        assert ordinary["status"] == "completed"
        assert status["version"] < ordinary["version"]
        assert (status["total"], status["completed"], status["failed"]) == (8, 8, 0)
        assert status["axes"] == request["axes"]
        points = client.get(f"/api/jobs/{key}/points").json()
        assert len(points) == 8
        assert len({item["signature"] for item in points}) == 8
        assert {item["config"]["scheduler"]["max_running_requests"] for item in points} == {1, 2}
        assert all(item["metrics"] is not None for item in points)
        invalid = client.post("/api/scans", json={"base": config, "axes": [
            {"key": "inference.model.num_layers", "values": [3]}]})
        assert invalid.status_code == 422
        duplicate = client.post("/api/scans", json={"base": config, "axes": [request["axes"][0]] * 2})
        assert duplicate.status_code == 422
        empty = client.post("/api/scans", json={"base": config, "axes": []})
        assert empty.status_code == 422
        another = client.post("/api/scans", json=request).json()["id"]
        cancelled = client.delete(f"/api/jobs/{another}")
        assert cancelled.json()["status"] == "cancelled"
        assert client.get(f"/api/jobs/{another}/result").status_code == 409


def test_two_workers_run_distinct_points(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMULATOR_WORKERS", "2")
    with TestClient(create_app(cache_path=tmp_path / "results.sqlite3")) as client:
        first = client.post("/api/simulate", json=make_payload(num_requests=8)).json()["id"]
        second = client.post(
            "/api/simulate", json=make_payload(seed=2, num_requests=8)
        ).json()["id"]
        assert wait_job(client, first)["status"] == "completed"
        assert wait_job(client, second)["status"] == "completed"
        assert len(client.app.state.jobs._slots) == 2
        assert client.get(f"/api/jobs/{second}/result").status_code == 200


def test_cancel_running_worker_and_replace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMULATOR_WORKERS", "1")
    with TestClient(create_app(cache_path=tmp_path / "results.sqlite3")) as client:
        heavy = make_payload(num_requests=12_000)
        key = client.post("/api/simulate", json=heavy).json()["id"]
        deadline = time.monotonic() + 10
        while client.get(f"/api/jobs/{key}").json()["status"] == "queued":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert client.get(f"/api/jobs/{key}").json()["status"] == "running"
        cancelled = client.delete(f"/api/jobs/{key}")
        assert cancelled.json()["status"] == "cancelled"
        small = client.post("/api/simulate", json=make_payload(num_requests=1)).json()["id"]
        assert wait_job(client, small)["status"] == "completed"
        assert client.get(f"/api/jobs/{key}").json()["status"] == "cancelled"
