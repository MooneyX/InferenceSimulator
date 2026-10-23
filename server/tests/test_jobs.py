"""后台调度接口与参数扫描的端到端测试。"""

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from simcore import SimConfig
from simserver import create_app
from simserver.job_store import JobStore
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
        for ident in (key, second_key):
            assert client.get(f"/api/jobs/{ident}").json()["request_progress"] == {
                "total": 2, "completed": 2, "aborted": 0, "skipped": 0,
            }
        assert client.get("/api/jobs/not-found").status_code == 404


def test_archived_jobs_survive_restart_and_cache_clear(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMULATOR_WORKERS", "1")
    path = tmp_path / "results.sqlite3"
    with TestClient(create_app(cache_path=path)) as client:
        single_id = client.post("/api/simulate", json=make_payload(num_requests=2)).json()["id"]
        assert wait_job(client, single_id)["code_version_status"] == "current"
        original = client.get(f"/api/jobs/{single_id}/result").content
        scan_id = client.post(
            "/api/scans",
            json={
                "base": {**make_payload(num_requests=1), "num_instances": 3},
                "axes": [{"key": "num_instances", "values": [1, 2]}],
            },
        ).json()["id"]
        scan_summary = wait_job(client, scan_id)
        assert scan_summary["status"] == "completed"
        assert scan_summary["base"]["num_instances"] == 3
        assert client.get("/api/jobs").json()[0]["base"] == scan_summary["base"]
        changes = client.app.state.jobs.changes(0, timeout=0)[1]
        assert next(item for item in changes if item["id"] == scan_id)["base"] == scan_summary["base"]
        points = client.get(f"/api/jobs/{scan_id}/points").json()
        page = client.get(f"/api/jobs/{scan_id}/tasks?offset=1&limit=1").json()
        assert page["total"] == 2
        assert page["items"][0]["index"] == 1
        assert page["items"][0]["values"] == {"num_instances": 2}
        archived_point = client.get(f"/api/jobs/{scan_id}/points/1/result").content
        assert page["items"][0]["result_available"] is True
        client.delete("/api/simulation-cache")
    with (
        patch("simserver.jobs.engine_fingerprint", return_value="new-engine"),
        TestClient(create_app(cache_path=path)) as restarted,
    ):
        assert [item["id"] for item in restarted.get("/api/jobs").json()] == [
            scan_id,
            single_id,
        ]
        summary = restarted.get(f"/api/jobs/{single_id}").json()
        assert summary["code_version_status"] == "stale"
        assert summary["code_fingerprint"] != "new-engine"
        assert restarted.get(f"/api/jobs/{single_id}/result").content == original
        assert restarted.get(f"/api/jobs/{scan_id}/points").json() == points
        assert restarted.get(f"/api/jobs/{scan_id}/points/1/result").content == archived_point
        assert restarted.get(f"/api/jobs/{scan_id}/tasks?offset=1&limit=1").json() == page
        archived = restarted.get(f"/api/jobs/{scan_id}").json()
        assert archived["code_version_status"] == "stale"
        assert archived["base"] == scan_summary["base"]
        assert archived["axes"] == scan_summary["axes"]


@pytest.mark.parametrize("base_snapshot", ["missing", "null", "present"])
def test_restart_marks_unfinished_scan_interrupted_with_partial_points(
    tmp_path: Path, base_snapshot: str,
) -> None:
    path = tmp_path / "results.sqlite3"
    store = JobStore(path.with_name("jobs.sqlite3"))
    summary = {
        "id": "unfinished",
        "kind": "scan",
        "status": "running",
        "total": 3,
        "completed": 1,
        "failed": 0,
        "error": None,
        "cache": None,
        "lookup_ms": None,
        "version": 4,
        "axes": [{"key": "num_instances", "values": [1, 2, 3]}],
        "code_fingerprint": "old-engine",
    }
    base = SimConfig.model_validate(make_payload(num_requests=1)).model_dump(mode="json")
    expected_base = base if base_snapshot == "present" else None
    if base_snapshot != "missing":
        summary["base"] = expected_base
    point = {"values": {"num_instances": 1}, "status": "completed", "signature": "old"}
    store.save(summary, point=point)
    with TestClient(create_app(cache_path=path)) as client:
        recovered = client.get("/api/jobs/unfinished").json()
        assert recovered["base"] == expected_base
        assert store.get("unfinished")["base"] == expected_base
        assert client.get("/api/jobs").json()[0]["base"] == expected_base
        assert client.app.state.jobs.changes(0, timeout=0)[1][0]["base"] == expected_base
        assert recovered["status"] == "interrupted"
        assert recovered["completed"] == 1
        assert recovered["failed"] == 0
        assert recovered["version"] > 4
        assert recovered["code_version_status"] == "stale"
        assert recovered["point_archive_version"] == 0
        page = client.get("/api/jobs/unfinished/tasks").json()
        assert page == {"total": 1, "items": [{"index": 0, "values": point["values"],
                  "status": "completed", "error": None, "result_available": False,
                  "request_progress": None}]}
        assert client.get("/api/jobs/unfinished/points/0/result").status_code == 409
        assert (
            client.get("/api/jobs/unfinished/points").json()[0]["signature"] == point["signature"]
        )
        assert client.delete("/api/jobs/unfinished").json()["status"] == "interrupted"
    with TestClient(create_app(cache_path=path)) as client:
        recovered = client.get("/api/jobs/unfinished").json()
        assert recovered["status"] == "interrupted"
        assert recovered["base"] == expected_base
        assert (
            client.get("/api/jobs/unfinished/points").json()[0]["signature"] == point["signature"]
        )


def test_interrupted_scan_preserves_new_point_results_and_marks_remaining_not_run(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.sqlite3"
    store = JobStore(path.with_name("jobs.sqlite3"))
    summary = {"id": "partial", "kind": "scan", "status": "running", "total": 3,
               "completed": 1, "failed": 0, "error": None, "cache": None,
               "lookup_ms": None, "version": 1, "code_fingerprint": "previous",
               "point_archive_version": 1,
               "axes": [{"key": "num_instances", "values": [1, 2, 3]}]}
    store.save(summary, point={"values": {"num_instances": 2}, "status": "completed"},
               point_index=1, point_result='{"meta": {"num_requests": 2}}')
    with TestClient(create_app(cache_path=path)) as client:
        page = client.get("/api/jobs/partial/tasks").json()
        assert [item["status"] for item in page["items"]] == ["not_run", "completed", "not_run"]
        assert [item["values"]["num_instances"] for item in page["items"]] == [1, 2, 3]
        assert client.get("/api/jobs/partial/points/1/result").json()["meta"] == {"num_requests": 2}
        assert client.get("/api/jobs/partial/points/0/result").status_code == 409


def test_legacy_point_table_is_migrated_without_losing_rows(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE job_points (
                job_id TEXT NOT NULL, sequence INTEGER NOT NULL, point TEXT NOT NULL,
                PRIMARY KEY (job_id, sequence)
            )
        """)
    store = JobStore(path)
    store.save({"id": "old", "kind": "scan", "completed": 1, "failed": 0,
                "version": 1, "axes": [{"key": "num_instances", "values": [1]}]},
               point={"values": {"num_instances": 1}, "status": "completed"})
    assert store.point_page("old", 0, 20, legacy=True)[0]["result_available"] is False
    assert store.points("old")[0]["values"] == {"num_instances": 1}


def test_scan_point_failures_and_duplicate_values_have_distinct_indices(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SIMULATOR_WORKERS", "2")
    with TestClient(create_app(cache_path=tmp_path / "results.sqlite3")) as client:
        key = client.post("/api/scans", json={"base": make_payload(num_requests=1),
            "axes": [{"key": "num_instances", "values": [1, 1, 0]}]}).json()["id"]
        summary = wait_job(client, key)
        assert (summary["status"], summary["completed"], summary["failed"]) == ("completed", 2, 1)
        items = client.get(f"/api/jobs/{key}/tasks").json()["items"]
        assert [item["index"] for item in items] == [0, 1, 2]
        assert [item["status"] for item in items] == ["completed", "completed", "failed"]
        assert all(item["result_available"] for item in items[:2])
        assert items[2]["error"]
        assert client.get(f"/api/jobs/{key}/points/0/result").status_code == 200
        assert client.get(f"/api/jobs/{key}/points/1/result").status_code == 200
        assert client.get(f"/api/jobs/{key}/points/2/result").status_code == 409


def test_scan_fifo_points_and_cancel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMULATOR_WORKERS", "1")
    with TestClient(create_app(cache_path=tmp_path / "results.sqlite3")) as client:
        config = make_payload(num_requests=1)
        request = {
            "base": config,
            "axes": [
                {"key": "scheduler.max_running_requests", "values": [1, 2]},
                {"key": "scheduler.pd_mutually_exclusive", "values": [True, False]},
                {"key": "num_instances", "values": [1]},
                {"key": "inference.max_output_tokens", "values": [65536]},
                {"key": "scheduler.max_prefill_tokens", "values": [16384, 32768]},
            ],
        }
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
        tasks = client.get(f"/api/jobs/{key}/tasks?offset=3&limit=2").json()
        assert tasks["total"] == 8
        assert [item["index"] for item in tasks["items"]] == [3, 4]
        assert [item["values"]["scheduler.max_prefill_tokens"] for item in tasks["items"]] == [
            32768, 16384
        ]
        assert all(item["result_available"] for item in tasks["items"])
        assert client.get(f"/api/jobs/{key}/points/4/result").json()["metrics"]
        assert client.get(f"/api/jobs/{key}/points/8/result").status_code == 404
        assert client.get(f"/api/jobs/{key}/tasks?limit=101").status_code == 422
        assert client.get(f"/api/jobs/{single}/tasks").status_code == 409
        invalid = client.post(
            "/api/scans",
            json={"base": config, "axes": [{"key": "inference.model.num_layers", "values": [3]}]},
        )
        assert invalid.status_code == 422
        duplicate = client.post(
            "/api/scans", json={"base": config, "axes": [request["axes"][0]] * 2}
        )
        assert duplicate.status_code == 422
        empty = client.post("/api/scans", json={"base": config, "axes": []})
        assert empty.status_code == 422
        another = client.post("/api/scans", json=request).json()["id"]
        cancelled = client.delete(f"/api/jobs/{another}")
        assert cancelled.json()["status"] == "cancelled"
        assert client.get(f"/api/jobs/{another}/result").status_code == 409
        pending = client.get(f"/api/jobs/{another}/tasks").json()["items"]
        assert pending[-1]["status"] == "not_run"
        assert client.get(f"/api/jobs/{another}/points/7/result").status_code == 409


def test_two_workers_run_distinct_points(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMULATOR_WORKERS", "2")
    with TestClient(create_app(cache_path=tmp_path / "results.sqlite3")) as client:
        first = client.post("/api/simulate", json=make_payload(num_requests=8)).json()["id"]
        second = client.post("/api/simulate", json=make_payload(seed=2, num_requests=8)).json()[
            "id"
        ]
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
