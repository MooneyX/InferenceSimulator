"""请求进度通过任务通道更新，且不会提前释放 worker 或丢失停止前的快照。"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from simserver import create_app
from simserver.jobs import Job, JobManager, Slot


PROGRESS = {"total": 10, "completed": 4, "aborted": 1, "skipped": 1}


def add_running_job(manager, kind="single"):
    job = Job("progress-job", kind, None, None, False, 3 if kind == "scan" else 1,
              status="running", point_archive_version=1 if kind == "scan" else 0,
              axes=[{"key": "workload.synthetic.num_requests", "values": [10, 20, 30]}]
              if kind == "scan" else [])
    with manager._condition:
        manager._jobs[job.id] = job
        manager._changed(job)
    return job


@pytest.mark.parametrize("kind", ["single", "scan"])
def test_progress_messages_keep_worker_assigned_and_failure_preserves_counts(tmp_path, kind):
    manager = JobManager(tmp_path / "results.sqlite3", workers=1)
    try:
        job = add_running_job(manager, kind)
        assignment = (job.id, 0, {"workload.synthetic.num_requests": 10})
        pipe = Mock()
        slot = Slot(Mock(), pipe, assignment)
        manager._slots.append(slot)
        pipe.poll.return_value = True
        stage = 0

        def receive():
            nonlocal stage
            stage += 1
            if stage == 1:
                return {"type": "progress", "request_progress": PROGRESS}
            if stage == 2:
                assert slot.assignment == assignment
                summary = manager.get(job.id)
                assert summary["status"] == "running"
                assert summary["completed"] == summary["failed"] == 0
                if kind == "single":
                    assert summary["request_progress"] == PROGRESS
                else:
                    assert manager.point_page(job.id, 0, 1)["items"][0]["request_progress"] == PROGRESS
                return {"ok": False, "error": "worker failed"}
            raise EOFError()

        pipe.recv.side_effect = receive
        manager._watch(slot)
        assert slot.assignment is None
        summary = manager.get(job.id)
        assert summary["failed"] == 1
        stored = summary if kind == "single" else manager.point_page(job.id, 0, 1)["items"][0]
        assert stored["request_progress"] == PROGRESS
        assert stored["status"] == "failed"
    finally:
        manager.close()


@pytest.mark.parametrize("kind", ["single", "scan"])
@pytest.mark.parametrize("cancel", [False, True])
def test_progress_survives_cancel_or_restart_and_is_visible_in_api(tmp_path, kind, cancel):
    path = tmp_path / "results.sqlite3"
    with TestClient(create_app(cache_path=path)) as client:
        manager = client.app.state.jobs
        job = add_running_job(manager, kind)
        version = job.version
        with manager._condition:
            manager._progress(job, 0, PROGRESS)
            if kind == "scan":
                manager._progress(job, 1, {**PROGRESS, "total": 20, "completed": 8})
        new_version, changes = manager.changes(version, timeout=0)
        assert new_version > version
        assert changes[0]["id"] == job.id
        current = client.get(f"/api/jobs/{job.id}").json()
        assert current["completed"] == current["failed"] == 0
        if kind == "scan":
            page = client.get(f"/api/jobs/{job.id}/tasks?offset=1&limit=1").json()
            assert page["total"] == 3
            assert page["items"][0]["request_progress"]["total"] == 20
            assert page["items"][0]["request_progress"]["completed"] == 8
        else:
            assert current["request_progress"] == PROGRESS
            assert client.get("/api/jobs").json()[0]["request_progress"] == PROGRESS
        if cancel:
            client.delete(f"/api/jobs/{job.id}")
            with manager._condition:
                manager._progress(job, 0, {**PROGRESS, "completed": 9})
    with TestClient(create_app(cache_path=path)) as client:
        expected_status = "cancelled" if cancel else "interrupted"
        summary = client.get(f"/api/jobs/{job.id}").json()
        assert summary["status"] == expected_status
        if kind == "single":
            assert summary["request_progress"] == PROGRESS
        else:
            items = client.get(f"/api/jobs/{job.id}/tasks").json()["items"]
            assert [item["status"] for item in items] == [expected_status, expected_status, "not_run"]
            assert items[0]["request_progress"] == PROGRESS
            assert items[2]["request_progress"] is None
            assert not any(item["result_available"] for item in items)


def test_finished_scan_moves_progress_into_archive_without_inflating_point_count(tmp_path):
    manager = JobManager(tmp_path / "results.sqlite3", workers=1)
    try:
        job = add_running_job(manager, "scan")
        final = {"total": 10, "completed": 7, "aborted": 1, "skipped": 2}
        with manager._condition:
            manager._progress(job, 0, PROGRESS)
            manager._record(job, 0, {"workload.synthetic.num_requests": 10}, {
                "ok": True, "request_progress": final,
                "payload": {"result": "{}", "num_completed": 7},
            })
        assert manager.store.progress_page(job.id, 0, 3) == {}
        assert len(manager.points(job.id)) == 1
        assert manager.get(job.id)["completed"] == 1
        point = manager.point_page(job.id, 0, 1)["items"][0]
        assert point["request_progress"] == final
        assert point["status"] == "completed"
        assert point["result_available"] is True
    finally:
        manager.close()
