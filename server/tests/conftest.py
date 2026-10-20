"""服务层测试不读写本地部署的结果缓存。"""

from pathlib import Path
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_result_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMULATOR_CACHE_PATH", str(tmp_path / "results.sqlite3"))
    monkeypatch.setenv("SIMULATOR_WORKERS", "2")


def await_task(client: TestClient, key: str, timeout: float = 60) -> dict:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        job = client.get(f"/api/jobs/{key}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        sleep(0.05)
    raise AssertionError(f"任务 {key} 超时")


def run_simulation(client: TestClient, payload: dict, *, force: bool = False):
    submit = client.post("/api/simulate?force=true" if force else "/api/simulate", json=payload)
    if submit.status_code != 202:
        return submit
    job = await_task(client, submit.json()["id"])
    if job["status"] != "completed":
        raise AssertionError(f"后台仿真失败：{job['error']}")
    return client.get(f"/api/jobs/{job['id']}/result")
