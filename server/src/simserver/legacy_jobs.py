"""部署前从旧版运行中服务归档尚在内存里的任务。"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from simserver.job_store import JobStore

BASE_URL = "http://127.0.0.1:8000"


def fetch(path: str) -> bytes:
    with urlopen(BASE_URL + path, timeout=120) as response:
        return response.read()


def main() -> int:
    try:
        jobs = json.loads(fetch("/api/jobs"))
    except URLError as exc:
        if isinstance(exc, HTTPError):
            raise
        print("旧版服务未运行，无需迁移任务。")
        return 0
    if not jobs or "code_fingerprint" in jobs[0]:
        print("没有需要迁移的旧版内存任务。")
        return 0
    cache_path = Path(os.environ.get("SIMULATOR_CACHE_PATH", ".simulation-cache/results.sqlite3"))
    history_path = Path(os.environ.get("SIMULATOR_JOBS_PATH", str(cache_path.with_name("jobs.sqlite3"))))
    store = JobStore(history_path)
    existing = {job["id"] for job in store.list()}
    imported = 0
    for job in reversed(jobs):
        if job["id"] in existing:
            continue
        key = job["id"]
        summary = json.loads(fetch(f"/api/jobs/{key}"))
        result = (fetch(f"/api/jobs/{key}/result").decode("utf-8")
                  if summary["kind"] == "single" and summary["status"] == "completed" else None)
        points = (json.loads(fetch(f"/api/jobs/{key}/points"))
                  if summary["kind"] == "scan" else [])
        summary["code_fingerprint"] = ""
        if store.import_legacy(summary, result, points):
            imported += 1
    print(f"已归档 {imported} 个旧版任务；旧任务的代码指纹无法追溯，标记为版本未知。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
