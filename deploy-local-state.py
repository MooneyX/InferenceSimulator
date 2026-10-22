"""部署停服前只读检查持久化任务状态，不启动服务或修改任务。"""

import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


def jobs_path() -> Path:
    history_path = os.environ.get("SIMULATOR_JOBS_PATH")
    if history_path:
        return Path(history_path)
    cache_path = Path(os.environ.get("SIMULATOR_CACHE_PATH", ".simulation-cache/results.sqlite3"))
    return cache_path.with_name("jobs.sqlite3")


def active_jobs(path: Path) -> list[dict]:
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        jobs = [json.loads(row[0]) for row in connection.execute("SELECT summary FROM jobs")]
    for job in jobs:
        if job["status"] not in ACTIVE_STATUSES | TERMINAL_STATUSES:
            raise ValueError(f"未知任务状态：{job['status']}")
    return [job for job in jobs if job["status"] in ACTIVE_STATUSES]


def main() -> int:
    path = jobs_path()
    try:
        jobs = active_jobs(path)
        if jobs:
            print(f"存在 {len(jobs)} 个未结束任务，取消本次部署，服务保持运行：")
            for job in jobs:
                print(f"  {job['id']} | {job['kind']} | {job['status']} | "
                      f"进度 {job['completed'] + job['failed']}/{job['total']}")
            return 2
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        print(f"无法确认本地任务状态（{path.resolve()}）：{exc}。未执行部署。", file=sys.stderr)
        return 1
    print("没有排队或运行中的任务，继续部署。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
