"""临时上传存储与后台扫描；部署在单服务进程中。"""

import atexit
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import uuid4

from simcore.schemas import TracePlan
from simserver.trace_processing import compile_plan, scan_trace

MAX_TRACE_BYTES = 3 * 1024**3
MAX_PLAN_BYTES = 64 * 1024**2
_MAX_AGE_S = 24 * 3600


class TraceStore:
    def __init__(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="simulator-traces-")
        atexit.register(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="trace-scan")
        self._lock = Lock()
        self._jobs: dict[str, dict] = {}
        self._plans: dict[str, Path] = {}
        self._created: dict[str, float] = {}

    def _purge(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [
                key
                for key, created in self._created.items()
                if now - created > _MAX_AGE_S
                and self._jobs.get(key, {}).get("status") != "processing"
            ]
            for key in expired:
                self._created.pop(key)
                job = self._jobs.pop(key, None)
                plan = self._plans.pop(key, None)
                if job:
                    (self.root / f"{key}.jsonl").unlink(missing_ok=True)
                if plan:
                    plan.unlink(missing_ok=True)

    def new_path(self, suffix: str) -> tuple[str, Path]:
        self._purge()
        key = uuid4().hex
        with self._lock:
            self._created[key] = time.monotonic()
        return key, self.root / f"{key}{suffix}"

    def scan(self, key: str, path: Path) -> None:
        with self._lock:
            self._jobs[key] = {"status": "processing", "sessions": []}
        self._pool.submit(self._scan_job, key, path)

    def _scan_job(self, key: str, path: Path) -> None:
        try:
            sessions = scan_trace(path)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            with self._lock:
                self._jobs[key] = {"status": "failed", "sessions": [], "error": str(exc)}
        else:
            with self._lock:
                self._jobs[key] = {"status": "ready", "sessions": sessions}

    def job(self, key: str) -> dict:
        self._purge()
        with self._lock:
            if key not in self._jobs:
                raise KeyError("未找到 trace，或上传已过期")
            return self._jobs[key].copy()

    def export(self, key: str, selected: list[str]) -> TracePlan:
        if self.job(key)["status"] != "ready":
            raise ValueError("trace 尚未扫描完成")
        return compile_plan(self.root / f"{key}.jsonl", selected, max_bytes=MAX_PLAN_BYTES)

    def add_plan(self, key: str, path: Path) -> TracePlan:
        plan = TracePlan.model_validate_json(path.read_bytes())
        with self._lock:
            self._plans[key] = path
        return plan

    def plan(self, key: str) -> TracePlan:
        self._purge()
        with self._lock:
            path = self._plans.get(key)
        if path is None:
            raise KeyError("未找到计划，或计划已过期")
        return TracePlan.model_validate_json(path.read_bytes())
