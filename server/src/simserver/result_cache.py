"""服务层的内容寻址结果缓存；不向仿真内核引入数据库 I/O。"""

import json
import platform
import sqlite3
import zlib
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import closing, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from time import perf_counter, time_ns
from typing import Any, Literal

import pydantic

import simcore
from simcore import SimConfig, SimResult
from simcore.schemas import TracePlan

CACHE_FORMAT_VERSION = 1
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def engine_fingerprint() -> str:
    digest = sha256()
    digest.update(f"{platform.python_version()}:{pydantic.__version__}".encode())
    roots = (Path(simcore.__file__).parent, Path(__file__).parent)
    for root in roots:
        for source in sorted(root.rglob("*.py")):
            digest.update(source.relative_to(root).as_posix().encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedSimulation:
    content: str
    status: Literal["hit", "miss", "shared"]
    lookup_ms: float


class ResultCache:
    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        fingerprint: str | None = None,
    ) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self._fingerprint = fingerprint
        self._lock = RLock()
        self._initialized = False
        self._inflight: dict[str, Future[str]] = {}
        self._generation = 0

    def identity(self, config: SimConfig, plan: TracePlan | None) -> tuple[str, str]:
        with self._lock:
            if self._fingerprint is None:
                self._fingerprint = engine_fingerprint()
        inputs = config.model_dump(mode="json")
        if plan is not None:
            plan_digest = sha256(canonical_json(plan.model_dump(mode="json")).encode()).hexdigest()
            inputs["workload"]["trace"] = {"plan_digest": plan_digest}
        normalized = canonical_json(
            {
                "format": CACHE_FORMAT_VERSION,
                "engine": self._fingerprint,
                "config": inputs,
            }
        )
        return sha256(normalized.encode()).hexdigest(), normalized

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            if not self._initialized:
                connection.execute("PRAGMA auto_vacuum = FULL")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS results (
                        key TEXT PRIMARY KEY,
                        inputs TEXT NOT NULL,
                        result BLOB NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        created_ns INTEGER NOT NULL,
                        accessed_ns INTEGER NOT NULL
                    )
                """)
                connection.execute("CREATE INDEX IF NOT EXISTS results_lru ON results(accessed_ns)")
                connection.commit()
                self._initialized = True
            with connection:
                yield connection

    def _read(self, key: str) -> str | None:
        with self._database() as connection:
            row = connection.execute("SELECT result FROM results WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            connection.execute("UPDATE results SET accessed_ns = ? WHERE key = ?", (time_ns(), key))
            return zlib.decompress(row[0]).decode()

    def _write(self, key: str, inputs: str, result: str) -> None:
        compressed = zlib.compress(result.encode())
        size = len(compressed) + len(inputs.encode())
        with self._database() as connection:
            connection.execute("DELETE FROM results WHERE key = ?", (key,))
            if size > self.max_bytes:
                return
            now = time_ns()
            connection.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)",
                (key, inputs, compressed, size, now, now),
            )
            total = connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM results"
            ).fetchone()[0]
            while total > self.max_bytes:
                oldest_key, oldest_size = connection.execute(
                    "SELECT key, size_bytes FROM results ORDER BY accessed_ns, key LIMIT 1"
                ).fetchone()
                connection.execute("DELETE FROM results WHERE key = ?", (oldest_key,))
                total -= oldest_size

    @staticmethod
    def _render(result: str, config: SimConfig) -> str:
        return result.replace(
            '{"meta":{',
            '{"meta":{"config":' + config.model_dump_json() + ",",
            1,
        )

    def run(
        self,
        config: SimConfig,
        plan: TracePlan | None,
        compute: Callable[[], SimResult],
        *,
        force: bool = False,
    ) -> CachedSimulation:
        started = perf_counter()
        key, inputs = self.identity(config, plan)
        with self._lock:
            cached = None if force else self._read(key)
            lookup_ms = (perf_counter() - started) * 1000
            existing = self._inflight.get(key)
            owner = existing is None
            future = existing if existing is not None else Future[str]()
            if cached is None and owner:
                self._inflight[key] = future
            generation = self._generation
        if cached is not None:
            content = self._render(cached, config)
            return CachedSimulation(content, "hit", (perf_counter() - started) * 1000)
        if not owner:
            return CachedSimulation(self._render(future.result(), config), "shared", lookup_ms)
        try:
            result = compute()
            neutral = result.model_dump_json(exclude={"meta": {"config"}})
            with self._lock:
                if generation == self._generation:
                    self._write(key, inputs, neutral)
                future.set_result(neutral)
            return CachedSimulation(result.model_dump_json(), "miss", lookup_ms)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                del self._inflight[key]

    def clear(self) -> None:
        with self._lock:
            with self._database() as connection:
                connection.execute("DELETE FROM results")
            self._generation += 1
