"""服务层的内容寻址结果缓存；不向仿真内核引入数据库 I/O。"""

import json
import logging
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
from time import perf_counter, sleep, time_ns
from typing import Any, Literal

import pydantic

import simcore
from simcore import SimConfig, SimResult
from simcore.schemas import TracePlan

CACHE_FORMAT_VERSION = 1
DEFAULT_MAX_BYTES = 512 * 1024 * 1024
DATABASE_TIMEOUT_SECONDS = 1
DATABASE_ATTEMPTS = 3
logger = logging.getLogger(__name__)


def _is_locked(exc: sqlite3.OperationalError) -> bool:
    code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
    return code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


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
        with closing(sqlite3.connect(self.path, timeout=DATABASE_TIMEOUT_SECONDS)) as connection:
            if not self._initialized:
                connection.execute("PRAGMA auto_vacuum = FULL")
                if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                    connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("BEGIN IMMEDIATE")
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
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS cache_meta (id INTEGER PRIMARY KEY, "
                    "generation INTEGER NOT NULL)"
                )
                connection.execute("INSERT OR IGNORE INTO cache_meta VALUES (1, 0)")
                connection.commit()
                self._initialized = True
            with connection:
                yield connection

    def _execute[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        attempt = 0
        while True:
            try:
                with self._database() as connection:
                    return operation(connection)
            except sqlite3.OperationalError as exc:
                attempt += 1
                if not _is_locked(exc) or attempt >= DATABASE_ATTEMPTS:
                    raise
                sleep(0.05 * attempt)

    @contextmanager
    def _optional_cache(self, operation: str) -> Iterator[None]:
        try:
            yield
        except sqlite3.OperationalError as exc:
            if not _is_locked(exc):
                raise
            logger.warning("Result cache %s skipped after lock retries: %s", operation,
                           self.path, exc_info=True)

    def _read(self, key: str) -> str | None:
        row = self._execute(
            lambda connection: connection.execute(
                "SELECT result FROM results WHERE key = ?", (key,)
            ).fetchone()
        )
        if row is None:
            return None
        with self._optional_cache("access time update"):
            self._execute(lambda connection: connection.execute(
                "UPDATE results SET accessed_ns = ? WHERE key = ?", (time_ns(), key)
            ))
        return zlib.decompress(row[0]).decode()

    def _write(self, key: str, inputs: str, result: str, generation: int) -> None:
        compressed = zlib.compress(result.encode())
        size = len(compressed) + len(inputs.encode())

        def write(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT generation FROM cache_meta WHERE id = 1"
            ).fetchone()[0]
            if current != generation:
                return
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

        self._execute(write)

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
            cached = None
            shared_generation = None
            with self._optional_cache("lookup"):
                cached = None if force else self._read(key)
                if cached is None:
                    shared_generation = self._execute(lambda connection: connection.execute(
                        "SELECT generation FROM cache_meta WHERE id = 1"
                    ).fetchone()[0])
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
                if generation == self._generation and shared_generation is not None:
                    with self._optional_cache("write"):
                        self._write(key, inputs, neutral, shared_generation)
                future.set_result(neutral)
            return CachedSimulation(result.model_dump_json(), "miss", lookup_ms)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                del self._inflight[key]

    def clear(self) -> None:
        def clear(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE cache_meta SET generation = generation + 1 WHERE id = 1")
            connection.execute("DELETE FROM results")

        with self._lock:
            self._execute(clear)
            self._generation += 1
