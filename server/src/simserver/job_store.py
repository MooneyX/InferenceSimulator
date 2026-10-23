"""与可淘汰的结果缓存分离的持久任务归档。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    summary TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    result TEXT
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS jobs_version ON jobs(version)")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS job_points (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    point TEXT NOT NULL,
                    point_index INTEGER,
                    result TEXT,
                    PRIMARY KEY (job_id, sequence),
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                )
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(job_points)")}
            if "point_index" not in columns:
                connection.execute("ALTER TABLE job_points ADD COLUMN point_index INTEGER")
            if "result" not in columns:
                connection.execute("ALTER TABLE job_points ADD COLUMN result TEXT")
            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS job_points_index
                ON job_points(job_id, point_index)
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS job_request_progress (
                    job_id TEXT NOT NULL,
                    point_index INTEGER NOT NULL,
                    progress TEXT NOT NULL,
                    PRIMARY KEY (job_id, point_index),
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                )
            """)
            connection.execute("PRAGMA user_version = 3")

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            yield connection

    def save(
        self,
        summary: dict[str, Any],
        *,
        result: str | None = None,
        point: dict[str, Any] | None = None,
        point_index: int | None = None,
        point_result: str | None = None,
        request_progress: dict[str, int] | None = None,
    ) -> None:
        with self._database() as connection:
            connection.execute(
                """
                INSERT INTO jobs (id, summary, version, result) VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary=excluded.summary, version=excluded.version,
                    result=COALESCE(excluded.result, jobs.result)
            """,
                (
                    summary["id"],
                    json.dumps(summary, ensure_ascii=False),
                    summary["version"],
                    result,
                ),
            )
            if request_progress is not None:
                connection.execute(
                    "INSERT INTO job_request_progress (job_id, point_index, progress) "
                    "VALUES (?, ?, ?) ON CONFLICT(job_id, point_index) "
                    "DO UPDATE SET progress=excluded.progress",
                    (summary["id"], point_index, json.dumps(request_progress)),
                )
            if point is not None:
                connection.execute(
                    "DELETE FROM job_request_progress WHERE job_id = ? AND point_index = ?",
                    (summary["id"], point_index),
                )
                connection.execute(
                    """
                    INSERT INTO job_points (job_id, sequence, point, point_index, result)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        summary["id"],
                        summary["completed"] + summary["failed"],
                        json.dumps(point, ensure_ascii=False),
                        point_index,
                        point_result,
                    ),
                )

    def list(self) -> list[dict[str, Any]]:
        with self._database() as connection:
            return [
                json.loads(row[0])
                for row in connection.execute("SELECT summary FROM jobs ORDER BY sequence DESC")
            ]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._database() as connection:
            row = connection.execute("SELECT summary FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def changes(self, version: int) -> list[dict[str, Any]]:
        with self._database() as connection:
            return [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT summary FROM jobs WHERE version > ? ORDER BY version", (version,)
                )
            ]

    def result(self, job_id: str) -> str | None:
        with self._database() as connection:
            row = connection.execute("SELECT result FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row[0]

    def points(self, job_id: str) -> list[dict[str, Any]]:
        with self._database() as connection:
            return [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT point FROM job_points WHERE job_id = ? ORDER BY sequence", (job_id,)
                )
            ]

    def point_page(
        self, job_id: str, start: int, limit: int, *, legacy: bool
    ) -> dict[int, dict[str, Any]]:
        with self._database() as connection:
            if legacy:
                rows = connection.execute(
                    "SELECT sequence - 1, point, 0 FROM job_points "
                    "WHERE job_id = ? AND sequence > ? AND sequence <= ? ORDER BY sequence",
                    (job_id, start, start + limit),
                )
            else:
                rows = connection.execute(
                    "SELECT point_index, point, result IS NOT NULL FROM job_points "
                    "WHERE job_id = ? AND point_index >= ? AND point_index < ? "
                    "ORDER BY point_index",
                    (job_id, start, start + limit),
                )
            return {index: {**json.loads(point), "result_available": bool(available)}
                    for index, point, available in rows}

    def progress_page(self, job_id: str, start: int, limit: int) -> dict[int, dict[str, int]]:
        with self._database() as connection:
            return {index: json.loads(progress) for index, progress in connection.execute(
                "SELECT point_index, progress FROM job_request_progress "
                "WHERE job_id = ? AND point_index >= ? AND point_index < ?",
                (job_id, start, start + limit),
            )}

    def point_result(self, job_id: str, index: int) -> str | None:
        with self._database() as connection:
            row = connection.execute(
                "SELECT result FROM job_points WHERE job_id = ? AND point_index = ?",
                (job_id, index),
            ).fetchone()
        return row[0] if row else None
