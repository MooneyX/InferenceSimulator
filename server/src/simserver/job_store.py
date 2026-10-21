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
                    PRIMARY KEY (job_id, sequence),
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                )
            """)
            connection.execute("PRAGMA user_version = 1")

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            with connection:
                yield connection

    def save(self, summary: dict[str, Any], *, result: str | None = None,
             point: dict[str, Any] | None = None) -> None:
        with self._database() as connection:
            connection.execute("""
                INSERT INTO jobs (id, summary, version, result) VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary=excluded.summary, version=excluded.version,
                    result=COALESCE(excluded.result, jobs.result)
            """, (summary["id"], json.dumps(summary, ensure_ascii=False), summary["version"], result))
            if point is not None:
                connection.execute("""
                    INSERT INTO job_points (job_id, sequence, point) VALUES (?, ?, ?)
                """, (summary["id"], summary["completed"] + summary["failed"],
                      json.dumps(point, ensure_ascii=False)))

    def import_legacy(self, summary: dict[str, Any], result: str | None,
                      points: list[dict[str, Any]]) -> bool:
        with self._database() as connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO jobs (id, summary, version, result) VALUES (?, ?, ?, ?)",
                (summary["id"], json.dumps(summary, ensure_ascii=False), summary["version"], result),
            )
            if inserted.rowcount == 0:
                return False
            for sequence, point in enumerate(points, start=1):
                connection.execute(
                    "INSERT INTO job_points (job_id, sequence, point) VALUES (?, ?, ?)",
                    (summary["id"], sequence, json.dumps(point, ensure_ascii=False)),
                )
        return True

    def list(self) -> list[dict[str, Any]]:
        with self._database() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT summary FROM jobs ORDER BY sequence DESC"
            )]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._database() as connection:
            row = connection.execute("SELECT summary FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def changes(self, version: int) -> list[dict[str, Any]]:
        with self._database() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT summary FROM jobs WHERE version > ? ORDER BY version", (version,)
            )]

    def result(self, job_id: str) -> str | None:
        with self._database() as connection:
            row = connection.execute("SELECT result FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row[0]

    def points(self, job_id: str) -> list[dict[str, Any]]:
        with self._database() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT point FROM job_points WHERE job_id = ? ORDER BY sequence", (job_id,)
            )]
