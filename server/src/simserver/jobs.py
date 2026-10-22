"""单服务进程内的 FIFO 仿真调度器；每个计算点由独立可替换的进程槽执行。"""

from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import product
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any
from uuid import uuid4

from simcore import SimConfig, simulate
from simcore.schemas import TracePlan
from simserver.job_store import JobStore
from simserver.result_cache import DEFAULT_MAX_BYTES, ResultCache, engine_fingerprint
from simserver.sweep import ScanRequest, build_point


def _worker(connection: Connection, cache_path: str, cache_size: int) -> None:
    parent = mp.parent_process()
    if parent is not None:

        def stop_with_parent() -> None:
            wait([parent.sentinel])
            os._exit(1)

        threading.Thread(target=stop_with_parent, name="worker-parent-watch", daemon=True).start()
    cache = ResultCache(Path(cache_path), max_bytes=cache_size)
    try:
        while True:
            command = connection.recv()
            if command is None:
                break
            try:
                config = SimConfig.model_validate(command["config"])
                plan = (
                    TracePlan.model_validate(command["plan"])
                    if command["plan"] is not None
                    else None
                )
                result = cache.run(
                    config,
                    plan,
                    lambda config=config, plan=plan: simulate(config, trace_plan=plan),
                    force=command["force"],
                )
                if command["kind"] == "scan":
                    body = json.loads(result.content)
                    signature, _ = cache.identity(config, plan)
                    payload: dict[str, Any] = {
                        "signature": signature,
                        "config": config.model_dump(mode="json"),
                        "metrics": body["metrics"],
                        "num_completed": body["meta"]["num_completed"],
                        "num_aborted": body["meta"]["num_aborted"],
                        "num_skipped": body["meta"]["num_skipped"],
                        "wall_clock_ms": body["meta"]["wall_clock_ms"],
                    }
                else:
                    payload = {"result": result.content}
                connection.send(
                    {
                        "ok": True,
                        "payload": payload,
                        "cache": result.status,
                        "lookup_ms": result.lookup_ms,
                    }
                )
            except Exception as exc:
                connection.send({"ok": False, "error": str(exc)})
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


@dataclass
class Job:
    id: str
    kind: str
    base: SimConfig | None
    plan: TracePlan | None
    force: bool
    total: int
    axes: list[dict[str, Any]] = field(default_factory=list)
    status: str = "queued"
    completed: int = 0
    failed: int = 0
    error: str | None = None
    result: str | None = None
    cache: str | None = None
    lookup_ms: float | None = None
    version: int = 0
    code_fingerprint: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "error": self.error,
            "cache": self.cache,
            "lookup_ms": self.lookup_ms,
            "version": self.version,
            "code_fingerprint": self.code_fingerprint,
            "axes": self.axes if self.kind == "scan" else None,
        }


@dataclass
class Slot:
    process: mp.Process
    pipe: Connection
    assignment: tuple[str, dict[str, Any]] | None = None


class JobManager:
    def __init__(
        self,
        cache_path: Path,
        cache_size: int = DEFAULT_MAX_BYTES,
        workers: int | None = None,
        history_path: Path | None = None,
    ) -> None:
        self.cache_path = cache_path
        self.cache_size = cache_size
        self.store = JobStore(history_path or cache_path.with_name("jobs.sqlite3"))
        self.code_fingerprint = engine_fingerprint()
        self.workers = workers if workers is not None else int(os.getenv("SIMULATOR_WORKERS", "8"))
        if self.workers < 1:
            raise ValueError("SIMULATOR_WORKERS 必须至少为 1")
        self._context = mp.get_context("spawn")
        self._condition = threading.Condition(threading.RLock())
        self._jobs: dict[str, Job] = {}
        self._queue: deque[tuple[str, Iterator[dict[str, Any]]]] = deque()
        self._slots: list[Slot] = []
        self._version = 0
        self._closed = False
        self._started = False
        for record in reversed(self.store.list()):
            job = Job(
                record["id"],
                record["kind"],
                None,
                None,
                False,
                record["total"],
                axes=record["axes"] or [],
                status=record["status"],
                completed=record["completed"],
                failed=record["failed"],
                error=record["error"],
                cache=record["cache"],
                lookup_ms=record["lookup_ms"],
                version=record["version"],
                code_fingerprint=record["code_fingerprint"],
            )
            self._jobs[job.id] = job
            self._version = max(self._version, job.version)
        with self._condition:
            for job in self._jobs.values():
                if job.status in ("queued", "running"):
                    job.status = "interrupted"
                    job.error = "服务重启，任务已中断"
                    self._changed(job)
        self._dispatcher = threading.Thread(
            target=self._dispatch, name="simulation-dispatch", daemon=True
        )
        self._dispatcher.start()
        atexit.register(self.close)

    def _present(self, summary: dict[str, Any]) -> dict[str, Any]:
        fingerprint = summary["code_fingerprint"]
        status = (
            "unknown"
            if not fingerprint
            else ("current" if fingerprint == self.code_fingerprint else "stale")
        )
        return {**summary, "code_version_status": status}

    def _changed(self, job: Job, *, point: dict[str, Any] | None = None) -> None:
        version = self._version + 1
        job.version = version
        try:
            self.store.save(job.summary(), result=job.result, point=point)
        except Exception:
            job.version = self._version
            raise
        self._version = version
        self._condition.notify_all()

    def submit(self, config: SimConfig, plan: TracePlan | None, *, force: bool = False) -> str:
        job = Job(
            uuid4().hex, "single", config, plan, force, 1, code_fingerprint=self.code_fingerprint
        )
        with self._condition:
            self._changed(job)
            self._jobs[job.id] = job
            self._queue.append((job.id, iter(({},))))
            self._started = True
            self._condition.notify_all()
        return job.id

    def submit_scan(self, request: ScanRequest, plan: TracePlan | None) -> str:
        axes = [axis.model_dump(mode="json") for axis in request.axes]
        total = 1
        for axis in axes:
            total *= len(axis["values"])
        job = Job(
            uuid4().hex,
            "scan",
            request.base,
            plan,
            False,
            total,
            axes=axes,
            code_fingerprint=self.code_fingerprint,
        )

        def combinations() -> Iterator[dict[str, Any]]:
            for values in product(*(axis["values"] for axis in axes)):
                yield {axis["key"]: value for axis, value in zip(axes, values, strict=True)}

        with self._condition:
            self._changed(job)
            self._jobs[job.id] = job
            self._queue.append((job.id, combinations()))
            self._started = True
            self._condition.notify_all()
        return job.id

    def _spawn(self) -> Slot:
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=_worker, args=(child, str(self.cache_path), self.cache_size), daemon=True
        )
        process.start()
        child.close()
        slot = Slot(process, parent)
        self._slots.append(slot)
        threading.Thread(
            target=self._watch, args=(slot,), daemon=True, name=f"simulation-worker-{process.pid}"
        ).start()
        return slot

    def _next(self) -> tuple[Job, dict[str, Any]] | None:
        while self._queue:
            job_id, iterator = self._queue[0]
            job = self._jobs[job_id]
            if job.status == "cancelled":
                self._queue.popleft()
                continue
            try:
                values = next(iterator)
            except StopIteration:
                self._queue.popleft()
                continue
            return job, values
        return None

    def _finish_if_done(self, job: Job) -> None:
        if job.status != "cancelled" and job.completed + job.failed == job.total:
            job.status = "completed" if job.kind == "scan" or job.failed == 0 else "failed"

    def _dispatch(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                if self._started and len(self._slots) < self.workers:
                    try:
                        self._spawn()
                    except OSError:
                        self._condition.wait(timeout=0.5)
                active = sum(slot.assignment is not None for slot in self._slots)
                if not self._queue or active >= self.workers:
                    if self._started and len(self._slots) < self.workers:
                        continue
                    self._condition.wait(timeout=0.5)
                    continue
                slot = next((item for item in self._slots if item.assignment is None), None)
                if slot is None:
                    self._condition.wait(timeout=0.5)
                    continue
                try:
                    job_and_values = self._next()
                    if job_and_values is None:
                        continue
                    job, values = job_and_values
                    assert job.base is not None
                    config = (
                        build_point(job.base, values, job.plan) if job.kind == "scan" else job.base
                    )
                    slot.pipe.send(
                        {
                            "kind": job.kind,
                            "config": config.model_dump(mode="json"),
                            "plan": job.plan.model_dump(mode="json") if job.plan else None,
                            "force": job.force,
                        }
                    )
                    slot.assignment = (job.id, values)
                    job.status = "running"
                    self._changed(job)
                except Exception as exc:
                    if job_and_values is not None:
                        job, values = job_and_values
                        self._record(job, values, {"ok": False, "error": str(exc)})
                    else:
                        self._condition.wait(timeout=0.5)

    def _record(self, job: Job, values: dict[str, Any], message: dict[str, Any]) -> None:
        if job.status == "cancelled":
            return
        point = None
        if message["ok"]:
            if job.kind == "single":
                job.result = message["payload"]["result"]
                job.cache = message["cache"]
                job.lookup_ms = message["lookup_ms"]
            else:
                point = {"values": values, "status": "completed", **message["payload"]}
            job.completed += 1
        else:
            job.error = message["error"]
            job.failed += 1
            if job.kind == "scan":
                point = {"values": values, "status": "failed", "error": message["error"]}
        self._finish_if_done(job)
        self._changed(job, point=point)

    def _watch(self, slot: Slot) -> None:
        while not self._closed:
            try:
                if not slot.pipe.poll(0.2):
                    if not slot.process.is_alive():
                        raise EOFError("worker 进程已退出")
                    continue
                message = slot.pipe.recv()
            except (EOFError, OSError) as exc:
                with self._condition:
                    if slot not in self._slots:
                        return
                    self._slots.remove(slot)
                    if slot.assignment:
                        job_id, values = slot.assignment
                        self._record(self._jobs[job_id], values, {"ok": False, "error": str(exc)})
                    slot.pipe.close()
                    self._condition.notify_all()
                return
            with self._condition:
                if slot not in self._slots:
                    return
                if slot.assignment:
                    job_id, values = slot.assignment
                    slot.assignment = None
                    self._record(self._jobs[job_id], values, message)
                self._condition.notify_all()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            return self._present(self.store.get(job_id))

    def list(self) -> list[dict[str, Any]]:
        with self._condition:
            return [self._present(item) for item in self.store.list()]

    def result(self, job_id: str) -> tuple[str, str | None, float | None]:
        with self._condition:
            job = self.store.get(job_id)
            content = self.store.result(job_id)
            if job["kind"] != "single" or job["status"] != "completed" or content is None:
                raise ValueError("仿真结果尚未就绪")
            return content, job["cache"], job["lookup_ms"]

    def points(self, job_id: str) -> list[dict[str, Any]]:
        with self._condition:
            if self.store.get(job_id)["kind"] != "scan":
                raise ValueError("这不是参数扫描任务")
            return self.store.points(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self._jobs[job_id]
            if job.status in ("completed", "failed", "cancelled", "interrupted"):
                return self.get(job_id)
            job.status = "cancelled"
            self._queue = deque(
                (ident, iterator) for ident, iterator in self._queue if ident != job_id
            )
            for slot in list(self._slots):
                if slot.assignment and slot.assignment[0] == job_id:
                    self._slots.remove(slot)
                    slot.process.terminate()
                    slot.process.join(timeout=1)
                    if slot.process.is_alive():
                        slot.process.kill()
                        slot.process.join()
                    slot.pipe.close()
            self._changed(job)
            return self.get(job_id)

    def changes(self, version: int, timeout: float = 15.0) -> tuple[int, list[dict[str, Any]]]:
        with self._condition:
            if self._version <= version and not self._closed:
                self._condition.wait(timeout=timeout)
            current = self._version
            return current, [self._present(item) for item in self.store.changes(version)]

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for slot in self._slots:
                slot.process.terminate()
                slot.process.join(timeout=1)
                if slot.process.is_alive():
                    slot.process.kill()
                    slot.process.join()
                slot.pipe.close()
            self._slots.clear()
            self._condition.notify_all()
