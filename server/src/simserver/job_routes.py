"""任务 HTTP 协议；计算始终在可终止的后台进程中进行。"""

import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from simcore import SimConfig, SimResult
from simcore.schemas import TracePlan
from simserver.jobs import JobManager
from simserver.routes import ErrorResponse, FieldValidationResponse
from simserver.sweep import PARAMETERS, ScanAxis, ScanRequest

router = APIRouter(prefix="/api")


class JobReceipt(BaseModel):
    id: str


class JobSummary(BaseModel):
    id: str
    kind: Literal["single", "scan"]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    total: int
    completed: int
    failed: int
    error: str | None
    cache: str | None
    lookup_ms: float | None
    version: int
    axes: dict[str, ScanAxis] | None


class ScanPoint(BaseModel):
    values: dict[str, Any]
    status: Literal["completed", "failed"]
    signature: str | None = None
    config: SimConfig | None = None
    metrics: dict[str, Any] | None = None
    num_completed: int | None = None
    num_aborted: int | None = None
    num_skipped: int | None = None
    wall_clock_ms: float | None = None
    error: str | None = None


def _manager(request: Request) -> JobManager:
    return request.app.state.jobs


def _plan(request: Request, config: SimConfig) -> TracePlan | None:
    if config.workload.mode != "trace":
        return None
    try:
        plan_id = config.workload.trace.plan_id
        if not plan_id:
            raise ValueError("Trace 仿真须提供回放计划")
        return request.app.state.trace_store.plan(plan_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _get(request: Request, key: str) -> dict[str, Any]:
    try:
        return _manager(request).get(key)
    except KeyError as exc:
        raise HTTPException(404, "未找到任务") from exc


@router.post(
    "/simulate", response_model=JobReceipt, status_code=202, summary="提交单场后台仿真",
    responses={404: {"model": ErrorResponse},
               422: {"model": ErrorResponse | FieldValidationResponse}},
)
def submit_simulation(config: SimConfig, request: Request,
                      force: bool = Query(False)) -> JobReceipt:
    plan = _plan(request, config)
    return JobReceipt(id=_manager(request).submit(config, plan, force=force))


class ScanParameter(BaseModel):
    key: str
    label: str
    kind: str
    modes: list[str]


@router.get("/scans/parameters", response_model=list[ScanParameter])
def scan_parameters() -> list[ScanParameter]:
    return [ScanParameter(key=key, label=value.label, kind=value.kind, modes=list(value.modes))
            for key, value in PARAMETERS.items()]


@router.post(
    "/scans", response_model=JobReceipt, status_code=202, summary="提交参数组合扫描",
    responses={404: {"model": ErrorResponse},
               422: {"model": ErrorResponse | FieldValidationResponse}},
)
def submit_scan(config: ScanRequest, request: Request) -> JobReceipt:
    plan = _plan(request, config.base)
    return JobReceipt(id=_manager(request).submit_scan(config, plan))


@router.get("/jobs", response_model=list[JobSummary])
def list_jobs(request: Request) -> list[dict[str, Any]]:
    return _manager(request).list()


@router.get("/jobs/events", summary="任务状态 SSE 推送")
async def job_events(request: Request) -> StreamingResponse:
    manager = _manager(request)

    async def stream():
        initial = manager.list()
        version = max((item["version"] for item in initial), default=0)
        yield f"event: snapshot\ndata: {json.dumps(initial, ensure_ascii=False)}\n\n"
        while not await request.is_disconnected():
            version, changed = await asyncio.to_thread(manager.changes, version, 10.0)
            if changed:
                for item in changed:
                    yield f"event: update\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
            else:
                yield ": heartbeat\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/jobs/{key}", response_model=JobSummary)
def get_job(key: str, request: Request) -> dict[str, Any]:
    return _get(request, key)


@router.delete("/jobs/{key}", response_model=JobSummary)
def cancel_job(key: str, request: Request) -> dict[str, Any]:
    _get(request, key)
    return _manager(request).cancel(key)


@router.get("/jobs/{key}/result", response_model=SimResult)
def job_result(key: str, request: Request) -> Response:
    _get(request, key)
    try:
        content, cache, lookup = _manager(request).result(key)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return Response(content=content, media_type="application/json",
                    headers={"X-Simulation-Cache": cache or "miss",
                             "X-Simulation-Cache-Lookup-Ms": f"{lookup or 0:.3f}"})


@router.get("/jobs/{key}/points", response_model=list[ScanPoint])
def scan_points(key: str, request: Request) -> list[dict[str, Any]]:
    _get(request, key)
    try:
        return _manager(request).points(key)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
