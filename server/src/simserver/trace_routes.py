"""原始轨迹上传、后台摘要与请求计划资源接口。"""

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, ValidationError

from simcore.schemas import TracePlan
from simserver.trace_processing import PlanTooLargeError
from simserver.trace_store import MAX_PLAN_BYTES, MAX_TRACE_BYTES, TraceStore

router = APIRouter(prefix="/api")


class SessionSummary(BaseModel):
    id: str
    requests: int
    subagent_groups: int
    input_tokens: int
    output_tokens: int
    start_s: float
    end_s: float
    open_loop_qps: float | None
    set_hit_rate: float
    prefix_hit_rate: float
    inferred_dependencies: int
    timed_roots: int
    ambiguous_dependencies: int


class TraceJob(BaseModel):
    id: str
    status: Literal["processing", "ready", "failed"]
    session_count: int = 0
    error: str | None = None


class SessionPage(BaseModel):
    total: int
    sessions: list[SessionSummary]


class PlanSelection(BaseModel):
    sessions: list[str] = Field(min_length=1)


class PlanReceipt(BaseModel):
    id: str
    sessions: int
    requests: int


def _store(request: Request) -> TraceStore:
    return request.app.state.trace_store


async def _receive(request: Request, *, suffix: str, maximum: int) -> tuple[str, Path]:
    store = _store(request)
    key, path = store.new_path(suffix)
    size = 0
    try:
        with path.open("wb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > maximum:
                    raise HTTPException(413, "文件超过此服务允许的上传大小")
                output.write(chunk)
        if size == 0:
            raise HTTPException(422, "文件为空")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return key, path


@router.post("/traces", response_model=TraceJob, status_code=202)
async def upload_trace(request: Request) -> TraceJob:
    key, path = await _receive(request, suffix=".jsonl", maximum=MAX_TRACE_BYTES)
    _store(request).scan(key, path)
    return TraceJob(id=key, status="processing")


@router.get("/traces/{key}", response_model=TraceJob)
def trace_status(key: str, request: Request) -> TraceJob:
    try:
        job = _store(request).job(key)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return TraceJob(
        id=key, status=job["status"], session_count=len(job["sessions"]), error=job.get("error")
    )


@router.get("/traces/{key}/sessions", response_model=SessionPage)
def sessions(
    key: str,
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> SessionPage:
    try:
        job = _store(request).job(key)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job["status"] != "ready":
        raise HTTPException(409, "trace 尚未准备就绪")
    items = job["sessions"]
    return SessionPage(total=len(items), sessions=items[offset : offset + limit])


@router.post("/traces/{key}/plans", response_model=TracePlan)
def export_plan(key: str, selection: PlanSelection, request: Request) -> Response:
    try:
        body = _store(request).export(key, selection.sessions).model_dump_json().encode("utf-8")
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PlanTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if len(body) > MAX_PLAN_BYTES:
        raise HTTPException(413, "选中的计划超过 64 MiB，请减少会话数量")
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="trace-plan.json"'},
    )


@router.post("/replay-plans", response_model=PlanReceipt, status_code=201)
async def upload_plan(request: Request) -> PlanReceipt:
    key, path = await _receive(request, suffix=".json", maximum=MAX_PLAN_BYTES)
    try:
        plan = _store(request).add_plan(key, path)
    except (ValidationError, ValueError) as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    return PlanReceipt(id=key, sessions=len(plan.sessions), requests=len(plan.requests))
