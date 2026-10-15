"""HTTP 路由：``POST /api/simulate``。

契约见 ``docs/design/api.md``。本模块只做 HTTP 边界工作：请求体由 FastAPI 按
``SimConfig`` 校验（失败即 422），调用内核，再映射内核异常为 HTTP 状态码。
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel

from simcore import SimConfig, SimResult, simulate
from simcore.capacity import resolve_config
from simcore.schemas import ModelSpec, ResolutionReport
from simserver.presets import HARDWARE, MODELS

__all__ = ["ErrorResponse", "router"]


class ErrorResponse(BaseModel):
    """服务层错误响应体，与 FastAPI ``HTTPException`` 的默认格式一致。

    属于 HTTP 边界类型，仅用于 OpenAPI 登记，不进入 ``simcore.schemas``。
    """

    detail: str


class FieldValidationResponse(BaseModel):
    """FastAPI 对请求字段校验失败时返回的 detail 列表。"""

    detail: list[dict[str, Any]]


_SIMULATION_ERRORS = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "Trace 请求计划不存在或已过期",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse | FieldValidationResponse,
        "description": "字段校验错误列表，或解析模型/硬件时的业务错误消息",
    },
    status.HTTP_501_NOT_IMPLEMENTED: {
        "model": ErrorResponse,
        "description": "此部署/计时模式尚未实现",
    },
}


router = APIRouter(prefix="/api")


class HardwareCatalogEntry(BaseModel):
    name: str
    gpu_count: int
    memory_bytes_per_gpu: int
    peak_fp16_tflops: float
    peak_hbm_gb_s: float
    peak_interconnect_gb_s: float


class PresetCatalog(BaseModel):
    models: list[ModelSpec]
    hardware: list[HardwareCatalogEntry]


@router.get("/presets", response_model=PresetCatalog, summary="模型与硬件参考目录")
def presets_endpoint() -> PresetCatalog:
    return PresetCatalog(
        models=list(MODELS),
        hardware=[HardwareCatalogEntry.model_validate(item) for item in HARDWARE],
    )


@router.post(
    "/resolve",
    response_model=ResolutionReport,
    summary="预览单实例 L1/L2、全场共享 L3 容量与运行可行性",
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: _SIMULATION_ERRORS[
            status.HTTP_422_UNPROCESSABLE_CONTENT
        ]
    },
)
def resolve_endpoint(config: SimConfig) -> ResolutionReport:
    try:
        return resolve_config(config)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post(
    "/simulate",
    response_model=SimResult,
    summary="同步执行一场仿真",
    responses=_SIMULATION_ERRORS,
)
def simulate_endpoint(config: SimConfig, request: Request) -> Response:
    """执行整场仿真并返回区间记录、指标与告警。

    - 用同步 ``def``：仿真是纯 CPU 计算，FastAPI 会将其放入线程池执行，
      不阻塞事件循环。
    - 直接返回内核 ``model_dump_json`` 的字节：跳过 FastAPI 对响应模型的二次
      校验（逐条 ``RequestRecord`` 重新校验在大规模仿真下开销可观），并保证
      响应与内核序列化逐字节一致。``response_model`` 仅用于 OpenAPI 登记。
    """
    try:
        trace_plan = None
        if config.workload.mode == "trace":
            trace = config.workload.trace
            if not trace.plan_id:
                raise ValueError("Trace 仿真须提供 workload.trace.plan_id")
            trace_plan = request.app.state.trace_store.plan(trace.plan_id)
        result = simulate(config, trace_plan=trace_plan)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return Response(content=result.model_dump_json(), media_type="application/json")
