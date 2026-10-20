"""参考配置、容量预览与结果缓存管理路由。"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel

from simcore import SimConfig
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


_SIMULATION_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "Trace 请求计划不存在或已过期",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse | FieldValidationResponse,
        "description": "字段校验错误列表，或解析模型/硬件时的业务错误消息",
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


@router.delete("/simulation-cache", status_code=204, summary="清空本地仿真结果缓存")
def clear_simulation_cache(request: Request) -> Response:
    request.app.state.result_cache.clear()
    return Response(status_code=204)
