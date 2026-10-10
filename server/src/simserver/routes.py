"""HTTP 路由：``POST /api/simulate``。

契约见 ``docs/design/api.md``。本模块只做 HTTP 边界工作：请求体由 FastAPI 按
``SimConfig`` 校验（失败即 422），调用内核，再映射内核异常为 HTTP 状态码。
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel

from simcore import SimConfig, SimResult, simulate
from simcore.capacity import resolve_config
from simcore.runner import simulate_resolved
from simcore.schemas import ModelSpec, ResolvedConfig, ResolvedRequest, ResolvedRun
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


_RESOLVED_ERRORS = {
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


class ScenarioCatalogEntry(BaseModel):
    id: str
    model_name: str
    hardware_name: str
    gpu_count: int
    stage: str


class PresetCatalog(BaseModel):
    models: list[ModelSpec]
    hardware: list[HardwareCatalogEntry]
    scenarios: list[ScenarioCatalogEntry]


@router.get("/presets", response_model=PresetCatalog, summary="模型与硬件参考目录")
def presets_endpoint() -> PresetCatalog:
    return PresetCatalog(
        models=list(MODELS),
        hardware=[HardwareCatalogEntry.model_validate(item) for item in HARDWARE],
        scenarios=[
            ScenarioCatalogEntry(
                id="qwen3-8b-b300-1",
                model_name="Qwen/Qwen3-8B",
                hardware_name="NVIDIA B300",
                gpu_count=1,
                stage="single_gpu_example",
            ),
            ScenarioCatalogEntry(
                id="qwen3-235b-b300-8",
                model_name="Qwen/Qwen3-235B-A22B",
                hardware_name="NVIDIA B300",
                gpu_count=8,
                stage="requires_tp_ep",
            ),
        ],
    )


@router.post(
    "/resolve",
    response_model=ResolvedConfig,
    summary="解析单实例容量与计时规格",
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: _RESOLVED_ERRORS[
            status.HTTP_422_UNPROCESSABLE_CONTENT
        ]
    },
)
def resolve_endpoint(request: ResolvedRequest) -> ResolvedConfig:
    try:
        return resolve_config(request)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post(
    "/simulate/resolved",
    response_model=ResolvedRun,
    summary="按解析规格运行单实例仿真",
    responses=_RESOLVED_ERRORS,
)
def simulate_resolved_endpoint(request: ResolvedRequest) -> ResolvedRun:
    try:
        resolution = resolve_config(request)
        result, analysis = simulate_resolved(resolution)
        return ResolvedRun(result=result, resolution=resolution, analysis=analysis)
    except NotImplementedError as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post(
    "/simulate",
    response_model=SimResult,
    summary="同步执行一场仿真",
    responses={
        status.HTTP_501_NOT_IMPLEMENTED: {
            "model": ErrorResponse,
            "description": "workload.mode=trace（预留接缝，MVP 未实现）",
        },
    },
)
def simulate_endpoint(config: SimConfig) -> Response:
    """执行整场仿真并返回区间记录、指标与告警。

    - 用同步 ``def``：仿真是纯 CPU 计算，FastAPI 会将其放入线程池执行，
      不阻塞事件循环。
    - 直接返回内核 ``model_dump_json`` 的字节：跳过 FastAPI 对响应模型的二次
      校验（逐条 ``RequestRecord`` 重新校验在大规模仿真下开销可观），并保证
      响应与内核序列化逐字节一致。``response_model`` 仅用于 OpenAPI 登记。
    """
    try:
        result = simulate(config)
    except NotImplementedError as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return Response(content=result.model_dump_json(), media_type="application/json")
