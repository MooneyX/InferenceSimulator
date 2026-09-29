"""HTTP 路由：``POST /api/simulate``。

契约见 ``docs/design/api.md``。本模块只做 HTTP 边界工作：请求体由 FastAPI 按
``SimConfig`` 校验（失败即 422），调用内核，再映射内核异常为 HTTP 状态码。
"""

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel

from simcore import SimConfig, SimResult, simulate
from simcore.kv_cache import CacheOOMError

__all__ = ["ErrorResponse", "router"]


class ErrorResponse(BaseModel):
    """服务层错误响应体，与 FastAPI ``HTTPException`` 的默认格式一致。

    属于 HTTP 边界类型，仅用于 OpenAPI 登记，不进入 ``simcore.schemas``。
    """

    detail: str


router = APIRouter(prefix="/api")


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
    except (CacheOOMError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return Response(content=result.model_dump_json(), media_type="application/json")
