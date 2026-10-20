"""FastAPI 应用工厂。

采用工厂函数而非模块级单例：导入本模块无副作用，测试可按需构造独立实例。
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from simserver.job_routes import router as job_router
from simserver.jobs import JobManager
from simserver.result_cache import DEFAULT_MAX_BYTES, ResultCache
from simserver.routes import router
from simserver.trace_routes import router as trace_router
from simserver.trace_store import TraceStore

__all__ = ["create_app"]


def create_app(*, cache_path: Path | None = None) -> FastAPI:
    """构造服务层应用实例。

    ``separate_input_output_schemas=False``：请求与响应复用同一组 Pydantic 模型
    （如 ``SimConfig`` 同时出现在请求体与 ``meta.config`` 回显中），关闭拆分使
    OpenAPI 中每个模型只有一个同名 schema，前端生成的 TS 类型与内核类型一一对应。
    """
    @asynccontextmanager
    async def lifespan(current: FastAPI):
        try:
            yield
        finally:
            current.state.jobs.close()

    app = FastAPI(
        lifespan=lifespan,
        title="InferenceSimulator",
        summary="LLM Serving 系统级离散事件仿真服务",
        version="0.1.0",
        separate_input_output_schemas=False,
    )
    app.state.trace_store = TraceStore()
    path = cache_path if cache_path is not None else Path(
        os.environ.get("SIMULATOR_CACHE_PATH", ".simulation-cache/results.sqlite3")
    )
    max_bytes = int(os.environ.get("SIMULATOR_CACHE_MAX_BYTES", str(DEFAULT_MAX_BYTES)))
    app.state.result_cache = ResultCache(path, max_bytes=max_bytes)
    app.state.jobs = JobManager(path, cache_size=max_bytes)
    app.include_router(router)
    app.include_router(job_router)
    app.include_router(trace_router)
    return app
