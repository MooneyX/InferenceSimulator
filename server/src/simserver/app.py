"""FastAPI 应用工厂。

采用工厂函数而非模块级单例：导入本模块无副作用，测试可按需构造独立实例。
"""

from fastapi import FastAPI

__all__ = ["create_app"]


def create_app() -> FastAPI:
    """构造服务层应用实例。

    ``separate_input_output_schemas=False``：请求与响应复用同一组 Pydantic 模型
    （如 ``SimConfig`` 同时出现在请求体与 ``meta.config`` 回显中），关闭拆分使
    OpenAPI 中每个模型只有一个同名 schema，前端生成的 TS 类型与内核类型一一对应。
    """
    return FastAPI(
        title="InferenceSimulator",
        summary="LLM Serving 系统级离散事件仿真服务",
        version="0.1.0",
        separate_input_output_schemas=False,
    )
