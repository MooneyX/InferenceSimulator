"""simserver —— simcore 仿真内核的 HTTP 服务层（FastAPI）。

职责仅限 HTTP 边界：请求体校验、调用 ``simcore.simulate``、响应序列化与错误映射。
不做任何仿真计算；数据类型全部来自 ``simcore.schemas``，不另行定义。

启动::

    uv run uvicorn simserver.app:create_app --factory
"""

from simserver.app import create_app

__version__ = "0.1.0"

__all__ = ["__version__", "create_app"]
