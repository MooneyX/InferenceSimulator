"""OpenAPI 文档导出：前端 TS 类型生成的输入。

类型同步链路（见 ``docs/design/api.md``「类型同步」）::

    simcore.schemas (Pydantic) → FastAPI OpenAPI → openapi.json → openapi-typescript → TS

导出结果是确定性的（固定缩进、LF 换行、末尾换行），可提交入库并由测试比对新鲜度。

用法::

    uv run python -m simserver.openapi_export <输出路径>
"""

import json
import sys
from pathlib import Path

from simserver.app import create_app

__all__ = ["render_openapi", "write_openapi"]


def render_openapi() -> str:
    """渲染 OpenAPI 文档为确定性 JSON 文本。

    保留 FastAPI 生成的键序（与代码声明顺序一致，便于阅读 diff）；
    ``ensure_ascii=False`` 使中文描述保持可读。
    """
    spec = create_app().openapi()
    return json.dumps(spec, indent=2, ensure_ascii=False) + "\n"


def write_openapi(path: Path) -> None:
    """将 OpenAPI 文档写入 ``path``，统一 UTF-8 与 LF 换行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_openapi(), encoding="utf-8", newline="\n")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m simserver.openapi_export <output-path>", file=sys.stderr)
        return 2
    write_openapi(Path(argv[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
