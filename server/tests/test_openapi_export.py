"""openapi_export 模块单测：确定性、schema 命名与入库文档的新鲜度。"""

import json
from pathlib import Path
from typing import Any

import pytest

from simserver.openapi_export import main, render_openapi, write_openapi

COMMITTED = Path(__file__).resolve().parents[2] / "web" / "openapi.json"


def test_render_deterministic() -> None:
    assert render_openapi() == render_openapi()


def test_render_format() -> None:
    text = render_openapi()
    assert text.endswith("}\n")
    assert "\r" not in text
    assert "仿真" in text, "中文描述应保持原文，不转义为 \\uXXXX"


def test_schema_names_not_split() -> None:
    """``separate_input_output_schemas=False`` 生效：不出现 -Input / -Output 变体。"""
    schemas: dict[str, Any] = json.loads(render_openapi())["components"]["schemas"]
    assert {"SimConfig", "SimResult", "RequestRecord", "Metrics", "ErrorResponse"} <= set(schemas)
    assert not [name for name in schemas if name.endswith(("-Input", "-Output"))]
    assert "ResolutionReport" in schemas
    assert not {
        "ResolvedRequest",
        "ResolvedConfig",
        "ResolvedRun",
        "ReplaySettings",
        "ReplayRun",
        "ResolvedReplayRun",
    } & set(schemas)


def test_workload_common_fields_and_engine_output_limit_schema() -> None:
    schemas = json.loads(render_openapi())["components"]["schemas"]
    common = schemas["WorkloadConfig"]["properties"]
    assert {"mode", "arrival_mode", "seed", "session_rate", "target_qps"} <= set(common)
    assert common["mode"]["default"] == "synthetic"
    assert common["arrival_mode"]["default"] == "closed_loop"
    assert common["seed"]["default"] == 0
    assert set(schemas["SyntheticWorkloadConfig"]["properties"]) == {
        "num_requests",
        "prompt_len",
        "turn_increment_len",
        "output_len",
        "session",
    }
    assert set(schemas["TraceWorkloadConfig"]["properties"]) == {"plan_id"}
    assert schemas["InferenceConfig"]["properties"]["max_output_tokens"]["default"] == 65536
    assert "mode" not in schemas["InferenceConfig"]["properties"]
    assert "mode" not in schemas["SharedL3Config"]["properties"]
    assert not {
        "PhysicalInferenceConfig",
        "PhysicalL3Config",
        "ManualInferenceConfig",
        "ManualKVConfig",
        "ManualL3Config",
    } & set(schemas)
    config = schemas["SimConfig"]["properties"]
    assert config["inference"]["$ref"] == "#/components/schemas/InferenceConfig"


@pytest.mark.parametrize(
    "schema,field,default",
    [
        ("HardwareSpec", "kernel_overhead_ms", 0.005),
        ("StorageSpec", "l2_latency_ms", 0.01),
        ("StorageSpec", "l3_latency_ms", 0.1),
        ("SharedL3Config", "lookup_latency_ms", 0.1),
    ],
)
def test_fixed_latency_defaults_in_openapi(schema: str, field: str, default: float) -> None:
    schemas = json.loads(render_openapi())["components"]["schemas"]
    property_schema = schemas[schema]["properties"][field]
    assert property_schema["default"] == default
    assert property_schema["minimum"] == 0
    assert "未经实测校准" in property_schema["description"]


def test_unified_routes_document_business_and_field_errors() -> None:
    spec = json.loads(render_openapi())
    for path in ("/api/resolve", "/api/simulate"):
        errors = spec["paths"][path]["post"]["responses"]["422"]["content"]["application/json"][
            "schema"
        ]
        assert {item["$ref"] for item in errors["anyOf"]} == {
            "#/components/schemas/ErrorResponse",
            "#/components/schemas/FieldValidationResponse",
        }
    assert spec["paths"]["/api/simulate"]["post"]["responses"]["202"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/JobReceipt"}


def test_only_unified_simulation_routes_remain() -> None:
    spec = json.loads(render_openapi())
    paths = spec["paths"]
    assert {path for path in paths if "simulate" in path} == {"/api/simulate"}
    preview = paths["/api/resolve"]["post"]
    assert preview["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SimConfig",
    }
    assert preview["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ResolutionReport",
    }
    assert {
        "/api/traces",
        "/api/traces/{key}/sessions",
        "/api/traces/{key}/plans",
        "/api/replay-plans",
    } <= set(paths)


def test_write_uses_lf(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "openapi.json"
    write_openapi(target)
    assert target.read_bytes() == render_openapi().encode("utf-8")


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "openapi.json"
    assert main([str(target)]) == 0
    assert target.is_file()
    assert main([]) == 2
    assert "usage" in capsys.readouterr().err


def test_committed_openapi_up_to_date() -> None:
    """入库的 web/openapi.json 必须与当前模型一致；失败时在 web/ 下执行 ``npm run gen:api``。"""
    assert COMMITTED.is_file(), f"缺少 {COMMITTED}，请在 web/ 下执行 npm run gen:api"
    # read_text 默认做通用换行转换，兼容 autocrlf 检出
    assert COMMITTED.read_text(encoding="utf-8") == render_openapi(), (
        "web/openapi.json 已过期，请在 web/ 下执行 npm run gen:api 并提交生成结果"
    )
