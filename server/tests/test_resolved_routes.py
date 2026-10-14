"""统一推理入口的物理容量预览与三种推理模式覆盖。"""

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from simcore import SimConfig, SimResult
from simcore.schemas import TracePlan, TraceTurn
from simserver import create_app


def payload() -> dict[str, Any]:
    return {
        "num_instances": 1,
        "workload": {
            "mode": "synthetic",
            "session_rate": 1,
            "seed": 0,
            "synthetic": {
                "num_requests": 1,
                "prompt_len": {"dist": "fixed", "value": 4},
                "turn_increment_len": {"dist": "fixed", "value": 1},
                "output_len": {"dist": "fixed", "value": 2},
            },
        },
        "scheduler": {"max_running_requests": 1},
        "inference": {
            "mode": "physical", "page_size": 4, "tp_size": 1, "ep_size": 1,
            "model": {
                "name": "toy", "num_layers": 2, "hidden_size": 16,
                "num_attention_heads": 2, "num_kv_heads": 1, "head_dim": 8,
                "ffn_intermediate_size": 32, "vocab_size": 16,
            },
            "hardware": {
                "name": "toy", "gpu_count": 1, "memory_bytes_per_gpu": 2_000_000,
                "effective_fp16_tflops": 1, "effective_hbm_gb_s": 1,
            },
            "storage": {
                "l1_available_gb": 0.001,
                "l2_available_gb": 0.000000512, "l3_available_gb": 0.000000256,
                "l2_to_l1_gb_s": 0.001, "l3_to_l2_gb_s": 0.002,
            },
            "timing": {"mode": "analytical"},
        },
    }


def inference_payload(mode: str) -> dict[str, Any]:
    request = payload()
    request["inference"]["page_size"] = 64
    timing = {
        "mode": "empirical", "prefill_c1": 0.01, "prefill_c2": 0,
        "decode_d1": 0.01, "decode_d2": 0,
    }
    if mode == "manual":
        request["inference"] = {
            "mode": "manual", "page_size": 64,
            "kv": {"max_total_tokens": 8192}, "timing": timing,
        }
    else:
        request["inference"]["storage"].update(l2_available_gb=0, l3_available_gb=0)
        if mode == "physical_empirical":
            request["inference"]["timing"] = timing
    return request


def attach_trace(
    client: TestClient, request: dict[str, Any], mode: str = "open_loop", output_len: int = 1,
) -> None:
    plan = TracePlan(block_size=64, sessions=("session-a",), requests=(
        TraceTurn(request_id=0, session_id=0, arrival_offset_ms=0,
                  prompt_len=129, output_len=output_len, hash_ids=(0, 1)),
        TraceTurn(request_id=1, session_id=0, arrival_offset_ms=1000,
                  parent_request_id=0, think_time_ms=100,
                  prompt_len=193, output_len=1, hash_ids=(0, 1, 2)),
    ))
    upload = client.post("/api/replay-plans", content=plan.model_dump_json().encode())
    assert upload.status_code == 201, upload.text
    request["workload"] = {
        "mode": "trace", "arrival_mode": mode, "session_rate": 1, "seed": 3,
        "trace": {"plan_id": upload.json()["id"]},
    }


@pytest.mark.parametrize("inference_mode", ["manual", "physical_empirical", "physical_analytical"])
@pytest.mark.parametrize("workload_mode", ["synthetic", "open_loop", "closed_loop"])
def test_three_inference_modes_use_one_run_contract(
    inference_mode: str, workload_mode: str,
) -> None:
    with TestClient(create_app()) as client:
        request = inference_payload(inference_mode)
        if workload_mode != "synthetic":
            attach_trace(client, request, workload_mode)
        response = client.post("/api/simulate", json=request)
        assert response.status_code == 200, response.text
        result = SimResult.model_validate_json(response.content)
        assert result.meta.config == SimConfig.model_validate(request)
        assert result.meta.num_completed == (1 if workload_mode == "synthetic" else 2)
        assert (result.resolution is not None) == (inference_mode != "manual")
        assert (result.analysis is not None) == (inference_mode == "physical_analytical")
        if result.resolution is not None:
            preview = client.post("/api/resolve", json=request)
            assert preview.status_code == 200, preview.text
            assert result.resolution.model_dump(mode="json") == preview.json()
        with patch("simserver.routes.simulate", return_value=result):
            repeated = client.post("/api/simulate", json=request)
        assert repeated.text == result.model_dump_json()


@pytest.mark.parametrize("inference_mode", ["manual", "physical_empirical", "physical_analytical"])
@pytest.mark.parametrize("workload_mode", ["synthetic", "open_loop", "closed_loop"])
def test_engine_output_limit_is_request_failure_in_every_mode(
    inference_mode: str, workload_mode: str,
) -> None:
    with TestClient(create_app()) as client:
        request = inference_payload(inference_mode)
        request["inference"]["max_output_tokens"] = 1
        if workload_mode != "synthetic":
            attach_trace(client, request, workload_mode, output_len=2)
        response = client.post("/api/simulate", json=request)
        assert response.status_code == 200, response.text
        result = SimResult.model_validate_json(response.content)
        assert result.meta.config.inference.max_output_tokens == 1
        assert len(result.aborted_requests) == 1
        assert result.aborted_requests[0].reason == "output_exceeds_engine_limit"
        assert result.aborted_requests[0].generated_tokens == 0
        assert result.meta.num_completed == (1 if workload_mode == "open_loop" else 0)
        assert result.meta.num_skipped == (1 if workload_mode == "closed_loop" else 0)
        if result.requests:
            assert result.requests[0].prompt_len == 193
            assert result.requests[0].output_len == 1


@pytest.mark.parametrize("inference_mode", ["manual", "physical_empirical", "physical_analytical"])
@pytest.mark.parametrize("workload_mode", ["synthetic", "trace"])
@pytest.mark.parametrize("error,status", [(ValueError("invalid"), 422), (NotImplementedError("unsupported"), 501)])
def test_unified_run_maps_kernel_errors_for_every_mode(
    inference_mode: str, workload_mode: str, error: Exception, status: int,
) -> None:
    with TestClient(create_app()) as client:
        request = inference_payload(inference_mode)
        if workload_mode == "trace":
            attach_trace(client, request)
        with patch("simserver.routes.simulate", side_effect=error):
            response = client.post("/api/simulate", json=request)
        assert response.status_code == status
        assert response.json() == {"detail": str(error)}


def test_trace_physical_preview_does_not_require_uploaded_plan() -> None:
    request = inference_payload("physical_analytical")
    request["workload"] = {"mode": "trace", "session_rate": 1, "trace": {}}
    preview = TestClient(create_app()).post("/api/resolve", json=request)
    assert preview.status_code == 200, preview.text
    assert preview.json()["runnable"] is True


@pytest.mark.parametrize("path", [
    "/api/simulate/resolved", "/api/replay-plans/unknown/simulate",
    "/api/replay-plans/unknown/simulate/resolved",
])
def test_old_run_routes_are_removed(path: str) -> None:
    assert TestClient(create_app()).post(path, json=payload()).status_code == 404


def test_preview_and_run_resolve_identically() -> None:
    client = TestClient(create_app())
    request = payload()
    preview = client.post("/api/resolve", json=request)
    assert preview.status_code == 200, preview.text
    run = client.post("/api/simulate", json=request)
    assert run.status_code == 200, run.text
    data = run.json()
    assert data["resolution"] == preview.json()
    assert data["meta"]["config"] == SimConfig.model_validate(request).model_dump(mode="json")
    assert set(data["resolution"]) == {
        "capacity", "max_total_tokens", "l2_capacity_tokens", "l3_capacity_tokens", "runnable", "reason",
    }
    assert data["analysis"]["steps"] == data["meta"]["num_engine_steps"]
    assert data["meta"]["num_completed"] == 1
    assert data["meta"]["wall_clock_ms"] >= 0
    request["inference"]["timing"] = {
        "mode": "empirical", "prefill_c1": 1, "prefill_c2": 0, "decode_d1": 2, "decode_d2": 1,
    }
    empirical = client.post("/api/simulate", json=request).json()
    assert empirical["analysis"] is None
    assert empirical["meta"]["wall_clock_ms"] >= 0


def test_custom_attention_groups_preview_and_run() -> None:
    client = TestClient(create_app())
    request = payload()
    mla = {"kind": "mla", "layers": 1, "num_attention_heads": 2,
           "q_rank": 4, "kv_rank": 4, "rope_dim": 2, "nope_dim": 6, "value_dim": 8}
    dsa = {**mla, "kind": "mla_dsa", "topk": 4, "index_heads": 2,
           "index_dim": 128, "full_indexer_layers": 1}
    request["inference"]["model"]["attention_layout"] = "grouped"
    request["inference"]["model"]["attention_groups"] = [mla, dsa]
    request["inference"]["storage"].update({"l2_available_gb": 0, "l3_available_gb": 0})
    preview = client.post("/api/resolve", json=request)
    assert preview.status_code == 200, preview.text
    assert preview.json()["capacity"]["kv_bytes_per_token"] == 156
    run = client.post("/api/simulate", json=request)
    assert run.status_code == 200, run.text
    assert run.json()["resolution"] == preview.json()
    request["inference"]["model"]["attention_groups"][1]["layers"] = 2
    assert client.post("/api/resolve", json=request).status_code == 422


def test_hy4_bf16_tp8_ep8_preview_and_simulation() -> None:
    client = TestClient(create_app())
    request = payload()
    request["inference"]["model"] = client.get("/api/presets").json()["models"][2]
    request["inference"]["hardware"].update({
        "gpu_count": 8, "memory_bytes_per_gpu": 288_000_000_000,
        "effective_interconnect_gb_s": 600, "collective_latency_ms": 0.01,
        "ep_communication_latency_ms": 0.02,
    })
    request["inference"]["storage"].update({"l1_available_gb": 260, "l2_available_gb": 0, "l3_available_gb": 0})
    request["inference"].update({"tp_size": 8, "ep_size": 8})
    preview = client.post("/api/resolve", json=request)
    assert preview.status_code == 200, preview.text
    assert preview.json()["capacity"]["rank_reports"][0]["kv_bytes_per_token"] == 92_628
    result = client.post("/api/simulate", json=request)
    assert result.status_code == 200, result.text
    assert result.json()["resolution"] == preview.json()
    assert result.json()["analysis"]["ep_exchanges"] > 0
    custom = dict(request, inference=dict(request["inference"], model=dict(
        request["inference"]["model"], name="custom", revision=None, attention_layout="grouped",
    )))
    custom_preview = client.post("/api/resolve", json=custom)
    assert custom_preview.status_code == 200, custom_preview.text
    assert custom_preview.json()["capacity"] == preview.json()["capacity"]
    assert custom_preview.json() == preview.json()
    custom_run = client.post("/api/simulate", json=custom)
    assert custom_run.status_code == 200, custom_run.text
    assert custom_run.json()["analysis"] == result.json()["analysis"]
    request["inference"]["model"]["kv_dtype"] = "fp8"
    assert client.post("/api/resolve", json=request).status_code == 422
    request["inference"]["model"]["kv_dtype"] = "bf16"
    request["inference"]["model"]["revision"] = "user-revision"
    request["inference"]["model"]["hidden_size"] = 4096
    request["inference"]["model"]["attention_groups"][0]["topk"] = 1024
    modified = client.post("/api/resolve", json=request)
    assert modified.status_code == 200, modified.text
    assert "model" not in modified.json()
    assert (
        modified.json()["capacity"]["weights_resident_bytes"]
        < preview.json()["capacity"]["weights_resident_bytes"]
    )
    run_modified = client.post("/api/simulate", json=request)
    assert run_modified.status_code == 200, run_modified.text
    assert run_modified.json()["resolution"] == modified.json()


def test_tp_analytical_run_matches_preview_and_empirical_stays_unavailable() -> None:
    client = TestClient(create_app())
    request = payload()
    request["inference"]["model"].update(
        {
            "name": "Qwen3-8B",
            "num_layers": 36,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "ffn_intermediate_size": 12288,
            "vocab_size": 151936,
            "tie_word_embeddings": False,
        }
    )
    request["inference"]["hardware"].update(
        {
            "gpu_count": 4,
            "memory_bytes_per_gpu": 288_000_000_000,
            "effective_interconnect_gb_s": 100,
            "collective_latency_ms": 0.1,
        }
    )
    request["inference"]["storage"].update({"l1_available_gb": 20, "l2_available_gb": 1, "l3_available_gb": 0.5})
    request["inference"]["tp_size"] = 4
    preview = client.post("/api/resolve", json=request)
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["runnable"] is True
    assert "tp_size" not in body
    assert len(body["capacity"]["rank_reports"]) == 4
    run = client.post("/api/simulate", json=request)
    assert run.status_code == 200, run.text
    data = run.json()
    assert data["resolution"] == body
    assert data["meta"]["num_completed"] == 1
    assert data["analysis"]["communication_ms"] > 0
    assert data["analysis"]["collectives"] == 2 * 36 * data["analysis"]["steps"]
    request["inference"]["timing"] = {
        "mode": "empirical", "prefill_c1": 1, "prefill_c2": 0, "decode_d1": 2, "decode_d2": 1,
    }
    assert client.post("/api/resolve", json=request).json()["runnable"] is False
    refused = client.post("/api/simulate", json=request)
    assert refused.status_code == 501
    assert "四系数" in refused.json()["detail"]


def test_qwen_moe_ep_preview_requires_parameters_then_runs_analytical() -> None:
    client = TestClient(create_app())
    catalog = client.get("/api/presets").json()
    request = payload()
    request["inference"]["model"] = catalog["models"][1]
    request["inference"]["hardware"].update(
        {
            "gpu_count": 8,
            "memory_bytes_per_gpu": 288_000_000_000,
            "effective_interconnect_gb_s": 600,
        }
    )
    request["inference"]["storage"].update(
        {
            "l1_available_gb": 200,
            "l2_available_gb": 0,
            "l3_available_gb": 0,
        }
    )
    request["inference"].update({"tp_size": 8, "ep_size": 8})
    response = client.post("/api/resolve", json=request)
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["runnable"] is False
    assert "ep_size" not in preview
    assert len(preview["capacity"]["rank_reports"]) == 8
    expected = 94 * 128 * 3 * 4096 * 1536 // 8 * 2
    expert_weights = {item["expert_weights_bytes"] for item in preview["capacity"]["rank_reports"]}
    assert expert_weights == {expected}
    assert preview["capacity"]["estimated_total_parameters"] == 235_093_634_560
    assert preview["capacity"]["estimated_active_parameters"] == 22_190_763_520
    assert preview["capacity"]["estimated_expert_parameters"] == 227_096_395_776
    assert preview["capacity"]["weights_resident_bytes"] >= 2 * 235_093_634_560
    incomplete = client.post("/api/simulate", json=request)
    assert incomplete.status_code == 422
    request["inference"]["hardware"].update(
        {
            "collective_latency_ms": 0.01,
            "ep_communication_latency_ms": 0.02,
        }
    )
    ready = client.post("/api/resolve", json=request)
    assert ready.status_code == 200 and ready.json()["runnable"] is True
    run = client.post("/api/simulate", json=request)
    assert run.status_code == 200, run.text
    data = run.json()
    assert data["resolution"] == ready.json()
    assert data["meta"]["num_completed"] == 1
    assert data["analysis"]["ep_dispatch_ms"] > 0
    assert data["analysis"]["ep_combine_ms"] > 0
    assert data["analysis"]["ep_exchanges"] > 0
    request["inference"]["timing"] = {
        "mode": "empirical", "prefill_c1": 1, "prefill_c2": 0, "decode_d1": 2, "decode_d2": 1,
    }
    assert client.post("/api/resolve", json=request).json()["runnable"] is False
    assert client.post("/api/simulate", json=request).status_code == 501


def test_presets_and_unsupported_modes() -> None:
    client = TestClient(create_app())
    catalog = client.get("/api/presets")
    assert catalog.status_code == 200
    assert {item["name"] for item in catalog.json()["models"]} == {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-235B-A22B",
        "tencent/Hy4-preview",
    }
    assert catalog.json()["hardware"][0]["gpu_count"] == 8
    assert catalog.json()["hardware"][0]["memory_bytes_per_gpu"] == 288_000_000_000
    assert catalog.json()["hardware"][0]["peak_fp16_tflops"] == 2250
    assert catalog.json()["hardware"][0]["peak_hbm_gb_s"] == 8000
    assert catalog.json()["hardware"][0]["peak_interconnect_gb_s"] == 1800
    assert catalog.json()["models"][0]["vocab_size"] == 151936
    assert catalog.json()["models"][1]["expert_intermediate_size"] == 1536
    for model, layers, heads, kv_heads in (
        (catalog.json()["models"][0], 36, 32, 8),
        (catalog.json()["models"][1], 94, 64, 4),
    ):
        assert model["attention_layout"] == "grouped"
        assert len(model["attention_groups"]) == 1
        group = model["attention_groups"][0]
        assert (
            group["kind"],
            group["layers"],
            group["num_attention_heads"],
            group["num_kv_heads"],
            group["head_dim"],
        ) == ("gqa", layers, heads, kv_heads, 128)
        assert group["qk_norm"] is True
        assert group["output_gate"] is False
        assert group["sink_dtype"] is None
        assert group["topk"] is None
    assert "total_parameters" not in catalog.json()["models"][0]
    assert set(catalog.json()) == {"models", "hardware"}
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert "ScenarioCatalogEntry" not in schemas
    assert set(schemas["PresetCatalog"]["properties"]) == {"models", "hardware"}
    assert "effective_fp8_tflops" in schemas["HardwareSpec"]["properties"]
    for model in catalog.json()["models"][:2]:
        request = payload()
        request["inference"]["model"] = model
        request["inference"]["hardware"].update(
            {
                "gpu_count": 8,
                "memory_bytes_per_gpu": 288_000_000_000,
                "effective_interconnect_gb_s": 600,
                "collective_latency_ms": 0.01,
                "ep_communication_latency_ms": 0.02,
            }
        )
        request["inference"]["storage"].update(
            {"l1_available_gb": 200, "l2_available_gb": 0, "l3_available_gb": 0}
        )
        request["inference"].update({"tp_size": 8, "ep_size": 8 if model["moe_layers"] else 1})
        explicit = client.post("/api/resolve", json=request)
        assert explicit.status_code == 200, explicit.text
        legacy = dict(request, inference=dict(request["inference"], model=dict(
            model, attention_layout="full_gqa", attention_groups=None,
        )))
        original = client.post("/api/resolve", json=legacy)
        assert original.status_code == 200, original.text
        assert explicit.json()["capacity"] == original.json()["capacity"]
        assert client.post("/api/simulate", json=request).json()["analysis"] == (
            client.post("/api/simulate", json=legacy).json()["analysis"]
        )
    request = payload()
    missing_vocab = payload()
    del missing_vocab["inference"]["model"]["vocab_size"]
    assert client.post("/api/resolve", json=missing_vocab).status_code == 422
    request["inference"]["hardware"]["gpu_count"] = 8
    response = client.post("/api/resolve", json=request)
    assert response.status_code == 422
    assert "TP 数" in response.json()["detail"]
    request["inference"]["hardware"]["gpu_count"] = 1
    for field in (
        "total_parameters",
        "active_parameters",
        "expert_parameters_total",
        "resident_weights_bytes",
    ):
        request["inference"]["model"][field] = 1
        rejected = client.post("/api/resolve", json=request)
        assert rejected.status_code == 422
        assert rejected.json()["detail"][0]["loc"] == ["body", "inference", "physical", "model", field]
        del request["inference"]["model"][field]
    request["inference"]["model"]["moe_layers"] = 2
    assert client.post("/api/simulate", json=request).status_code == 422
