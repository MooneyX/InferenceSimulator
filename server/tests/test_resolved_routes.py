"""自动解析入口与旧仿真 API 的隔离。"""

from typing import Any

from fastapi.testclient import TestClient

from simserver import create_app


def payload() -> dict[str, Any]:
    return {
        "config": {
            "workload": {"mode": "synthetic", "synthetic": {
                "num_requests": 1, "arrival_rate": 1,
                "prompt_len": {"dist": "fixed", "value": 4},
                "output_len": {"dist": "fixed", "value": 2},
                "max_new_tokens": 2, "seed": 0,
            }},
            "engine": {"max_running_requests": 1, "max_total_tokens": 4, "page_size": 4},
            "perf_model": {"prefill_c1": 1, "decode_d1": 2, "decode_d2": 1},
        },
        "model": {
            "name": "toy", "num_layers": 2, "hidden_size": 16,
            "num_attention_heads": 2, "num_kv_heads": 1, "head_dim": 8,
            "ffn_intermediate_size": 32, "total_parameters": 1000,
        },
        "hardware": {
            "name": "toy", "gpu_count": 1, "memory_bytes_per_gpu": 2_000_000,
            "effective_fp16_tflops": 1, "effective_hbm_gb_s": 1,
        },
        "storage": {
            "l1_available_gb": 0.001, "l2_available_gb": 0.000000512,
            "l3_available_gb": 0.000000256, "l2_to_l1_gb_s": 0.001,
            "l3_to_l2_gb_s": 0.002,
        },
        "timing": "analytical",
    }


def test_preview_and_run_resolve_identically() -> None:
    client = TestClient(create_app())
    request = payload()
    preview = client.post("/api/resolve", json=request)
    assert preview.status_code == 200, preview.text
    run = client.post("/api/simulate/resolved", json=request)
    assert run.status_code == 200, run.text
    data = run.json()
    assert data["resolution"] == preview.json()
    assert data["result"]["meta"]["config"] == data["resolution"]["config"]
    assert data["analysis"]["steps"] == data["result"]["meta"]["num_engine_steps"]
    assert data["result"]["meta"]["num_completed"] == 1
    assert client.post("/api/simulate", json=request["config"]).status_code == 200
    request["timing"] = "empirical"
    assert client.post("/api/simulate/resolved", json=request).json()["analysis"] is None


def test_tp_preview_available_but_run_returns_not_implemented() -> None:
    client = TestClient(create_app())
    request = payload()
    request["model"].update({
        "name": "Qwen3-8B", "num_layers": 36, "hidden_size": 4096,
        "num_attention_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "ffn_intermediate_size": 12288, "total_parameters": 8_200_000_000,
        "vocab_size": 151936, "tie_word_embeddings": False,
    })
    request["hardware"].update({
        "gpu_count": 4, "memory_bytes_per_gpu": 288_000_000_000,
        "effective_interconnect_gb_s": 100, "collective_latency_ms": 0.1,
    })
    request["storage"].update({"l1_available_gb": 20, "l2_available_gb": 1,
                               "l3_available_gb": 0.5})
    request["tp_size"] = 4
    preview = client.post("/api/resolve", json=request)
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["runnable"] is False
    assert body["tp_size"] == 4
    assert len(body["capacity"]["rank_reports"]) == 4
    request["timing"] = "empirical"
    run = client.post("/api/simulate/resolved", json=request)
    assert run.status_code == 501
    assert "容量预览" in run.json()["detail"]


def test_presets_and_unsupported_modes() -> None:
    client = TestClient(create_app())
    catalog = client.get("/api/presets")
    assert catalog.status_code == 200
    assert {item["name"] for item in catalog.json()["models"]} == {
        "Qwen/Qwen3-8B", "Qwen/Qwen3-235B-A22B",
    }
    assert catalog.json()["hardware"][0]["gpu_count"] == 8
    assert catalog.json()["hardware"][0]["memory_bytes_per_gpu"] == 288_000_000_000
    assert catalog.json()["hardware"][0]["peak_fp16_tflops"] == 2250
    assert catalog.json()["hardware"][0]["peak_hbm_gb_s"] == 8000
    assert catalog.json()["models"][0]["total_parameters"] == 8_200_000_000
    assert catalog.json()["models"][1]["active_parameters"] == 22_000_000_000
    assert catalog.json()["scenarios"][1]["stage"] == "requires_tp_ep"
    request = payload()
    request["hardware"]["gpu_count"] = 8
    response = client.post("/api/resolve", json=request)
    assert response.status_code == 422
    assert "TP 数" in response.json()["detail"]
    request["hardware"]["gpu_count"] = 1
    request["model"]["moe_layers"] = 2
    assert client.post("/api/simulate/resolved", json=request).status_code == 422
