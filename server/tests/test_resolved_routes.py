"""自动解析入口与旧仿真 API 的隔离。"""

from typing import Any

from fastapi.testclient import TestClient

from simserver import create_app


def payload() -> dict[str, Any]:
    return {
        "config": {
            "num_instances": 1,
            "workload": {
                "mode": "synthetic",
                "synthetic": {
                    "num_requests": 1,
                    "session_rate": 1,
                    "prompt_len": {"dist": "fixed", "value": 4},
                    "turn_increment_len": {"dist": "fixed", "value": 1},
                    "output_len": {"dist": "fixed", "value": 2},
                    "max_new_tokens": 2,
                    "seed": 0,
                },
            },
            "engine": {"max_running_requests": 1, "max_total_tokens": 4, "page_size": 4},
            "perf_model": {"prefill_c1": 1, "decode_d1": 2, "decode_d2": 1},
        },
        "model": {
            "name": "toy",
            "num_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_kv_heads": 1,
            "head_dim": 8,
            "ffn_intermediate_size": 32,
            "vocab_size": 16,
        },
        "hardware": {
            "name": "toy",
            "gpu_count": 1,
            "memory_bytes_per_gpu": 2_000_000,
            "effective_fp16_tflops": 1,
            "effective_hbm_gb_s": 1,
        },
        "storage": {
            "l1_available_gb": 0.001,
            "l2_available_gb": 0.000000512,
            "l3_available_gb": 0.000000256,
            "l2_to_l1_gb_s": 0.001,
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
    assert data["result"]["meta"]["wall_clock_ms"] >= 0
    assert client.post("/api/simulate", json=request["config"]).status_code == 200
    request["timing"] = "empirical"
    empirical = client.post("/api/simulate/resolved", json=request).json()
    assert empirical["analysis"] is None
    assert empirical["result"]["meta"]["wall_clock_ms"] >= 0


def test_tp_analytical_run_matches_preview_and_empirical_stays_unavailable() -> None:
    client = TestClient(create_app())
    request = payload()
    request["model"].update(
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
    request["hardware"].update(
        {
            "gpu_count": 4,
            "memory_bytes_per_gpu": 288_000_000_000,
            "effective_interconnect_gb_s": 100,
            "collective_latency_ms": 0.1,
        }
    )
    request["storage"].update({"l1_available_gb": 20, "l2_available_gb": 1, "l3_available_gb": 0.5})
    request["tp_size"] = 4
    preview = client.post("/api/resolve", json=request)
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["runnable"] is True
    assert body["tp_size"] == 4
    assert len(body["capacity"]["rank_reports"]) == 4
    run = client.post("/api/simulate/resolved", json=request)
    assert run.status_code == 200, run.text
    data = run.json()
    assert data["resolution"] == body
    assert data["result"]["meta"]["num_completed"] == 1
    assert data["analysis"]["communication_ms"] > 0
    assert data["analysis"]["collectives"] == 2 * 36 * data["analysis"]["steps"]
    request["timing"] = "empirical"
    assert client.post("/api/resolve", json=request).json()["runnable"] is False
    refused = client.post("/api/simulate/resolved", json=request)
    assert refused.status_code == 501
    assert "四系数" in refused.json()["detail"]


def test_qwen_moe_ep_preview_requires_parameters_then_runs_analytical() -> None:
    client = TestClient(create_app())
    catalog = client.get("/api/presets").json()
    request = payload()
    request["model"] = catalog["models"][1]
    request["hardware"].update(
        {
            "gpu_count": 8,
            "memory_bytes_per_gpu": 288_000_000_000,
            "effective_interconnect_gb_s": 600,
        }
    )
    request["storage"].update(
        {
            "l1_available_gb": 200,
            "l2_available_gb": 0,
            "l3_available_gb": 0,
        }
    )
    request.update({"tp_size": 8, "ep_size": 8})
    response = client.post("/api/resolve", json=request)
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["runnable"] is False
    assert preview["ep_size"] == 8
    assert len(preview["capacity"]["rank_reports"]) == 8
    expected = 94 * 128 * 3 * 4096 * 1536 // 8 * 2
    expert_weights = {item["expert_weights_bytes"] for item in preview["capacity"]["rank_reports"]}
    assert expert_weights == {expected}
    assert preview["capacity"]["estimated_total_parameters"] == 235_093_634_560
    assert preview["capacity"]["estimated_active_parameters"] == 22_190_763_520
    assert preview["capacity"]["estimated_expert_parameters"] == 227_096_395_776
    assert preview["capacity"]["weights_resident_bytes"] >= 2 * 235_093_634_560
    incomplete = client.post("/api/simulate/resolved", json=request)
    assert incomplete.status_code == 422
    request["hardware"].update(
        {
            "collective_latency_ms": 0.01,
            "ep_communication_latency_ms": 0.02,
        }
    )
    ready = client.post("/api/resolve", json=request)
    assert ready.status_code == 200 and ready.json()["runnable"] is True
    run = client.post("/api/simulate/resolved", json=request)
    assert run.status_code == 200, run.text
    data = run.json()
    assert data["resolution"] == ready.json()
    assert data["result"]["meta"]["num_completed"] == 1
    assert data["analysis"]["ep_dispatch_ms"] > 0
    assert data["analysis"]["ep_combine_ms"] > 0
    assert data["analysis"]["ep_exchanges"] > 0
    request["timing"] = "empirical"
    assert client.post("/api/resolve", json=request).json()["runnable"] is False
    assert client.post("/api/simulate/resolved", json=request).status_code == 501


def test_presets_and_unsupported_modes() -> None:
    client = TestClient(create_app())
    catalog = client.get("/api/presets")
    assert catalog.status_code == 200
    assert {item["name"] for item in catalog.json()["models"]} == {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-235B-A22B",
    }
    assert catalog.json()["hardware"][0]["gpu_count"] == 8
    assert catalog.json()["hardware"][0]["memory_bytes_per_gpu"] == 288_000_000_000
    assert catalog.json()["hardware"][0]["peak_fp16_tflops"] == 2250
    assert catalog.json()["hardware"][0]["peak_hbm_gb_s"] == 8000
    assert catalog.json()["hardware"][0]["peak_interconnect_gb_s"] == 1800
    assert catalog.json()["models"][0]["vocab_size"] == 151936
    assert catalog.json()["models"][1]["expert_intermediate_size"] == 1536
    assert "total_parameters" not in catalog.json()["models"][0]
    assert catalog.json()["scenarios"][1]["stage"] == "requires_tp_ep"
    request = payload()
    missing_vocab = payload()
    del missing_vocab["model"]["vocab_size"]
    assert client.post("/api/resolve", json=missing_vocab).status_code == 422
    request["hardware"]["gpu_count"] = 8
    response = client.post("/api/resolve", json=request)
    assert response.status_code == 422
    assert "TP 数" in response.json()["detail"]
    request["hardware"]["gpu_count"] = 1
    for field in (
        "total_parameters",
        "active_parameters",
        "expert_parameters_total",
        "resident_weights_bytes",
    ):
        request["model"][field] = 1
        rejected = client.post("/api/resolve", json=request)
        assert rejected.status_code == 422
        assert rejected.json()["detail"][0]["loc"] == ["body", "model", field]
        del request["model"][field]
    request["model"]["moe_layers"] = 2
    assert client.post("/api/simulate/resolved", json=request).status_code == 422
