"""本地处理器与服务端原始轨迹/计划往返。"""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import await_task, run_simulation
from simserver import create_app
from simserver.trace_processing import PlanTooLargeError, compile_plan, scan_trace


def sample() -> bytes:
    session = {
        "id": "session-a",
        "models": ["test-model"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "s",
                "model": "test-model",
                "in": 129,
                "out": 1,
                "hash_ids": [0, 1],
                "api_time": 1.0,
            },
            {
                "t": 2.0,
                "type": "s",
                "model": "test-model",
                "in": 193,
                "out": 1,
                "hash_ids": [0, 1, 2],
                "api_time": 1.0,
                "think_time": 1.0,
            },
            {
                "t": 3.0,
                "type": "subagent",
                "agent_id": "helper",
                "requests": [
                    {
                        "t": 3.0,
                        "type": "n",
                        "model": "test-model",
                        "in": 129,
                        "out": 0,
                        "hash_ids": [0, 1],
                        "api_time": 1.0,
                        "think_time": 0.0,
                    },
                ],
            },
        ],
    }
    return (json.dumps(session) + "\n").encode()


def config() -> dict:
    return {
        "num_instances": 1,
        "workload": {
            "mode": "synthetic",
            "session_rate": 1,
            "seed": 0,
            "synthetic": {
                "num_requests": 1,
                "prompt_len": {"dist": "fixed", "value": 129},
                "turn_increment_len": {"dist": "fixed", "value": 1},
                "output_len": {"dist": "fixed", "value": 1},
            },
        },
        "scheduler": {"max_running_requests": 4},
        "inference": {
            "page_size": 64,
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
                "memory_bytes_per_gpu": 2_000_000,
                "effective_fp16_tflops": 1,
                "effective_hbm_gb_s": 1,
            },
            "storage": {"l1_available_gb": (10_464 + 2048 * 64) / 1_000_000_000},
            "timing": {
                "mode": "empirical",
                "prefill_c1": 0.01,
                "prefill_c2": 0,
                "decode_d1": 0.01,
                "decode_d2": 0,
            },
        },
    }


@pytest.mark.parametrize("field", ["t", "api_time", "think_time"])
@pytest.mark.parametrize("invalid", [-1, float("inf"), float("nan")])
def test_invalid_original_request_times_are_rejected(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    raw = json.loads(sample())
    raw["requests"][1][field] = invalid
    source = tmp_path / "trace.jsonl"
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="原始时间或思考时间无效"):
        scan_trace(source)
    with pytest.raises(ValueError, match="原始时间或思考时间无效"):
        compile_plan(source, ["session-a"])


def test_missing_api_time_is_rejected_before_topology(tmp_path: Path) -> None:
    raw = json.loads(sample())
    del raw["requests"][0]["api_time"]
    source = tmp_path / "trace.jsonl"
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="原始时间或思考时间无效"):
        scan_trace(source)
    with pytest.raises(ValueError, match="原始时间或思考时间无效"):
        compile_plan(source, ["session-a"])


def test_local_summary_parent_and_export(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    source.write_bytes(sample())
    summary = scan_trace(source)[0]
    assert summary["requests"] == 3
    assert summary["input_tokens"] == 451
    assert summary["open_loop_qps"] == pytest.approx(2 / 3)
    assert summary["inferred_dependencies"] == 1
    assert summary["timed_roots"] == 2
    plan = compile_plan(source, ["session-a"])
    assert [item.parent_request_id for item in plan.requests] == [None, 0, None]
    assert plan.requests[1].think_time_ms == 1000
    assert plan.requests[2].output_len == 0


def test_upload_preview_export_and_reupload_plan() -> None:
    with TestClient(create_app()) as client:
        receipt = client.post("/api/traces", content=sample())
        assert receipt.status_code == 202
        key = receipt.json()["id"]
        for _ in range(100):
            state = client.get(f"/api/traces/{key}").json()
            if state["status"] != "processing":
                break
            time.sleep(0.01)
        assert state["status"] == "ready", state
        page = client.get(f"/api/traces/{key}/sessions").json()
        assert page["total"] == 1
        assert page["sessions"][0]["inferred_dependencies"] == 1
        exported = client.post(f"/api/traces/{key}/plans", json={"sessions": ["session-a"]})
        assert exported.status_code == 200
        uploaded = client.post("/api/replay-plans", content=exported.content)
        assert uploaded.status_code == 201
        plan_id = uploaded.json()["id"]
        request = config()
        request["workload"] = {
            "mode": "trace",
            "arrival_mode": "closed_loop",
            "session_rate": 1,
            "seed": 1,
            "trace": {"plan_id": plan_id},
        }
        result = run_simulation(client, request)
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["meta"]["num_completed"] == 2
        assert body["aborted_requests"][0]["reason"] == "invalid_trace_output"
        assert body["meta"]["config"]["workload"]["mode"] == "trace"
        request["workload"].update(arrival_mode="open_loop", target_qps=4)
        scaled = run_simulation(client, request)
        assert scaled.status_code == 200, scaled.text
        assert scaled.json()["metrics"]["arrival_qps"] == pytest.approx(4)
        assert scaled.json()["meta"]["num_requests"] == 3
        request["workload"]["arrival_mode"] = "closed_loop"
        invalid = client.post("/api/simulate", json=request)
        assert invalid.status_code == 422


def test_trace_plan_uses_parsed_capacity_and_timing() -> None:
    resolved = config()
    resolved["inference"] = {
        "page_size": 64,
        "tp_size": 1,
        "ep_size": 1,
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
        "storage": {"l1_available_gb": 0.001},
        "timing": {"mode": "analytical"},
    }
    with TestClient(create_app()) as client:
        plan = compile_plan_from_sample(client)
        resolved["workload"] = {
            "mode": "trace",
            "arrival_mode": "open_loop",
            "session_rate": 1,
            "seed": 2,
            "trace": {"plan_id": plan},
        }
        preview = client.post("/api/resolve", json=resolved)
        response = run_simulation(client, resolved)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["resolution"] == preview.json()
        assert "mode" not in data["meta"]["config"]["inference"]
        assert data["meta"]["num_requests"] == 3
        assert data["analysis"]["steps"] == data["meta"]["num_engine_steps"]
        assert data["resolution"]["max_total_tokens"] != 2048
        resolved["workload"]["target_qps"] = 4
        scaled = run_simulation(client, resolved)
        assert scaled.status_code == 200, scaled.text
        assert scaled.json()["metrics"]["arrival_qps"] == pytest.approx(4)
        assert scaled.json()["analysis"] is not None
        resolved["workload"].pop("target_qps")
        empirical_timing = config()["inference"]["timing"]
        resolved["inference"]["timing"] = empirical_timing
        empirical = run_simulation(client, resolved)
        assert empirical.status_code == 200, empirical.text
        assert empirical.json()["analysis"] is None
        resolved["inference"]["hardware"].update(
            {
                "gpu_count": 2,
                "effective_interconnect_gb_s": 100,
                "collective_latency_ms": 0.1,
            }
        )
        resolved["inference"]["tp_size"] = 2
        resolved["inference"]["timing"] = {"mode": "analytical"}
        tp = run_simulation(client, resolved)
        assert tp.status_code == 200, tp.text
        assert tp.json()["analysis"]["communication_ms"] > 0
        resolved["inference"]["timing"] = empirical_timing
        refused = client.post("/api/simulate", json=resolved)
        assert refused.status_code == 202
        assert await_task(client, refused.json()["id"])["status"] == "failed"


def compile_plan_from_sample(client: TestClient) -> str:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        source = Path(directory) / "trace.jsonl"
        source.write_bytes(sample())
        plan = compile_plan(source, ["session-a"])
    response = client.post("/api/replay-plans", content=plan.model_dump_json().encode())
    assert response.status_code == 201
    return response.json()["id"]


def test_ambiguous_time_match_stays_timed_root(tmp_path: Path) -> None:
    raw = json.loads(sample())
    raw["requests"] = [
        {**raw["requests"][0], "api_time": 2.0},
        {**raw["requests"][0], "t": 1.0, "api_time": 1.0},
        {**raw["requests"][1], "t": 3.0, "think_time": 1.0},
    ]
    path = tmp_path / "ambiguous.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    assert scan_trace(path)[0]["ambiguous_dependencies"] == 1
    assert compile_plan(path, ["session-a"]).requests[2].parent_request_id is None
    with pytest.raises(PlanTooLargeError):
        compile_plan(path, ["session-a"], max_bytes=128)


def test_invalid_plan_and_missing_trace() -> None:
    with TestClient(create_app()) as client:
        assert client.post("/api/replay-plans", content=b"not-json").status_code == 422
        assert client.get("/api/traces/unknown").status_code == 404
        assert client.post("/api/traces", content=b"").status_code == 422
        plan = {
            "version": 1,
            "block_size": 64,
            "sessions": ["session-a"],
            "requests": [
                {
                    "request_id": 0,
                    "session_id": 0,
                    "arrival_offset_ms": 0,
                    "prompt_len": 128,
                    "output_len": 1,
                    "hash_ids": [1, 2],
                },
                {
                    "request_id": 1,
                    "session_id": 0,
                    "arrival_offset_ms": 1,
                    "prompt_len": 128,
                    "output_len": 1,
                    "hash_ids": [3, 2],
                },
            ],
        }
        response = client.post("/api/replay-plans", content=json.dumps(plan).encode())
        assert response.status_code == 422
        assert "不同输入上下文" in response.json()["detail"]
