"""本地处理器与服务端原始轨迹/计划往返。"""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from simserver import create_app
from simserver.trace_processing import PlanTooLargeError, compile_plan, scan_trace


def sample() -> bytes:
    session = {
        "id": "session-a", "models": ["test-model"], "block_size": 64,
        "hash_id_scope": "local", "requests": [
            {"t": 0.0, "type": "s", "model": "test-model", "in": 129,
             "out": 1, "hash_ids": [0, 1], "api_time": 1.0},
            {"t": 2.0, "type": "s", "model": "test-model", "in": 193,
             "out": 1, "hash_ids": [0, 1, 2], "api_time": 1.0, "think_time": 1.0},
            {"t": 3.0, "type": "subagent", "agent_id": "helper", "requests": [
                {"t": 3.0, "type": "n", "model": "test-model", "in": 129,
                 "out": 0, "hash_ids": [0, 1], "api_time": 1.0, "think_time": 0.0},
            ]},
        ],
    }
    return (json.dumps(session) + "\n").encode()


def config() -> dict:
    return {
        "num_instances": 1,
        "workload": {"mode": "synthetic", "synthetic": {
            "num_requests": 1, "session_rate": 1, "seed": 0,
            "prompt_len": {"dist": "fixed", "value": 129},
            "turn_increment_len": {"dist": "fixed", "value": 1},
            "output_len": {"dist": "fixed", "value": 1},
        }},
        "engine": {"max_running_requests": 4, "max_total_tokens": 2048},
        "perf_model": {"prefill_c1": 0.01, "prefill_c2": 0,
                       "decode_d1": 0.01, "decode_d2": 0},
    }


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
        result = client.post(
            f"/api/replay-plans/{plan_id}/simulate",
            json={"config": config(), "mode": "session", "session_rate": 1, "seed": 1},
        )
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["meta"]["num_completed"] == 2
        assert body["aborted_requests"][0]["reason"] == "invalid_trace_output"
        assert body["meta"]["config"]["workload"]["mode"] == "trace"


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
            "version": 1, "block_size": 64, "sessions": ["session-a"],
            "requests": [
                {"request_id": 0, "session_id": 0, "arrival_offset_ms": 0,
                 "prompt_len": 128, "output_len": 1, "hash_ids": [1, 2]},
                {"request_id": 1, "session_id": 0, "arrival_offset_ms": 1,
                 "prompt_len": 128, "output_len": 1, "hash_ids": [3, 2]},
            ],
        }
        response = client.post("/api/replay-plans", content=json.dumps(plan).encode())
        assert response.status_code == 422
        assert "不同输入上下文" in response.json()["detail"]
