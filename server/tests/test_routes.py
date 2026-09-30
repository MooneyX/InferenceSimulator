"""routes 模块单测：``POST /api/simulate`` 的成功路径、可复现性、错误映射与 OpenAPI 登记。"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from simcore import SimConfig, SimResult, simulate
from simserver import create_app

URL = "/api/simulate"


def make_payload(seed: int = 0, num_requests: int = 100, budget: int = 16_000) -> dict[str, Any]:
    return {
        "workload": {
            "mode": "synthetic",
            "synthetic": {
                "num_requests": num_requests,
                "arrival_rate": 30.0,
                "prompt_len": {"dist": "lognormal", "mean": 256.0, "sigma": 0.8},
                "output_len": {"dist": "poisson", "mean": 40.0},
                "seed": seed,
            },
        },
        "engine": {"max_batch_size": 16, "kv_token_budget": budget},
        "perf_model": {
            "prefill_c1": 0.05,
            "prefill_c2": 1e-5,
            "decode_d1": 8.0,
            "decode_d2": 0.002,
        },
    }


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


class TestSuccess:
    def test_returns_full_result(self, client: TestClient) -> None:
        response = client.post(URL, json=make_payload())
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        body = response.json()
        assert set(body) == {"meta", "metrics", "requests", "warnings"}
        assert len(body["requests"]) == 100
        assert body["metrics"]["total_input_tokens"] == sum(
            request["prompt_len"] for request in body["requests"]
        )
        assert body["metrics"]["l1_capacity_blocks"] == 16_000
        assert set(body["requests"][0]) >= {
            "id",
            "arrival",
            "prefill_start",
            "prefill_end",
            "finish",
        }

    def test_matches_kernel_serialization(self, client: TestClient) -> None:
        """响应字节即内核 ``SimResult`` 的 JSON 序列化。"""
        payload = make_payload(seed=3)
        expected = simulate(SimConfig.model_validate(payload)).model_dump_json()
        assert client.post(URL, json=payload).text == expected

    def test_response_validates_as_sim_result(self, client: TestClient) -> None:
        result = SimResult.model_validate_json(client.post(URL, json=make_payload()).content)
        assert result.meta.config == SimConfig.model_validate(make_payload())

    def test_same_seed_bit_identical(self, client: TestClient) -> None:
        payload = make_payload(seed=7)
        assert client.post(URL, json=payload).content == client.post(URL, json=payload).content

    def test_multi_turn_prefix_metrics_returned(self, client: TestClient) -> None:
        payload = make_payload(num_requests=2, budget=80)
        payload["workload"]["synthetic"].update(
            {
                "prompt_len": {"dist": "fixed", "value": 16},
                "output_len": {"dist": "fixed", "value": 8},
                "session": {"num_sessions": 1, "target_ideal_prefix_hit_rate": 0.25},
            }
        )
        payload["engine"]["prefix_cache"] = {"block_size_tokens": 2}
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["metrics"]["target_ideal_prefix_hit_rate"] == 0.25
        assert body["metrics"]["actual_hit_tokens"] == 8
        assert body["requests"][1]["session_id"] == 0
        assert body["requests"][1]["parent_request_id"] == 0
        assert "input_token_ids" not in body["requests"][1]
        assert "input_chunk_ids" not in body["requests"][1]

    def test_infeasible_cache_budget_rejected(self, client: TestClient) -> None:
        response = client.post(URL, json=make_payload(num_requests=50, budget=500))
        assert response.status_code == 422
        assert "超过 L1 容量" in response.json()["detail"]


class TestErrors:
    def test_unreachable_ideal_hit_target_reports_r_max(self, client: TestClient) -> None:
        payload = make_payload(num_requests=2)
        payload["workload"]["synthetic"]["session"] = {
            "num_sessions": 2,
            "target_ideal_prefix_hit_rate": 0.5,
        }
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert "r_max=0" in response.json()["detail"]

    def test_content_seed_is_rejected(self, client: TestClient) -> None:
        payload = make_payload(num_requests=1)
        payload["workload"]["synthetic"]["session"] = {"content_seed": 123}
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert "content_seed" in response.text

    def test_cross_session_shared_prefix_is_rejected(self, client: TestClient) -> None:
        payload = make_payload(num_requests=2)
        payload["workload"]["synthetic"]["session"] = {
            "num_sessions": 2,
            "allow_cross_session_shared_prefix": True,
        }
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert "不支持 allow_cross_session_shared_prefix=true" in response.json()["detail"]

    def test_trace_mode_not_implemented(self, client: TestClient) -> None:
        payload = make_payload()
        payload["workload"] = {"mode": "trace"}
        response = client.post(URL, json=payload)
        assert response.status_code == 501
        assert "trace" in response.json()["detail"]

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("engine", "max_batch_size"), 0),
            (("workload", "synthetic", "arrival_rate"), -1.0),
            (("workload", "synthetic", "prompt_len"), {"dist": "uniform", "value": 3}),
            (("engine", "unknown_field"), 1),
        ],
        ids=["out-of-range", "negative-rate", "bad-discriminator", "extra-field"],
    )
    def test_invalid_config_rejected(
        self, client: TestClient, path: tuple[str, ...], value: object
    ) -> None:
        payload = make_payload()
        node: dict[str, Any] = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert response.json()["detail"]

    def test_synthetic_mode_without_payload_rejected(self, client: TestClient) -> None:
        payload = make_payload()
        payload["workload"] = {"mode": "synthetic"}
        assert client.post(URL, json=payload).status_code == 422

    def test_get_not_allowed(self, client: TestClient) -> None:
        assert client.get(URL).status_code == 405


@pytest.fixture(scope="module")
def operation(client: TestClient) -> dict[str, Any]:
    spec: dict[str, Any] = client.get("/openapi.json").json()
    op: dict[str, Any] = spec["paths"][URL]["post"]
    return op


class TestOpenAPI:
    def test_request_body_is_sim_config(self, operation: dict[str, Any]) -> None:
        schema = operation["requestBody"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/SimConfig"}

    def test_response_is_sim_result(self, operation: dict[str, Any]) -> None:
        schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/SimResult"}

    def test_error_responses_documented(self, operation: dict[str, Any]) -> None:
        assert {"422", "501"} <= set(operation["responses"])
        schema = operation["responses"]["501"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/ErrorResponse"}
