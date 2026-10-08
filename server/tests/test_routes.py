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
                "max_new_tokens": 128,
                "seed": seed,
            },
        },
        "engine": {"max_running_requests": 16, "max_total_tokens": budget, "page_size": 1},
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
    def test_new_engine_fields_echo_and_counts(self, client: TestClient) -> None:
        payload = make_payload(num_requests=1)
        payload["workload"]["synthetic"]["prompt_len"] = {"dist": "fixed", "value": 16}
        payload["workload"]["synthetic"]["output_len"] = {"dist": "fixed", "value": 1}
        payload["engine"] = {"max_running_requests": 1, "max_total_tokens": 192}
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["config"]["engine"]["page_size"] == 64
        assert body["meta"]["num_completed"] == 1
        assert body["meta"]["num_aborted"] == body["meta"]["num_skipped"] == 0
        assert body["aborted_requests"] == body["skipped_requests"] == []

    def test_enabled_tiers_use_existing_route_and_echo_metrics(self, client: TestClient) -> None:
        payload = make_payload(num_requests=1)
        payload["workload"]["synthetic"]["prompt_len"] = {"dist": "fixed", "value": 3}
        payload["workload"]["synthetic"]["output_len"] = {"dist": "fixed", "value": 1}
        payload["workload"]["synthetic"]["max_new_tokens"] = 1
        payload["engine"].update(
            max_running_requests=1,
            max_total_tokens=12,
            page_size=4,
            l2_capacity_tokens=4,
            l3_capacity_tokens=8,
        )
        payload["perf_model"].update(
            kv_bytes_per_token=16,
            l2_to_l1_bandwidth_bytes_per_ms=64,
            l3_to_l2_bandwidth_bytes_per_ms=32,
        )
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        result = SimResult.model_validate_json(response.content)
        assert result.meta.num_completed == 1
        assert result.meta.config.engine.l2_capacity_tokens == 4
        assert result.meta.config.engine.l3_capacity_tokens == 8
        assert (result.metrics.l2_capacity_blocks, result.metrics.l3_capacity_blocks) == (1, 2)
        assert (result.metrics.l2_free_blocks, result.metrics.l3_free_blocks) == (1, 2)

    def test_returns_full_result(self, client: TestClient) -> None:
        response = client.post(URL, json=make_payload())
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        body = response.json()
        assert set(body) == {
            "meta",
            "metrics",
            "requests",
            "aborted_requests",
            "skipped_requests",
            "warnings",
        }
        assert (
            "allow_cross_session_shared_prefix"
            not in body["meta"]["config"]["workload"]["synthetic"]["session"]
        )
        assert body["meta"]["num_requests"] == 100
        assert body["meta"]["num_completed"] == len(body["requests"])
        assert (
            body["meta"]["num_completed"]
            + body["meta"]["num_aborted"]
            + body["meta"]["num_skipped"]
            == 100
        )
        assert body["metrics"]["total_input_tokens"] == sum(
            request["prompt_len"] for request in body["requests"]
        )
        assert body["metrics"]["l1_capacity_blocks"] == 16_000
        assert body["metrics"]["l1_free_blocks"] + body["metrics"]["l1_resident_blocks"] == 16_000
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
        payload["engine"]["page_size"] = 2
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["metrics"]["target_ideal_prefix_hit_rate"] == 0.25
        assert body["metrics"]["actual_hit_tokens"] == 8
        assert body["requests"][1]["session_id"] == 0
        assert body["requests"][1]["parent_request_id"] == 0
        assert "input_token_ids" not in body["requests"][1]
        assert "input_chunk_ids" not in body["requests"][1]

    def test_request_level_abort_returns_200_and_does_not_block_others(
        self, client: TestClient
    ) -> None:
        payload = make_payload(num_requests=2, budget=192)
        payload["workload"]["synthetic"]["prompt_len"] = {"dist": "fixed", "value": 128}
        payload["workload"]["synthetic"]["output_len"] = {"dist": "fixed", "value": 1}
        payload["engine"]["page_size"] = 64
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        result = SimResult.model_validate_json(response.content)
        assert result.meta.num_requests == 2
        assert result.meta.num_completed == 0
        assert result.meta.num_aborted == 2
        assert [item.reason for item in result.aborted_requests] == [
            "input_exceeds_kv_capacity",
            "input_exceeds_kv_capacity",
        ]
        assert result.metrics.ttft_mean is None


class TestErrors:
    @pytest.mark.parametrize("capacity", [63, 130])
    def test_invalid_page_capacity_returns_422(self, client: TestClient, capacity: int) -> None:
        payload = make_payload(num_requests=1)
        payload["engine"] = {"max_running_requests": 1, "max_total_tokens": capacity}
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert "max_total_tokens" in response.text

    @pytest.mark.parametrize(
        ("engine_updates", "perf_updates", "message"),
        [
            ({"l2_capacity_tokens": 3}, {}, "l2_capacity_tokens"),
            ({"l3_capacity_tokens": 2}, {}, "l3_capacity_tokens"),
            ({"l2_capacity_tokens": 2}, {}, "kv_bytes_per_token"),
            (
                {"l2_capacity_tokens": 2},
                {"kv_bytes_per_token": 16},
                "l2_to_l1_bandwidth_bytes_per_ms",
            ),
            (
                {"l2_capacity_tokens": 2, "l3_capacity_tokens": 2},
                {"kv_bytes_per_token": 16, "l2_to_l1_bandwidth_bytes_per_ms": 128},
                "l3_to_l2_bandwidth_bytes_per_ms",
            ),
        ],
    )
    def test_invalid_tiered_configuration_returns_422(
        self,
        client: TestClient,
        engine_updates: dict[str, int],
        perf_updates: dict[str, int],
        message: str,
    ) -> None:
        payload = make_payload(num_requests=1)
        payload["engine"].update(engine_updates)
        payload["perf_model"].update(perf_updates)
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert message in response.text

    @pytest.mark.parametrize("legacy", ["max_batch_size", "kv_token_budget", "prefix_cache"])
    def test_legacy_engine_inputs_are_rejected(self, client: TestClient, legacy: str) -> None:
        payload = make_payload(num_requests=1)
        payload["engine"][legacy] = 1
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert legacy in response.text

    @pytest.mark.parametrize("exclusive", [True, False])
    def test_pd_switch_is_echoed_and_produces_completed_results(
        self, client: TestClient, exclusive: bool
    ) -> None:
        payload = make_payload(num_requests=1)
        payload["workload"]["synthetic"]["prompt_len"] = {"dist": "fixed", "value": 5}
        payload["workload"]["synthetic"]["output_len"] = {"dist": "fixed", "value": 2}
        payload["engine"].update({"page_size": 4, "pd_mutually_exclusive": exclusive})
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        result = SimResult.model_validate_json(response.content)
        assert result.meta.config.engine.pd_mutually_exclusive is exclusive
        assert result.meta.num_completed == 1 and result.meta.num_engine_steps == 2

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

    @pytest.mark.parametrize("value", [True, False])
    def test_cross_session_field_is_rejected_as_extra(
        self, client: TestClient, value: bool
    ) -> None:
        payload = make_payload(num_requests=2)
        payload["workload"]["synthetic"]["session"] = {
            "num_sessions": 2,
            "allow_cross_session_shared_prefix": value,
        }
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert any(
            error["type"] == "extra_forbidden"
            and error["loc"]
            == ["body", "workload", "synthetic", "session", "allow_cross_session_shared_prefix"]
            for error in response.json()["detail"]
        )

    def test_trace_mode_not_implemented(self, client: TestClient) -> None:
        payload = make_payload()
        payload["workload"] = {"mode": "trace"}
        response = client.post(URL, json=payload)
        assert response.status_code == 501
        assert "trace" in response.json()["detail"]

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("engine", "max_running_requests"), 0),
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
