"""routes 模块单测：``POST /api/simulate`` 的成功路径、可复现性、错误映射与 OpenAPI 登记。"""

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from simcore import SimConfig, SimResult, simulate
from simserver import create_app

URL = "/api/simulate"


def make_payload(seed: int = 0, num_requests: int = 100, budget: int = 16_000) -> dict[str, Any]:
    return {
        "num_instances": 1,
        "workload": {
            "mode": "synthetic",
            "session_rate": 30.0,
            "seed": seed,
            "synthetic": {
                "num_requests": num_requests,
                "prompt_len": {"dist": "lognormal", "mean": 256.0, "sigma": 0.8},
                "turn_increment_len": {"dist": "fixed", "value": 8},
                "output_len": {"dist": "poisson", "mean": 40.0},
            },
        },
        "scheduler": {"max_running_requests": 16},
        "inference": {
            "page_size": 1,
            "model": {"name": "toy", "num_layers": 2, "hidden_size": 16,
                      "num_attention_heads": 2, "num_kv_heads": 1, "head_dim": 8,
                      "ffn_intermediate_size": 32, "vocab_size": 16},
            "hardware": {"name": "toy", "memory_bytes_per_gpu": 2_000_000,
                         "effective_fp16_tflops": 1, "effective_hbm_gb_s": 1},
            "storage": {"l1_available_gb": (10_464 + budget * 64) / 1_000_000_000,
                        "l2_available_gb": 0},
            "timing": {"mode": "empirical", "prefill_c1": 0.05,
                       "prefill_c2": 1e-5, "decode_d1": 8.0, "decode_d2": 0.002},
        },
    }


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


class TestSuccess:
    def test_closed_loop_session_rate_and_request_qps(self, client: TestClient) -> None:
        payload = make_payload(num_requests=4)
        payload["workload"]["session_rate"] = 2
        payload["workload"]["synthetic"]["session"] = {
            "num_sessions": 2, "think_time_ms": 200,
        }
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["config"]["workload"]["target_qps"] is None
        assert body["meta"]["config"]["workload"]["session_rate"] == 2
        assert body["metrics"]["arrival_qps"] > 0

    def test_physical_fields_echo_and_counts(self, client: TestClient) -> None:
        payload = make_payload(num_requests=1)
        payload["workload"]["synthetic"]["prompt_len"] = {"dist": "fixed", "value": 16}
        payload["workload"]["synthetic"]["output_len"] = {"dist": "fixed", "value": 1}
        payload["scheduler"] = {"max_running_requests": 1}
        del payload["inference"]["page_size"]
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["config"]["inference"]["page_size"] == 64
        assert "mode" not in body["meta"]["config"]["inference"]
        assert body["meta"]["num_completed"] == 1
        assert body["meta"]["num_aborted"] == body["meta"]["num_skipped"] == 0
        assert body["aborted_requests"] == body["skipped_requests"] == []
        assert body["resolution"]["max_total_tokens"] == 16_000
        assert body["analysis"] is None

    def test_enabled_tiers_use_existing_route_and_echo_metrics(self, client: TestClient) -> None:
        payload = make_payload(num_requests=1)
        payload["workload"]["synthetic"]["prompt_len"] = {"dist": "fixed", "value": 3}
        payload["workload"]["synthetic"]["output_len"] = {"dist": "fixed", "value": 1}
        payload["inference"]["max_output_tokens"] = 1
        payload["scheduler"]["max_running_requests"] = 1
        payload["inference"]["page_size"] = 4
        payload["shared_l3"] = {"available_gb": 8 * 64 / 1_000_000_000}
        payload["inference"]["storage"].update(
            l1_available_gb=(10_464 + 12 * 64) / 1_000_000_000,
            l2_available_gb=4 * 64 / 1_000_000_000,
            l2_to_l1_gb_s=64 / 1_000_000,
            l3_to_l2_gb_s=32 / 1_000_000,
        )
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        result = SimResult.model_validate_json(response.content)
        assert result.meta.num_completed == 1
        assert result.resolution is not None
        assert result.meta.config.shared_l3 is not None
        assert result.resolution.l2_capacity_tokens == 4
        assert result.resolution.l3_capacity_tokens == 8
        assert result.meta.config.shared_l3.available_gb == payload["shared_l3"]["available_gb"]
        assert "mode" not in response.json()["meta"]["config"]["shared_l3"]
        assert (result.metrics.l2_capacity_blocks, result.metrics.l3_capacity_blocks) == (1, 2)
        assert (result.metrics.l2_free_blocks, result.metrics.l3_free_blocks) == (1, 2)

    def test_two_instances_have_independent_capacity_and_request_owners(
        self, client: TestClient
    ) -> None:
        payload = make_payload(num_requests=4)
        payload["num_instances"] = 2
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["config"]["num_instances"] == 2
        assert [item["instance_id"] for item in body["instances"]] == [0, 1]
        assert (
            body["metrics"]["l1_capacity_blocks"]
            == 2 * body["instances"][0]["metrics"]["l1_capacity_blocks"]
        )
        assert (
            sum(item["num_completed"] for item in body["instances"])
            == body["meta"]["num_completed"]
        )
        assert {item["instance_id"] for item in body["requests"]} == {0, 1}
        for metrics in [body["metrics"], *(item["metrics"] for item in body["instances"])]:
            for tier in ("l1", "l2", "l3"):
                expected = (metrics[f"{tier}_hit_tokens"] / metrics["total_input_tokens"]
                            if metrics["total_input_tokens"] else 0)
                assert metrics[f"{tier}_prefix_hit_rate"] == pytest.approx(expected)
            assert sum(metrics[f"{tier}_prefix_hit_rate"] for tier in ("l1", "l2", "l3")) == (
                pytest.approx(metrics["actual_prefix_hit_rate"])
            )

    def test_returns_full_result(self, client: TestClient) -> None:
        response = client.post(URL, json=make_payload())
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        body = response.json()
        assert set(body) == {
            "meta",
            "metrics",
            "instances",
            "requests",
            "aborted_requests",
            "skipped_requests",
            "warnings",
            "resolution",
            "analysis",
        }
        assert (
            "allow_cross_session_shared_prefix"
            not in body["meta"]["config"]["workload"]["synthetic"]["session"]
        )
        assert body["meta"]["config"]["workload"]["arrival_mode"] == "closed_loop"
        assert body["meta"]["config"]["inference"]["max_output_tokens"] == 65536
        assert body["meta"]["num_requests"] == 100
        assert body["meta"]["wall_clock_ms"] >= 0
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
        result = SimResult.model_validate_json(response.content)
        assert result.meta.config == SimConfig.model_validate(make_payload())

    def test_matches_kernel_serialization(self, client: TestClient) -> None:
        """响应字节即本次内核 ``SimResult`` 的 JSON 序列化。"""
        payload = make_payload(seed=3)
        expected = simulate(SimConfig.model_validate(payload))
        with patch("simserver.routes.simulate", return_value=expected):
            response = client.post(URL, json=payload)
        assert response.text == expected.model_dump_json()

    def test_same_seed_bit_identical(self, client: TestClient) -> None:
        payload = make_payload(seed=7)
        first = SimResult.model_validate_json(client.post(URL, json=payload).content)
        second = SimResult.model_validate_json(client.post(URL, json=payload).content)
        exclude = {"meta": {"wall_clock_ms"}}
        assert first.model_dump_json(exclude=exclude) == second.model_dump_json(exclude=exclude)

    def test_multi_turn_prefix_metrics_returned(self, client: TestClient) -> None:
        payload = make_payload(num_requests=2, budget=80)
        payload["workload"]["synthetic"].update(
            {
                "prompt_len": {"dist": "fixed", "value": 16},
                "output_len": {"dist": "fixed", "value": 8},
                "session": {"num_sessions": 1},
                "turn_increment_len": {"dist": "fixed", "value": 1},
            }
        )
        payload["inference"]["page_size"] = 2
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["metrics"]["constructed_ideal_prefix_hit_rate"] == pytest.approx(22 / 41)
        assert body["metrics"]["actual_hit_tokens"] == 22
        assert body["requests"][1]["prompt_len"] == 25
        assert body["requests"][1]["session_id"] == 0
        assert body["requests"][1]["parent_request_id"] == 0
        assert "input_token_ids" not in body["requests"][1]
        assert "input_chunk_ids" not in body["requests"][1]

    def test_request_level_abort_returns_200_and_does_not_block_others(
        self, client: TestClient
    ) -> None:
        payload = make_payload(num_requests=2, budget=192)
        payload["workload"]["synthetic"]["prompt_len"] = {"dist": "fixed", "value": 193}
        payload["workload"]["synthetic"]["output_len"] = {"dist": "fixed", "value": 1}
        payload["inference"]["page_size"] = 64
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
    @pytest.mark.parametrize("branch,legacy,value", [
        ("synthetic", "arrival_mode", "closed_loop"),
        ("synthetic", "seed", 0),
        ("synthetic", "session_rate", 1),
        ("synthetic", "arrival_rate", 1),
        ("synthetic", "max_new_tokens", 128),
        ("trace", "mode", "open_loop"),
        ("trace", "seed", 0),
        ("trace", "session_rate", 1),
        ("trace", "target_qps", 1),
    ])
    def test_old_nested_workload_fields_are_rejected(
        self, client: TestClient, branch: str, legacy: str, value: object,
    ) -> None:
        payload = make_payload(num_requests=1)
        if branch == "trace":
            payload["workload"] = {"mode": "trace", "session_rate": 1, "trace": {}}
        payload["workload"][branch][legacy] = value
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert any(
            error["type"] == "extra_forbidden"
            and error["loc"] == ["body", "workload", branch, legacy]
            for error in response.json()["detail"]
        )

    def test_instance_count_is_required(self, client: TestClient) -> None:
        payload = make_payload(num_requests=1)
        del payload["num_instances"]
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert "num_instances" in response.text

    @pytest.mark.parametrize("field", ["inference", "shared_l3"])
    @pytest.mark.parametrize("url", [URL, "/api/resolve"])
    def test_removed_physical_mode_field_is_rejected(
        self, client: TestClient, field: str, url: str,
    ) -> None:
        payload = make_payload(num_requests=1)
        if field == "shared_l3":
            payload[field] = {"available_gb": 0, "mode": "physical"}
        else:
            payload[field]["mode"] = "physical"
        response = client.post(url, json=payload)
        assert response.status_code == 422
        assert any(
            error["type"] == "extra_forbidden"
            and error["loc"] == ["body", field, "mode"]
            for error in response.json()["detail"]
        )

    @pytest.mark.parametrize("storage,shared,message", [
        ({"l1_available_gb": 0.00001}, None, "不足一个本地 KV 页"),
        ({"l2_available_gb": 0.00000001}, None, "至少容纳一个 KV 页"),
        ({"l2_available_gb": 0, "l2_to_l1_gb_s": None},
         {"available_gb": 0.000000128}, "启用 L3 时必须先启用 L2"),
        ({"l2_available_gb": 0.000000128, "l2_to_l1_gb_s": None},
         None, "L2 → L1 带宽"),
        ({"l2_available_gb": 0.000000128, "l2_to_l1_gb_s": 1},
         {"available_gb": 0.000000128}, "L3 → L2 带宽"),
    ])
    def test_invalid_tiered_configuration_returns_422(
        self, client: TestClient, storage: dict[str, float | None],
        shared: dict[str, object] | None, message: str,
    ) -> None:
        payload = make_payload(num_requests=1)
        payload["inference"]["storage"].update(storage)
        payload["shared_l3"] = shared
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert message in response.text

    @pytest.mark.parametrize("legacy", ["max_batch_size", "kv_token_budget", "prefix_cache"])
    def test_legacy_engine_inputs_are_rejected(self, client: TestClient, legacy: str) -> None:
        payload = make_payload(num_requests=1)
        payload["scheduler"][legacy] = 1
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
        payload["inference"]["page_size"] = 4
        payload["scheduler"]["pd_mutually_exclusive"] = exclusive
        response = client.post(URL, json=payload)
        assert response.status_code == 200
        result = SimResult.model_validate_json(response.content)
        assert result.meta.config.scheduler.pd_mutually_exclusive is exclusive
        assert result.meta.num_completed == 1 and result.meta.num_engine_steps == 2

    def test_removed_hit_target_is_rejected(self, client: TestClient) -> None:
        payload = make_payload(num_requests=2)
        payload["workload"]["synthetic"]["session"] = {
            "num_sessions": 2,
            "target_ideal_prefix_hit_rate": 0.5,
        }
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert "target_ideal_prefix_hit_rate" in response.text

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

    @pytest.mark.parametrize("plan_id", [None, "missing"])
    def test_trace_plan_id_is_required_and_must_exist(
        self, client: TestClient, plan_id: str | None,
    ) -> None:
        payload = make_payload()
        payload["workload"] = {
            "mode": "trace", "arrival_mode": "open_loop", "session_rate": 1, "seed": 0,
            "trace": {"plan_id": plan_id},
        }
        response = client.post(URL, json=payload)
        assert response.status_code == (422 if plan_id is None else 404)
        assert response.json()["detail"]

    def test_physical_preview_matches_run(self, client: TestClient) -> None:
        payload = make_payload(num_requests=1)
        preview = client.post("/api/resolve", json=payload)
        run = client.post(URL, json=payload)
        assert preview.status_code == run.status_code == 200
        assert run.json()["resolution"] == preview.json()

    @pytest.mark.parametrize("field", ["engine", "perf_model", "config"])
    def test_removed_top_level_inputs_rejected(self, client: TestClient, field: str) -> None:
        payload = make_payload()
        payload[field] = {}
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert any(error["loc"] == ["body", field] for error in response.json()["detail"])

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("scheduler", "max_running_requests"), 0),
            (("workload", "session_rate"), -1.0),
            (("workload", "synthetic", "prompt_len"), {"dist": "uniform", "value": 3}),
            (("scheduler", "unknown_field"), 1),
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

    @pytest.mark.parametrize(
        "path,value,expected",
        [
            (("scheduler", "max_running_requests"), 0,
             ["body", "scheduler", "max_running_requests"]),
            (("inference", "timing", "decode_d1"), -1,
             ["body", "inference", "timing", "empirical", "decode_d1"]),
            (("inference", "kv"), {"max_total_tokens": 128}, ["body", "inference", "kv"]),
        ],
    )
    def test_field_validation_paths_follow_unified_config(
        self, client: TestClient, path: tuple[str, ...], value: object, expected: list[str],
    ) -> None:
        payload = make_payload()
        node = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        response = client.post(URL, json=payload)
        assert response.status_code == 422
        assert any(error["loc"] == expected for error in response.json()["detail"])

    def test_synthetic_mode_without_payload_rejected(self, client: TestClient) -> None:
        payload = make_payload()
        payload["workload"] = {"mode": "synthetic"}
        assert client.post(URL, json=payload).status_code == 422

    def test_get_not_allowed(self, client: TestClient) -> None:
        assert client.get(URL).status_code == 405


@pytest.fixture
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
        assert {"404", "422", "501"} <= set(operation["responses"])
        schema = operation["responses"]["501"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/ErrorResponse"}
