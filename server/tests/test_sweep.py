from itertools import permutations

import pytest
from pydantic import ValidationError

from simcore import SimConfig
from simserver.presets import HARDWARE, HARDWARE_PRESET_FIELDS, MODELS
from simserver.sweep import PRESETS, ScanRequest, build_point
from test_routes import make_payload


def test_presets_apply_before_scalars_in_every_axis_order() -> None:
    base = SimConfig.model_validate(make_payload(num_requests=1))
    original = base.model_dump()
    entries = [
        ("inference.model", "tencent/Hy4-preview"),
        ("inference.hardware", {"name": "Huawei Ascend 950PR", "parallelism": 4}),
        ("inference.hardware.memory_bytes_per_gpu", 160),
        ("inference.hardware.kernel_overhead_ms", 0.123),
    ]
    expected = None
    for order in permutations(entries):
        point = build_point(base, dict(order), None)
        if expected is None:
            expected = point
        assert point == expected
        assert point.inference.hardware.memory_bytes_per_gpu == 160_000_000_000
        assert point.inference.hardware.kernel_overhead_ms == 0.123
        assert point.inference.tp_size == point.inference.ep_size == point.inference.hardware.gpu_count == 4
        assert point.inference.model == next(model for model in MODELS if model.name == "tencent/Hy4-preview")
    assert base.model_dump() == original


@pytest.mark.parametrize("preset", MODELS, ids=lambda preset: preset.name)
def test_model_preset_replaces_all_model_fields_without_changing_deployment(preset) -> None:
    data = make_payload(num_requests=1)
    data["inference"]["model"] = MODELS[2].model_dump(mode="json")
    data["inference"]["model"]["weight_dtype"] = "fp16"
    base = SimConfig.model_validate(data)
    point = build_point(base, {"inference.model": preset.name}, None)
    assert point.inference.model == preset
    assert point.model_dump(exclude={"inference": {"model"}}) == base.model_dump(exclude={"inference": {"model"}})
    assert point.inference.model is not preset
    assert point.inference.model.attention_groups is not PRESETS["inference.model"][preset.name]["attention_groups"]


@pytest.mark.parametrize("preset", HARDWARE, ids=lambda preset: preset["name"])
def test_hardware_preset_scope_and_all_field_overrides(preset) -> None:
    base = SimConfig.model_validate(make_payload(num_requests=1))
    point = build_point(base, {"inference.hardware": {"name": preset["name"], "parallelism": 1}}, None)
    hardware = point.inference.hardware.model_dump()
    assert {key: hardware[key] for key in HARDWARE_PRESET_FIELDS} == {
        key: preset[key] for key in HARDWARE_PRESET_FIELDS
    }
    assert point.inference.hardware.gpu_count == base.inference.hardware.gpu_count
    assert point.model_dump(exclude={"inference": {"hardware"}}) == base.model_dump(exclude={"inference": {"hardware"}})
    overrides = {
        "inference.hardware.memory_bytes_per_gpu": 192,
        "inference.hardware.effective_fp16_tflops": 123,
        "inference.hardware.effective_fp8_tflops": None,
        "inference.hardware.effective_hbm_gb_s": 234,
        "inference.hardware.effective_interconnect_gb_s": None,
        "inference.hardware.collective_latency_ms": 0,
        "inference.hardware.ep_communication_latency_ms": None,
        "inference.hardware.kernel_overhead_ms": 0.5,
        "inference.hardware": {"name": preset["name"], "parallelism": 1},
    }
    request = ScanRequest(base=base, axes=[{"key": key, "values": [value]} for key, value in overrides.items()])
    result = build_point(request.base, overrides, None).inference.hardware.model_dump()
    for key, value in overrides.items():
        if key != "inference.hardware":
            assert result[key.split(".")[-1]] == (192_000_000_000 if key.endswith("memory_bytes_per_gpu") else value)


@pytest.mark.parametrize("bound", [True, False])
@pytest.mark.parametrize(("name", "ep"), [(MODELS[0].name, 1), (MODELS[2].name, 8)])
def test_parallelism_uses_selected_model(name: str, ep: int, bound: bool) -> None:
    data = make_payload(num_requests=1)
    data["inference"]["model"] = MODELS[2].model_dump(mode="json")
    base = SimConfig.model_validate(data)
    axis = {"inference.hardware": {"name": HARDWARE[0]["name"], "parallelism": 8}} if bound else {"inference.parallelism": 8}
    point = build_point(base, {**axis, "inference.model": name}, None)
    assert point.inference.ep_size == ep
    assert point.inference.tp_size == point.inference.hardware.gpu_count == 8


@pytest.mark.parametrize(("key", "values"), [
    ("inference.model", ["unknown"]), ("inference.hardware", [None]),
    ("inference.model", [1]), ("inference.hardware", [True]),
    ("inference.model", [["Qwen/Qwen3-8B"]]),
    ("inference.hardware", []), ("num_instances", ["2"]),
    ("inference.hardware.effective_hbm_gb_s", ["1000"]),
    ("inference.hardware.effective_hbm_gb_s", [float("inf")]),
    ("scheduler.pd_mutually_exclusive", ["true"]),
    ("num_instances", [{"name": "NVIDIA B300", "parallelism": 8}]),
    ("inference.hardware", ["NVIDIA B300"]),
    ("inference.hardware", [{"name": "unknown", "parallelism": 8}]),
    ("inference.hardware", [{"name": "NVIDIA B300"}]),
    ("inference.hardware", [{"name": "NVIDIA B300", "parallelism": 8, "gpu_count": 16}]),
    *[("inference.hardware", [{"name": "NVIDIA B300", "parallelism": value}])
      for value in [0, -1, 1.5, True, "8", None]],
    ("inference.hardware", [{"name": "NVIDIA B300", "parallelism": n} for n in [8, 16]]),
])
def test_rejects_unknown_presets_and_wrong_axis_types(key: str, values: list) -> None:
    with pytest.raises(ValidationError):
        ScanRequest(base=make_payload(num_requests=1), axes=[{"key": key, "values": values}])


@pytest.mark.parametrize("reverse", [True, False])
def test_hardware_and_independent_parallelism_are_mutually_exclusive(reverse: bool) -> None:
    axes = [
        {"key": "inference.hardware", "values": [{"name": "NVIDIA B300", "parallelism": 8}]},
        {"key": "inference.parallelism", "values": [4, 16]},
    ]
    with pytest.raises(ValidationError, match="不能同时扫描独立并行度"):
        ScanRequest(base=make_payload(num_requests=1), axes=axes[::-1] if reverse else axes)
