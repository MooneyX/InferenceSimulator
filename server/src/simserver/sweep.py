"""扫描维度目录：仅接受显式登记的输入参数，不开放任意对象路径。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from simcore import SimConfig
from simcore.schemas import TracePlan


@dataclass(frozen=True)
class Parameter:
    label: str
    kind: str = "number"
    modes: tuple[str, ...] = (
        "synthetic/open_loop", "synthetic/closed_loop", "trace/open_loop", "trace/closed_loop"
    )


ALL = Parameter.modes
SYNTHETIC = ("synthetic/open_loop", "synthetic/closed_loop")
CLOSED = ("synthetic/closed_loop",)
TRACE = ("trace/open_loop", "trace/closed_loop")

PARAMETERS: dict[str, Parameter] = {
    "workload.synthetic.num_requests": Parameter("请求数量", "integer", SYNTHETIC),
    "workload.synthetic.session.num_sessions": Parameter("会话数量", "integer", CLOSED),
    "workload.target_qps": Parameter("目标 QPS", modes=("synthetic/open_loop", "trace/open_loop")),
    "workload.synthetic.session.think_time_ms": Parameter("思考间隔 (ms)", modes=CLOSED),
    "workload.synthetic.prompt_len": Parameter("平均输入长度", modes=SYNTHETIC),
    "workload.synthetic.turn_increment_len": Parameter("平均新增长度", modes=SYNTHETIC),
    "workload.synthetic.output_len": Parameter("平均输出长度", modes=SYNTHETIC),
    "workload.session_rate": Parameter("会话投放频率", modes=("synthetic/closed_loop", *TRACE)),
    "inference.parallelism": Parameter("每实例 GPU / TP / EP 数", "integer"),
    "inference.hardware.memory_bytes_per_gpu": Parameter("单卡显存 (GB)"),
    "inference.hardware.effective_fp16_tflops": Parameter("有效 FP16/BF16 算力 (TFLOPS)"),
    "inference.hardware.effective_fp8_tflops": Parameter("有效 FP8 算力 (TFLOPS)"),
    "inference.hardware.effective_hbm_gb_s": Parameter("有效 HBM 带宽 (GB/s)"),
    "inference.hardware.effective_interconnect_gb_s": Parameter("有效互联带宽 (GB/s)"),
    "inference.hardware.collective_latency_ms": Parameter("TP 集合通信延迟 (ms)"),
    "inference.hardware.ep_communication_latency_ms": Parameter("EP 通信延迟 (ms)"),
    "inference.hardware.kernel_overhead_ms": Parameter("算子固定开销 (ms)"),
    "inference.storage.l1_available_gb": Parameter("L1 可用容量 (GB)"),
    "inference.storage.l2_available_gb": Parameter("L2 可用容量 (GB)"),
    "inference.storage.runtime_reserve_bytes": Parameter("运行时预留 (GB)"),
    "inference.storage.l2_to_l1_gb_s": Parameter("L2 → L1 带宽 (GB/s)"),
    "inference.storage.l3_to_l2_gb_s": Parameter("L3 → L2 带宽 (GB/s)"),
    "inference.storage.l2_latency_ms": Parameter("L2 固定延迟 (ms)"),
    "inference.storage.l3_latency_ms": Parameter("L3 固定延迟 (ms)"),
    "shared_l3.available_gb": Parameter("共享 L3 总容量 (TB)"),
    "shared_l3.lookup_latency_ms": Parameter("L3 查找延迟 (ms)"),
    "num_instances": Parameter("推理实例数量", "integer"),
    "scheduler.max_running_requests": Parameter("最多同时运行请求", "integer"),
    "inference.max_output_tokens": Parameter("引擎输出上限 (token)", "integer"),
    "scheduler.max_prefill_tokens": Parameter("单次 Prefill 额度 (token)", "integer"),
    "scheduler.l3_prefetch_min_tokens": Parameter("L3 预取最小长度 (token)", "integer"),
    "inference.page_size": Parameter("每页 token 数", "integer"),
    "scheduler.pd_mutually_exclusive": Parameter("Prefill / Decode 分步执行", "boolean"),
}


class ScanAxis(BaseModel):
    key: str
    values: list[Any] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def check_values(cls, values: list[Any]) -> list[Any]:
        import math

        for value in values:
            if value is not None and (type(value) not in (int, float, bool) or
                                      (type(value) is float and not math.isfinite(value))):
                raise ValueError("扫描值必须是有限数字、布尔值或 null")
        return values


class ScanRequest(BaseModel):
    base: SimConfig
    axes: list[ScanAxis] = Field(min_length=1)

    @model_validator(mode="after")
    def check_axes(self) -> ScanRequest:
        selected = [axis.key for axis in self.axes]
        if len(selected) != len(set(selected)):
            raise ValueError("各维度不能选择同一参数")
        mode = f"{self.base.workload.mode}/{self.base.workload.arrival_mode}"
        for axis in self.axes:
            parameter = PARAMETERS.get(axis.key)
            if parameter is None or mode not in parameter.modes:
                raise ValueError(f"当前负载模式不支持参数 {axis.key}")
            for value in axis.values:
                if value is None:
                    if axis.key not in NULLABLE:
                        raise ValueError(f"{axis.key} 不接受 null")
                elif parameter.kind == "boolean":
                    if type(value) is not bool:
                        raise ValueError(f"{axis.key} 需要布尔值")
                elif type(value) is bool or (parameter.kind == "integer" and
                                             (type(value) is float or type(value) is not int)):
                    raise ValueError(f"{axis.key} 需要{parameter.kind}数值")
        return self


NULLABLE = {
    "inference.hardware.effective_fp8_tflops", "inference.hardware.effective_interconnect_gb_s",
    "inference.hardware.collective_latency_ms", "inference.hardware.ep_communication_latency_ms",
    "inference.storage.l2_to_l1_gb_s", "inference.storage.l3_to_l2_gb_s",
}


def build_point(base: SimConfig, values: dict[str, Any], plan: TracePlan | None) -> SimConfig:
    data = base.model_dump(mode="python")
    for key, value in values.items():
        if key == "inference.parallelism":
            data["inference"]["hardware"]["gpu_count"] = value
            data["inference"]["tp_size"] = value
            data["inference"]["ep_size"] = value if base.inference.model.moe_layers else 1
            continue
        if key.startswith("workload.synthetic.") and key.endswith("_len"):
            length = data["workload"]["synthetic"]
            for part in key.split(".")[2:]:
                length = length[part]
            length["value" if length["dist"] == "fixed" else "mean"] = value
            continue
        parts = key.split(".")
        target = data
        for part in parts[:-1]:
            if target.get(part) is None and part == "shared_l3":
                target[part] = {"available_gb": 0, "lookup_latency_ms": 0}
            target = target[part]
        if key == "shared_l3.available_gb":
            value *= 1000
        if key in (
            "inference.hardware.memory_bytes_per_gpu", "inference.storage.runtime_reserve_bytes"
        ):
            value *= 1_000_000_000
            if not float(value).is_integer():
                raise ValueError(f"{key} 换算后必须为整数字节")
            value = int(value)
        target[parts[-1]] = value
    config = SimConfig.model_validate(data)
    if plan is not None and config.inference.page_size != plan.block_size:
        raise ValueError("Trace 计划块大小必须与每页 token 数一致")
    return config
