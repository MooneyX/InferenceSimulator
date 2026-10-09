"""单 GPU Dense/GQA 的逐步 Roofline 解析计时（未经校准）。"""

from dataclasses import dataclass

from simcore.perf_model import StepWorkload
from simcore.schemas import HardwareSpec, ModelSpec


@dataclass(frozen=True, slots=True)
class StepCost:
    duration_ms: float
    flops: float
    hbm_bytes: float
    compute_ms: float
    memory_ms: float


def step_cost(workload: StepWorkload, model: ModelSpec, hardware: HardwareSpec) -> StepCost:
    """逐层顺序执行投影、因果注意力和 FFN；层内访存/计算可重叠。"""
    if not workload.prefill and not workload.decode:
        return StepCost(0.0, 0.0, 0.0, 0.0, 0.0)
    h = model.hidden_size
    q = model.num_attention_heads * model.head_dim
    kv = model.num_kv_heads * model.head_dim
    ffn = model.ffn_intermediate_size
    tokens = sum(item.new_tokens for item in workload.prefill) + len(workload.decode)
    attention_pairs = sum(
        item.new_tokens * item.reused_tokens
        + item.new_tokens * (item.new_tokens + 1) / 2
        for item in workload.prefill
    ) + sum(item.context_after_decode for item in workload.decode)
    projection_weights = h * (2 * q + 2 * kv)
    ffn_weights = 3 * h * ffn
    projection_flops = 2 * tokens * projection_weights
    attention_flops = 4 * q * attention_pairs
    ffn_flops = 2 * tokens * ffn_weights
    weight_bytes = 2  # 第一阶段 BF16/FP16 权重
    kv_bytes = 2  # 第一阶段 BF16/FP16 KV
    projection_hbm = projection_weights * weight_bytes
    kv_reads = sum(item.context_after for item in workload.prefill) + sum(
        item.context_after_decode for item in workload.decode
    )
    attention_hbm = 2 * kv * kv_bytes * (kv_reads + tokens)
    ffn_hbm = ffn_weights * weight_bytes
    effective_flops = hardware.effective_fp16_tflops * 1e12 / 1000  # FLOPs/ms
    effective_bandwidth = hardware.effective_hbm_gb_s * 1e6  # bytes/ms
    layer_device = 0.0
    layer_compute = 0.0
    layer_memory = 0.0
    for flops, nbytes in (
        (projection_flops, projection_hbm),
        (attention_flops, attention_hbm),
        (ffn_flops, ffn_hbm),
    ):
        compute = flops / effective_flops
        memory = nbytes / effective_bandwidth
        layer_compute += compute
        layer_memory += memory
        layer_device += hardware.kernel_overhead_ms + max(compute, memory)
    return StepCost(
        layer_device * model.num_layers,
        (projection_flops + attention_flops + ffn_flops) * model.num_layers,
        (projection_hbm + attention_hbm + ffn_hbm) * model.num_layers,
        layer_compute * model.num_layers,
        layer_memory * model.num_layers,
    )
