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
    communication_ms: float = 0.0
    communication_bytes: float = 0.0
    collectives: int = 0


def step_cost(
    workload: StepWorkload, model: ModelSpec, hardware: HardwareSpec, tp_size: int = 1
) -> StepCost:
    """逐层顺序执行投影、因果注意力和 FFN；TP 按本地工作量和通信计时。"""
    if not workload.prefill and not workload.decode:
        return StepCost(0.0, 0.0, 0.0, 0.0, 0.0)
    if tp_size > 1:
        return _tp_step_cost(workload, model, hardware, tp_size)
    h = model.hidden_size
    q = model.num_attention_heads * model.head_dim
    kv = model.num_kv_heads * model.head_dim
    ffn = model.ffn_intermediate_size
    tokens = sum(item.new_tokens for item in workload.prefill) + len(workload.decode)
    attention_pairs = sum(
        item.new_tokens * item.reused_tokens + item.new_tokens * (item.new_tokens + 1) / 2
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


def _tp_step_cost(
    workload: StepWorkload, model: ModelSpec, hardware: HardwareSpec, tp_size: int
) -> StepCost:
    """单实例同构 TP：各 rank 执行相同的局部工作，逐层两次等效归约。"""
    h, d, f = model.hidden_size, model.head_dim, model.ffn_intermediate_size
    q_heads = model.num_attention_heads // tp_size
    kv_heads = max(1, model.num_kv_heads // tp_size)
    tokens = sum(item.new_tokens for item in workload.prefill) + len(workload.decode)
    pairs = sum(
        item.new_tokens * item.reused_tokens + item.new_tokens * (item.new_tokens + 1) / 2
        for item in workload.prefill
    ) + sum(item.context_after_decode for item in workload.decode)
    kv_reads = sum(item.context_after for item in workload.prefill) + sum(
        item.context_after_decode for item in workload.decode
    )
    # Q/K/V 与输出投影在不同的同步边界，不能合成一次 Roofline 阶段。
    qkv_weights = h * (q_heads + 2 * kv_heads) * d
    output_weights = h * q_heads * d
    ffn_weights = 3 * h * (f // tp_size)
    stages = (
        (2 * tokens * qkv_weights, 2 * qkv_weights),
        (4 * q_heads * d * pairs, 4 * kv_heads * d * (kv_reads + tokens)),
        (2 * tokens * output_weights, 2 * output_weights),
        (2 * tokens * ffn_weights, 2 * ffn_weights),
    )
    flops_per_ms = hardware.effective_fp16_tflops * 1e9
    bytes_per_ms = hardware.effective_hbm_gb_s * 1e6
    layer_compute = sum(flops / flops_per_ms for flops, _ in stages)
    layer_memory = sum(size / bytes_per_ms for _, size in stages)
    layer_device = sum(
        hardware.kernel_overhead_ms + max(flops / flops_per_ms, size / bytes_per_ms)
        for flops, size in stages
    )
    # S 是每 rank 参与归约的完整隐藏状态；按 ring 的每 rank 有效链路流量计。
    message_bytes = tokens * h * 2
    bytes_per_collective = 2 * (tp_size - 1) / tp_size * message_bytes
    if hardware.effective_interconnect_gb_s is None or hardware.collective_latency_ms is None:
        raise ValueError("TP 计时需要单 rank 互联带宽和集合通信固定延迟")
    link_bytes_per_ms = hardware.effective_interconnect_gb_s * 1e6
    communication_per_collective = (
        hardware.collective_latency_ms + bytes_per_collective / link_bytes_per_ms
    )
    collectives = 2 * model.num_layers
    communication = collectives * communication_per_collective
    return StepCost(
        duration_ms=model.num_layers * layer_device + communication,
        flops=model.num_layers * sum(flops for flops, _ in stages),
        hbm_bytes=model.num_layers * sum(size for _, size in stages),
        compute_ms=model.num_layers * layer_compute,
        memory_ms=model.num_layers * layer_memory,
        communication_ms=communication,
        communication_bytes=collectives * bytes_per_collective,
        collectives=collectives,
    )
