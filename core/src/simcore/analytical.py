"""Dense TP 与平均负载 MoE/EP 的逐步解析计时（未经校准）。"""

from dataclasses import dataclass, replace

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
    tp_communication_ms: float = 0.0
    tp_communication_bytes: float = 0.0
    ep_dispatch_ms: float = 0.0
    ep_combine_ms: float = 0.0
    ep_dispatch_bytes: float = 0.0
    ep_combine_bytes: float = 0.0
    ep_exchanges: int = 0


def step_cost(
    workload: StepWorkload,
    model: ModelSpec,
    hardware: HardwareSpec,
    tp_size: int = 1,
    ep_size: int = 1,
) -> StepCost:  # 单步负载、模型、硬件及 TP/EP 并行度 → 分项耗时与资源成本
    """按真实单步工作量计算各层依赖；MoE 替换 Dense FFN。"""
    if not workload.prefill and not workload.decode:
        return StepCost(0.0, 0.0, 0.0, 0.0, 0.0)
    if model.attention_layout == "hy4_preview":
        return _hy4_step_cost(workload, model, hardware, tp_size, ep_size)
    if model.attention_groups is not None:
        return _group_step_cost(workload, model, hardware, tp_size, ep_size)
    if model.moe_layers:
        return _moe_step_cost(workload, model, hardware, tp_size, ep_size)
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


def _group_step_cost(
    workload: StepWorkload, model: ModelSpec, hardware: HardwareSpec, tp_size: int, ep_size: int
) -> StepCost:
    """按组累加注意力阶段 Roofline，FFN 与 TP/EP 仍用原有模型。"""
    base = step_cost(
        workload,
        model.model_copy(update={"attention_layout": "full_gqa", "attention_groups": None}),
        hardware, tp_size, ep_size
    )
    tokens = sum(item.new_tokens for item in workload.prefill) + len(workload.decode)
    pairs = sum(
        item.new_tokens * item.reused_tokens + item.new_tokens * (item.new_tokens + 1) / 2
        for item in workload.prefill
    ) + sum(item.context_after_decode for item in workload.decode)
    kv_reads = sum(item.context_after for item in workload.prefill) + sum(
        item.context_after_decode for item in workload.decode
    )
    h, d = model.hidden_size, model.head_dim
    q_heads = model.num_attention_heads // tp_size
    kv_heads = max(1, model.num_kv_heads // tp_size)
    single_dense = not model.moe_layers and tp_size == 1
    flops_per_ms = hardware.effective_fp16_tflops * 1e9
    bytes_per_ms = hardware.effective_hbm_gb_s * 1e6

    def stage(flops: float, nbytes: float, layers: int) -> tuple[float, float, float, float, float]:
        compute, memory = flops / flops_per_ms, nbytes / bytes_per_ms
        return (layers * (hardware.kernel_overhead_ms + max(compute, memory)),
                layers * flops, layers * nbytes, layers * compute, layers * memory)

    def gqa_stages(heads: int, kv: int, dim: int, layers: int) -> list[tuple[float, ...]]:
        qkv = h * (heads + 2 * kv) * dim
        out = h * heads * dim
        if single_dense:
            projections = [stage(2 * tokens * (qkv + out), 2 * (qkv + out), layers)]
        else:
            projections = [stage(2 * tokens * qkv, 2 * qkv, layers),
                           stage(2 * tokens * out, 2 * out, layers)]
        return [
            *projections,
            stage(4 * heads * dim * pairs, 4 * kv * dim * (kv_reads + tokens), layers),
        ]

    old = gqa_stages(q_heads, kv_heads, d, model.num_layers)
    current = []
    for group in model.attention_groups:
        heads, layers = group.num_attention_heads // tp_size, group.layers
        if group.kind == "gqa":
            current.extend(gqa_stages(
                heads, max(1, group.num_kv_heads // tp_size), group.head_dim, layers
            ))
            continue
        q_rank, kv_rank, rope, nope, value = (
            group.q_rank, group.kv_rank, group.rope_dim, group.nope_dim, group.value_dim
        )
        a = h * (q_rank + kv_rank + rope)
        b = q_rank * heads * (nope + rope) + kv_rank * heads * (nope + value)
        out = h * heads * value
        current.append(stage(2 * tokens * (a + b), 2 * (a + b), layers))
        if group.kind == "mla_dsa":
            uncapped_pairs = 0
            for item in workload.prefill:
                uncapped = min(item.new_tokens, max(0, group.topk - item.reused_tokens))
                uncapped_pairs += (
                    uncapped * item.reused_tokens + uncapped * (uncapped + 1) / 2
                    + (item.new_tokens - uncapped) * group.topk
                )
            uncapped_pairs += sum(
                min(group.topk, item.context_after_decode) for item in workload.decode
            )
            index_weights = group.index_heads * group.index_dim * q_rank + h * (
                group.index_dim + group.index_heads
            )
            index_bytes = group.index_dim + group.index_dim // 128 * 4
            current.append(stage(
                2 * tokens * index_weights + 2 * group.index_heads * group.index_dim * pairs,
                2 * index_weights + index_bytes * (kv_reads + tokens), group.full_indexer_layers,
            ))
            attention_pairs = uncapped_pairs
        else:
            attention_pairs = pairs
        current.append(stage(
            2 * heads * (nope + rope + value) * attention_pairs,
            (kv_rank + rope) * 2 * (attention_pairs + tokens), layers,
        ))
        current.append(stage(2 * tokens * out, 2 * out, layers))
    delta = [sum(row[i] for row in current) - sum(row[i] for row in old) for i in range(5)]
    return replace(
        base, duration_ms=base.duration_ms + delta[0], flops=base.flops + delta[1],
        hbm_bytes=base.hbm_bytes + delta[2], compute_ms=base.compute_ms + delta[3],
        memory_ms=base.memory_ms + delta[4],
    )


def _hy4_step_cost(
    workload: StepWorkload, model: ModelSpec, hardware: HardwareSpec, tp_size: int, ep_size: int
) -> StepCost:
    """HY4 MLA/DSA 未校准粗估：替换 GQA 注意力三阶段，保留 MoE/TP/EP 基线。"""
    from dataclasses import replace

    base = _moe_step_cost(workload, model, hardware, tp_size, ep_size)
    h, heads, d = model.hidden_size, model.num_attention_heads // tp_size, model.head_dim
    kv_heads = max(1, model.num_kv_heads // tp_size)
    tokens = sum(item.new_tokens for item in workload.prefill) + len(workload.decode)
    full_pairs = sum(
        item.new_tokens * item.reused_tokens + item.new_tokens * (item.new_tokens + 1) / 2
        for item in workload.prefill
    ) + sum(item.context_after_decode for item in workload.decode)
    kv_reads = sum(item.context_after for item in workload.prefill) + sum(
        item.context_after_decode for item in workload.decode
    )
    topk = 2048
    sparse_pairs = 0
    for item in workload.prefill:
        # 第 i 个新 token 最多访问 min(topk, reused+i) 个位置。
        uncapped = min(item.new_tokens, max(0, topk - item.reused_tokens))
        sparse_pairs += (
            uncapped * item.reused_tokens + uncapped * (uncapped + 1) / 2
            + (item.new_tokens - uncapped) * topk
        )
    sparse_pairs += sum(min(topk, item.context_after_decode) for item in workload.decode)
    full_layers = sum(i == 0 or (i - 1) % 4 == 0 for i in range(model.num_layers))
    flops_per_ms = hardware.effective_fp16_tflops * 1e9
    bytes_per_ms = hardware.effective_hbm_gb_s * 1e6

    def stage(
        flops: float, nbytes: float, repeats: int
    ) -> tuple[float, float, float, float, float]:
        compute, memory = flops / flops_per_ms, nbytes / bytes_per_ms
        return (
            repeats * (hardware.kernel_overhead_ms + max(compute, memory)),
            repeats * flops, repeats * nbytes, repeats * compute, repeats * memory,
        )

    old_qkv = h * (heads + 2 * kv_heads) * d
    old_out = h * heads * d
    old_stages = (
        stage(2 * tokens * old_qkv, 2 * old_qkv, model.num_layers),
        stage(4 * heads * d * full_pairs, 4 * kv_heads * d * (kv_reads + tokens), model.num_layers),
        stage(2 * tokens * old_out, 2 * old_out, model.num_layers),
    )
    # A 投影和 full indexer 在 TP rank 上复制；B、输出与门控按头分片。
    a_weights = h * (2048 + 512 + 64)
    q_b = 2048 * heads * 256
    kv_b = 512 * heads * (192 + 256)
    gate = h * heads * 256
    out = h * heads * 256
    index_weights = 32 * 128 * 2048 + h * (128 + 32)
    new_stages = (
        stage(2 * tokens * (a_weights + q_b + kv_b),
              2 * (a_weights + q_b + kv_b), model.num_layers),
        stage(2 * tokens * index_weights + 2 * 32 * 128 * full_pairs,
              2 * index_weights + 132 * (kv_reads + tokens), full_layers),
        stage(4 * heads * 256 * sparse_pairs,
              (512 + 64) * 2 * (sparse_pairs + tokens) + 4 * sparse_pairs,
              model.num_layers),
        stage(2 * tokens * (gate + out), 2 * (gate + out), model.num_layers),
        # SGLang 的独立共享专家在该 EP 放置口径下每 rank 常驻并参与每个 MoE 层。
        stage(2 * tokens * 3 * h * model.expert_intermediate_size,
              2 * 3 * h * model.expert_intermediate_size, model.moe_layers),
    )
    delta = [
        sum(row[i] for row in new_stages) - sum(row[i] for row in old_stages)
        for i in range(5)
    ]
    return replace(
        base,
        duration_ms=base.duration_ms + delta[0],
        flops=base.flops + delta[1],
        hbm_bytes=base.hbm_bytes + delta[2],
        compute_ms=base.compute_ms + delta[3],
        memory_ms=base.memory_ms + delta[4],
    )


def _tp_step_cost(
    workload: StepWorkload, model: ModelSpec, hardware: HardwareSpec, tp_size: int
) -> StepCost:  # 单步负载、模型、硬件与 TP 规模 → 各 rank 及通信成本
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
    # Q/K/V 与输出投影中间隔着注意力计算，须分别计入 Roofline 阶段。
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
        tp_communication_ms=communication,
        tp_communication_bytes=collectives * bytes_per_collective,
    )


def _moe_step_cost(
    workload: StepWorkload,
    model: ModelSpec,
    hardware: HardwareSpec,
    tp_size: int,
    ep_size: int,
) -> StepCost:  # 单步负载、模型、硬件、TP/EP 规模 → 含专家通信的分项成本
    """无重叠基线：各 rank 平均路由，逐层等待通信与专家计算。"""
    h, d = model.hidden_size, model.head_dim
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
    qkv_weights = h * (q_heads + 2 * kv_heads) * d
    output_weights = h * q_heads * d
    dense_layers = model.num_layers - model.moe_layers
    flops_per_ms = hardware.effective_fp16_tflops * 1e9
    bytes_per_ms = hardware.effective_hbm_gb_s * 1e6
    flops = hbm = compute = memory = device = 0.0

    def stage(work_flops: float, nbytes: float, repeats: int = 1) -> None:  # 单阶段 FLOPs、字节与重复层数 → 累加计算/显存成本；无返回值
        nonlocal flops, hbm, compute, memory, device
        flops += repeats * work_flops
        hbm += repeats * nbytes
        compute += repeats * work_flops / flops_per_ms
        memory += repeats * nbytes / bytes_per_ms
        device += repeats * (
            hardware.kernel_overhead_ms + max(work_flops / flops_per_ms, nbytes / bytes_per_ms)
        )

    stage(
        2 * tokens * qkv_weights, 2 * qkv_weights, model.num_layers
    )  # Q/K/V 投影：记录各层本步 token 的投影计算与权重读取
    stage(
        4 * q_heads * d * pairs, 4 * kv_heads * d * (kv_reads + tokens), model.num_layers
    )  # 注意力：按配对数记录计算，按 KV 规模记录显存读写
    stage(
        2 * tokens * output_weights, 2 * output_weights, model.num_layers
    )  # 输出投影：记录各层注意力输出的投影计算与权重读取
    if dense_layers:
        ffn_weights = 3 * h * (model.ffn_intermediate_size // tp_size)
        stage(
            2 * tokens * ffn_weights, 2 * ffn_weights, dense_layers
        )  # 普通前馈网络（Dense FFN）：只记录普通层的计算与权重读取

    # Gate 在每个 rank 对本步所有 token 的全部专家打分；专家参数只读本地份额。
    gate_weights = h * model.num_experts
    stage(
        2 * tokens * gate_weights, 2 * gate_weights, model.moe_layers
    )  # MoE 门控（Gate）：记录 MoE 层的专家打分与权重读取
    local_expert_weights = model.num_experts // ep_size * 3 * h * model.expert_intermediate_size
    stage(
        2 * tokens * model.experts_per_token / ep_size * 3 * h * model.expert_intermediate_size,
        2 * local_expert_weights,
        model.moe_layers,
    )  # MoE 专家计算：记录 MoE 层选中专家的计算与本地权重读取

    tp_ms = tp_bytes = 0.0
    collectives = 0
    ep_dispatch_ms = ep_combine_ms = ep_dispatch_bytes = ep_combine_bytes = 0.0
    ep_exchanges = 0
    if tp_size > 1:
        if hardware.effective_interconnect_gb_s is None or hardware.collective_latency_ms is None:
            raise ValueError("TP 计时需要单 rank 互联带宽和集合通信固定延迟")
        message_bytes = tokens * h * 2
        bytes_per_collective = 2 * (tp_size - 1) / tp_size * message_bytes
        collectives = model.num_layers + dense_layers
        tp_bytes = collectives * bytes_per_collective
        tp_ms = collectives * (
            hardware.collective_latency_ms
            + bytes_per_collective / (hardware.effective_interconnect_gb_s * 1e6)
        )
    if ep_size > 1:
        if (
            hardware.effective_interconnect_gb_s is None
            or hardware.ep_communication_latency_ms is None
        ):
            raise ValueError("EP 计时需要单 rank 互联带宽和 EP 单次通信固定开销")
        # 不生成具体路由：每 token 在无放回 Top-K 中命中某一目标 rank 的概率。
        experts_per_rank = model.num_experts // ep_size  # 每个 EP rank 驻留的专家数
        missed = 1.0  # 无放回选择均未命中某个远端 rank 的概率
        for selected in range(model.experts_per_token):
            missed *= (model.num_experts - experts_per_rank - selected) / (
                model.num_experts - selected
            )
        remote_selections = tokens * model.experts_per_token * (ep_size - 1) / (ep_size * ep_size)
        if workload.prefill:
            remote_payloads = tokens * (ep_size - 1) / ep_size * (1 - missed)
        else:
            # 纯 Decode 按远端专家选择次数估计逻辑 payload。
            remote_payloads = remote_selections
        # BF16/FP16 向量与计算精度一致；每次远端选择另计 4B ID + 4B 权重/索引。
        bytes_per_exchange = remote_payloads * h * 2 + remote_selections * 8
        ep_dispatch_bytes = ep_combine_bytes = bytes_per_exchange * model.moe_layers
        ep_exchanges = 2 * model.moe_layers
        one_exchange = hardware.ep_communication_latency_ms + bytes_per_exchange / (
            hardware.effective_interconnect_gb_s * 1e6
        )
        ep_dispatch_ms = ep_combine_ms = model.moe_layers * one_exchange
    communication = tp_ms + ep_dispatch_ms + ep_combine_ms
    return StepCost(
        duration_ms=device + communication,
        flops=flops,
        hbm_bytes=hbm,
        compute_ms=compute,
        memory_ms=memory,
        communication_ms=communication,
        communication_bytes=tp_bytes + ep_dispatch_bytes + ep_combine_bytes,
        collectives=collectives,
        tp_communication_ms=tp_ms,
        tp_communication_bytes=tp_bytes,
        ep_dispatch_ms=ep_dispatch_ms,
        ep_combine_ms=ep_combine_ms,
        ep_dispatch_bytes=ep_dispatch_bytes,
        ep_combine_bytes=ep_combine_bytes,
        ep_exchanges=ep_exchanges,
    )
