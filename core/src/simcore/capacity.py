"""从全注意力规格解析单卡或 TP/EP 组的固定大小 KV 页容量。"""

from decimal import ROUND_FLOOR, Decimal
from typing import cast

from simcore.runtime import (
    AnalyticalSpec,
    RuntimeConfig,
    TransferLink,
    TransferLinks,
)
from simcore.schemas import (
    AttentionGroup,
    CapacityReport,
    EmpiricalTimingConfig,
    EngineConfig,
    ModelSpec,
    PerfModelConfig,
    RankCapacityReport,
    ResolutionReport,
    SimConfig,
)

_GB = Decimal(1_000_000_000)


def _bytes_from_gb(value: float) -> int:  # 十进制 GB 数值 → 向下取整的字节数
    return int((Decimal(str(value)) * _GB).to_integral_value(rounding=ROUND_FLOOR))


def _dtype_bytes(dtype: str | None) -> int:
    return 4 if dtype == "fp32" else 2


def _norm_parameters(kind: str, width: int) -> int:
    return width * {"none": 0, "rms": 1, "layer": 2}[kind]


def _group_attention_budget(
    group: AttentionGroup,
    hidden_size: int,
    tp_size: int,
    hierarchical: bool,
    weight_dtype: str = "bf16",
) -> tuple[int, int, int, int]:
    """返回每 rank KV、分片/复制权重字节及全组逻辑参数数。"""
    heads, layers = group.num_attention_heads, group.layers
    if heads % tp_size:
        raise ValueError("注意力组的 Query 头数须能按 TP 均分")
    local_heads = heads // tp_size
    weight_bytes = _dtype_bytes(weight_dtype)
    norm_bytes = _dtype_bytes(group.norm_dtype or weight_dtype)
    if group.kind == "gqa":
        kv_heads, dim = cast(int, group.num_kv_heads), cast(int, group.head_dim)
        if (kv_heads >= tp_size and kv_heads % tp_size) or (
            kv_heads < tp_size and tp_size % kv_heads
        ):
            raise ValueError("注意力组的 KV 头数与 TP 不满足均分或复制条件")
        local_kv = max(1, kv_heads // tp_size)
        local_params = layers * hidden_size * (2 * local_heads + 2 * local_kv) * dim
        logical = layers * hidden_size * (2 * heads + 2 * kv_heads) * dim
        norm_params = layers * 2 * dim if group.qk_norm else 0
        kv_bytes = layers * 2 * local_kv * dim * 2
        replicated_bytes = norm_params * norm_bytes
    else:
        q_rank, kv_rank, rope, nope, value = (
            cast(int, value)
            for value in (
                group.q_rank,
                group.kv_rank,
                group.rope_dim,
                group.nope_dim,
                group.value_dim,
            )
        )
        local_params = layers * (
            q_rank * local_heads * (nope + rope)
            + kv_rank * local_heads * (nope + value)
            + hidden_size * local_heads * value * (2 if group.output_gate else 1)
        )
        a_params = layers * hidden_size * (q_rank + kv_rank + rope)
        norm_params = layers * (q_rank + kv_rank) if group.qk_norm else 0
        logical = local_params * tp_size + a_params
        replicated_bytes = a_params * weight_bytes + norm_params * norm_bytes
        kv_bytes = layers * (kv_rank + rope) * 2
        if group.kind == "mla_dsa":
            full = cast(int, group.full_indexer_layers)
            index_dim, index_heads = cast(int, group.index_dim), cast(int, group.index_heads)
            index_params = full * (
                index_heads * index_dim * q_rank + hidden_size * (index_dim + index_heads)
            )
            index_norm_params = full * _norm_parameters(group.index_norm, index_dim)
            replicated_bytes += index_params * weight_bytes + index_norm_params * _dtype_bytes(
                group.index_norm_dtype or weight_dtype
            )
            logical += index_params + index_norm_params
            index_layers = layers if hierarchical else full
            kv_bytes += index_layers * (index_dim + index_dim // 128 * 4)
    sharded_bytes = local_params * weight_bytes
    if group.sink_dtype is not None:
        sharded_bytes += layers * local_heads * _dtype_bytes(group.sink_dtype)
        logical += layers * heads
    return kv_bytes, sharded_bytes, replicated_bytes, logical + norm_params


def _weight_budget(model: ModelSpec, tp_size: int) -> tuple[int, int, int, int, int, int]:
    """按组件与存储精度直接累加权重，TP/EP 采用当前同构放置口径。"""
    h, layers, moe = model.hidden_size, model.num_layers, model.moe_layers
    dense_layers = layers - moe
    if model.vocab_size % tp_size:
        raise ValueError("词表大小必须能被 TP 数整除")
    if dense_layers and model.ffn_intermediate_size % tp_size:
        raise ValueError("Dense FFN 中间维度必须能被 TP 数整除")
    if moe:
        if not (model.num_experts and 1 <= model.experts_per_token <= model.num_experts):
            raise ValueError("MoE 需要专家数及有效的每 token 激活专家数")
        if model.expert_intermediate_size is None:
            raise ValueError("MoE 需要专家中间维度以放置全部驻留专家权重")
        if model.num_experts % tp_size:
            raise ValueError("MoE 专家数必须能被 TP=EP 数整除")
    groups = model.attention_groups or [
        AttentionGroup(
            kind="gqa",
            layers=layers,
            num_attention_heads=model.num_attention_heads,
            num_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
        )
    ]
    budgets = [
        _group_attention_budget(group, h, tp_size, False, model.weight_dtype) for group in groups
    ]
    weight_bytes = _dtype_bytes(model.weight_dtype)
    dense_params = dense_layers * 3 * h * model.ffn_intermediate_size
    embedding_params = (1 if model.tie_word_embeddings else 2) * model.vocab_size * h
    sharded = (
        sum(row[1] for row in budgets) + (dense_params + embedding_params) // tp_size * weight_bytes
    )
    norm_params = (
        layers * model.norms.block_per_layer + int(model.norms.final_norm)
    ) * _norm_parameters(model.norms.kind, h)
    replicated = sum(row[2] for row in budgets)
    replicated += norm_params * _dtype_bytes(model.norms.dtype or model.weight_dtype)
    router_params = moe * h * model.num_experts
    correction_params = moe * model.num_experts if model.router.correction_bias else 0
    replicated += router_params * _dtype_bytes(model.router.dtype or model.weight_dtype)
    replicated += correction_params * _dtype_bytes(model.router.correction_dtype)
    shared_params = (
        moe * model.num_shared_experts * 3 * h * cast(int, model.shared_expert_intermediate_size)
        if model.num_shared_experts
        else 0
    )
    replicated += shared_params * weight_bytes
    ihc_params = 0
    if model.ihc is not None:
        mult = model.ihc.mult
        ihc_params = layers * 2 * (2 * mult * mult * h + 2 + 2 * mult) + mult * mult * h + 1 + mult
        replicated += ihc_params * _dtype_bytes(model.ihc.dtype)
    expert_params = (
        moe * model.num_experts * 3 * h * cast(int, model.expert_intermediate_size) if moe else 0
    )
    expert = expert_params // tp_size * weight_bytes
    non_expert = (
        sum(row[3] for row in budgets)
        + dense_params
        + embedding_params
        + norm_params
        + router_params
        + correction_params
        + shared_params
        + ihc_params
    )
    active_expert = expert_params // model.num_experts * model.experts_per_token if moe else 0
    return (
        sharded,
        replicated,
        expert,
        non_expert + expert_params,
        non_expert + active_expert,
        expert_params,
    )


def resolve_config(config: SimConfig) -> ResolutionReport:
    """预览物理推理配置；报告与原输入、运行状态相互独立。"""
    request = config.inference
    model, hardware, storage = request.model, request.hardware, request.storage
    page_size = request.page_size
    tp_size = request.tp_size
    is_moe = model.moe_layers > 0
    if hardware.gpu_count != tp_size:
        raise ValueError("TP 数必须等于分配给当前实例的 GPU 卡数")
    if model.attention_layout not in ("full_gqa", "grouped", "hy4_preview"):
        raise ValueError("仅支持全注意力 GQA/MQA、注意力组合或 HY4-Preview BF16")
    if is_moe:
        if request.ep_size != tp_size:
            raise ValueError("MoE 首版要求 EP 数等于 TP 数和 GPU 卡数")
    elif request.ep_size != 1 or any(
        (
            model.num_experts,
            model.experts_per_token,
            model.expert_intermediate_size,
        )
    ):
        raise ValueError("Dense 模型不应配置 EP 或 MoE 专家字段")
    if model.weight_dtype == "fp8" or model.kv_dtype == "fp8":
        raise ValueError("首版仅支持 BF16/FP16 权重与 KV；FP8 需要专门的量化布局")
    if model.attention_groups is not None:
        kv_bytes = sum(
            _group_attention_budget(group, model.hidden_size, tp_size, storage.l2_available_gb > 0)[
                0
            ]
            for group in model.attention_groups
        )
    else:
        local_kv_heads = max(1, model.num_kv_heads // tp_size)
        kv_bytes = (
            2 * model.num_layers * local_kv_heads * model.head_dim * 2
        )  # 单 rank 每 token 的 K/V BF16 字节
    group_kv_bytes = kv_bytes * tp_size  # 整个 TP 组每 token 的 KV 字节
    page_bytes = group_kv_bytes * page_size  # 下层按整组页分配容量
    l1 = hardware.memory_bytes_per_gpu
    l2 = _bytes_from_gb(storage.l2_available_gb)
    shared = config.shared_l3
    l3 = _bytes_from_gb(shared.available_gb) if shared is not None else 0
    rank_reports: list[RankCapacityReport] = []
    sharded, replicated, expert, total_parameters, active_parameters, expert_parameters = (
        _weight_budget(model, tp_size)
    )
    local_page_bytes = kv_bytes * page_size
    l1_kv = l1 - sharded - replicated - expert
    if l1_kv < local_page_bytes:
        raise ValueError("至少一张 GPU 扣除权重后不足一个本地 KV 页")
    l1_pages = l1_kv // local_page_bytes
    if tp_size > 1 or is_moe:
        rank_reports = [
            RankCapacityReport(
                rank=rank,
                sharded_weights_bytes=sharded,
                replicated_weights_bytes=replicated,
                expert_weights_bytes=expert,
                kv_bytes_per_token=kv_bytes,
                l1_kv_budget_bytes=l1_kv,
                l1_capacity_pages=l1_pages,
            )
            for rank in range(tp_size)
        ]
    weights = (sharded + replicated + expert) * tp_size
    if l3 and not l2:
        raise ValueError("启用 L3 时必须先启用 L2")
    if (l2 and l2 < page_bytes) or (l3 and l3 < page_bytes):
        raise ValueError("启用的 L2/L3 容量必须至少容纳一个 KV 页")
    if l2 and storage.l2_to_l1_gb_s is None:
        raise ValueError("启用 L2 时必须配置 L2 → L1 带宽")
    if l3 and storage.l3_to_l2_gb_s is None:
        raise ValueError("启用 L3 时必须配置 L3 → L2 带宽")
    l2_bw = storage.l2_to_l1_gb_s if l2 else None
    l3_bw = storage.l3_to_l2_gb_s if l3 else None

    l2_pages, l3_pages = l2 // page_bytes, l3 // page_bytes
    runnable = (
        (tp_size == 1 or request.timing.mode == "analytical")
        and (not is_moe or request.timing.mode == "analytical")
        and (
            tp_size == 1
            or (
                hardware.effective_interconnect_gb_s is not None
                and hardware.collective_latency_ms is not None
            )
        )
        and (not is_moe or request.ep_size == 1 or hardware.ep_communication_latency_ms is not None)
    )
    reason = None
    if not runnable:
        reason = (
            "TP/EP 或 MoE 仅支持解析计时；四系数无法表达逐卡计算与通信"
            if request.timing.mode == "empirical"
            else "多卡解析运行须提供互联带宽、TP 集合通信延迟及所需的 EP 单次通信开销"
        )
    return ResolutionReport(
        max_total_tokens=l1_pages * page_size,
        l2_capacity_tokens=l2_pages * page_size,
        l3_capacity_tokens=l3_pages * page_size,
        runnable=runnable,
        reason=reason,
        capacity=CapacityReport(
            kv_bytes_per_token=group_kv_bytes,
            kv_bytes_per_page=page_bytes,
            weights_resident_bytes=weights,
            estimated_total_parameters=total_parameters,
            estimated_active_parameters=active_parameters,
            estimated_expert_parameters=expert_parameters,
            l1_budget_bytes=l1,
            l1_kv_budget_bytes=l1_kv,
            l2_budget_bytes=l2,
            l3_budget_bytes=l3,
            l1_capacity_pages=l1_pages,
            l2_capacity_pages=l2_pages,
            l3_capacity_pages=l3_pages,
            l2_to_l1_tokens_per_ms=(
                l2_bw * 1_000_000 / group_kv_bytes if l2_bw is not None else None
            ),
            l3_to_l2_tokens_per_ms=(
                l3_bw * 1_000_000 / group_kv_bytes if l3_bw is not None else None
            ),
            rank_reports=rank_reports,
        ),
    )


def prepare_runtime(config: SimConfig) -> RuntimeConfig:
    """运行前解析一次容量、GPU 计时与每页链路成本，不改写输入。"""
    inference = config.inference
    scheduler = config.scheduler
    shared = config.shared_l3
    lookup_latency = shared.lookup_latency_ms if shared is not None else 0.0
    report = resolve_config(config)
    if not report.runnable:
        if inference.timing.mode == "empirical":
            raise NotImplementedError(report.reason)
        raise ValueError(report.reason)
    storage = inference.storage
    engine = EngineConfig(
        max_running_requests=scheduler.max_running_requests,
        max_prefill_tokens=scheduler.max_prefill_tokens,
        pd_mutually_exclusive=scheduler.pd_mutually_exclusive,
        l3_prefetch_min_tokens=scheduler.l3_prefetch_min_tokens,
        page_size=inference.page_size,
        max_output_tokens=inference.max_output_tokens,
        max_total_tokens=report.max_total_tokens,
        l2_capacity_tokens=report.l2_capacity_tokens,
        l3_capacity_tokens=report.l3_capacity_tokens,
    )
    coefficients = (
        inference.timing.model_dump(exclude={"mode"})
        if isinstance(inference.timing, EmpiricalTimingConfig)
        else {"prefill_c1": 0.0, "prefill_c2": 0.0, "decode_d1": 0.0, "decode_d2": 0.0}
    )
    perf = PerfModelConfig(
        **coefficients,
        kv_bytes_per_token=report.capacity.kv_bytes_per_token,
        l2_to_l1_bandwidth_bytes_per_ms=(
            storage.l2_to_l1_gb_s * 1_000_000
            if engine.l2_capacity_tokens and storage.l2_to_l1_gb_s is not None
            else None
        ),
        l3_to_l2_bandwidth_bytes_per_ms=(
            storage.l3_to_l2_gb_s * 1_000_000
            if engine.l3_capacity_tokens and storage.l3_to_l2_gb_s is not None
            else None
        ),
        l2_latency_ms=storage.l2_latency_ms,
        l3_latency_ms=storage.l3_latency_ms,
        l3_lookup_latency_ms=lookup_latency,
    )
    capacity = report.capacity
    l2 = (
        TransferLink(inference.page_size / capacity.l2_to_l1_tokens_per_ms, storage.l2_latency_ms)
        if capacity.l2_to_l1_tokens_per_ms is not None
        else None
    )
    l3 = (
        TransferLink(inference.page_size / capacity.l3_to_l2_tokens_per_ms, storage.l3_latency_ms)
        if capacity.l3_to_l2_tokens_per_ms is not None
        else None
    )
    analytical = (
        AnalyticalSpec(inference.model, inference.hardware, inference.tp_size, inference.ep_size)
        if inference.timing.mode == "analytical"
        else None
    )
    return RuntimeConfig(engine, perf, TransferLinks(l2, l3), analytical, report)
