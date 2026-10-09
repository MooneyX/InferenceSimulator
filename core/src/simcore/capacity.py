"""从全注意力规格解析单卡或 TP 组的固定大小 KV 页容量。"""

from decimal import ROUND_FLOOR, Decimal

from simcore.schemas import (
    CapacityReport,
    ModelSpec,
    RankCapacityReport,
    ResolvedConfig,
    ResolvedRequest,
)

_GB = Decimal(1_000_000_000)


def _bytes_from_gb(value: float) -> int:
    return int((Decimal(str(value)) * _GB).to_integral_value(rounding=ROUND_FLOOR))


def _tp_weights(model: ModelSpec, tp_size: int) -> tuple[int, int, int]:
    """返回每 rank 的分片/复制权重字节及尚未归属的参数数目。"""
    if model.vocab_size is None:
        raise ValueError("TP 容量预览需要模型 vocab_size 与词嵌入共享规则")
    if model.num_attention_heads % tp_size or model.ffn_intermediate_size % tp_size:
        raise ValueError("Query 头数和 FFN 中间维度必须能被 TP 数整除")
    h, q, kv, d, f = (
        model.hidden_size,
        model.num_attention_heads,
        model.num_kv_heads,
        model.head_dim,
        model.ffn_intermediate_size,
    )
    if (kv >= tp_size and kv % tp_size) or (kv < tp_size and tp_size % kv):
        raise ValueError("KV 头数与 TP 数不满足均分或按头复制条件")
    if model.vocab_size % tp_size:
        raise ValueError("第一版词表大小必须能被 TP 数整除")
    local_kv_heads = max(1, kv // tp_size)
    sharded = (
        model.num_layers
        * (
            h * ((q // tp_size + 2 * local_kv_heads) * d)
            + h * (q // tp_size * d)
            + 3 * h * (f // tp_size)
        )
        + (1 if model.tie_word_embeddings else 2) * model.vocab_size // tp_size * h
    )
    # 两次层归一化、末层归一化及每层 Q/K 归一化。
    replicated = model.num_layers * (2 * h + 2 * d) + h
    known_total = (
        model.num_layers * (h * (2 * q * d + 2 * kv * d) + 3 * h * f)
        + (1 if model.tie_word_embeddings else 2) * model.vocab_size * h
        + replicated
    )
    weights = model.resident_weights_bytes or model.total_parameters * 2
    if weights < known_total * 2:
        raise ValueError("模型驻留权重小于结构可推导的权重；请核对参数量与精度")
    extra_bytes = weights - known_total * 2
    # 未归属的差额保守地计入每张卡，不虚构其分片方式。
    return sharded * 2, replicated * 2 + extra_bytes, extra_bytes // 2


def resolve_config(request: ResolvedRequest) -> ResolvedConfig:
    model, hardware, storage = request.model, request.hardware, request.storage
    page_size = request.config.engine.page_size
    tp_size = request.tp_size
    if hardware.gpu_count != tp_size:
        raise ValueError("TP 数必须等于分配给当前实例的 GPU 卡数")
    if model.attention_layout != "full_gqa" or model.moe_layers:
        raise ValueError("TP 容量预览仅支持 Dense 全注意力 GQA/MQA 模型；MoE 等待 EP")
    if model.weight_dtype == "fp8" or model.kv_dtype == "fp8":
        raise ValueError("第一阶段仅支持 BF16/FP16 权重与 KV；FP8 需要专门的量化布局")
    if tp_size > 1 and (
        hardware.effective_interconnect_gb_s is None or hardware.collective_latency_ms is None
    ):
        raise ValueError("TP 容量预览需要单 rank 有效互联带宽及每次集合通信固定延迟")

    local_kv_heads = max(1, model.num_kv_heads // tp_size)
    kv_bytes = 2 * model.num_layers * local_kv_heads * model.head_dim * 2
    group_kv_bytes = kv_bytes * tp_size
    page_bytes = group_kv_bytes * page_size
    weights = model.resident_weights_bytes or model.total_parameters * 2
    l1 = _bytes_from_gb(storage.l1_available_gb)
    l2 = _bytes_from_gb(storage.l2_available_gb)
    l3 = _bytes_from_gb(storage.l3_available_gb)
    if l1 > hardware.memory_bytes_per_gpu:
        raise ValueError("L1 可用容量不能超过单卡显存")
    rank_reports: list[RankCapacityReport] = []
    unassigned = 0
    if tp_size == 1:
        l1_kv = l1 - weights - storage.runtime_reserve_bytes
        if l1_kv < page_bytes:
            raise ValueError("L1 扣除驻留权重与运行时预留后不足一个 KV 页")
        l1_pages = l1_kv // page_bytes
    else:
        sharded, replicated, unassigned = _tp_weights(model, tp_size)
        local_page_bytes = kv_bytes * page_size
        l1_kv = l1 - sharded - replicated - storage.runtime_reserve_bytes
        if l1_kv < local_page_bytes:
            raise ValueError("至少一张 GPU 扣除权重和预留后不足一个本地 KV 页")
        l1_pages = l1_kv // local_page_bytes
        rank_reports = [
            RankCapacityReport(
                rank=rank,
                sharded_weights_bytes=sharded,
                replicated_weights_bytes=replicated,
                kv_bytes_per_token=kv_bytes,
                l1_kv_budget_bytes=l1_kv,
                l1_capacity_pages=l1_pages,
            )
            for rank in range(tp_size)
        ]
        weights = (sharded + replicated) * tp_size
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
    engine = request.config.engine.model_copy(
        update={
            "max_total_tokens": l1_pages * page_size,
            "l2_capacity_tokens": l2_pages * page_size,
            "l3_capacity_tokens": l3_pages * page_size,
        }
    )
    perf = request.config.perf_model.model_copy(
        update={
            "kv_bytes_per_token": group_kv_bytes,
            "l2_to_l1_bandwidth_bytes_per_ms": (l2_bw * 1_000_000 if l2_bw is not None else None),
            "l3_to_l2_bandwidth_bytes_per_ms": (l3_bw * 1_000_000 if l3_bw is not None else None),
            "l2_latency_ms": storage.l2_latency_ms,
            "l3_latency_ms": storage.l3_latency_ms,
        }
    )
    config = request.config.model_copy(update={"engine": engine, "perf_model": perf})
    # model_copy 不运行 Pydantic 校验；统一经过正式配置校验入口。
    config = type(config).model_validate(config.model_dump())
    return ResolvedConfig(
        config=config,
        model=model,
        hardware=hardware,
        storage=storage,
        timing=request.timing,
        tp_size=tp_size,
        runnable=tp_size == 1 or request.timing == "analytical",
        capacity=CapacityReport(
            kv_bytes_per_token=group_kv_bytes,
            kv_bytes_per_page=page_bytes,
            weights_resident_bytes=weights,
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
            unassigned_parameters=unassigned,
        ),
    )
