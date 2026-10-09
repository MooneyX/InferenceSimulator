"""从单 GPU 全注意力规格解析固定大小的 KV 页池与恢复链路。"""

from decimal import ROUND_FLOOR, Decimal

from simcore.schemas import CapacityReport, ResolvedConfig, ResolvedRequest

_GB = Decimal(1_000_000_000)


def _bytes_from_gb(value: float) -> int:
    return int((Decimal(str(value)) * _GB).to_integral_value(rounding=ROUND_FLOOR))


def resolve_config(request: ResolvedRequest) -> ResolvedConfig:
    model, hardware, storage = request.model, request.hardware, request.storage
    page_size = request.config.engine.page_size
    if hardware.gpu_count != 1:
        raise ValueError("第一阶段仅支持单 GPU；多卡规格可保存但尚不可运行")
    if model.attention_layout != "full_gqa" or model.moe_layers:
        raise ValueError("第一阶段仅支持 Dense 全注意力 GQA/MQA 模型")
    if model.weight_dtype == "fp8" or model.kv_dtype == "fp8":
        raise ValueError("第一阶段仅支持 BF16/FP16 权重与 KV；FP8 需要专门的量化布局")

    kv_bytes = 2 * model.num_layers * model.num_kv_heads * model.head_dim * 2
    page_bytes = kv_bytes * page_size
    weights = model.resident_weights_bytes or model.total_parameters * 2
    l1 = _bytes_from_gb(storage.l1_available_gb)
    l2 = _bytes_from_gb(storage.l2_available_gb)
    l3 = _bytes_from_gb(storage.l3_available_gb)
    if l1 > hardware.memory_bytes_per_gpu:
        raise ValueError("L1 可用容量不能超过单卡显存")
    l1_kv = l1 - weights - storage.runtime_reserve_bytes
    if l1_kv < page_bytes:
        raise ValueError("L1 扣除驻留权重与运行时预留后不足一个 KV 页")
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

    l1_pages, l2_pages, l3_pages = (l1_kv // page_bytes, l2 // page_bytes, l3 // page_bytes)
    engine = request.config.engine.model_copy(update={
        "max_total_tokens": l1_pages * page_size,
        "l2_capacity_tokens": l2_pages * page_size,
        "l3_capacity_tokens": l3_pages * page_size,
    })
    perf = request.config.perf_model.model_copy(update={
        "kv_bytes_per_token": kv_bytes,
        "l2_to_l1_bandwidth_bytes_per_ms": (
            l2_bw * 1_000_000 if l2_bw is not None else None
        ),
        "l3_to_l2_bandwidth_bytes_per_ms": (
            l3_bw * 1_000_000 if l3_bw is not None else None
        ),
        "l2_latency_ms": storage.l2_latency_ms,
        "l3_latency_ms": storage.l3_latency_ms,
    })
    config = request.config.model_copy(update={"engine": engine, "perf_model": perf})
    # model_copy 不运行 Pydantic 校验；统一经过正式配置校验入口。
    config = type(config).model_validate(config.model_dump())
    return ResolvedConfig(
        config=config, model=model, hardware=hardware, storage=storage, timing=request.timing,
        capacity=CapacityReport(
            kv_bytes_per_token=kv_bytes, kv_bytes_per_page=page_bytes,
            weights_resident_bytes=weights, l1_budget_bytes=l1, l1_kv_budget_bytes=l1_kv,
            l2_budget_bytes=l2, l3_budget_bytes=l3,
            l1_capacity_pages=l1_pages, l2_capacity_pages=l2_pages, l3_capacity_pages=l3_pages,
            l2_to_l1_tokens_per_ms=(l2_bw * 1_000_000 / kv_bytes if l2_bw is not None else None),
            l3_to_l2_tokens_per_ms=(l3_bw * 1_000_000 / kv_bytes if l3_bw is not None else None),
        ),
    )
