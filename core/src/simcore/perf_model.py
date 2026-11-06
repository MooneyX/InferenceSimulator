"""性能模型：单步工作量按层序逐层估算 Roofline、TP/EP 通信成本（未经校准）。"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from simcore.schemas import AttentionGroup, HardwareSpec, ModelSpec

__all__ = [
    "DecodeWork",
    "LayerKind",
    "LayerPlan",
    "PrefillWork",
    "StepCost",
    "StepWorkload",
    "build_layer_plan",
    "pipelined_finish_ms",
    "step_cost",
]


@dataclass(frozen=True, slots=True)
class PrefillWork:
    request_id: int
    reused_tokens: int
    new_tokens: int
    context_after: int

    def __post_init__(self) -> None:  # 校验 Prefill 请求身份、复用量和输入规模；无返回值
        if self.request_id < 0 or self.reused_tokens < 0 or self.new_tokens < 1:
            raise ValueError("Prefill 请求 ID、复用量或新计算量不合法")
        if self.context_after != self.reused_tokens + self.new_tokens:
            raise ValueError("Prefill 上下文长度必须等于复用量与新计算量之和")


@dataclass(frozen=True, slots=True)
class DecodeWork:
    request_id: int
    context_after_decode: int

    def __post_init__(self) -> None:  # 校验 Decode 请求 ID 与上下文长度；无返回值
        if self.request_id < 0 or self.context_after_decode < 1:
            raise ValueError("Decode 请求 ID 或上下文长度不合法")


@dataclass(frozen=True, slots=True)
class StepWorkload:
    phase: Literal["PREFILL", "DECODE", "MIXED"]
    prefill: tuple[PrefillWork, ...]
    decode: tuple[DecodeWork, ...]
    allocated_pages: int
    batch_requests: int
    restore_only: tuple[int, ...] = ()

    def __post_init__(self) -> None:  # 校验 GPU 步阶段、请求集合与页数；无返回值
        prefill_ids = tuple(item.request_id for item in self.prefill)
        decode_ids = tuple(item.request_id for item in self.decode)
        phase = "MIXED" if prefill_ids and decode_ids else "PREFILL" if prefill_ids else "DECODE"
        if (not prefill_ids and not decode_ids) or self.phase != phase:
            raise ValueError("单步阶段须与非空 Prefill/Decode 集合一致")
        if (
            prefill_ids != tuple(sorted(set(prefill_ids)))
            or decode_ids != tuple(sorted(set(decode_ids)))
            or set(prefill_ids) & set(decode_ids)
            or self.restore_only != tuple(sorted(set(self.restore_only)))
            or (set(self.restore_only) & (set(prefill_ids) | set(decode_ids)))
        ):
            raise ValueError("Prefill/Decode/Restore 请求须各自按 ID 排序且不可重叠")
        if self.allocated_pages < 0 or self.batch_requests != (
            len(prefill_ids) + len(decode_ids) + len(self.restore_only)
        ):
            raise ValueError("单步页申请量或批内请求数不合法")


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
    # 按层序的（连续层数, 单层耗时 ms），合计为 duration_ms
    layer_ms: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class LayerKind:
    """一类同构层：注意力组下标、是否 MoE 及 DSA 索引器是否完整计算。"""

    group: int
    moe: bool
    full_indexer: bool


# 单层非权重类计算阶段与工作量无关的系数。P 为本阶段配对数：topk 为 0 取全配对，否则取截断配对。
# FLOPs = 每 token × tokens + 每配对 × P；
# 字节 = 固定 + 每 KV 读 × (kv_reads + tokens) + 每 (配对 + token) × (P + tokens) + 每配对 × P。
_Stage = tuple[float, float, float, float, float, float, int]


@dataclass(frozen=True, slots=True)
class _KindStages:
    """一类层的阶段系数。权重类阶段（每 token 读一遍权重矩阵）FLOPs = c × tokens、字节 = c，
    计算与访存孰大只取决于 tokens 与 FLOPS/带宽之比而与 c 无关，故合并为系数和与阶段数。"""

    weight_sum: int
    weight_stages: int
    stages: tuple[_Stage, ...]
    collectives: int  # 每层 TP 同步次数：Dense 层注意力与 FFN 后各一次，MoE 层仅注意力后一次
    moe: bool  # MoE 层在 EP>1 时另计 dispatch + combine


@dataclass(frozen=True, slots=True)
class LayerPlan:
    """模型层序合并相邻同类层后的结构；运行前构建一次，供逐层计时与流水复用。"""

    model: ModelSpec
    groups: tuple[AttentionGroup, ...]
    kinds: tuple[LayerKind, ...]
    kind_counts: tuple[int, ...]
    segments: tuple[tuple[int, int], ...]  # 按层序的（连续层数, kinds 下标）
    topks: tuple[int, ...]  # 层序中出现的 DSA top-k
    moe_layers: int
    # (TP, EP) → 每类层的阶段系数，首次计时时生成
    stage_cache: dict[tuple[int, int], tuple[_KindStages, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def num_layers(self) -> int:  # 层序 → 模型总层数
        return self.model.num_layers


def build_layer_plan(model: ModelSpec) -> LayerPlan:  # 模型规格 → 合并相邻同类层的层序结构
    """未配置注意力组时以顶层 GQA 字段构造唯一注意力组。"""
    groups = tuple(
        model.attention_groups
        or (
            AttentionGroup(
                kind="gqa",
                num_attention_heads=model.num_attention_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim,
            ),
        )
    )
    index: dict[LayerKind, int] = {}
    counts: list[int] = []
    segments: list[list[int]] = []
    for segment in model.layers:
        kind = LayerKind(segment.attention_group, segment.ffn == "moe", segment.full_indexer)
        position = index.setdefault(kind, len(index))
        if position == len(counts):
            counts.append(0)
        counts[position] += segment.count
        if segments and segments[-1][1] == position:
            segments[-1][0] += segment.count
        else:
            segments.append([segment.count, position])
    dsa_groups = [group for group in groups if group.kind == "mla_dsa"]
    topks = sorted({cast(int, group.topk) for group in dsa_groups})
    return LayerPlan(
        model,
        groups,
        tuple(index),
        tuple(counts),
        tuple((count, position) for count, position in segments),
        tuple(topks),
        model.moe_layers,
    )


def _layer_stages(plan: LayerPlan, tp_size: int, ep_size: int) -> tuple[_KindStages, ...]:
    """层序结构与 TP/EP → 每类层各计算阶段的系数；按 (TP, EP) 缓存在层序结构上。"""
    model = plan.model
    h = model.hidden_size
    merged = tp_size == 1 and not plan.moe_layers  # 单卡 Dense 将 Q/K/V 与输出投影合为一个阶段
    result: list[_KindStages] = []
    for kind in plan.kinds:
        group = plan.groups[kind.group]
        heads = group.num_attention_heads // tp_size
        weight_sizes: list[int] = []  # 各权重类阶段每 rank 的权重元素数（BF16/FP16）
        stages: list[_Stage] = []
        weights = weight_sizes.append
        if group.kind == "gqa":
            kv = max(1, (group.num_kv_heads or 0) // tp_size)
            dim = group.head_dim or 0
            qkv = h * (heads + 2 * kv) * dim
            out = h * heads * dim
            if merged:
                weights(qkv + out)
            else:
                weights(qkv)
                weights(out)
            stages.append((0, 4 * heads * dim, 0, 4 * kv * dim, 0, 0, 0))
        else:
            q_rank, kv_rank = group.q_rank or 0, group.kv_rank or 0
            rope, nope, value = group.rope_dim or 0, group.nope_dim or 0, group.value_dim or 0
            weights(
                h * (q_rank + kv_rank + rope)
                + (q_rank * heads * (nope + rope) + kv_rank * heads * (nope + value))
            )
            topk = index_read = 0
            if group.kind == "mla_dsa":
                index_heads, index_dim = group.index_heads or 0, group.index_dim or 0
                if kind.full_indexer:
                    index_weights = index_heads * index_dim * q_rank + h * (index_dim + index_heads)
                    stages.append(
                        (
                            2 * index_weights,
                            2 * index_heads * index_dim,
                            2 * index_weights,
                            index_dim + index_dim // 128 * 4,
                            0,
                            0,
                            0,
                        )
                    )
                topk = cast(int, group.topk)
                index_read = 4  # 每个有效候选一次 INT32 选位索引读取
            stages.append(
                (0, 2 * heads * (nope + rope + value), 0, 0, (kv_rank + rope) * 2, index_read, topk)
            )
            weights(h * heads * value * (2 if group.output_gate else 1))
        if kind.moe:
            expert_size = 3 * h * (model.expert_intermediate_size or 0)
            weights(h * model.num_experts)  # 每 rank 对本步所有 token 的全部专家打分
            stages.append(
                (
                    2 * model.experts_per_token / ep_size * expert_size,
                    0,
                    2 * (model.num_experts // ep_size) * expert_size,
                    0,
                    0,
                    0,
                    0,
                )
            )
            if model.num_shared_experts:
                shared = model.shared_expert_intermediate_size or 0
                weights(model.num_shared_experts * 3 * h * shared)
        else:
            weights(3 * h * (model.ffn_intermediate_size // tp_size))
        result.append(
            _KindStages(
                2 * sum(weight_sizes),
                len(weight_sizes),
                tuple(stages),
                (1 if kind.moe else 2) if tp_size > 1 else 0,
                kind.moe,
            )
        )
    plan.stage_cache[(tp_size, ep_size)] = cached = tuple(result)
    return cached


def pipelined_finish_ms(
    start_ms: float, ready_ms: Sequence[float], layer_ms: Sequence[tuple[int, float]]
) -> float:  # 步起点、各层 KV 到齐时刻、逐层耗时 → 末层计算完成时刻
    """第 i 层计算须等第 i-1 层算完且第 i 层 KV 到齐：F_i = max(F_{i-1}, R_i) + c_i。"""
    finish = start_ms
    layer = 0
    for count, duration in layer_ms:
        for ready in ready_ms[layer : layer + count]:
            finish = (finish if finish > ready else ready) + duration
        layer += count
    return finish


def step_cost(
    workload: StepWorkload,
    plan: LayerPlan,
    hardware: HardwareSpec,
    tp_size: int = 1,
    ep_size: int = 1,
) -> StepCost:  # 单步负载、层序结构、硬件及 TP/EP 并行度 → 分项耗时、资源成本与逐层耗时
    """每类同构层按真实单步工作量计一次单层成本，再按层数汇总；同步通信串行计入单层耗时。"""
    model = plan.model
    h = model.hidden_size
    tokens = len(workload.decode) # 记录本步生成的新token总数, decode每步只生成一个token
    pairs = 0.0
    kv_reads = 0
    for item in workload.prefill:
        tokens += item.new_tokens
        pairs += item.new_tokens * item.reused_tokens + item.new_tokens * (item.new_tokens + 1) / 2
        kv_reads += item.context_after
    decode_context = sum(item.context_after_decode for item in workload.decode)
    pairs += decode_context
    kv_reads += decode_context
    flops_per_ms = hardware.effective_fp16_tflops * 1e9
    bytes_per_ms = hardware.effective_hbm_gb_s * 1e6
    overhead = hardware.kernel_overhead_ms

    collective_ms = collective_bytes = 0.0
    if tp_size > 1:
        if hardware.effective_interconnect_gb_s is None or hardware.collective_latency_ms is None:
            raise ValueError("TP 计时需要单 rank 互联带宽和集合通信固定延迟")
        # 每 rank 参与归约的完整隐藏状态；按 ring 的每 rank 有效链路流量计。
        collective_bytes = 2 * (tp_size - 1) / tp_size * tokens * h * 2
        collective_ms = hardware.collective_latency_ms + collective_bytes / (
            hardware.effective_interconnect_gb_s * 1e6
        )
    exchange_ms = exchange_bytes = 0.0
    ep_active = ep_size > 1 and plan.moe_layers > 0
    if ep_active:
        if (
            hardware.effective_interconnect_gb_s is None
            or hardware.ep_communication_latency_ms is None
        ):
            raise ValueError("EP 计时需要单 rank 互联带宽和 EP 单次通信固定开销")
        # 不生成具体路由：每 token 在无放回 Top-K 中命中某一目标 rank 的概率。
        experts_per_rank = model.num_experts // ep_size
        missed = 1.0  # 无放回选择均未命中某个远端 rank 的概率
        for selected in range(model.experts_per_token): # 计算单个token不会选中任何远端专家, 不产生ep通信的概率
            missed *= (model.num_experts - experts_per_rank - selected) / (
                model.num_experts - selected
            )
        remote_selections = tokens * model.experts_per_token * (ep_size - 1) / (ep_size * ep_size) # 单个token选中远端专家的期望次数
        if workload.prefill:
            remote_payloads = tokens * (ep_size - 1) / ep_size * (1 - missed)
        else:
            remote_payloads = remote_selections  # 纯 Decode 按远端专家选择次数估计逻辑 payload
        # BF16/FP16 向量与计算精度一致；每次远端选择另计 4B ID + 4B 权重/索引。
        exchange_bytes = remote_payloads * h * 2 + remote_selections * 8 # hidden state向量 + 路由元数据
        exchange_ms = hardware.ep_communication_latency_ms + exchange_bytes / (
            hardware.effective_interconnect_gb_s * 1e6
        )

    # 各 DSA top-k 截断后的配对数；0 对应全配对。
    pair_counts: dict[int, float] = {0: pairs}
    for topk in plan.topks:
        sparse = 0.0
        for item in workload.prefill:
            uncapped = min(item.new_tokens, max(0, topk - item.reused_tokens))
            sparse += (
                uncapped * item.reused_tokens
                + uncapped * (uncapped + 1) / 2
                + (item.new_tokens - uncapped) * topk
            )
        sparse += sum(min(topk, item.context_after_decode) for item in workload.decode)
        pair_counts[topk] = sparse
    kv_tokens = kv_reads + tokens
    kind_stages = plan.stage_cache.get((tp_size, ep_size)) or _layer_stages(plan, tp_size, ep_size)

    duration = flops = hbm = compute = memory = 0.0
    collectives = 0
    kind_ms: list[float] = []
    for count, coefficients in zip(plan.kind_counts, kind_stages, strict=True):
        weight_sum = coefficients.weight_sum
        layer_flops = weight_sum * tokens
        layer_hbm = weight_sum
        layer_compute = layer_flops / flops_per_ms
        layer_memory = weight_sum / bytes_per_ms
        layer_device = coefficients.weight_stages * overhead + (
            layer_compute if layer_compute > layer_memory else layer_memory
        )
        for stage in coefficients.stages:
            per_token, per_pair, fixed, per_kv, per_pair_token, per_pair_byte, topk = stage
            stage_pairs = pair_counts[topk]
            work_flops = per_token * tokens + per_pair * stage_pairs
            nbytes = (
                fixed
                + per_kv * kv_tokens
                + per_pair_token * (stage_pairs + tokens)
                + per_pair_byte * stage_pairs
            )
            stage_compute = work_flops / flops_per_ms
            stage_memory = nbytes / bytes_per_ms
            layer_flops += work_flops
            layer_hbm += nbytes
            layer_compute += stage_compute
            layer_memory += stage_memory
            layer_device += overhead + (
                stage_compute if stage_compute > stage_memory else stage_memory
            )
        layer_ms = layer_device
        if coefficients.collectives:
            layer_ms += coefficients.collectives * collective_ms
        if coefficients.moe and ep_active:
            layer_ms += 2 * exchange_ms
        kind_ms.append(layer_ms)
        duration += count * layer_ms
        flops += count * layer_flops
        hbm += count * layer_hbm
        compute += count * layer_compute
        memory += count * layer_memory
        collectives += count * coefficients.collectives
    tp_ms = collectives * collective_ms
    tp_bytes = collectives * collective_bytes
    ep_layers = plan.moe_layers if ep_active else 0
    ep_ms = ep_layers * exchange_ms
    ep_bytes = ep_layers * exchange_bytes
    return StepCost(
        duration,
        flops,
        hbm,
        compute,
        memory,
        tp_ms + 2 * ep_ms,
        tp_bytes + 2 * ep_bytes,
        collectives,
        tp_ms,
        tp_bytes,
        ep_ms,
        ep_ms,
        ep_bytes,
        ep_bytes,
        2 * ep_layers,
        tuple([(count, kind_ms[position]) for count, position in plan.segments]),
    )
