"""性能模型：把 token 规模映射为耗时（ms）。

本模块全部为**纯函数**——输出仅取决于显式入参，无全局状态、无缓存、无随机性。
这是 ``docs/design/kernel.md`` 的硬约束，为后续内核替换（如 Rust）预留接缝：
只要系数一致，任何语言的等价实现都必须给出逐位相同的结果。

依赖层级：仅依赖 ``schemas``。
"""

from collections.abc import Iterable

from simcore.schemas import PerfModelConfig

__all__ = ["decode_step_time", "prefill_cost", "prefill_time", "step_time"]


def prefill_time(prompt_tokens: int, cfg: PerfModelConfig) -> float:
    """单个请求的 prefill 耗时（ms）。

    ``prefill_c1 * P + prefill_c2 * P^2``。二次项刻画 attention 随序列长度的超线性增长，
    ``prefill_c2`` 默认 0 时退化为线性模型。

    Args:
        prompt_tokens: prompt token 数，须 >= 1。
        cfg: 性能模型系数。

    Raises:
        ValueError: ``prompt_tokens`` < 1。
    """
    if prompt_tokens < 1:
        raise ValueError(f"prompt_tokens 须 >= 1，得到 {prompt_tokens}")
    return cfg.prefill_c1 * prompt_tokens + cfg.prefill_c2 * prompt_tokens * prompt_tokens


def prefill_cost(context_before: int, new_prefill_tokens: int, cfg: PerfModelConfig) -> float:
    """前缀命中后仍需读取历史上下文：二次项按历史与新输入的面积差计算。"""
    if context_before < 0 or new_prefill_tokens < 1:
        raise ValueError("context_before 须 >= 0 且 new_prefill_tokens 须 >= 1")
    return cfg.prefill_c1 * new_prefill_tokens + cfg.prefill_c2 * new_prefill_tokens * (
        2 * context_before + new_prefill_tokens
    )


def decode_step_time(batch_total_tokens: int, cfg: PerfModelConfig) -> float:
    """一次 decode 迭代的耗时（ms）。

    ``decode_d1 + decode_d2 * T``。常数项为与 batch 无关的固定开销（kernel launch 等），
    线性项随 batch 内 KV token 总量增长（访存带宽受限）。

    注意：耗时按**整个 batch 一次迭代**计，不是每请求。batch 为空时无意义，应由调用方
    避免（引擎在 batch 为空时直接跳跃时钟，不执行 step）。

    Args:
        batch_total_tokens: 本步 batch 内所有请求的 KV token 总量，须 >= 1。
        cfg: 性能模型系数。

    Raises:
        ValueError: ``batch_total_tokens`` < 1。
    """
    if batch_total_tokens < 1:
        raise ValueError(f"batch_total_tokens 须 >= 1，得到 {batch_total_tokens}")
    return cfg.decode_d1 + cfg.decode_d2 * batch_total_tokens


def step_time(
    admitted_prompt_tokens: Iterable[int],
    batch_total_tokens: int,
    cfg: PerfModelConfig,
    *,
    has_decode: bool = True,
    prefill_contexts: Iterable[int] | None = None,
) -> float:
    """一个 ``EngineStep`` 的总步长（ms）——融合批语义。

    prefill 与 decode 在同一次迭代中融合执行（chunked-prefill 式），故取两者最大值
    而非相加::

        max(Σ prefill_cost(已复用上下文, 新计算输入), decode_step_time(batch_total_tokens))

    Args:
        admitted_prompt_tokens: 本步新准入请求的 prompt 长度序列；无新准入时为空。
        batch_total_tokens: 本步 batch 的 KV token 总量（含本步准入请求的 prompt）。
        cfg: 性能模型系数。
        has_decode: 本步是否存在 decode 工作。纯 prefill 步（batch 内全是刚准入、
            尚无 token 可解码的请求）置 False，此时步长仅由 prefill 决定。
        prefill_contexts: 与新计算输入一一对应的实际已复用 token 数，默认全为 0。

    Raises:
        ValueError: 既无准入请求又无 decode 工作（空步不应发生）。
    """
    admitted = list(admitted_prompt_tokens)
    if not admitted and not has_decode:
        raise ValueError("空步：既无准入请求也无 decode 工作")
    contexts = list(prefill_contexts) if prefill_contexts is not None else [0] * len(admitted)
    if len(contexts) != len(admitted):
        raise ValueError("prefill_contexts 与 admitted_prompt_tokens 数量须一致")
    # 逐项求和而非 sum(generator)：顺序固定，保证浮点累加结果可复现
    prefill_total = 0.0
    for context, prompt_tokens in zip(contexts, admitted, strict=True):
        prefill_total += prefill_cost(context, prompt_tokens, cfg)
    decode_total = decode_step_time(batch_total_tokens, cfg) if has_decode else 0.0
    return max(prefill_total, decode_total)
