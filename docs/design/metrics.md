# 当前指标计算口径

`metrics.compute_metrics` 以**已完成请求**的区间记录计算时延、前缀命中和首次 Prefill，并结合全场回撤重算量与 L1 缓存快照；已发送但失败和未发送而跳过的请求不进入成功请求指标分母。字段定义由 `core/src/simcore/schemas.py` 提供，HTTP 传输见 [api.md](api.md)。下述单位除吞吐（请求/秒）、块数和 token 数外均为 ms。

## 时延与吞吐

- 每请求 `TTFT = prefill_end - arrival`，返回均值、P50、P99。分位数先升序排序，再在位置 `(n-1)*q` 的两侧样本上线性插值。
- 每个 `output_len > 1` 请求的 `TPOT = (finish-prefill_end)/(output_len-1)`；全场 TPOT 是这些请求的算术平均。只有单 token 输出时返回 `null`。
- 有成功请求时 `makespan = 最晚成功完成时刻 - 最早已发送请求到达时刻`（含失败请求的到达），`throughput = completed_requests/(makespan/1000)`；`makespan=0` 时吞吐为 `null`。无成功请求时 makespan 为 0，TTFT 均值/P50/P99、TPOT、吞吐均为 `null`。吞吐是实测完成速率，**不是**目标到达率 `arrival_rate`。

## 前缀复用与容量

- `total_input_tokens = sum(prompt_len)`，`ideal_hit_tokens = sum(ideal_reuse_tokens)`，`matched_cache_tokens = sum(matched_cache_tokens)`，`actual_hit_tokens = sum(actual_hit_tokens)`；`actual_prefix_hit_rate = actual_hit_tokens/total_input_tokens`。`target_ideal_prefix_hit_rate`、`constructed_ideal_prefix_hit_rate` 与量化误差来自负载计划；构造命中率与实际命中率不能互相替代。
- `actual_prefill_tokens` 只计成功请求的**首次**输入计算；`saved_prefill_tokens = total_input_tokens-actual_prefill_tokens = actual_hit_tokens`。必须满足 `actual_hit_tokens <= ideal_hit_tokens <= total_input_tokens` 和 `actual_prefill_tokens+actual_hit_tokens=total_input_tokens`。`replayed_prefill_tokens` 单独累计包括**最终失败请求**在内的已执行回撤重算，不混入首次命中率分母。所有全场 token 比率都以成功请求输入 token 总量加权，不取每请求比率的简单平均；全失败时这些计数和比率为零。
- `requests_with_hit_ratio` 为成功且实际复用 token 数大于零的请求比例；`requests_fully_cached_ratio` 为成功请求中**可缓存完整输入块数大于零且所有完整块都在缓存中匹配**的比例；即使完整块全命中，最后一个输入 token 仍需重算。
- 当前仅有 L1 实际命中；L2/L3 命中、restore 与 promotion/demotion 字段保留零值，不表示多级缓存已启用。结果还包含 L1 驱逐数、驻留峰值和结束快照的容量、驻留、活动、可淘汰、空闲和碎片；`free_blocks + resident_blocks = capacity_blocks`，`active_blocks + evictable_blocks = resident_blocks`。`reserved_future_blocks` 固定为 0，**空闲容量不扣除**未来输出估算。物理容量和逻辑上下文须分开计算，规则见 [kv-prefix-reuse.md](kv-prefix-reuse.md)。
