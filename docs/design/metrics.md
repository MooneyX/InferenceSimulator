# 当前指标计算口径

`metrics.compute_metrics` 汇总已完成请求的区间记录和 L1 缓存快照；字段定义由 `core/src/simcore/schemas.py` 提供，HTTP 传输见 [api.md](api.md)。下述单位除吞吐（请求/秒）、块数和 token 数外均为 ms。

## 时延与吞吐

- 每请求 `TTFT = prefill_end - arrival`，返回均值、P50、P99。分位数先升序排序，再在位置 `(n-1)*q` 的两侧样本上线性插值。
- 每个 `output_len > 1` 请求的 `TPOT = (finish-prefill_end)/(output_len-1)`；全场 TPOT 是这些请求的算术平均。只有单 token 输出时返回 `null`。
- `makespan = max(finish)-min(arrival)`；`throughput = completed_requests/(makespan/1000)`。`makespan=0` 时吞吐为 `null`。吞吐是实测完成速率，**不是**配置中的目标到达率 `arrival_rate`。

## 前缀复用与容量

- `total_input_tokens = sum(prompt_len)`，`ideal_hit_tokens = sum(ideal_reuse_tokens)`，`matched_cache_tokens = sum(matched_cache_tokens)`，`actual_hit_tokens = sum(actual_hit_tokens)`；`actual_prefix_hit_rate = actual_hit_tokens/total_input_tokens`。`target_ideal_prefix_hit_rate`、`constructed_ideal_prefix_hit_rate` 与量化误差来自负载计划；构造命中率与实际命中率不能互相替代。
- `actual_prefill_tokens = sum(actual_prefill_tokens)`，`saved_prefill_tokens = total_input_tokens-actual_prefill_tokens`。必须满足 `actual_hit_tokens <= ideal_hit_tokens <= total_input_tokens` 和 `actual_prefill_tokens+actual_hit_tokens=total_input_tokens`。所有全场 token 比率都以输入 token 总量加权，不取每请求比率的简单平均。
- `requests_with_hit_ratio` 为实际复用 token 数大于零的请求比例；`requests_fully_cached_ratio` 为**可缓存完整输入块数大于零且所有完整块都在缓存中匹配**的请求比例，按请求数为分母；即使完整块全命中，最后一个输入 token 仍需重算。
- 当前仅有 L1 实际命中；L2/L3 命中字段预留为零，不表示多级缓存已启用。结果还包含 L1 驱逐数、驻留峰值和结束快照的容量、驻留、活动、可淘汰、空闲、未来预留及碎片；空闲容量扣除未来预留。物理容量和逻辑上下文必须分开计算，规则见 [kv-prefix-reuse.md](kv-prefix-reuse.md)。
