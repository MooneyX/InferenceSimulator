# 当前指标计算口径

`metrics.compute_metrics` 以**已完成请求**的区间记录计算时延、分层前缀命中和首次 Prefill，并结合全场回撤重算量、L1/L2/L3 缓存快照与迁移、restore 计数；已发送但失败和未发送而跳过的请求不进入成功请求指标分母。多个实例共享一条到达时间线：请求记录和逐实例指标标注 `instance_id`，全场延迟、分位数和吞吐从合并请求重新计算；本地 L1/L2 容量和末态驻留按实例求和，迁移量按各实例实际操作求和。全场 `l1/l2_peak_resident_blocks` 为各实例峰值之和，是容量规划上界，**不是同时峰值**。共享 L3 容量、末态驻留、实际峰值和淘汰仅从共享池读取一次；逐实例对应七个 L3 池统计字段均为 `null`，命中及传输量仍归实例。字段定义由 `core/src/simcore/schemas.py` 提供，HTTP 传输见 [api.md](api.md)。下述单位除吞吐（请求/秒）、块数、token 数及传输字节数外均为 ms。

`compute_metrics` 的计划、页大小、负载模式、三级快照及发送/迁移/重算统计均为必填输入；全场关闭的缓存层仍提供零容量快照；逐实例的共享 L3 快照传 `None`，对应七个池状态字段均为 `null`。成功请求的 `RequestRecord.actual_prefill_tokens` 是必填非负整数，不再从缺省值回算。

## 时延与吞吐

- 每请求 `TTFT = prefill_end - arrival`，返回均值、P50、P99。分位数先升序排序，再在位置 `(n-1)*q` 的两侧样本上线性插值。
- 每个 `output_len > 1` 请求的 `TPOT = (finish-prefill_end)/(output_len-1)`；全场 TPOT 是这些请求的算术平均，`tpot_p99` 为逐请求 TPOT 的 P99。只有单 token 输出时两者返回 `null`。
- ITL（相邻输出 token 间隔）以每个间隔为一个样本，对全部样本统计 `itl_mean`、`itl_p50`、`itl_p99`、`itl_max`，与 SGLang `bench_serving.calculate_metrics` 拼接全部请求 `itl` 列表的口径一致（参考 SGLang `v0.4.6` `python/sglang/bench_serving.py`），输出越长的请求权重越大。仿真中首个输出 token 在 Prefill 步结束时产生，此后每个 Decode 步结束产生一个 token，间隔为本步终点减该请求上一 token 时刻；回撤后的排队与重算等待并入恢复后首个间隔，PD 互斥时被 Prefill 步挡住的等待同理。因此每请求间隔之和等于 `finish-prefill_end`，`itl_mean = Σ(finish-prefill_end)/Σ(output_len-1)`。同一步内上一 token 时刻相同的请求间隔相同，引擎按 `(间隔, 样本数)` 分组记录，样本量与步数同阶；分位数按组权重计算，与展开样本后的线性插值严格相等。与 SGLang 的差异：已产生 Decode 间隔后因 `decode_kv_exhausted` 失败的请求，其间隔仍留在样本中（该失败只在单请求也无法继续时发生）。无样本时四项为 `null`。
- 有成功请求时 `makespan = 最晚成功完成时刻 - 最早已发送请求到达时刻`（含失败请求的到达），`throughput = completed_requests/(makespan/1000)`；`makespan=0` 时吞吐为 `null`。无成功请求时 makespan 为 0，TTFT 均值/P50/P99、TPOT、ITL、吞吐均为 `null`。吞吐是实测完成速率，**不是**目标到达率 `workload.target_qps`。

开环模式的 `arrival_qps` 是已投放 N 个请求（包括发送后失败者）的 `(N-1)*1000/(最后投放时刻-首次投放时刻)`；泊松到达下它通常不等于配置的期望速率 q。N=1 或首末时刻相同时为 `null`。闭环模式同样报告已发送请求（不含因父轮失败而跳过的后代）的首末到达观测 QPS；它不是 `session_rate`，也不保证达到任何目标请求 QPS。`num_sessions` 为创建的根会话数，`root_request_ratio=num_sessions/N`，两者计入发送后失败的根。`mean_prompt_tokens`、`mean_output_tokens` 则只对已完成请求统计；全失败时为 `null`。目标 QPS、发送 QPS 与完成吞吐的时间分母不同，不可混用。

## 前缀复用与容量

- `total_input_tokens = sum(prompt_len)`，`ideal_hit_tokens = sum(ideal_reuse_tokens)`，`matched_cache_tokens = sum(matched_cache_tokens)`，`actual_hit_tokens = sum(actual_hit_tokens)`；`actual_prefix_hit_rate = actual_hit_tokens/total_input_tokens`。`l1/l2/l3_prefix_hit_rate` 分别为对应层的 `hit_tokens/total_input_tokens`，三者之和等于 `actual_prefix_hit_rate`；它们是各层对成功请求输入的命中贡献率，不是各层 lookup 成功率。无成功请求时四个比率均为 0，逐实例也用自身成功请求输入 token 作分母。两种模式均不配置理想命中目标，`constructed_ideal_prefix_hit_rate` 是全部已构造请求（含失败/跳过轮次的计划）的可继承完整页 token / 全部计划输入 token。`child_ideal_prefix_hit_rate` 与 `child_actual_prefix_hit_rate` 只对已完成子轮统计，分母是这些子轮的输入 token 总数；无成功子轮时为 `null`。构造率与成功请求实际率的样本范围不同，不应将两者之差简单解释为 KV 淘汰损失。
- `actual_prefill_tokens` 只计成功请求的**首次**输入计算；`saved_prefill_tokens = total_input_tokens-actual_prefill_tokens = actual_hit_tokens`。必须满足 `actual_hit_tokens <= ideal_hit_tokens <= total_input_tokens` 和 `actual_prefill_tokens+actual_hit_tokens=total_input_tokens`。`replayed_prefill_tokens` 单独累计包括**最终失败请求**在内的已执行回撤重算，不混入首次命中率分母。所有全场 token 比率都以成功请求输入 token 总量加权，不取每请求比率的简单平均；全失败时这些计数和比率为零。
- `requests_with_hit_ratio` 为成功且实际复用 token 数大于零的请求比例；`requests_fully_cached_ratio` 为成功请求中**可缓存完整输入块数大于零且所有完整块都在缓存中匹配**的比例；即使完整块全命中，最后一个输入 token 仍需重算。
- 每个成功请求的 `l1_hit_tokens+l2_hit_tokens+l3_hit_tokens=actual_hit_tokens`，全场分层命中是成功请求逐层之和；归因以**首次准入实际跳过**的输入页来源为准，已完成 L3 预取后实际复用的页仍归 L3，回撤重算只进入 `replayed_prefill_tokens`。请求共享同一低层页可各自命中，但 `promotions` 按唯一 ID 实际上移的迁移次数计（L3→L2 与 L2→L1 分别计一次，两层结构的 L3→L1 计一次）；`demotions` 按逐级下移次数计，`l1_evictions` 包含移至 L2（两层结构为 L3）的页。`restore_bytes` 为**已完成** L2→L1 提升页与 L3→L2 预取页的字节数之和；`restore_time_ms` 为两条链路的服务时长之和：L3→L2 每批（至多 128 页）计固定延迟加传输量/速率（含 L3 来源的降速与抖动），L2→L1 每步逐层计 `l2_latency_ms+本批该层字节/速率` 之和，不含链路排队、lookup 或 write-back。传输与 GPU 计算可重叠，因此该服务量不等于 makespan 增量；解析结果的 `analysis.restore_ms` 使用同一口径，L2→L1 回载及两层结构 L3→L1 读取未被计算掩盖、拉长 GPU 步长的部分单列为 `analysis.restore_exposed_ms`。
- 每层返回容量、驻留、活动、可淘汰、空闲、驻留峰值与逐出次数；`free_blocks + resident_blocks = capacity_blocks`，`active_blocks + evictable_blocks = resident_blocks`，峰值不小于末态驻留且不超过容量。在途传输的目的占位计入驻留及活动，来源页在完成迁入前仍占原层容量且受保护；占位不是额外可命中的缓存副本。L1 还返回碎片及固定为 0 的 `reserved_future_blocks`，**空闲容量不扣除**未来输出估算，但会扣除实际传输占位；零容量层的计数全部为零。即使无成功请求，分层容量、驱逐、迁移及恢复量仍反映实际运行。物理容量和逻辑上下文须分开计算，规则见 [kv-prefix-reuse.md](kv-prefix-reuse.md)。

## 两层结构 L3 读取停顿

用于观察 SSD 直连 HBM（L2 关闭、L3 启用）时 L3 读取及其波动对 GPU 步的影响；三层结构 L2→L1 回载不计入。

- 停顿：某个含 L3→L1 读取的 GPU 步中，读取未被计算掩盖、拉长步长的时长。整批读取为“末批读完时刻 − 步起点”；逐层读取为“流水后步长 − 纯计算耗时”。两层结构下 `analysis.restore_exposed_ms` 即其总和。
- `l3_read_steps` 为含 L3 读取的步数，`l3_stall_ms` 为停顿之和，`l3_stall_p50/p99/max` 为这些步的停顿分布（无读取步时为 `null`），`l3_stall_fraction = l3_stall_ms / 全部 GPU 步长之和`（未执行任何步时为 `null`）。多实例全场合并各实例逐步样本，占比分母为各实例步长之和，不使用 makespan，避免并行实例被重复计时。
- 停顿属物理执行开销，无成功请求时仍统计。单步停顿会同时拉长同批全部 Decode 请求的间隔，因此应与 `itl_p99`/`itl_max` 对照观察。
