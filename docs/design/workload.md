# 合成负载与 session 客户端

本文只描述现行合成负载如何规划轮次、决定理想继承量，以及在模拟时间内提交请求。完整块 ID 的编码规则见 [chunk-hash-encoding.md](chunk-hash-encoding.md)，事件堆的处理顺序见 [events.md](events.md)。

## 计划与到达

`build_plan(SyntheticWorkloadConfig, EngineConfig) -> WorkloadPlan` 一次性生成不可变计划，不访问 Engine 或 KV cache。`num_requests=N` 是所有轮次数，`num_sessions=S` 是根轮次数，要求 `1 <= S <= N`。每个 session 至少一个根，首期为线性父子链：先给每个 session 分配 `floor(N/S)` 轮，多出的 `N % S` 轮按拓扑随机流的稳定顺序分配。根记录 `root_arrival_time_ms`，子轮次只记录父请求与 `think_time_ms`，不预先计算其到达时刻。`TurnPlan` 记录输入长度、截断后的目标 `output_len`、可见 `max_new_tokens` 和 `ideal_reuse_blocks`，不预生成块 ID。

`arrival_rate=q` 的单位是所有轮次的目标请求/秒，根的泊松速率为 `q*S/N` session/秒；首根同样采样指数间隔，不固定在时刻 0。这个换算仅给出持续注入的长期名义速率，单次有限仿真的实际到达率和完成吞吐不保证等于 `q`。`S=N` 时退化为单轮请求到达口径。根到达、长度、拓扑、思考时间及理想命中分配分别由 `stable_derive(seed, domain)` 派生独立随机流，保持相同配置与种子的计划可复现。`session.think_time_ms` 是子轮思考时间的指数分布均值（毫秒），每个子轮独立采样一次；设为 `0` 时始终无等待。根轮不采样思考时间，子轮实际到达仍由父轮完成时刻决定。

`prompt_len`、`output_len` 使用 fixed、poisson 或 lognormal 长度分布，采样结果至少为 1 token；lognormal 配置中的 `mean` 表示原分布期望值。对输入长 `P` 先计算可见上限 `M=max(0,min(配置 max_new_tokens, max_total_tokens-(ceil(P/B)+1)*B))`，实际输出目标 `O=min(采样输出长,M)`；引擎持有 `O`，调度器只看到 `M`。`M<1` 的请求在实际发送时失败，子轮只有在父轮完成后才发送。手写 `Arrival.max_new_tokens` 默认 128，经 `run_engine`/`plan_from_arrivals` 适配为各自独立的根 session，同样截断。`trace` 模式目前未实现。

## 理想继承块分配

设完整块大小为 `B`，请求输入长 `P_i`，父轮实际建立的 KV 长度为 `L_parent(i)=P_parent+O_parent-1`（最后采样的 token 尚未建立 KV）。根的上限为零；子轮次的最大可继承块数为

```text
M_i = min(floor((P_parent+O_parent-1)/B), floor((P_i-1)/B))
D = sum(P_i); M = sum(M_i); r_max = B*M/D
```

`target_ideal_prefix_hit_rate > r_max` 明确报不可行错误并给出 `r_max`。在 `[0,M]` 中取使 `abs(B*K/D-target)` 最小的整数总块数 `K`，距离相等时取较小值；再按 `K*M_i/M` 先取整、按余数和独立随机流的稳定顺序分配余量。输出构造命中率及量化误差；理想继承量仅约束块身份的构造，**不代表物理缓存已命中**。跨 session 公共前缀不支持：根请求不继承块，不同根请求生成不同的新块 ID；只有同 session 父子轮次可继承完整块。配置不提供跨 session 共享字段，额外字段无论取值均由 schema 拒绝。

## 客户端的发送契约

`ClientDriver` 不读取缓存或引擎的 waiting/running 状态。`initial_wakeups()` 返回根轮唤醒；`on_wakeup()` 仅在发送时构造 `InferenceRequest`：根据完成父轮保存的请求身份和已建立长度，按需推导前 `ideal_reuse_blocks` 个**完整**页 ID 并传递为 `inherited_prefix_ids`；其余页 ID 由引擎按当前请求 ID 与页序号生成，不发送整条 `input_chunk_ids`。请求携带 `max_new_tokens`，不携带实际输出目标或未来到达时间。

`on_request_completed(RequestCompletion)` 使用完成消息中的 `generated_tokens` 保存父轮已建立长度 `prompt_len+generated_tokens-1`，再登记子轮 `finish_time_ms+think_time_ms` 的唤醒；父轮失败时递归将尚未发送的后代记为 `skipped`，不触发发送。客户端不预存未来完整请求、不调用真实 `sleep()`、不自行维护事件堆。
