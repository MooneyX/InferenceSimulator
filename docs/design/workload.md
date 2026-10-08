# 合成负载与 session 客户端

本文只描述现行合成负载如何规划轮次、决定理想继承量，以及在模拟时间内提交请求。完整块 ID 的编码规则见 [chunk-hash-encoding.md](chunk-hash-encoding.md)，事件堆的处理顺序见 [events.md](events.md)。

## 计划与到达

`build_plan(SyntheticWorkloadConfig, EngineConfig) -> WorkloadPlan` 一次性生成不可变计划，不访问 Engine 或 KV cache。`num_requests=N` 是所有轮次数，`num_sessions=S` 是根轮次数，要求 `1 <= S <= N`。每个 session 至少一个根，首期为线性父子链：先给每个 session 分配 `floor(N/S)` 轮，多出的 `N % S` 轮按拓扑随机流的稳定顺序分配。根记录 `root_arrival_time_ms`，子轮次只记录父请求与 `think_time_ms`，不预先计算其到达时刻。`TurnPlan` 记录 `prompt_len`、`output_len` 和 `ideal_reuse_blocks`，不预生成输出块 ID。

`arrival_rate=q` 的单位是所有轮次的目标请求/秒，根的泊松速率为 `q*S/N` session/秒；首根同样采样指数间隔，不固定在时刻 0。这个换算仅给出持续注入的长期名义速率，单次有限仿真的实际到达率和完成吞吐不保证等于 `q`。`S=N` 时退化为单轮请求到达口径。根到达、长度、拓扑、思考时间及理想命中分配分别由 `stable_derive(seed, domain)` 派生独立随机流，保持相同配置与种子的计划可复现。`session.think_time_ms` 是子轮思考时间的指数分布均值（毫秒），每个子轮独立采样一次；设为 `0` 时始终无等待。根轮不采样思考时间，子轮实际到达仍由父轮完成时刻决定。

`prompt_len`、`output_len` 使用 fixed、poisson 或 lognormal 长度分布，采样结果至少为 1 token；lognormal 配置中的 `mean` 表示原分布期望值。合成负载由 `build_plan` 构造 session 计划；手写 `Arrival` 到达序列可经 `run_engine` 调用 `plan_from_arrivals` 适配为各自独立的根 session。`trace` 模式目前未实现。

## 理想继承块分配

设完整块大小为 `B`，请求输入长 `P_i`，父轮次的完整逻辑上下文长 `L_parent(i)=P_parent+O_parent`。根的上限为零；子轮次的最大可继承块数为

```text
M_i = min(floor(L_parent(i)/B), floor((P_i-1)/B))
D = sum(P_i); M = sum(M_i); r_max = B*M/D
```

`target_ideal_prefix_hit_rate > r_max` 明确报不可行错误并给出 `r_max`。在 `[0,M]` 中取使 `abs(B*K/D-target)` 最小的整数总块数 `K`，距离相等时取较小值；再按 `K*M_i/M` 先取整、按余数和独立随机流的稳定顺序分配余量。输出构造命中率及量化误差；理想继承量仅约束块身份的构造，**不代表物理缓存已命中**。跨 session 公共前缀不支持：根请求不继承块，不同根请求生成不同的新块 ID；只有同 session 父子轮次可继承完整块。配置不提供跨 session 共享字段，额外字段无论取值均由 schema 拒绝。

## 客户端的发送契约

`ClientDriver` 独占 session 已完成块 ID 上下文，不读取缓存或引擎的 waiting/running 状态。`initial_wakeups()` 返回根轮次唤醒；`on_wakeup()` 只在发送时刻构造 `InferenceRequest`：对子轮次复制父请求前 `ideal_reuse_blocks` 个**完整**块 ID，之后生成本请求的新块 ID；不复制未满尾块。每条请求的 `input_chunk_ids` 恰有 `floor(P_i/B)` 项。

父轮次完成后，`on_request_completed(RequestCompletion)` 根据已知输入与计划输出长度重建完整输出块 ID 上下文，再登记子轮次 `finish_time_ms + think_time_ms` 的唤醒。完成消息只携带请求 ID 和完成时间；当前合成引擎严格生成计划中的输出长度。客户端不预存未来完整请求、不调用真实 `sleep()`、不自行维护事件堆。
