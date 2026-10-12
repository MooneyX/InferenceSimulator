# 合成/Trace 负载与统一客户端

本文描述合成负载与 Trace 如何准备计划、决定理想复用量，以及由统一客户端在模拟时间内提交请求。完整块 ID 的编码规则见 [chunk-hash-encoding.md](chunk-hash-encoding.md)，事件堆的处理顺序见 [events.md](events.md)。

## 开环多轮模式（网页默认）

配置 `arrival_mode="open_loop"`、根轮 `prompt_len`、子轮 `turn_increment_len`、每轮 `output_len`，不设置 `session.num_sessions` 或 `session.think_time_ms`。`num_requests=N` 指全部投放次数；`arrival_rate=q` 指目标发送 QPS。`Client.initial_wakeups` 使用从 `seed` 独立派生的开环到达随机流，逐次采样均值 `1000/q` ms 的指数间隔并累加为到达时刻（首个请求也采样，不固定在 0）；因此固定请求数 N 的实际发送 QPS 只是有限样本的观测值，不保证等于 q。到达不依赖引擎吞吐；单请求无可测的投放间隔。完成结算先于同刻投放。若有完成且未续写的会话，按完成时刻、session ID 优先选一个子轮；否则创建新根会话。单会话一次只有一个请求在途，父轮失败不会生成后代，下一投放节拍可以新建会话。即使引擎已过载，仍会投放全部 N 次；完成吞吐并不保证等于 q。N 次后不再新投放，但引擎排空才结束。

根轮输入 `P_0=sample(prompt_len)`；子轮输入 `P_child=P_parent+O_parent+sample(turn_increment_len)`，其中 `O_parent` 是父轮实际生成长度。三类长度按 seed 派生独立随机流。超出本实例 KV 容量时，子轮请求仍按节拍发送并失败，该会话不再续写；当前模式不做上下文截断或摘要。前缀在父轮完成后依据完整页身份继承，理想复用量为 `B*min(floor((P_parent+O_parent-1)/B),floor((P_child-1)/B))` token。理想命中率是**构造结果**，不设目标；物理命中还取决于缓存驻留。根请求不跨会话共享前缀。运行中动态确定会话数与根占比。

## 有限会话闭环模式（`arrival_mode="session"`）

`build_plan` 消费 `SyntheticWorkloadConfig` 与一次性准备好的运行容量/页大小，一次性生成不可变 `WorkloadPlan`，不访问 Engine 或 KV cache。`num_requests=N` 是所有轮次数，`num_sessions=S` 是根轮次数，要求 `1 <= S <= N`。每个 session 至少一个根，首期为线性父子链：先给每个 session 分配 `floor(N/S)` 轮，多出的 `N % S` 轮按拓扑随机流的稳定顺序分配。根记录 `root_arrival_time_ms`，子轮次只记录父请求与 `think_time_ms`，不预先计算其到达时刻。`TurnPlan` 记录输入长度、截断后的目标 `output_len`、可见 `max_new_tokens` 和 `ideal_reuse_blocks`，不预生成块 ID。

闭环不配置目标请求 QPS；`session_rate` 直接设定根会话的泊松到达频率（会话/秒），首根同样采样指数间隔，不固定在时刻 0。子轮须等父轮实际完成后，再经过独立采样的平均思考间隔投放；实际请求 QPS 由所有已发送轮次的首末时间计算，而非由会话速率推导。`S=N` 时每个会话仅有一个根请求。根到达、根输入、子轮新增内容、输出、拓扑及思考时间分别由 `stable_derive(seed, domain)` 派生独立随机流，保持相同配置与种子的计划可复现。`session.think_time_ms` 是子轮思考时间的指数分布均值（毫秒），每个子轮独立采样一次；设为 `0` 时始终无等待。根轮不采样思考时间，子轮实际到达仍由父轮完成时刻决定。

`prompt_len`（仅根轮）、`turn_increment_len`（仅子轮）和 `output_len` 均使用 fixed、poisson 或 lognormal 长度分布，采样结果至少为 1 token；lognormal 配置中的 `mean` 表示原分布期望值。每个子轮的计划输入 `P_child=P_parent+O_parent+sample(turn_increment_len)`，其中 `O_parent` 已按父轮容量和输出上限截断；不截断或摘要超长上下文。对本轮输入长 `P` 计算可见上限 `M=max(0,min(配置 max_new_tokens, max_total_tokens-(ceil(P/B)+1)*B))`，实际输出目标 `O=min(采样输出长,M)`；引擎持有 `O`，调度器只看到 `M`。`M<1` 的请求仍在父轮成功后按计划发送并失败，子轮只有在父轮实际完成后才发送；失败父轮的后代标记跳过。Trace 回放向统一 `simulate` 入口注入 `TracePlan` 资源，回放设置来自 `SimConfig.workload.trace`，不使用合成长度分布。

## Trace 请求计划回放

`simserver.trace_processing` 可在本地逐行扫描 AgentX JSONL：每行一原始会话，展开 `subagent` 分组，汇总请求量、token 数、按原到达窗口计算的会话 QPS、历史块 set-hit 与最长连续前缀 prefix-hit。仅在同一主对话或同一 subagent 分组内，用原始 `t_parent + api_time_parent + think_time_child ≈ t_child`（容差 1 ms）推断唯一投放依赖；不以输入复用决定父子关系。无唯一父请求的请求保留相对本会话起点的到达偏移。报告 HTML 不参与扫描。

`TracePlan(version=1)` 存所选原始会话 ID、64-token 块大小和每次请求的本场 ID、会话编号、父 ID、相对到达偏移（ms）、子思考时间（ms）、原始输入输出长度及输入块 ID。上传前校验块 ID 的会话局部命名空间、位置/前驱一致性与父子拓扑；原始模型名不参与仿真，统一使用用户指定的引擎与性能配置。没有输出块 ID，不推断尚未计算的输出块。计划不按引擎容量截短输入或输出；零输出与超出输出可见上限在运行时记请求级失败。

`simulate(config, trace_plan=plan)` 从 `config.workload.trace` 读取 `mode`、`session_rate`、`seed` 和可选 `target_qps`，对每个所选原始会话采样均值 `1000/session_rate` ms 的指数间隔。开环先得到所有会话合并后的到达时刻 `t_i=会话起点+原始偏移`，父请求失败不阻止后续固定到达；可选正数 `target_qps=q` 时，按 `k=(N-1)*1000/[q*(max(t)-min(t))]`、`t'_i=min(t)+k*(t_i-min(t))` 缩放**整条**开环时间线。要求至少两条请求且原始时间跨度大于零；未配置时保留原到达间隔。缩放仅约束首末请求的发送 QPS，保留间隔比例，不保证均匀到达或完成吞吐。闭环不接受 `target_qps`，有父请求在仿真完成后加原始思考时间发送，无父分支按同一会话起点加偏移发送。不同分支共用原会话的实例 KV；跨原会话同数字块 ID 不可互相命中。trace 的 `constructed_ideal_prefix_hit_rate` 是整份所选计划中其他请求具有相同输入块 ID 的连续完整前缀**潜在上界**，可能包含当前请求到达后才出现的块，并非按历史可用块计算的原轨迹 prefix-hit；实际命中只取运行时已发布且尚在缓存中的连续页。

## 理想继承块

设完整块大小为 `B`，请求输入长 `P_i`，父轮实际建立的 KV 长度为 `L_parent(i)=P_parent+O_parent-1`（最后采样的 token 尚未建立 KV）。根不继承；子轮继承父轮已建立且包含在本轮输入中的**所有**完整页：

```text
K_i = min(floor((P_parent+O_parent-1)/B), floor((P_i-1)/B))
constructed_ideal_prefix_hit_rate = B*sum(K_i)/sum(P_i)
```

子轮新增长度至少 1 token，父轮输出为 0 时 `K_i=0`，该子轮不会实际发送。该比率是生成计划的结果，不再作为配置目标或量化分配值；理想继承只约束块身份，**不代表物理缓存已命中**。跨 session 公共前缀不支持：根请求不继承块，不同根请求生成不同的新块 ID；只有同 session 父子轮次可继承完整块。配置不提供跨 session 共享字段，额外字段无论取值均由 schema 拒绝。

## 客户端的发送契约

单个 `Client` 不读取缓存或引擎的 waiting/running 状态。`workload_mode` 区分合成/Trace，`arrival_mode` 区分开环/闭环，初始化后不切换；不使用策略子类。`initial_wakeups()` 返回初始投放，`initial_send_kind` 标明普通发送或合成开环 tick；所有投放均调用 `on_wakeup()`，返回 `(InferenceRequest, output_len)`，实际输出目标单独交给引擎，不放进请求。合成模式仅在发送时构造 `InferenceRequest`：根据完成父轮保存的请求身份和已建立长度，按需推导前 `ideal_reuse_blocks` 个**完整**页 ID 并传递为 `inherited_prefix_ids`；其余页 ID 由引擎按当前请求 ID 与页序号生成，不发送整条 `input_chunk_ids`。请求携带 `max_new_tokens`，不携带实际输出目标或未来到达时间。

`on_request_completed(RequestCompletion)` 统一结算在途请求。合成模式保存父轮请求身份与已建立长度 `prompt_len+generated_tokens-1`，Trace 只保存完成 ID，不重构原始块身份。闭环返回子轮 `finish_time_ms+think_time_ms` 的唤醒；合成开环只把会话放入待续堆，Trace 开环不改变固定时间线，两者均返回空元组。`on_request_aborted` 在闭环递归返回未发送后代的 `SkippedRequest`，开环返回空元组且不取消后续投放。客户端不预存未来完整请求、不调用真实 `sleep()`、不自行维护事件堆；协调器不按具体客户端类分派。
