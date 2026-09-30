# 仿真内核设计

## 仿真层级

LLM Serving 系统级：请求到达 → 调度 → prefill/decode → continuous batching → KV cache 约束。单 GPU。

## DES 事件模型

全局仿真时钟和事件队列由 `SimulationCoordinator` 独占，队列承载三类事件：

- `ClientSend`：某个 session 轮次到达发送时刻；协调器调用 `ClientDriver.on_wakeup`，再以当前时刻调用 `engine.submit`；
- `EngineStep`：引擎迭代步。batch 非空时请求协调器在步末安排下一步；batch 为空时不压事件，时钟直接跳跃到下一个 `ClientSend`；
- `RequestDone`：请求完成结算、KV publish/release，并产生 `RequestCompletion`。

所有请求都属于 session，并通过 `ClientSend` 实际提交。未来发送事件可以位于协调器事件堆中，但完整请求只能在 `ClientSend` 触发后构造并交给 Engine。Engine 不拥有全局事件队列，其 waiting queue 只能包含已经调用 `engine.submit` 的请求。

同刻事件全序为 `(time, kind, seq)`：

```text
RequestDone / KV publish-release
→ RequestCompletion 路由与客户端状态更新
→ ClientSend / engine.submit
→ EngineStep
```

该顺序使引擎步能看到同刻释放的 KV 和同刻真正到达的请求。客户端收到父请求完成结果后注册 `finish_time + think_time` 的后继 `ClientSend`；这是模拟事件，不调用真实 `sleep()`。同刻同类事件按创建序号 `seq` 处理。

`EngineStep` 内执行：调度决策（准入）→ 推进 batch 内所有请求的 decode → 结算完成请求。一次事件推进整个 batch，使堆操作次数约等于客户端发送数、引擎迭代数与请求完成数之和，而非请求 token 步数。

**约束**：客户端和 Engine 不得各自维护可独立推进的时钟或事件堆；性能模型的步长计算必须是纯函数（无全局状态依赖），为后续内核替换（如 Rust）预留。

### 请求生命周期

- 准入步：执行 prefill，`prefill_start` = 步起点，`prefill_end` = 步终点，同时产出首 token。
- 其后每个 decode 步产出 1 token，全部 `output_len` 个 token 产出后完成；`output_len = 1` 的请求在准入步结束即完成。
- batch = 全部 KV 驻留请求。完成请求在步末释放，故每步开始时驻留请求均处于 decode 阶段。

## 调度与 KV cache 语义

- 完整块身份判等、lookup 和 paged KV block manager 始终开启，不提供关闭开关；
- 调度策略：FCFS，受 `max_batch_size` 与 L1 KV block 容量双重约束；
- 发送时客户端携带完整输入块 ID：继承父请求的连续完整块前缀，其余由 `chunk:{request_id}:{block_index}` 直接编码；准入时基于同一快照只读 `probe`，再以 `prepare_batch` / `commit` 原子准入；
- L1 容量不足时按确定性 LRU 淘汰 inactive block；active block 不可淘汰。L2/L3 降级属于后续扩展；
- 请求完成时解除活动引用，输入及已生成输出形成的可复用完整 block 保留为 evictable，未满的 partial block 释放；
- 无法通过驱逐获得容量时禁止准入；单请求在空 L1 中仍无法容纳时返回明确的不可执行/OOM 结果；
- 严格 FCFS：队头放不下时停止本步准入，不跳过队头去准入后续较小请求。

## 性能模型

系数均来自 config，单位 ms：

- prefill 耗时 = `prefill_c1 · prompt_tokens + prefill_c2 · prompt_tokens²`（`prefill_c2` 默认 0）
- decode 单步耗时 = `decode_d1 + decode_d2 · batch_total_tokens`

### 融合批步长

一个 `EngineStep` 内可同时存在新准入请求的 prefill 与在跑请求的 decode。步长取两者最大值：

```
step_time = max(Σ prefill 耗时（本步准入的全部请求）, decode 单步耗时)
```

语义：prefill 与 decode 在同一次迭代中融合执行（chunked-prefill 式），而非串行相加。`batch_total_tokens` 计入本步准入请求的 prompt token。

除排队外不建模其他延迟来源（无网络、无采样开销）。

## 工作负载接口

现行合成负载经 `build_plan(SyntheticWorkloadConfig, PrefixCacheConfig) -> WorkloadPlan` 构造 session 计划；旧长度入口的 `SyntheticSource` 仍产出 `Arrival`，通过 `plan_from_arrivals` 适配为独立根 session。

- `build_plan` 按固定种子规划根 session 泊松到达、prompt/output 长度、拓扑、think time 与理想复用块数，不物化 token ID 或预生成输出块 ID。配置的 `arrival_rate` 是所有轮次目标请求 QPS，根泊松速率为 `arrival_rate * num_sessions / num_requests`（session/秒），不保证有限场实测请求 QPS 精确达标；
- `num_requests` 表示所有 session 的总轮次数，`num_sessions` 表示根请求数；每个请求必须且只属于一个 session；
- 单轮 session 是最小合法 workload；
- `WorkloadPlan` 只交给 `ClientDriver`，生成器永不接触 Engine 或 KV cache；
- 客户端返回根请求的 `ClientWakeup`；父请求完成回调后才返回后继唤醒事件。协调器负责入队，并在发送时刻唤醒客户端构造请求，随后调用 `engine.submit`；
- 不得把未来完整请求或预计到达时间预先放入 Engine；
- `TraceSource` 后续同样解析为 `WorkloadPlan`，并经 `ClientDriver` 和 `SimulationCoordinator` 重放。
