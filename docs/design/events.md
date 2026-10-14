# 离散事件时钟与协调器

`Coordinator` 独占仿真时钟、事件堆与事件创建序号；`Client` 和 `Engine` 不得自行推进时钟。事件以 `(time, kind, ref, seq)` 排序：同刻先提交 GPU 步结果，再按请求 ID 逐个完成结算、KV 释放与客户端回调，随后结算 L3 查询与页传输；合成/Trace、开环/闭环的所有发送统一使用 `ClientSend`，同刻按请求 ID 排序，最后运行引擎步。GPU 步与步完成事件的 `ref` 为实例 ID，传输完成事件为全局传输 ID，其余为请求/轮次 ID；同类事件先按 `ref`、再按创建序号排序：

```text
StepDone → RequestDone → L3LookupDone → L3L2Done → L2L1Done → ClientSend → EngineStep
```

- `ClientSend`：所有客户端模式统一使用的发送事件，业务载荷仅为发送时间和请求 ID，事件堆仍使用 `(time, kind, ref, seq)` 排序。客户端提供初始唤醒和闭环完成回调产生的后续唤醒，协调器统一入堆。触发时调用 `client.on_wakeup`，获取携带 `output_len` 的请求；Router 选择实例后，由 Coordinator 以当前时间向对应 `engine.submit`，不再单独登记输出目标。合成闭环和 Trace 从计划构造请求；合成开环只预排独立泊松到达时间和请求 ID，不等待上个请求完成，在发送时优先续写已完成会话，否则新建根，再动态采样请求内容。Engine 的 waiting 队列只含已真正发送的请求；未来事件可在堆中，但未来完整请求不能提前进入 Engine。
- `EngineStep`：协调器调用 `engine.start_step`；引擎选批并按需发起回载，回载未就绪且没有可计算工作时不产生 GPU 步。就绪批次交给内部 `_execute_step`（`selection` 必填）计算时长；协调器只安排步末的 `StepDone`，不在步首公开新前缀或结算请求。每个实例同刻最多挂起一个待执行步骤，GPU 忙碌时不提前开启下一步；独立查询和传输事件仍可推进。请求级 abort 后由客户端递归标记尚未发送的后代 skipped。
- `StepDone`：调用 `engine.finish_step`，发布本步 Prefill 建立的完整页，再为本步完成请求安排同刻 `RequestDone`，并按就绪状态安排下一步。
- `L3LookupDone` / `L3L2Done` / `L2L1Done`：分别完成查询、L3→L2 预取和 L2→L1 回载，由引擎处理状态与缓存迁入，协调器派发事件并唤醒可执行工作。同实例每条传输链路按页串行，传输不计入 GPU 步长。事件每次入堆后按全序重新取最小项；发送产生的零延迟 L3 查询先于剩余同刻发送处理，并可立即预留预取容量及链路位置，不存在“本时刻全部发送完再查询”的屏障。
- `RequestDone`：调用引擎 `complete` 完成结算，由缓存 `finish` 发布剩余完整页并释放 KV，`complete` 不返回值；随后在当前事件内立即调用 `client.on_request_completed(request_id, now_ms)`，客户端按已发送请求的固定 `output_len` 计算完整 KV 长度，将返回的唤醒统一安排为 `ClientSend`，不直接发送子轮。同刻请求按 ID 逐个结算并回调，不等待其他请求结算后才回调。零 think-time 子轮与同刻普通发送按轮次 ID 排序；父请求的结算、发布和释放仍先于所有发送，因此子轮可以观察父轮的完成发布。

客户端/引擎间只传不可变消息；协调器不构造块身份、不参与调度或缓存决策。调度事件仅用于推进仿真，不保存逐步诊断日志；结果通过请求区间和汇总指标提供。模块所有权和数据类型见 [architecture.md](architecture.md)，当前每步计算语义见 [kernel.md](kernel.md)。
