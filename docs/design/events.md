# 离散事件时钟与协调器

`Coordinator` 独占仿真时钟、事件堆与事件创建序号；`ClientDriver` 和 `Engine` 不得自行推进时钟。事件以 `(time, kind, ref, seq)` 排序：同刻先提交 GPU 步结果，再按请求 ID 完成结算与 KV 释放、回调客户端，随后结算 L3 查询与页传输；零思考时间子轮优先于普通发送，最后运行引擎步。GPU 步与步完成事件的 `ref` 为实例 ID，传输完成事件为全局传输 ID，其余为请求/轮次 ID；同类事件先按 `ref`、再按创建序号排序：

```text
StepDone → RequestDone → ClientCompletion → L3LookupDone → L3L2Done → L2L1Done
         → ClientChildSend → ClientSend → OpenLoopTick → EngineStep
```

- `ClientSend`：计划客户端（合成闭环及 Trace，包括 Trace 开环）在计划唤醒事件调用 `client.on_wakeup`；Router 选择实例后，由 Coordinator 登记输出目标并以当前时间向对应 `engine.submit`。Engine 的 waiting 队列只含已真正发送的请求；未来事件可在堆中，但未来完整请求不能提前进入 Engine。
- `OpenLoopTick`：仅用于合成开环的独立泊松投放时钟，间隔由指数分布采样，不是固定间隔，也不等待上个请求完成。投放时优先续写已完成会话，否则新建根；Coordinator 路由请求并在提交前登记其输出目标。
- `EngineStep`：协调器调用 `engine.start_step`；引擎选批并按需发起回载，回载未就绪且没有可计算工作时不产生 GPU 步。就绪批次交给内部 `_execute_step`（`selection` 必填）计算时长；协调器只安排步末的 `StepDone`，不在步首公开新前缀或结算请求。每个实例同刻最多挂起一个待执行步骤，GPU 忙碌时不提前开启下一步；独立查询和传输事件仍可推进。请求级 abort 后由客户端递归标记尚未发送的后代 skipped。
- `StepDone`：调用 `engine.finish_step`，发布本步 Prefill 建立的完整页，再为本步完成请求安排同刻 `RequestDone`，并按就绪状态安排下一步。
- `L3LookupDone` / `L3L2Done` / `L2L1Done`：分别完成查询、L3→L2 预取和 L2→L1 回载，由引擎处理状态与缓存迁入，协调器派发事件并唤醒可执行工作。同实例每条传输链路按页串行，传输不计入 GPU 步长。
- `RequestDone`：调用引擎 `complete` 完成结算，由缓存 `finish` 发布剩余完整页并释放 KV，生成携带 ID、完成时间及 `generated_tokens` 的 `RequestCompletion`；同刻的 `ClientCompletion` 再回调客户端，为零 think-time 子轮产生 `ClientChildSend`，因此它可以看到父请求已发布且释放的完整页。

客户端/引擎间只传不可变消息；协调器不构造块身份、不参与调度或缓存决策。调度事件仅用于推进仿真，不保存逐步诊断日志；结果通过请求区间和汇总指标提供。模块所有权和数据类型见 [architecture.md](architecture.md)，当前每步计算语义见 [kernel.md](kernel.md)。
