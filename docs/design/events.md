# 离散事件时钟与协调器

`Coordinator`（`SimulationCoordinator` 是其别名）独占仿真时钟、事件堆与事件创建序号；`ClientDriver` 和 `Engine` 不得自行推进时钟。事件以 `(time, kind, ref, seq)` 排序：同刻先按请求 ID 完成结算与 KV 释放，再回调客户端；完成时立刻唤醒的零思考时间子轮优先于普通发送，最后运行引擎步。同类事件先按引用的请求/轮次 ID、再按创建序号排序：

```text
RequestDone → ClientCompletion → ClientChildSend → ClientSend → OpenLoopTick → EngineStep
```

- `ClientSend`：闭环模式调用 `client.on_wakeup`，再以当前时间 `engine.submit`。开环模式由 `OpenLoopTick` 在固定间隔触发，优先续写已完成会话，否则新建根，并在提交前登记其输出目标。Engine 的 waiting 队列只含已真正发送的请求；未来事件可在堆中，但未来完整请求不能提前进入 Engine。
- `EngineStep`：引擎产生本步时长和完成请求 ID；协调器把完成事件排在步末，若仍有引擎工作则安排下一步。同刻最多挂起一个待执行步骤；忙碌时不在上一步结束前开启下一步，空闲期跳到下个客户端事件。请求级 abort 后由客户端递归标记尚未发送的后代 skipped。
- `RequestDone`：引擎完成结算、发布剩余完整页并释放 KV，生成携带 ID、完成时间及 `generated_tokens` 的 `RequestCompletion`；同刻的 `ClientCompletion` 再回调客户端，为零 think-time 子轮产生 `ClientChildSend`，因此它可以看到父请求已发布且释放的完整页。

客户端/引擎间只传不可变消息；协调器不构造块身份、不参与调度或缓存决策。内部 `EventLog` 可记录到达、步骤与完成事件，但默认仿真关闭记录且 HTTP 响应不暴露该流。模块所有权和数据类型见 [architecture.md](architecture.md)，当前每步计算语义见 [kernel.md](kernel.md)。
