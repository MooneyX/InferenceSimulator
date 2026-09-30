# 离散事件时钟与协调器

`Coordinator`（`SimulationCoordinator` 是其别名）独占仿真时钟、事件堆与事件创建序号；`ClientDriver` 和 `Engine` 不得自行推进时钟。事件以 `(time, kind, seq)` 排序，同刻先完成结算与 KV 释放，再路由客户端完成消息、发送请求，最后运行引擎步：

```text
RequestDone → ClientCompletion → ClientSend → EngineStep
```

- `ClientSend`：协调器调用 `client.on_wakeup`，再以当前时间 `engine.submit`。Engine 的 waiting 队列只含已真正发送的请求；未来事件可在堆中，但未来完整请求不能提前进入 Engine。
- `EngineStep`：引擎产生本步时长和完成请求 ID；协调器把完成事件排在步末，若仍有引擎工作则安排下一步。忙碌时不在上一步结束前开启下一步；空闲期跳到下个客户端事件。
- `RequestDone`：引擎完成结算与 KV release，生成只含 ID 和完成时间的 `RequestCompletion`；同刻的 `ClientCompletion` 再回调客户端，产生子轮次的 `ClientSend`。因此零 think time 的子轮次可以看到父请求已发布、释放的完整块。

同刻同类事件按单调创建序号稳定处理。客户端/引擎间只传不可变消息；协调器不构造块身份、不参与调度或缓存决策。模块所有权和数据类型见 [architecture.md](architecture.md)，当前每步计算语义见 [kernel.md](kernel.md)。
