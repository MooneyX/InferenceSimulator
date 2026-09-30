# 当前引擎与调度

本文描述现行单 Engine、一次性 Prefill 的步进和准入语义。全局时钟与事件全序见 [events.md](events.md)，session 的构造和发送见 [workload.md](workload.md)。目前同一步可以同时有新请求的 Prefill 和在跑请求的 Decode，但新请求不会跨步做 partial Prefill；F001 的完整 continuous batching 尚未实现。

## 请求生命周期

- 准入步：执行 prefill，`prefill_start` = 步起点，`prefill_end` = 步终点，同时产出首 token。
- 其后每个 decode 步产出 1 token，全部 `output_len` 个 token 产出后完成；`output_len = 1` 的请求在准入步结束即完成。
- batch = 全部 KV 驻留请求。完成请求在步末释放，故每步开始时驻留请求均处于 decode 阶段。

## 准入与步进

- `PrefixScheduler` 管理已到达请求的 FCFS 等待队列；`Engine` 维护运行请求，步首在 `max_batch_size` 和 L1 物理容量约束下只准入连续队头。队头暂时放不下时等待释放，不跳过较小的后续请求；单请求在空 L1 仍放不下则报不可执行/OOM。旧 `FCFSScheduler` 的 token 账本/强制超限路径不是统一仿真路径。
- 每步使用同一 L1 快照探测候选，并通过事务原子准入；同一步新发布的块对本步其他请求不可见。完整块的身份、事务、LRU 和释放规则由 [chunk-hash-encoding.md](chunk-hash-encoding.md) 与 [kv-prefix-reuse.md](kv-prefix-reuse.md) 定义。
- 准入请求在一个 step 内完成未命中输入的 Prefill，并产出首个 output token；步首已驻留的请求各推进一个 Decode token。步末结算已完成请求，未完成请求在后续 step 继续 Decode。步长由 [perf-model.md](perf-model.md) 定义。

除排队外，当前引擎不单独建模网络或采样开销；经验公式和步长口径见 [perf-model.md](perf-model.md)。
