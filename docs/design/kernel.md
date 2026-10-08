# 当前引擎与调度

本文描述现行单 Engine、一次性 Prefill 的步进和准入语义。全局时钟与事件全序见 [events.md](events.md)，session 的构造和发送见 [workload.md](workload.md)。目前同一步可以同时有新请求的 Prefill 和在跑请求的 Decode，但新请求不会跨步做 partial Prefill；F001 的完整 continuous batching 尚未实现。

## 请求生命周期

- 准入步：执行 prefill，`prefill_start` = 步起点，`prefill_end` = 步终点，同时产出首 token。
- 其后每个 decode 步产出 1 token，全部 `output_len` 个 token 产出后完成；`output_len = 1` 的请求在准入步结束即完成。
- batch = 全部 KV 驻留请求。完成请求在步末释放，故每步开始时驻留请求均处于 decode 阶段。

## 准入与步进

- `F001Scheduler` 管理已到达请求的 FCFS 等待队列；`Engine` 维护运行请求，步首在 `max_running_requests` 和 L1 物理容量约束下只准入连续队头。队头暂时放不下时等待或回撤，不跳过较小的后续请求；单请求无法执行时记录失败，不允许强制超预算准入。
- 调度器逐队头探测并临时保护已命中的完整页，按实际页容量选择候选；调度结果由 `AdmissionBatch` 表达。执行层确认准入后逐请求扩展 KV，同一步新发布的页对本步其他请求不可见。完整页的身份、容量、LRU 和释放规则见 [chunk-hash-encoding.md](chunk-hash-encoding.md) 与 [kv-prefix-reuse.md](kv-prefix-reuse.md)。
- 准入请求在一个 step 内完成未命中输入的 Prefill，并产出首个 output token；步首已驻留的请求各推进一个 Decode token。步末结算已完成请求，未完成请求在后续 step 继续 Decode。步长由 [perf-model.md](perf-model.md) 定义。

除排队外，当前引擎不单独建模网络或采样开销；经验公式和步长口径见 [perf-model.md](perf-model.md)。
