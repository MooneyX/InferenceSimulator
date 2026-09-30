# 调度与引擎实施计划（F001）

**状态：待实现。** 现行引擎已有动态准入与 batch 大小变化，但仍一次完成剩余输入的 Prefill，并使用 Prefill/Decode 耗时取最大值的近似；完整 token-level continuous batching、chunked prefill、P/D 互斥、retraction 都尚未接通。现行契约见 [当前引擎设计](../design/kernel.md)、[L1 接口](../design/kv-prefix-reuse.md) 和 [经验模型](../design/perf-model.md)。此计划只负责 `schemas` 中的调度字段、`scheduler`、`engine` 及其与缓存的调用编排；性能成本后端归 [性能模型计划](perf-model.md)。

## 目标与范围

首期复现固定 SGLang 版本的普通 decoder-only、单实例 unified scheduler 默认路径：Prefill-first，**每个 step 只执行 Prefill 或 Decode**，不混合。可同时驻留 waiting、partial-prefill 和 decoding 请求，但单步 P/D 互斥；不是在不同设备上做 Prefill/Decode 分离。非默认 mixed chunk、speculative、跨 Engine 路由另行规划。行为基线对应 `refs/sglang/` 提交 `0318a8d0af86ba14a05ca093aa43fafe446da23e`；不能将 vLLM V1 的 running-first mixed batching 当成此计划的默认行为。

## 配置和状态接口（CB1 冻结）

字段分别约束不同资源：`max_running_requests` 限制驻留请求数，`max_queued_requests` 限制 waiting，`max_total_tokens` 限制 KV pool；`max_prefill_tokens`、`prefill_max_requests` 限制本步 P batch，`chunked_prefill_size` 限制单次 chunk，`-1` 表示关闭 chunking。首期 `schedule_policy=fcfs`、`enable_mixed_chunk=false`、`prefill_decode_interval=0`、`num_continuous_decode_steps=1`；`schedule_conservativeness` 和 `retraction_policy=length` 用于容量与活性。其余 SGLang 配置若未改变本期行为，不要仅为占位而暴露。字段最终归属需先在设计阶段确定，避免 `EngineConfig`、调度配置与运行时配置重复持有。

现有输入的 `max_batch_size`、`kv_token_budget` 在兼容层分别映射到 `max_running_requests`、`max_total_tokens`；已存在的调用方保持可解析，核心调度与结果回显只用新字段。确定的请求状态至少有 waiting/prefilling/decoding/finished、剩余 prompt、已建立 KV、已产出 output、首次 Prefill 起止与逐步 token 进度。

冻结不可变 `StepWorkload`：`phase`、本步参与请求及其 Prefill token、Decode 请求及各自 context length、执行 batch size、步首 waiting/running 和 KV 状态、总 batched tokens。非空步必须满足：

```text
PREFILL => prefill_tokens > 0 and decode_tokens == 0
DECODE  => prefill_tokens == 0 and decode_tokens == batch_size
```

普通 Decode 请求每步推进一个 token。`StepWorkload` 不含 duration；经验模型和后续解析模型分别根据本步工作量计算耗时。`F002` 的端到端验收依赖此接口，接口冻结后可并行开发成本后端。

## 分阶段实施

1. **CB1：状态、互斥接口与单步兼容。** 增加请求级 Prefill 进度和不可变 `StepWorkload`，先将现有一次性 Prefill 与 Decode 拆成互斥 step，再冻结上述 phase 不变量；把调度决策与 duration 计算解耦，让现行四系数经验模型消费新的阶段工作量。旧配置输入仍可解析，但旧混合步的时间与步骤数会因互斥语义变更，不承诺逐字节保持旧结果。
2. **CB2：预算、分块与缓存容量接口。** 在 `schemas`/`scheduler` 中实施驻留数、请求数、token 及 KV 上限；长输入跨步做 Prefill，active chunk 优先继续。现行 L1 `prepare_batch/commit` 在准入时一次分配全部输入块、预留全部输出，并按完整上下文判 OOM；必须扩展事务或定义等价的分块准入/增长操作，明确每步新增块、未来 Decode 预留、容量快照、失败回滚与回撤释放。已处理 token 对应 KV 随步骤增长，不把尚未处理的整个 prompt 标作驻留。保守 admission 给驻留 Decode 留必要空间；当 Decode 下一 token 不能分配时按 `length` 策略回撤、释放并重新入队，单请求不可执行则明确终止。
3. **CB3：Prefill-first 调度。** 在 CB1 互斥语义和 CB2 分块容量接口上，先尝试构造**可执行的非空** P batch（继续 active chunk，再按 FCFS 选 waiting）；构造失败再执行 D。仅 waiting 非空不构成 P 可执行；被预算阻塞的 P 不能阻止 D 释放资源。Prefill 在某步完成的请求至早在后续 step Decode。默认 `prefill_decode_interval=0` 不保证持续无限 P 流下 D 不饥饿。
4. **CB4：L1 前缀复用编排。** admission 使用已扩展的同快照 probe/原子分块事务初始化命中进度，命中 token 不占 Prefill token budget；每个 chunk 完成时 `publish` 新的完整块，同 step 保持快照隔离；retraction/完成/终止调用 `release`，避免回撤后立即命中自身并反复占满的无进展循环。块身份、最长连续命中和 LRU 语义不变；不能将现行整请求事务当作完整分块实现。

## 验收

长输入跨多个 step 且实际处理的 token 守恒；每步不超请求/KV/token 预算，P/D 严格互斥；有可执行 P 时优先 P，否则可执行 D 仍能推进；Prefill 完成不在同一步 Decode；容量被阻塞时不会让有限、可执行的输入永远空转。KV 按实际进度增长并在回撤/完成时释放；完整块仅在 chunk 算完后发布，缓存命中不重复计算或占 Prefill 预算。同配置、seed 的步骤、事件和结果可复现，现行单轮入口有明确兼容行为，经验模型和未来 F002 均能消费 `StepWorkload`。API/schema 与前端配套由 [服务计划](server-api.md)、[前端计划](frontend.md) 跟踪。
