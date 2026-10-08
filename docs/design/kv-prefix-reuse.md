# 当前 L1 前缀缓存与物理块管理

本文只规定已接通的单 Engine、一次性 Prefill、L1 paged KV 路径。当前 Engine 只实例化 `L1KVCache`，没有 L2/L3 降级、提升或 restore。未来层级实现见 [KV 模块计划](../plans/kv-cache.md)。合成块 ID 的编码见 [chunk-hash-encoding.md](chunk-hash-encoding.md)，session 的理想继承量见 [workload.md](workload.md)。

## 容量与状态

`engine.page_size=B` 规定完整页身份与物理分配粒度；L1 容量为 `max_total_tokens/B` 页。准入时估算未来输出增长，但不预分配未来页；物理共享页只占一次容量，每个请求仍分别统计逻辑上下文，有效 token 与内部碎片分开计算。

缓存用完整页 ID 索引已发布页，为每个请求保存已计算长度 `K` 与已发布长度 `F`。只读 `probe` 查找连续命中前缀；`pin` 临时保护命中页，`cancel_pin` 撤销保护且不更新 LRU；确认准入后才刷新访问序号。未发布页是请求私有的，`publish` 遇到相同 ID 时合并物理页；无活动引用的已发布页可按访问顺序淘汰。

容量账本满足 `free_pages + resident_pages = capacity_pages`，未来输出只用于准入估算，不提前占用物理页。暂时不足时按 FCFS 等待或回撤运行请求；单请求无法执行时记录失败，不强制超预算分配。

## 连续前缀命中与准入

`probe(request, target_tokens)` 只读查找已发布页的最长连续前缀，首次准入始终留下最后一个输入 token 计算。`F001Scheduler` 按 FCFS 逐队头探测并 `pin` 命中页，结合运行请求、可用物理页与预计未来增长组成 `AdmissionBatch`；失败时 `cancel_pin`，不跳过队头。执行层 `admit` 确认候选后按需分配页，同一步新发布的页对本步其他候选不可见。

`matched_cache_tokens` 是连续匹配的完整页 token 数，`actual_hit_tokens` 是实际跳过的 Prefill token；当前首次准入的 `probe` 已限制命中不超过 `prompt_len-1`。`actual_prefill_tokens = prompt_len-actual_hit_tokens`。命中前缀不会缩短 Decode 逻辑上下文；当前 L1 命中没有 restore 成本，也不单独计身份编码/lookup 的 CPU 成本。当前耗时估算见 [perf-model.md](perf-model.md)。

## 发布、增长与释放

准入请求的剩余输入在一个 Prefill step 内计算，`extend` 按实际计算长度申请物理页；步末 `publish_batch` 按请求 ID 顺序索引已完成的完整页，相同 ID 的页合并。首个输出 token 在该步生成，后续每个 Decode step 继续 `extend` 一个 token 并发布完整页。发布的 ID 及客户端重建规则由 [块身份文档](chunk-hash-encoding.md) 定义。

完成时 `finish(request_id)` 发布剩余完整页并解除引用；回撤时 `withdraw(request_id)` 丢弃未发布页，已发布页保留为可淘汰缓存。父请求完成后，零 think time 的子轮次可以命中父页。当前输出与容量指标的计算口径见 [metrics.md](metrics.md)。
