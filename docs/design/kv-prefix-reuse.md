# 当前 L1 前缀缓存与物理块管理

本文只规定已接通的单 Engine、一次性 Prefill、L1 paged KV 路径。`HierarchicalKVCache` 名称及配置中的 L2/L3 字段是扩展接缝；当前 Engine 只实例化 L1，没有降级、提升或 restore。未来层级实现见 [KV 模块计划](../plans/kv-cache.md)。合成块 ID 的编码见 [chunk-hash-encoding.md](chunk-hash-encoding.md)，session 的理想继承量见 [workload.md](workload.md)。

## 容量与状态

`prefix_cache.block_size_tokens=B` 同时规定完整块身份与物理分配粒度；L1 容量由 `floor(kv_token_budget/B)` 解析，不能再维护第二套预算。活动请求按 `ceil((prompt_len+output_len)/B)` 规划最坏情况下的未来物理增长；已预留容量不伪装成已驻留 block。物理共享块只占一次容量，每个请求仍分别统计逻辑上下文；有效 token、物理 block capacity 与内部碎片必须分开计算。

缓存持有有序的请求 block table、完整块 ID 到物理 block 的反向索引，以及单调访问序号和 LRU。`content_hash` / `hash_index` 是内部历史字段名，值是**不透明的完整块 ID**，不代表真实内容摘要；同 ID 可索引多个物理块。`ACTIVE` block 有活动引用、不可淘汰；`EVICTABLE` block 无活动引用、可按 `(last_access_seq, block_id)` 的稳定顺序淘汰。仅成功提交的命中或 acquire 更新访问序号；只读 probe、miss、未准入和失败事务不得改变引用、索引或 LRU。未满的 partial block 由活动请求独占，不进入可共享索引，释放时回收。回收物理位置须先移除旧 ID 映射。

容量账本满足 `free_blocks + resident_blocks + reserved_future_blocks = l1_capacity_blocks`（`free` 表示未预留的可用容量），`active_blocks <= resident_blocks`。多请求准入去重共享物理块，并原子地预留未来输出增长；增长消耗预留，完成时归还。暂时不足时按严格 FCFS 等待释放；单请求在空 L1 仍无法容纳则明确报 OOM，不能靠超预算强制准入或无限空转。

## 连续前缀命中与原子准入

`snapshot() -> CacheSnapshot` 提供一致容量与索引视图。`probe(snapshot, input_chunk_ids) -> PrefixProbe` 从输入首个完整块开始，按 ID 依次找最长连续命中，首个 miss 即停止；同 ID 有多个物理位置时稳定选取最小 block ID。probe 本身不增加引用或刷新 LRU。

调度器在一个 `EngineStep` 内对候选共用步首 snapshot，按 `max_batch_size` 与物理容量只选择连续 FCFS 队头；`prepare_batch(snapshot, probes, allocations)` 规划命中引用、新块与必要淘汰，`commit(transaction)` 原子提交。失败不留下部分分配或错误的 LRU；本 step 新发布的 KV 不对同 step 的其他请求可见，同 ID 因此可能对应多个物理块。`StepPlan` 的实际 Prefill 只能由 probe/commit 结果导出，不能用目标理想命中率直接跳过计算。

`matched_cache_tokens` 是连续匹配的完整 block token 数；`actual_hit_tokens = min(matched_cache_tokens, prompt_len-1)` 是实际跳过的 Prefill token。即使整个输入块 KV 已匹配，也至少重算最后一个输入 token 来产生 logits；匹配的物理 block 不重复分配。`actual_prefill_tokens = prompt_len-actual_hit_tokens`。命中前缀不会缩短每请求的 Decode 逻辑上下文；当前 L1 命中没有 restore 成本，也不单独计身份编码/lookup 的 CPU 成本。当前耗时估算见 [perf-model.md](perf-model.md)。

## 发布、增长与释放

准入请求的剩余输入在当前引擎一个 Prefill step 内计算，步末批量 `publish` 已完成的未命中完整输入块。首个输出 token 在该步生成，后续在每个 Decode step 生成；每次输出先 `advance(request_id, 1)` 推进 KV，填满完整块时在步末发布相应 ID。发布仅索引已计算完成的完整块，接口允许逐块或批量发布；输入尾块可由输出填满并发布。发布的 ID 及客户端重建规则由 [块身份文档](chunk-hash-encoding.md) 统一定义。

`release(request_id)` 解除活动引用：已发布的完整块成为可淘汰驻留项，未满尾块直接回收。父请求完成并释放后，零 think time 的子轮次可以命中父块。当前输出与容量指标的计算口径见 [metrics.md](metrics.md)。调度器在 release 后可重新尝试被资源阻塞的队头；是否回撤请求属于未来 [调度计划](../plans/scheduler-engine.md)，不是缓存接口的隐含行为。
