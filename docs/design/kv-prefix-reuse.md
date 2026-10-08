# 当前 L1 前缀缓存与物理块管理

本文只规定已接通的单 Engine、一次性 Prefill、L1 paged KV 路径。当前 Engine 只实例化 `L1KVCache`，没有 L2/L3 降级、提升或 restore。未来层级实现见 [KV 模块计划](../plans/kv-cache.md)。合成块 ID 的编码见 [chunk-hash-encoding.md](chunk-hash-encoding.md)，session 的理想继承量见 [workload.md](workload.md)。

## 容量与状态

`engine.page_size=B` 同时规定完整页身份与物理分配粒度；`max_total_tokens` 必须是 `B` 的正整数倍，L1 容量为 `max_total_tokens/B` 页。全局索引只保存已发布完整页，页含引用数和 LRU 访问序号；请求仅保存已建立 KV 长度 `K` 和已发布且被引用的连续前沿 `F`，不维护物理地址页表。`F` 是 `B` 的整数倍，`0≤F≤K`；`[F,K)` 是私有 KV，其占页数为 `ceil(K/B)-F/B`。`resident_pages=已索引页数+各请求私有页数`，始终有 `free_pages+resident_pages=capacity_pages`。未来生成估算只进入调度预算，`reserved_future_blocks` 为零。

## 连续命中与引用

`probe` 顺序查找已发布完整页，到首个缺失页为止；首次输入最多命中 `floor((P-1)/B)*B` token，故至少重新计算一个输入 token。回撤恢复重算目标 `R=P+g-1`，最多命中 `floor(R/B)*B` token。页 ID 按逻辑位置由继承前缀或 `(request_id, 页序号)` 推导。`pin` 暂时为命中页加引用，失败时 `cancel_pin` 归还且不刷新 LRU；成功 `admit` 后才刷新访问序号。无引用的完整页仍驻留且可淘汰；驱逐按 `(last_access_seq, chunk_id)` 稳定升序，只有实际缺页时才按需取最旧页。`A=(free_pages+evictable_pages)*B` 是准入可用量，不等于当前空闲量；逐请求预算及回撤见 [kernel.md](kernel.md)。

成功请求首次准入时 `matched_cache_tokens=actual_hit_tokens`（均为连续 L1 命中），`actual_prefill_tokens=P-actual_hit_tokens`；回撤的额外 Prefill 单列为 `replayed_prefill_tokens`。共享物理页只占一页容量，不缩短 Decode 的逻辑上下文；L1 命中不计 restore 或查找 CPU 耗时。

## 发布、增长与释放

Prefill 在一个 step 内完成本请求剩余输入，`extend` 只为实际新增 KV 申请页。Prefill 步末 `publish_batch` 按请求 ID 顺序发布完整私有页；同批首次算出的相同 ID 在发布前各占私有页，发布时合并并归还重复物理页。**Decode 中填满的页仍是私有页，不逐步发布**；请求完成的 `RequestDone` 时由 `finish` 发布剩余完整页、释放所有共享引用与未满尾页。回撤时 `withdraw` 直接回收未发布私有页、解除引用，不发布其完整私有页；已发布页无引用后仍可留作缓存。新发布页在下一步才可供其他请求命中，零思考时间子轮能在父轮完成、释放后继承完整页。

当前只有 L1。结果中的 L2/L3 命中、restore bytes/time、promotion/demotion 字段为零值预留，**没有**启用低层缓存的输入配置或迁移路径；见 [KV3 计划](../plans/kv-cache.md)。容量和碎片指标见 [metrics.md](metrics.md)。
