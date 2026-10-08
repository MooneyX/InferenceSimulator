# 分层 KV 前缀缓存与物理块管理

单 Engine 的 `TieredKVCache` 负责 L1/L2/L3 已发布完整页、请求私有页和迁移。`L1KVCache` 是低层容量为零时复用同一实现的兼容入口；不保存真实 tensor、物理页地址或请求级页表。合成块 ID 编码见 [chunk-hash-encoding.md](chunk-hash-encoding.md)，session 理想继承量见 [workload.md](workload.md)。

## 容量与状态

`engine.page_size=B` 同时规定完整页身份与物理分配粒度；`max_total_tokens` 必须是 `B` 的正整数倍，L1 容量为 `max_total_tokens/B` 页。`l2_capacity_tokens`、`l3_capacity_tokens` 默认零，均须为 `B` 的非负整数倍，启用 L3 必须先启用 L2。全局索引对每个已发布 ID 只保存一条含 `tier/ref_count/last_access_seq` 的记录；各层仅维护驻留 ID、容量、计数和稳定 LRU，不复制页对象。请求仅保存已建立 KV 长度 `K` 和已发布且被引用的连续前沿 `F`；`F` 是 `B` 的整数倍，`0≤F≤K`，`[F,K)` 是只占 L1 的私有 KV，页数为 `ceil(K/B)-F/B`。每层 `free+resident=capacity`；L1 驻留包含已发布页与私有页，L2/L3 仅含已发布页。未来生成估算只进入调度预算，`reserved_future_blocks` 为零。

## 连续命中与引用

`probe` 按逻辑 ID 顺序跨层查找已发布完整页，到首个缺失页为止，并记录每个命中页查询时的来源；首次输入最多命中 `floor((P-1)/B)*B` token，故至少重算一个输入 token。回撤目标 `R=P+g-1` 最多命中 `floor(R/B)*B` token。页 ID 按逻辑位置由继承前缀或 `(request_id, 页序号)` 推导。`pin` 暂时为命中页加引用，失败时 `cancel_pin` 归还且不刷新 LRU；成功提交才将低层页提升并刷新访问序号。零引用且未受保护的完整页才可淘汰；各层按 `(last_access_seq, chunk_id)` 稳定升序选择。L1 的 `A=(free_pages+evictable_pages)*B` 是准入可用量，不等于当前空闲量；低层页的提升另占唯一 L1 容量，同批多个请求命中同一低层页只迁移一次。逐请求预算和回撤见 [kernel.md](kernel.md)。

成功请求只在首次准入时按查询来源计 `l1/l2/l3_hit_tokens`；三者之和为 `actual_hit_tokens`，`actual_prefill_tokens=P-actual_hit_tokens`。回撤重算单列为 `replayed_prefill_tokens`，不重复计首次命中。共享物理页只占一页容量，不缩短 Decode 的逻辑上下文；身份查找不计 CPU 耗时。

## 发布、增长与释放

Prefill 在一个 step 内完成本请求剩余输入，`extend` 只为实际新增 L1 私有页申请空间。提交前预检查完整批次的容量及迁移可行性，失败不执行部分迁移；提升源层的空位可与逐级下移的页交换，不要求中间额外页。L1 的零引用页移往 L2，L2 零引用页移往 L3，末层淘汰时删除全局索引；被活动请求引用的页不可降级。低层来源页提升后从原层移除，L3→L1 的传输经过两条链路，但不在 L2 永久驻留。

Prefill 步末 `publish_batch` 按请求 ID 顺序发布完整私有页；同批首次算出的相同 ID 在发布前各占私有页，发布时合并并归还重复物理页。若重算的新完整页与低层同 ID 零引用旧页相遇，保留新算的 L1 页并移除旧低层项，不计为 restore。**Decode 中填满的页仍是私有页，不逐步发布**；请求完成的 `RequestDone` 时由 `finish` 发布剩余完整页、释放共享引用及尾页。回撤时 `withdraw` 丢弃未发布私有页并解除引用；无引用已发布页仍可留在所属层。新发布页在下一步才可供其他请求命中。低层关闭时保持既有 L1 路径与指标口径；分层容量、迁移与 restore 指标见 [metrics.md](metrics.md)。
