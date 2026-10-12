# 分层 KV 前缀缓存与物理块管理

单 Engine 的 `TieredKVCache` 负责 L1/L2/L3 已发布完整页、请求私有页和异步传输占位、完成迁入；低层容量为零时仍使用该实现，不提供旧缓存类兼容入口。不保存真实 tensor、物理页地址或请求级页表。合成块 ID 编码见 [chunk-hash-encoding.md](chunk-hash-encoding.md)，session 理想继承量见 [workload.md](workload.md)。

## 容量与状态

`inference.page_size=B` 同时规定完整页身份与物理分配粒度；`max_total_tokens` 必须是 `B` 的正整数倍，L1 容量为 `max_total_tokens/B` 页。`l2_capacity_tokens`、`l3_capacity_tokens` 默认零，均须为 `B` 的非负整数倍，启用 L3 必须先启用 L2。全局索引对每个已发布 ID 只保存一条含 `tier/ref_count/last_access_seq` 的 `CacheEntry` 记录；各层仅维护驻留 ID、容量、计数和稳定 LRU，不复制页对象。请求仅保存已建立 KV 长度 `K` 和已发布且被引用的连续前沿 `F`；`F` 是 `B` 的整数倍，`0≤F≤K`，`[F,K)` 是只占 L1 的私有 KV，页数为 `ceil(K/B)-F/B`。每层 `free+resident=capacity`；L1 驻留包含已发布页、私有页及在途目的占位，L2/L3 驻留包含已发布页及在途目的占位。传输期间索引仍指向受保护的来源页，目的占位尚不可命中，完成后才迁入并释放来源容量。未来生成估算只进入调度预算，`reserved_future_blocks` 为零，与实际传输占位不同。

## 连续命中与引用

`probe` 按逻辑 ID 顺序跨层查找已发布完整页，到首个缺失页为止，直接返回按顺序排列的 `(chunk_id, tier)` 命中页元组；命中 token 数为元组长度乘 `B`，`tier` 固定为查询时来源；首次输入最多命中 `floor((P-1)/B)*B` token，故至少重算一个输入 token。回撤目标 `R=P+g-1` 最多命中 `floor(R/B)*B` token。页 ID 按逻辑位置由继承前缀或 `(request_id, 页序号)` 推导。`pin` 暂时为命中页加引用，失败时 `cancel_pin` 归还且不刷新 LRU；传输占位保护来源，完成迁入及成功准入才刷新访问序号。零引用且未受保护的完整页才可淘汰；各层按 `(last_access_seq, chunk_id)` 稳定升序选择。L1 的 `A=(free_pages+evictable_pages)*B` 是准入可用量，不等于当前空闲量；低层页的提升另占唯一 L1 容量，同批多个请求命中同一低层页按 ID 去重传输。逐请求预算和回撤见 [kernel.md](kernel.md)。

成功请求只在首次准入时按查询来源计 `l1/l2/l3_hit_tokens`；三者之和为 `actual_hit_tokens`，`actual_prefill_tokens=P-actual_hit_tokens`。回撤重算单列为 `replayed_prefill_tokens`，不重复计首次命中。共享物理页只占一页容量，不缩短 Decode 的逻辑上下文；身份查找不计 CPU 耗时。

## 发布、增长与释放

Prefill 在一个 GPU step 内完成本请求剩余输入，`extend` 只为实际新增 L1 私有页申请空间。缓存 `plan` 无副作用地预检完整批次提升与新计算所需容量，不提交同步提升。实际上行由 `reserve_transfer` 保护来源、预留目的占位，`complete_transfer` 在完成事件迁入并释放来源容量和传输引用，保留请求 pin 引用；不能提前把尚未迁出的来源空位用于占位。因此 `plan` 的理想试算成功也不保证实际 `reserve_transfer` 成功，源页保护仍可能使腾位失败。`ensure_free` 同步原子执行逐级驱逐/降级：L1 零引用页移往 L2，L2 零引用页移往 L3，末层淘汰时删除全局索引，被活动请求或在途传输保护的页不可降级。

启用 L3 后，请求先等待查询事件；连续 L3 命中段的 token 数严格大于 `l3_prefetch_min_tokens` 才发起逐页 L3→L2 预取。预取完成时页实际驻留 L2，之后通过选批和逐页 L2→L1 回载才可用于 GPU；未预取的 L3 页不直接复用。同一 ID 的在途传输去重，每个实例每条链路串行，GPU 等待期间可推进已有 Decode。服务时长和查询/排队口径见 [perf-model.md](perf-model.md)。

Prefill 的 `StepDone` 调用引擎 `finish_step`，由 `publish_batch` 按请求 ID 顺序发布完整私有页；步首不会公开。同批首次算出的相同 ID 在发布前各占私有页，发布时合并并归还重复物理页。若重算的新完整页与低层同 ID 零引用旧页相遇，保留新算的 L1 页并移除旧低层项，不计为 restore。**Decode 中填满的页仍是私有页，不逐步发布**；请求完成的 `RequestDone` 调用引擎 `complete`，由缓存 `finish` 发布剩余完整页、释放共享引用及尾页。回撤时 `withdraw` 丢弃未发布私有页并解除引用；无引用已发布页仍可留在所属层。发布后的完整页可在同刻后续调度中命中，零思考时间子轮也可观察完成发布。低层关闭时保持既有 L1 路径与指标口径；分层容量、迁移与 restore 指标见 [metrics.md](metrics.md)。
