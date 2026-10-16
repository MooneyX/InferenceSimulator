# 分层 KV 前缀缓存与物理块管理

## 所有权与资源池

`CachePool` 是 L1/L2/L3 共用的资源池类，以 `level` 和一次仿真内全局唯一的 `pool_id` 标识位置。每个池拥有完整页索引、容量、保护引用、稳定 LRU、在途目的占位和统计，不持有请求或仿真时钟。不保存真实 tensor、物理页地址或 token 序列。

每个 Engine 独占一个 `TieredKVCache` 请求管理器及其 L1/L2 池。`runner` 每场新建一个共享 L3 池，并注入所有 Engine；L3 容量为零表示关闭，不跨仿真共享。请求管理器保存请求、已建立 KV 长度 `K`、已发布连续前沿 `F`、私有页及候选引用的实际来源池。资源池不理解 Prefill/Decode；链路排队与完成事件由 Engine 和 Coordinator 推进。

同一 `chunk_id` 可在不同池内拥有副本。`CacheEntry.tier/pool_id` 仅标记该池内副本的位置，不表示全系统唯一驻留位置。当前本地 L1/L2 采用迁移而非长期备份；在途回载与本地重算相遇时允许暂时拥有两个本地副本，后续迁移合并容量和引用。L3 与本地副本独立。合成块 ID 编码见 [chunk-hash-encoding.md](chunk-hash-encoding.md)。

## 容量与状态

`inference.page_size=B` 同时规定完整页身份与分配粒度。物理配置按 L1/L2 预算和 `shared_l3.available_gb` 解析每层容量；L1 扣除权重和预留后至少容纳一页，运行时的 `max_total_tokens`、`l2_capacity_tokens` 和 `l3_capacity_tokens` 均按完整页向下取整。页字节成本是单个 TP 组保存一份逻辑页的字节数，共享 L3 不乘实例数量。启用 L3 要求启用本地 L2 及对应接入带宽。

`F` 是 `B` 的整数倍，`0≤F≤K`；`[F,K)` 是请求私有 KV，页数为 `ceil(K/B)-F/B`，只占 L1。每池满足 `free+resident=capacity`，`active+evictable=resident`。驻留包含完整页、L1 私有页及实际在途目的占位；目的占位尚不可命中。未来生成估算只进入调度预算，`reserved_future_blocks` 为零。

## 连续匹配与保护

不实现 RadixTree。请求管理器按逻辑页位置枚举 ID，每页先查 L1，再查 L2；`probe(include_l3=False)` 到首个本地缺失页结束。若仍有可缓存输入页，Engine 发起一次有延迟的 L3 查询；查询完成后从本地连续命中的终点继续，后续每页仍优先使用本地副本，否则查共享 L3，所有层均缺失时停止。返回顺序固定的 `(chunk_id, tier)` 元组，来源表示查询时位置。

首次最多命中 `floor((P-1)/B)*B` token，至少重算最后一个输入 token；回撤恢复目标 `R=P+g-1` 最多命中 `floor(R/B)*B`。页 ID 按位置从继承前缀、Trace 身份或请求 ID 推导，不物化整条新 ID 序列。

`pin` 保护命中副本并记录来源池；失败 `cancel_pin` 撤销保护但不刷新 LRU。候选页迁移后引用绑定跟随本地副本；L3 复制时仅转移该管理器自身的候选引用，不影响其他实例读者。零引用且没有在途保护的完整页才可淘汰，各池按 `(last_access_seq, chunk_id)` 稳定选择。查询不刷新 LRU，成功回载与准入才刷新。

## 迁移、复制与预取

`plan` 通过逐池计数、增量页覆盖和只读堆前沿预检完整批次，不复制全部索引、不改变真实池状态。`ensure_free` 原子提交同步降级：L1 零引用页迁往 L2；L2 淘汰页写入 L3 后释放本地副本；L3 按 ID 去重，末层淘汰只删除 L3 副本，不影响任何本地页。

`reserve_transfer` 保护来源并预留目标容量；`complete_transfer` 在完成事件中使目标副本可见、释放传输保护。本地 L2→L1 回载释放 L2 来源；L3→L2 是复制，保留 L3 来源及其他实例的引用。不同实例可同时读取同一 L3 页，在各自 L2 分别占位和驻留；同实例同 ID 的在途读取去重。目标池按页 ID 维护在途占位，多个实例同时写入同一 L3 页也只占一页；首次写入完成后副本可见，但保持其他未完成写入的保护引用。来源受保护时不能以其容量为其他传输腾位，失败的预检或预留不改变索引、计数或 LRU。

后续连续前缀中需从 L3 回载的 token 数严格大于 `scheduler.l3_prefetch_min_tokens` 时发起逐页预取；中间已有的本地页不重复传输。预取完成后还需 L2→L1 回载及准入，未预取的 L3 页不能直接供 GPU 使用。每实例每条接入链路串行，两条链路与 GPU 可独立推进；不模拟共享存储总吞吐竞争。下行降级同步且不计写回时长，不包含 D 异步卸载。计时见 [perf-model.md](perf-model.md)。

## 发布、增长与释放

Prefill 的 `StepDone` 调用 `publish_batch`，按请求 ID 顺序将完整私有页发布到本地 L1，步首不公开。同批重复私有页发布时合并并释放额外容量；重算的完整页替换本地 L2 的无引用旧副本，但不删除共享 L3 副本，也不计为 restore。

Decode 中填满的页仍私有，不逐步发布。`RequestDone` 调用 `finish` 发布剩余完整页、释放本地请求引用及尾页；回撤 `withdraw` 丢弃未发布私有页并解除引用。已发布的无引用页可继续留在池中。零思考时间子轮可观察完成发布，但本地发布本身不等于写入共享 L3。

首次成功准入按来源计 `l1/l2/l3_hit_tokens`，三者之和为 `actual_hit_tokens`，`actual_prefill_tokens=P-actual_hit_tokens`；回撤重算单列。共享 L3 的容量、驻留、实际峰值与淘汰只在全场指标计一次，逐实例对应池状态字段为 `null`，命中及传输量仍按实例统计。指标见 [metrics.md](metrics.md)。
