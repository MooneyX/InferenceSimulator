# 分层 KV 前缀缓存与物理块管理

## 所有权与数据结构

L1（各实例 HBM）与 L2（host）由前缀基数树 `radix_cache.RadixTree` 统一管理，L3 由按页管理的 `kv_cache.CachePool` 管理。仿真不保存真实 tensor、物理页地址或 token 序列，节点与页只以完整页 ID 判等，ID 编码见 [chunk-hash-encoding.md](chunk-hash-encoding.md)。

- **节点**：一段不分叉的连续完整页，key 为若干段连续 ID `(起始 ID, 页数)`，子节点按首页 ID 索引。ID 同时编码整条前缀，节点与请求的公共页数可二分确定。
- **节点元数据**直接存于节点：L2 副本 `l2` 与 L2 访问时间；全部实例的路径锁之和 `locks`；持有 HBM 副本的实例数 `hbm`；按实例的 `holders[instance]`（HBM 副本、该实例路径锁、HBM 访问时间、本实例驻留子节点数）。
- **树的归属**：`shared_l2_enabled` 且 L2 容量非零时，`runner` 为全场建一棵树，每个实例把整块共享 L2 视为自己的 L2，并在同一棵树中维护自己的 HBM 副本；否则每个实例一棵树。L2 容量与淘汰按树共享，HBM 容量账本 `HbmPool` 与 D-leaf 淘汰堆按实例。
- **请求管理器** `TieredKVCache` 是一个实例对树与共享 L3 的视图：保存请求、已建立 KV 长度 `K`、已发布前沿 `F`、请求锁终点节点、私有页和候选保护。资源池不理解 Prefill/Decode；链路排队与完成事件由 Engine 和 Coordinator 推进。
- **L3**：`runner` 每场新建一个共享 `CachePool`，按页索引、保护引用与稳定 LRU，同 ID 只存一份；容量为零表示关闭，不跨仿真共享。

对照 SGLang（提交 `188cf7da`）：`mem_cache/radix_cache.py` 的 `TreeNode`/`_split_node`/`_insert_helper`/`evict`/`inc_lock_ref`，以及 `mem_cache/unified_cache/unified_tree_core.py` 的 `value`/`host_value`、D-leaf（`_is_device_leaf`）与 H-leaf（`_is_host_leaf`）。SGLang 每个实例一棵树、节点只有一份 device `value`；本项目把 `value` 扩为按实例记录，以表达共享 L2。

## 容量与状态

`inference.page_size=B` 同时规定完整页身份与分配粒度。物理配置按单卡显存 `inference.hardware.memory_bytes_per_gpu`、L2 预算和 `shared_l3.available_gb` 解析每层容量；单卡显存扣除每卡全部模型权重后用于 L1 KV，不扣额外运行时空间，至少容纳一页，运行时的 `max_total_tokens`、`l2_capacity_tokens` 和 `l3_capacity_tokens` 均按完整页向下取整。页字节成本是单个 TP 组保存一份逻辑页的字节数，共享 L3 不乘实例数量，共享 L2 容量为每实例 L2 乘实例数。启用 L2 要求 L2→L1 带宽，启用 L3 要求 `inference.storage.l3_read_gb_s`。

L2 容量为 0 且启用 L3 时为两层结构（L1+L3），不另设开关（`TieredKVCache.direct_l3`）：L1 淘汰节点的页直接写入 L3，L3 命中页在准入时直接读入 L1，`shared_l2_enabled` 不起作用。SGLang HiCache 默认三层，L3 读出数据先落主机内存池；SGLang（commit `2b36c718`）中显存直连外部存储有两条路径：Direct L3 linker（`--enable-unified-cache-external-linker`，`mem_cache/unified_cache/unified_cache_linker.py`）逐层读取并与前向流水，LMCache 接入（`mem_cache/storage/lmcache/lmcache_unified_radix_cache.py`）整批读完再计算。`storage.l3_layerwise_load`（默认开启）选择前者，关闭时选择后者；两层结构是本项目按这两条路径建立的仿真模式，不是 HiCache 默认行为。

`F` 是 `B` 的整数倍，`0≤F≤K`；`[F,K)` 是请求私有 KV，页数为 `ceil(K/B)-F/B`，只占 L1。L1 驻留 = 本实例树中 HBM 副本页 + 私有页，满足 `free+resident=capacity`；可淘汰页为本实例驻留且未加锁的树页（含非叶节点，它们终将成为叶子），`active+evictable=resident`。L2 驻留 = L2 副本页 + 在途预取占位；可淘汰页为 L2 副本中无任何 HBM 副本且未加锁的页。未来生成估算只进入调度预算，`reserved_future_blocks` 为零。

## 匹配、切分与路径锁

- **匹配**：`probe` 自根沿请求页 ID 走树，节点在本实例 HBM 中为 L1、否则有 L2 副本为 L2、两者皆无则本地前缀结束。本实例 HBM 驻留集对前缀封闭，故命中形如“L1 页在前、L2 页在后”。`include_l3` 时其后逐页查共享 L3，首个缺失页结束；与 SGLang `prefetch_from_storage` 自本地匹配终点查询存储一致。返回 `PrefixMatch`：本地前缀终点节点、L1 页数、本地页数及其后各页 `(chunk_id, tier)`。
- **切分**：匹配止于节点内部（ID 分歧或达到可复用上限）时，按 SGLang `_match_prefix_helper` 在该处切分节点；插入同样切分。发布时另在 prompt 边界切分（SGLang `checkpoint(split_prompt=True)`）。切分后不合并。上半为新节点，原节点保留为下半，故已有锁终点不变；两半复制 L2 状态、锁计数与各实例副本状态。
- **LRU**：匹配与插入刷新路径上节点的访问时间（同一次操作同一时刻）：本实例 HBM 副本刷新 HBM 时间，L2 副本刷新 L2 时间。与 SGLang 一致，未准入的匹配也会刷新。
- **路径锁**：`pin` 从本地前缀终点沿路径到根为本实例加锁（SGLang `inc_lock_ref`），同时保护其后的 L3 命中页；锁同时阻止该实例的 HBM 淘汰与全树的 L2 淘汰。`admit` 要求命中页已全部在本实例 HBM，锁转由请求持有；发布新页后锁终点移到新末节点；完成或回撤时解锁。首次最多命中 `floor((P-1)/B)*B` token，至少重算最后一个输入 token；回撤恢复目标 `R=P+g-1` 最多命中 `floor(R/B)*B`。

## 写入、淘汰与回载

- **write-through**：对应 SGLang 默认 `--hicache-write-policy write_through`（`arg_groups/fields/memory.py`）。发布插入新页后，本请求路径上尚无 L2 副本的节点一并写入 L2；L2 空间不足时按 LRU 淘汰 H-leaf，仍不足则跳过本次写入（SGLang `_execute_kv_backup` 在 `evict_host` 不足时放弃备份）。写入同步完成，不计写回传输耗时。
- **HBM 淘汰**：`ensure_free` 按 HBM 访问时间淘汰本实例 D-leaf（本实例驻留、未加锁、无本实例驻留子节点），整节点释放，可能多于所需页数（SGLang `evict` 同样整节点释放）。有 L2 副本的节点只丢弃本实例 HBM 副本；无 L2 副本时三层结构直接丢弃，两层结构先逐页写入 L3。节点再无任何副本且无子节点时删除，并向上级联删除空占位节点。
- **L2 淘汰**：只淘汰 H-leaf：L2 驻留、无任何 HBM 副本、未加锁、无子节点（SGLang `_is_host_leaf`）。淘汰页逐页写入 L3：已有副本只刷新 LRU；L3 满时先淘汰 L3 LRU，L3 无可淘汰页时丢弃。L3 末层淘汰只删除 L3 副本。
- **L2→L1 回载**：选批后 Engine 先用 `ensure_free` 为本批唯一提升页与新计算页一次腾位，再对全部已保护候选 `promote`：沿锁定路径自上而下为本实例复制 L2 节点，L2 副本保留（SGLang `load_back` 后 `host_value` 不变）。同批候选共享的节点只复制一次；耗时由 GPU 步逐层流水计入，见 [perf-model.md](perf-model.md)。
- **L3→L2 预取**：后续连续前缀中需从 L3 回载的 token 数严格大于 `scheduler.l3_prefetch_min_tokens` 时发起，按前缀顺序每 128 页一批读取；中间已在途的页直接等待。`reserve_prefetch` 先保护 L3 来源再预留一页 L2 占位（腾挪 L2 时被淘汰页写 L3 不会挤掉来源），预留失败或来源已被淘汰时停止。完成时 `complete_prefetch` 按发起请求的页序号把连续段作为 L2 副本插入树：路径上缺失的前缀建为无副本的占位节点，已有 L2 副本的页直接释放占位。在途预取页的完成时刻挂在树上，共享 L2 时其他实例对同页等待同一读取。每实例 L3→L2 链路按批先来先服务串行、与 GPU 独立推进；不模拟共享存储总吞吐竞争。
- **两层结构 L3→L1**：选批匹配包含 L3；`promote` 把 L3 命中页作为本实例 HBM 副本插入树并后移锁终点，读取保护见下文。

## 与 SGLang 的差异

- **共享 L2 的切分点共享**：SGLang 每个实例一棵树，HBM 节点边界只取决于本实例的插入与匹配。共享 L2 时全场一棵树，其他实例的匹配或插入切开的节点也切开本实例的 HBM 节点，淘汰粒度变细（更少超量释放）。会话内 ID 只在会话内共享，`cache_affinity` 路由下几乎不发生；不共享 L2 时无此差异。共享 L2 本身是本项目扩展，SGLang host 池归属单个实例。
- **写回时机**：write-through 与淘汰写 L3 均同步且不计耗时；SGLang 由后台线程异步写入并在写入期间锁定节点。SGLang 启用存储后由 host 备份完成触发写存储，本项目仅在 L2 淘汰（两层结构为 L1 淘汰）时写 L3。
- **未建模**：priority 淘汰策略、`hit_count` 阈值（`write_through_selective`）、`write_back` 策略及其冗余 host 副本回收。L2 容量小于 L1 时，write-through 下 L2 主要被仍在 HBM 的副本占满，HBM 淘汰出的节点很快被后续写入挤到 L3，L2 命中显著降低；SGLang 对此同样告警（host 池小于 device 池时 L2 效果下降），默认 `hicache_ratio=2.0`。

### 两层结构的查找、保护与直接读取

- **到达只查找**：本地连续前缀不足时仍发起一次有延迟的 L3 查询；查询完成后不搬数据。连续 L3 命中 token 数严格大于 `l3_prefetch_min_tokens` 时，对命中的 L3 页加查找保护（L3 引用），请求随即就绪；否则不保护。
- **准入时读取**：选批 `probe` 包含 L3（L1→L3 连续匹配，不再重新查询 L3）；L3 部分不超过阈值时自首个 L3 页起重算。L3 命中页与 L2 提升页一样计入 L1 预算，选批后同步插入 L1；源 L3 页另加读取保护，逐层读取时至末层读完、整批读取时至所在 128 页批次读完（`L3_L1_DONE` 事件）才释放。查找保护在准入或中止时释放。
- **排队期间 L1 命中被淘汰**：这些页已写入 L3，准入时直接读回；不触发三层结构中“本地匹配变短则重新查询 L3”的机制。该处理与 SGLang 两层路径的细节可能不同（未核实）。
- 阈值语义：两种结构中 `l3_prefetch_min_tokens` 均为“命中超过该值才从 L3 读，否则重算”；SGLang 两层路径未见对应阈值，属本项目设定。

## 发布、增长与释放

Prefill 的 `StepDone` 调用 `publish_batch`，按请求 ID 顺序把 `[F/B, K/B)` 的完整私有页插入树，步首不公开；已在本实例树中的重复页释放额外容量，已有 L2 副本或其他实例 HBM 副本的节点只增加本实例副本。重算页不计为 restore。

Decode 中填满的页仍私有，不逐步发布。`RequestDone` 调用 `finish` 发布剩余完整页（并在 prompt 边界切分）、解锁并释放尾页；回撤 `withdraw` 丢弃未发布私有页并解锁。已发布的未加锁节点可继续留在树中。零思考时间子轮可观察完成发布，但发布本身不等于写入共享 L3。

首次成功准入按来源计 `l1/l2/l3_hit_tokens`，三者之和为 `actual_hit_tokens`，`actual_prefill_tokens=P-actual_hit_tokens`；回撤重算单列。共享 L3 的容量、驻留、实际峰值与淘汰只在全场指标计一次，逐实例对应池状态字段为 `null`，命中及传输量仍按实例统计。指标见 [metrics.md](metrics.md)。
