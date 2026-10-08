# KV 缓存模块实施计划（KV3）

**状态：待实现。** 实现单实例 L1/L2/L3 KV 元数据管理、分层准入、逐级迁移与 restore 计时。现有 F001 的请求 `K/F`、完整页身份、Prefill/Decode 时序和 L1 容量口径保持不变。

## 1. 配置与数据契约

- 在 `core/src/simcore/schemas.py` 的 `EngineConfig` 增加 `l2_capacity_tokens`、`l3_capacity_tokens`（默认 0）；两者须为 `page_size` 的非负整数倍。`max_total_tokens/page_size` 仍是唯一的 L1 容量；`L3>0` 要求 `L2>0`。零容量层不参与查找、迁移、占用或计费。
- 在 `PerfModelConfig` 增加 `kv_bytes_per_token`（正整数或 `None`，默认 `None`）、`l2_to_l1_bandwidth_bytes_per_ms`、`l3_to_l2_bandwidth_bytes_per_ms`（正数或 `None`，默认 `None`）、`l2_latency_ms`、`l3_latency_ms`（非负，默认 0）。L2 启用时要求 KV 字节/token 与 L2→L1 带宽为正；L3 启用时另要求 L3→L2 带宽为正；低层全关闭时新增参数不改变旧配置的行为。在 `SimConfig` 与直接构造 `Engine`/`run_engine` 的入口共用跨配置校验，拒绝无效组合。
- 单页传输量为 `page_size * kv_bytes_per_token` 字节。容量按整页计，未来输出预算不计入驻留。外部配置仍通过现有 `POST /api/simulate` 提交，响应继续采用 `SimResult`。

## 2. 单一 KV 管理器

在 `core/src/simcore/kv_cache.py` 建立 `TieredKVCache`，作为调度器和引擎唯一的 KV 状态入口。内部数据与所有权如下：

| 状态 | 所有者与内容 |
|---|---|
| 全局索引 `_entries: dict[str, CacheEntry]` | 已发布完整页的唯一记录；将 `L1Page` 调整为通用 `CacheEntry`，保留 `chunk_id`、当前 `tier`、`ref_count`、`last_access_seq`；一个 ID 同时只驻留一层。 |
| 请求状态 | 每请求的 `K/F`、输入身份、私有页计数及候选 pin；私有页只占 L1，不进入全局索引，仍按请求 ID 稳定顺序 `publish_batch`。 |
| 层级状态 `_tiers[L1|L2|L3]` | 每层独立的容量、驻留/可淘汰/空闲计数、峰值和稳定 LRU 候选堆；层对象只保存 ID 与计数，不能拥有另一份缓存项或请求状态。 |
| 迁移计数 | 各层逐出、提升、逐级降级与最终删除；在管理器内随一次提交更新，不由界面或指标层反推。 |

保留 `page_size`、`state`、`snapshot`、`probe`、`pin`、`cancel_pin`、`admit`、`extend`、`publish_batch`、`finish`、`withdraw`、`ensure_free` 的业务入口，将缓存类型及返回值适配分层信息。`L1KVCache` 作为零低层容量的兼容入口复用 `TieredKVCache`，不维护另一份状态，以保持现有调用和测试可用。只保存模拟容量和层次，不引入真实 KV tensor、物理页地址或请求级页表。

**状态不变量：**每个已发布 `chunk_id` 恰好有一个 `tier`；每层 `free + resident = capacity`；L1 `resident = 已发布 L1 页数 + 私有页数`，L2/L3 `resident = 该层已发布页数`；仅 `ref_count == 0` 且未被候选保护的页可淘汰。稳态运行请求引用的已发布页必须在 L1；低层页允许在候选 pin 期间暂有保护引用，但必须在提交时提升或在失败时撤销。禁用层计数恒为零。

## 3. 查询、准入、迁移与释放

1. **查找与来源。** `probe` 按逻辑 `chunk_id` 顺序跨全局索引查连续已发布前缀，遇首个缺页即停；返回命中 token、按页排序的 `(chunk_id, source_tier)`，不改变 LRU。首次 Prefill 仍最多命中 `floor((prompt_len-1)/page_size)*page_size`，回撤重算仍遵循目标 `K` 的完整页边界。请求级分层命中在此记录来源，后续迁移不回改归因。
2. **保护与预留。** `pin` 对命中页增加临时引用并保护来源；`cancel_pin` 解除引用但不刷新 LRU。调度本批维护待提升的唯一 `chunk_id` 集合；同一低层页被多个候选命中时，只预留一次 L1 容量、只传输一次，每个请求分别持有引用。被保护的低层页不得被降级、删除或用于腾空间。
3. **容量规划。** 每个候选检查 L1 的 `free + evictable`，把本批 Decode 新页、各 Prefill 新私有页、每候选原有一页安全余量，以及**唯一低层命中页的提升需求**一起纳入即时/未来预算；已存在于 L1 的命中页不再占第二份容量。计算 token 额度只统计新计算页，不统计提升页。预检查整个提交计划的各层可用容量、被保护页和逐级驱逐；计划失败时撤销当前候选 pin，不执行迁移、不更新 LRU 或计数，也不跳过 FCFS 队头。
4. **原子提交。** 对已接纳批次按确定性顺序规划并提交提升、为提升和新私有页腾出的空间及逐级降级；支持将待提升来源页腾出的槽位与待降级页交换，不要求迁移瞬间额外占一页。新完整页从 L1 发布；L1 可淘汰页移至 L2，L2 可淘汰页移至 L3，末层可淘汰页删除并从全局索引移除。低层命中提升到 L1 后移除原层驻留；迁入层刷新访问序号，同层 LRU 按 `(last_access_seq, chunk_id)` 稳定选取。不能迁移被活动请求引用的 L1 页。不可满足的计划在改变任何可见状态前返回容量不足，沿用现有等待、回撤和请求失败路径。
5. **发布与释放。** `extend` 只申请真实新增的 L1 私有页；`publish_batch` 同批同 ID 去重并立即归还重复私有页。若新算完整页与低层零引用旧项同 ID，保留新算的 L1 内容，删除旧层驻留并合并为唯一索引项，不将重算误计作 restore；若同 ID 已在本批提升到 L1，则按既有去重规则合并引用。`finish` 在 Done 事件发布剩余完整页、释放引用；`withdraw` 丢弃私有页并释放已发布引用，不改变输出进度或首次 token 时间。

## 4. 调度、引擎与时间

- `core/src/simcore/scheduler.py`：为 `PrefillCandidate`/`AdmissionBatch` 增加命中来源与待提升 ID；逐候选执行 `probe → pin → 预算/计划检查`，成功保留保护直至提交，失败 `cancel_pin`。选中候选间共享待提升集合，融合模式先固定步首 Decode 集合再计算容量。低层全关闭时保持原 `A/current_offset/total_offset`、FCFS 与回撤结果。
- **全命中回撤**：若待提升集合为空，沿用当前立即恢复 running 的零时长分支；若需提升，`AdmissionBatch` 增加按请求 ID 排序的 `restore_only` 集合，并与当步允许的 Prefill/Decode 一同提交。扩展 `StepWorkload` 的 restore 页数/来源及 `StepTrace` 的恢复请求和传输明细：只有 restore、没有 Prefill/Decode 时使用 `RESTORE` 阶段且计算耗时为零；有计算时沿用 `PREFILL`/`DECODE`/`MIXED` 阶段并加上 restore 耗时。互斥模式的恢复项按 Prefill 侧调度，不与 Decode 同步；融合模式允许同一步恢复和 Decode。请求在步末才转为 running，不采样新输出、不参加步首 Decode、不重置首次 Prefill 时间。保持 `Coordinator` 的完成、释放、客户端回调、下一步调度顺序。
- `core/src/simcore/engine.py`：在执行前完成批次容量与迁移预检查，提交后取得实际的唯一 `(chunk_id, source_tier)` 提升记录及降级/删除计数，供计时、trace 和结算使用；随后 `admit → extend → publish_batch`。新计算页和提升页不可重复扣减 L1 容量；完成请求只在 Done 事件 `finish`。Decode 缺空间仍先驱逐、后回撤，降级失败不得留下部分状态。
- `core/src/simcore/perf_model.py`：对每步唯一来源集合设 `n2` 为源自 L2 的页数、`n3` 为源自 L3 的页数、`S` 为单页字节数；按链路汇总，固定延迟每条被使用的链路每步计一次：

  ```text
  restore_bytes = (n2 + 2*n3) * S
  restore_time_ms = [n2+n3>0] * l2_latency_ms
                  + (n2+n3)*S / l2_to_l1_bandwidth_bytes_per_ms
                  + [n3>0] * l3_latency_ms
                  + n3*S / l3_to_l2_bandwidth_bytes_per_ms
  step_time_ms = restore_time_ms + workload_time(step, perf_model)
  ```

  L3 页经 L3→L2→L1 两条链路传输，不在 L2 永久驻留或增加额外页占用；纯传输步的计算耗时为零。身份查找与降级仅计元数据/次数，本期不另计 lookup 或 write-back 传输时间。低层全关闭时恢复耗时与字节均为零，现有两种 P/D 计时保持不变。

## 5. 结果、对外接口与测试

- `core/src/simcore/engine.py` 的 `EngineRun` 携带分层快照、逐层逐出、唯一提升/逐级降级计数及 restore 字节/耗时；`runner.py → metrics.py → schemas.py` 传递、汇总并校验。`RequestRecord.l1/l2/l3_hit_tokens` 只按**首次准入且实际跳过的 prompt token**归因；回撤重算单独计 `replayed_prefill_tokens`，不重复记首次命中。每个请求 `l1+l2+l3=actual_hit_tokens`，全场同样守恒。`promotions` 按实际提升到 L1 的唯一 ID 数计，`demotions` 按实际逐级下移次数计；`l1_evictions` 统计离开 L1 的页（包括降级）。`restore_bytes` 为上行链路字节之和。补 L2/L3 容量、驻留、空闲、峰值、逐出指标，并覆盖无成功请求时的汇总分支；现有 L1 指标口径不变。
- `server` 继续使用现有模拟路由和 `core.schemas`；配置非法返回 422。更新 `web/openapi.json`、`web/src/api/schema.d.ts`；前端配置默认关闭 L2/L3，容量/链路字段仅按实际启用层校验，显示分层命中、驻留、提升/降级和 restore 指标，不将总命中误标为 L1。外部职责分别见 [服务计划](server-api.md) 和 [前端计划](frontend.md)。
- `core/tests/test_l1_kv_cache.py`、`test_f001_scheduler.py`、`test_f001_engine.py`、`test_f001_recovery.py` 保留零低层回归；新增分层缓存、调度、成本及端到端测试：L1→L2→L3→删除、跨层连续前缀、末 token 限制、共享低层命中一次传输多次引用、同批私有页去重、低层同 ID 重算、L3 两链路计费、批次固定延迟一次计费、全命中 transfer-only、融合 Decode/restore、pin 失败回滚、容量不足回撤、确定性 LRU 及 Done 时释放。每次关键操作断言唯一驻留、各层容量/引用守恒；`L2=L3=0` 时逐字段比对原 L1 结果，两种 P/D 模式均覆盖。

## 6. 实施顺序

1. 配置校验、分层数据结构与零低层兼容测试。
2. 查询/保护、原子迁移与分层容量/回滚测试。
3. 调度预算、transfer-only、引擎时序与分层计时测试。
4. 结果守恒、服务类型、前端配置/展示及端到端回归。
