# KV 前缀复用与多级缓存：分步实现指导

## 1. 实施边界和交付顺序

按下表逐步合入；每步先完成对应模块的接口与测试，再接入下一步。`PC1`、`PC2`、`PC3A` 在现有单 Engine、一次性 Prefill 引擎上实现 L1，三者全部接通后才将 `F003` 视为完成。中间步骤不作为对外可用的“缓存已启用”版本；最终所有请求均走缓存路径，不保留运行时关闭开关。

| 顺序 | 阶段 | 交付物 | 依赖 |
|---|---|---|---|
| 1 | PC1-A：配置、计划、内容和 hash | 可复现的 session `WorkloadPlan` 与理想命中分配 | 无 |
| 2 | PC1-B：客户端和统一事件时钟 | 到时构造请求、完成后再发送子轮次 | 1 |
| 3 | PC2：L1 paged block manager | 独立可测的 snapshot/probe/原子 acquire/publish/release | 1 |
| 4 | PC3A-A：调度、引擎和性能接入 | L1 命中真实减少一次性 Prefill | 2、3 |
| 5 | PC3A-B：指标和产品接口 | 请求/全场口径、API/前端展示及回归 | 4 |
| 独立扩展 | KV3：L2/L3 扩展 | 独占式 write-back、restore 成本与分层指标 | 5；不阻塞 `F001` |
| 后续 | `F001.CB4` / `F002` | chunked prefill/retraction；解析性能模型 | `F001` 依赖 5，`F002` 的端到端接入依赖 `F001` |

当前落点为 `core/src/simcore/`：`schemas.py`、`workload.py`、`events.py`、`engine.py`、`scheduler.py`、`kv_cache.py`、`perf_model.py`、`metrics.py`、`runner.py`；新增 `prefix_hash.py`、`client.py`、`coordinator.py`。保持 `runner.simulate(SimConfig) -> SimResult` 和 `POST /api/simulate` 为对外入口。`kv_cache` 独占可变 KV 状态，`coordinator` 独占时钟/事件堆，`client` 独占 session 历史；三者不能互相代管。

### 所有步骤共用的口径

- `arrival_rate` 是用户输入的**所有轮次请求的目标 QPS**（请求/秒），不是根 session 的到达率或实际完成吞吐；仅用于在构造 `WorkloadPlan` 时换算根到达过程，不根据引擎结果反馈调参。有限场仿真的实际到达 QPS/完成吞吐不保证等于目标。
- `target_ideal_prefix_hit_rate` 属于 workload；`constructed_ideal_prefix_hit_rate` 由内容构造决定；`actual_prefix_hit_rate` 仅由请求实际准入时的缓存驻留决定，不反馈给生成器。全局比率统一为 `sum(hit_tokens) / sum(prompt_len)`，包括理想命中为 0 的根请求。
- `matched_cache_tokens` 是 hash 命中的完整 block token 数；`actual_hit_tokens = reused_prefill_tokens = min(matched_cache_tokens, prompt_len - 1)`。即使整个输入的 KV 都命中，也要计算最后一个输入 token 以获得 logits；已命中的物理 block 不重复分配。
- block size `B` 同时决定 hash、内容复用与 KV 物理分配粒度；partial block 可由活动请求持有但不跨请求共享。单轮 workload 表示为仅含根轮次的 session。
- 同配置、seed 和性能参数必须复现计划、token、hash、事件、淘汰和结果；避免依赖 Python 内置 `hash()` 与无序容器遍历。

## 2. PC1-A：配置、计划、内容与 hash

### `schemas.py`：先冻结数据契约

将 session 参数置于 `SyntheticWorkloadConfig` 下；在 `EngineConfig` 下增加 prefix cache 输入配置，解析后内部始终有 `prefix_cache`，无 `enabled` 字段。`num_requests=N` 为总轮次数，`num_sessions=S` 为根请求数，要求 `1 <= S <= N`，`arrival_rate=q` 为所有轮次请求的目标到达 QPS（有限正数，单位请求/秒）；保持字段名兼容旧请求。旧请求缺少新字段时默认 `num_sessions=num_requests`、`target_ideal_prefix_hit_rate=0`、`think_time_ms=0`、`block_size_tokens=1`（保证任何现有正数 `kv_token_budget` 可解析），L2/L3 容量为 0；显式配置较大 `B` 可模拟实际块粒度。入口兼容现有 `kv_token_budget`：`l1.capacity_blocks = floor(kv_token_budget / B)`，不得同时维护独立的第二套 L1 预算；`B >= 1`，解析后容量不足以放下一个 block 时明确报配置错误。L2/L3 在本阶段不启用，KV3 才激活。

```text
synthetic.session:
  num_sessions                   # 1 <= num_sessions <= num_requests
  topology: linear              # 首期生成线性链，数据结构保留分支
  think_time_ms                 # >= 0
  target_ideal_prefix_hit_rate  # [0, 1]
  content_seed: optional        # 不指定则从已有 seed 稳定派生
  allow_cross_session_shared_prefix: false

prefix_cache:
  block_size_tokens             # 全场唯一 B
  cache_salt
  eviction_policy: lru
  hierarchy_policy: exclusive_write_back
  l1: {capacity_blocks}         # resolved config 的唯一 L1 物理容量
  l2: {capacity_blocks, read_bandwidth_bytes_per_ms, fixed_latency_ms}
  l3: {capacity_blocks, read_bandwidth_bytes_per_ms, fixed_latency_ms}
  kv_bytes_per_token
```

内部使用不可变值对象（可置于 `schemas.py` 或同层专用模型文件；对外配置/结果仍以 `schemas.py` 为单一事实源）：

```text
TurnPlan(request_id, session_id, parent_request_id,
         root_arrival_time_ms | think_time_ms,
         prompt_len, output_len, ideal_reuse_blocks,
         unique_suffix_seed, output_token_plan)
WorkloadPlan(turns: request_id -> TurnPlan,
             root_request_ids, children: parent_request_id -> request_ids)
InferenceRequest(request_id, session_id, parent_request_id,
                 input_token_ids, output_token_plan,
                 input_chunk_hashes, ideal_reuse_blocks)
RequestCompletion(request_id, output_token_ids, finish_time_ms)
ClientWakeup(session_id, turn_id, send_time_ms)
```

根轮次只有到达时刻，子轮次只有 parent 和 think time；每个请求恰属一个 session。输入 token 数必须等于 `prompt_len`，完成后的输出 token 数必须等于 `output_len`；hash 列表只包含完整 block。完整 token/完整 hash 不经默认 API 回显。对现有仅有长度的 `Arrival` 调用方，在入口转换为单根 session 请求，使用稳定种子补齐内容；不要让它成为绕过 hash/cache 的第二条执行路径。

### `workload.py`：一次规划、到时发送

保留现有长度分布采样函数；生成器改为产生不可变 `WorkloadPlan`，不接触 Engine、cache 或事件队列。分别用 `stable_derive(seed, domain)` 派生根到达、轮次长度、session 拓扑、内容和命中分配的独立随机流，禁止某个维度新增采样扰动其他维度。根和各请求唯一后缀使用 session/request 域隔离，禁止随机内容偶然跨 session 命中；显式公共前缀开启时将这部分共享计入 ideal。

**由目标请求 QPS 构造计划**（首期线性 session）：

1. 令 `N=num_requests`、`S=num_sessions`、`q=arrival_rate`（请求/秒），校验 `N >= S >= 1`、`q > 0` 且有限。每个 session 至少一个根；先确定轮数 `n_s`，保证 `sum_s n_s=N`。首期均衡分配：`n_s = floor(N/S) + 1` 的 session 有 `N % S` 个，其余为 `floor(N/S)`；额外轮次给哪些 session 由拓扑独立随机流生成稳定顺序。按 session/轮次固定顺序分配 request ID，建立线性 parent/children 边。
2. 平均每个根带来 `m=N/S` 个请求，设根的泊松到达率 `lambda_root = q/m = q*S/N`（session/秒）。例如 `N=1000`、`S=200`、`q=50` 时每 session 平均 5 轮，根速率为 10 session/秒，而不是 50 session/秒。用根到达独立随机流依次采样 `delta_j ~ Exp(lambda_root/1000)`（毫秒），令 `t_j = sum_{k=0..j} delta_k`，`j=0..S-1`；首个根同样采样间隔，不强置 `t_0=0`。把 `t_j` 写入该 session 根轮次的 `root_arrival_time_ms`，只为根轮次生成到达时刻。
3. 用长度独立随机流确定所有轮次的 `prompt_len`、`output_len`，为子轮次只记录 `parent_request_id`、`think_time_ms`，再按下述 block 规则分配理想复用量和内容种子。计划仅包含 `S` 个已知根发送时刻及 `N-S` 个依赖父完成的轮次；不得预测子轮次的发送时刻，也不得为了追逐实测 QPS 重排、提前发送或反馈调整根到达。

该换算的含义是：若持续以此根速率注入同样平均轮数的 session，且所有轮次最终完成，则**长期平均**总请求到达率为 `lambda_root * (N/S) = q`；它不是对这一次有限仿真的精确 QPS 保证。泊松采样有波动，启动/收尾、think time、排队及处理能力会改变实际子轮次到达时刻；引擎无法完成所有轮次时更不能宣称目标达成。尤其 `S=1` 只有一次根到达采样，不能从单场验证目标 QPS；`S=N` 时 `lambda_root=q`，退化为现有单轮请求泊松到达口径。这里是按目标速率构造开放式根到达，不用 Little’s law 反推并发；Little’s law 只适合在稳定条件下检查在途请求数与吞吐/时延的关系。

长度/拓扑确定后，设 `P_i` 为输入长度、`L_parent(i)` 为父请求最终逻辑上下文长度：

```text
M_root = 0
M_i = floor(min(L_parent(i), P_i - 1) / B)  # 子请求最多可共享的 block
D = sum(P_i); M = sum(M_i); r_max = B * M / D
```

目标大于 `r_max` 时返回包含 `r_max` 的不可行错误，不静默截断。取 `[0, M]` 中令 `abs(B*K/D - target)` 最小的整数 `K*`，平局取较小值；`M=0` 时 `K*=0`。按 `q_i=K* M_i/M` 先取整、再按小数余量降序分配剩余 block；平局由独立分配随机流提供稳定顺序。保证 `0 <= K_i <= M_i`、`sum(K_i)=K*`，输出目标、`achieved=B*K*/D`、量化误差。

客户端构造子输入时，复制父请求完成后 `input + output` 上下文的前 `B*K_i` 个 token；若 `K_i<M_i`，下一 token 必须不同；余下用 request 域隔离的唯一 token 填充。生成器可以预定输出计划，但客户端只能在收到父请求实际完成结果后使用其 output token 构造子输入。部分继承表示从 block 边界分叉，不暗示自然语言对话完整继承。

### `prefix_hash.py`：只有内容决定身份

以稳定摘要计算完整输入及随后生成的 output 所形成的完整块：

```text
H_0 = Digest(namespace, ROOT, C_0)
H_j = Digest(namespace, H_(j-1), C_j)
```

`namespace` 至少覆盖模型/权重、tokenizer、KV dtype、adapter/LoRA、影响 KV 的多模态变换以及 `cache_salt`；未配置的模型维度在首期使用稳定固定值，但保留隔离字段。相同 chunk 处于不同历史前缀时不得得到相同身份。不得按 session ID、父请求 ID、长度或目标命中率直接判定 cache hit。注入 hasher 的测试接缝用于验证重复 hash 对索引/引用计数的健壮性，不要求首期 token-by-token 碰撞校验。

**本步完成检查**：同 seed 的计划/token/hash 完全一致；改动某一派生域不影响其他域；`sum(n_s)=N`、根数为 `S`、根间隔按 `q*S/N` 采样；`S=N` 保持单轮到达率口径，`S=1` 不断言单场实测 QPS；所有子轮次都不含预定到达时刻；命中块总量恰为 `K*`，不可行错误包含 `r_max`，无意外跨 session 前缀。

## 3. PC1-B：模拟客户端、协调器与 Engine 边界

### `client.py`

`ClientDriver` 只持有 `WorkloadPlan`、未发送轮次、in-flight ID 和已完成父请求的逻辑上下文；不读 cache 或 Engine waiting/running 状态：

```text
initial_wakeups() -> tuple[ClientWakeup, ...]
on_request_completed(completion, now_ms) -> tuple[ClientWakeup, ...]
on_wakeup(wakeup, now_ms) -> tuple[InferenceRequest, ...]
```

根轮次由 `initial_wakeups` 返回；完成父请求才为其 children 登记 `finish_time_ms + think_time_ms` 的发送事件；`on_wakeup` 仅在 `now_ms == send_time_ms` 构造请求并移除 ready 状态。相同 wakeup 不可重复提交；线性链首期每次至多一个子 wakeup，但接口保留分支/批量能力。

### `events.py`、`coordinator.py`、`engine.py`、`runner.py`

将 `_Engine` 中的 `heapq`、`EventFactory` 和时钟推进移到 `SimulationCoordinator`；在 `events.py` 增加 `CLIENT_SEND`、完成结果路由所需事件类型与单调序号。协调器在当前时刻调用 `client.on_wakeup` 并立即 `engine.submit(request, now_ms)`；Engine 入 waiting queue 并当场记录 `Arrival`，不能预存未来请求或读取 `WorkloadPlan`。Engine 保留 step、scheduler、KV 所有权，向协调器登记未来 EngineStep/RequestDone，完成时返回通用 `RequestCompletion`。`runner.simulate` 组合 workload、client、engine、coordinator 并最终汇总结果；不在客户端引入第二套事件堆或真实 `sleep()`。

同刻顺序固定为：`RequestDone + KV publish/release → completion 路由与 client 回调 → ClientSend/on_wakeup → engine.submit/Arrival → EngineStep`。若多请求同刻完成，所有完成结算先于该时刻的发送和 step；`think_time_ms=0` 的子请求可见父请求刚发布的 block。空闲期直接跳至下一事件；忙碌时不在前一步终点之前开启新 step。

**本步完成检查**：子请求在父完成前既不在 Engine waiting 中，也没有引擎侧的预计到达时间；同刻事件序列稳定；旧单轮调用可通过适配器运行。PC1-B 测试可用虚拟 Engine 验证路由，不能将无缓存路径作为最终产物。

## 4. PC2：L1 paged KV block manager

### `kv_cache.py`：以物理块替换 token 账本

现有 `KVCache` 的 `request_id -> token_count`、`force=True` 和超预算增长不能作为最终缓存实现。新增/替换为 `HierarchicalKVCache` 的 L1 模式；`KVBlock` 至少包含 `block_id`、`content_hash`、`valid_tokens`、`tier`、`state`、`ref_count`、`last_access_seq`、`owner_generation`。内部维护 `blocks`、每请求有序 `request_blocks`、`hash_index: chunk_hash -> ordered block_ids`、L1 free/evictable LRU、单调 `epoch` 和 `access_seq`。同一 hash 允许对应多个物理块；generation 防止回收位置的旧引用命中。

```text
snapshot() -> CacheSnapshot
probe(snapshot, input_chunk_hashes) -> PrefixProbe
prepare_batch(snapshot, probes, allocations) -> CacheTransaction
commit(transaction) -> BatchCacheResult     # 批量原子 acquire
publish(request_id, completed_block_hashes)
release(request_id)
```

`probe` 从输入首块查最长连续前缀，遇 miss 停止；相同 hash 的候选按稳定 block ID 选择，返回命中位置及来源层。它必须只读，不更新 LRU 或引用。`prepare_batch` 根据同一 snapshot 预留所有候选所需空间，`commit` 原子完成引用、未命中块分配和必要驱逐；失败不留下部分分配、错误 LRU 或破坏旧索引。`publish` 仅将已算完的完整 block 加入共享索引，允许每次传一个或多个；step 内新 block 不对同 step 其他请求可见。`release` 减少引用，完整可缓存块变为 evictable，partial 块释放。

L1 `ACTIVE` 引用数大于 0 且不可淘汰；`EVICTABLE` 引用数为 0，按 `(last_access_seq, block_id)` 选择 victim。只有成功提交的 acquire/hit 才刷新访问序号；probe、miss、未准入和失败事务不刷新。物理容量按 block 计算，共享块仅计一次，活动请求的逻辑 context 分别计数；分别统计有效 token、block capacity 与内部碎片。partial tail 仅活动请求独占，不进入 hash index。

**硬容量与进展**：准入和 step 规划须覆盖当前输入及之后 decode 的活动 KV 增长，不能沿用旧的“空 batch 强制准入并允许超限”。对已确定的 output_len 在准入时预留最坏情况下的未来增长空间，计入保留量但不伪装成已驻留物理块；多请求批量预留须去重共享物理块并原子提交，后续增长消耗预留。单请求独占空 L1 仍无法容纳其活动上下文时明确报不可执行/OOM；暂时放不下则保持 FCFS 队头等待，不跳过。每一步保证 `resident_blocks + reserved_future_blocks <= l1_capacity_blocks`，并在释放时归还预留，避免 decode 中途超限或无限空转。

**本步完成检查**：只读 probe、重复 hash、共享引用、partial tail、block ID 复用、LRU/失败回滚、容量守恒及未来增长预留均通过 cache 单元测试；不需要先接入 Engine。

## 5. PC3A-A：调度、执行与经验性能模型

### `scheduler.py`、`engine.py`

保持现有 FCFS 和一次性 Prefill。每个 EngineStep 从 L1 取一次 `CacheSnapshot`，对候选做只读 probe，再按 `max_batch_size` 和 cache 规划结果确定本步可准入的连续队头；同 batch 候选全部基于该 snapshot，成功后一次提交事务。不能在探测阶段 `reserve`，也不能让先遍历的请求看到同 step 的新 KV。未准入候选不改变 cache。`StepPlan` 应返回逐请求 matched/reused token、逻辑 context、新分配 block、实际 Prefill 工作量和可执行准入结果；Engine 不根据目标理想命中直接跳过 token。

Prefill step 结束时，按实际算完的 token 进度批量 `publish` 输入中形成的完整块；首次输出 token 在当前引擎 Prefill step 产生，后续输出 token 在 decode step 产生。仿真约定本步产出的 output token 在步末同时计入其 KV，累积成完整块即追加 hash 链并发布（包括最终 token），不在生成前发布。父请求完成、release 后，完整 output block 可命中后续子请求，partial output 不共享。`RequestCompletion` 必须返回实际生成的 `output_token_ids`。同一步重复计算出同 hash 的多个物理块不合并；到下一步再按稳定规则选取。

### `perf_model.py`

一次性 Prefill 的 `new_prefill_tokens = prompt_len - actual_hit_tokens`。保留历史上下文的二次成本，使用增量经验公式而不是把后缀视作独立短 prompt：

```text
prefill_cost(context_before, new_prefill_tokens)
  = prefill_c1 * new_prefill_tokens
  + prefill_c2 * ((context_before + new_prefill_tokens)^2 - context_before^2)
```

`context_before = actual_hit_tokens`；物理 KV 共享不缩短每个请求的 Decode 逻辑 context。L1 命中无 restore 成本；hash/lookup CPU 耗时首期取 0。维持现有 `step_time` 的 Prefill/Decode 融合规则，不能因为引入 cache 擅自将两者改为串行。

**本步完成检查**：首请求 miss，发布后重复前缀 hit 并减少 Prefill；同 step 互不可见；父 output 完整块可命中而 partial 不可；全 prompt 命中仍计算一个输入 token；KV 无超限且事件时序可复现。

## 6. PC3A-B：指标、服务与前端

### `events.py`、`schemas.py`、`metrics.py`、`runner.py`

请求记录增加 session/parent、理想复用 token、物理 matched token、实际 L1 复用 token、实际 Prefill token、逻辑 context、时间区间；step trace 记录查询、命中、分配/驱逐、步首/步末 active/evictable/free、内部碎片。`metrics.compute_metrics` 保留原 TTFT/TPOT/吞吐口径，再汇总目标/构造/量化误差/实际 token 命中率、有命中及“所有可缓存输入完整块均命中”的请求比例、实际/节省 Prefill、eviction 与峰值。可缓存完整块数为 0 的请求不计为“完全命中”。所有比率用输入 token 总数加权，不平均各请求比率。

配置回显中的 `arrival_rate` 是**目标请求 QPS**，已有 `throughput` 是**实测端到端完成吞吐**（`N / ((末请求完成 - 首根到达)/1000)`，请求/秒）；两者不等同，不以吞吐校准根到达率。若未来展示实测到达 QPS，应先定义固定观测窗口，再用窗口内实际提交请求数除以窗口秒数；不能将完成吞吐或首末到达跨度的商直接视为目标达成率。

L1 阶段仍以与后续层级兼容的分层字段输出：L2/L3 hit、restore、迁移均为 0。实际 hit token 仅按 probe 时来源层归因；若最后一个输入 token 必须重算，从其所在匹配块的来源层扣除该 token。至少维持：

```text
actual_hit_tokens <= ideal_hit_tokens <= total_input_tokens
actual_prefill_tokens + actual_hit_tokens = total_input_tokens
L1_hit_tokens + L2_hit_tokens + L3_hit_tokens = actual_hit_tokens
free_blocks + resident_blocks + reserved_future_blocks = tier_capacity_blocks  # L1
active_blocks <= L1_resident_blocks
```

若保留单独的“可用但尚未预留”与“未来已预留”账本，以上 `free_blocks` 专指可用且未预留的容量；L2/L3 没有未来预留时为 `free + resident = capacity`。显式跨 session 前缀也必须计入 ideal，不能破坏 `actual <= ideal`。

### `server/`、`web/`

对外结果模型经 `server/src/simserver/routes.py` 的原有 `/api/simulate` 返回，更新 `server/src/simserver/openapi_export.py` 的导出结果、`web/openapi.json`、`web/src/api/schema.d.ts`、`web/src/config/simulation.ts` 以及使用配置/指标的 UI。旧单轮请求的 `arrival_rate` 仍为请求/秒；多轮时明确标为“总请求目标 QPS”，不再解释为根 session/秒；其余旧字段保留含义与单位。新增参数提供旧请求可用的默认值，结果默认不传完整 token ID/hash。

**本步完成检查**：端到端复现性、上述守恒关系、API schema 与前端类型一致；旧单轮场景继续可运行，新增 session 配置在界面可提交，理想/实际命中与容量指标可查看。

## 7. KV3：在已完成的 L1 接口上扩展 L2/L3

此阶段才在 `schemas.py` 激活 L2/L3 容量、带宽、固定延迟及 `kv_bytes_per_token`；容量为 0 的层关闭。沿用相同 `WorkloadPlan`、链式 hash、probe 与指标分母，不重写客户端或引擎的到达时序。独占式 write-back：新块进 L1；L1 victim 降级 L2，L2 victim 降级 L3，L3 victim 删除；低层 hit 去重提升 L1 并移除原层位置，禁用层跳过。一个物理块同刻只属于一层；每层独立 LRU，降级块成为目标层最新驻留项，目标层满时递归逐出 victim，删除时清理 hash 反向索引。

`kv_cache.py` 的全局 hash directory 记录每块实际 tier；probe 从首块逐一查找，优先 L1→L2→L3，同层按稳定 block ID 选取并返回每块来源。相同低层物理块在同 batch 被多请求命中时只提升和传输一次，建立多个 L1 引用；低层命中必须先拿到 L1 空间，事务失败不得留下部分提升。`scheduler.py` 继续同 snapshot 原子规划；`engine.py` step 末发布及完成释放流程不变。

`perf_model.py` 在原 Prefill/Decode 融合步长之外串行加入 restore（首期不假设与计算重叠），相同物理块共享提升只收一次：

```text
t_step = t_cache_lookup + t_restore + max(t_prefill_total, t_decode)
t_L2 = latency_L2 + bytes_L2 / BW_L2_to_L1
t_L3 = latency_L3 + bytes_L3 / BW_L3_to_L2
     + latency_L2 + bytes_L3 / BW_L2_to_L1
```

`metrics.py` 增加 L1/L2/L3 的 token hit ratio、promotion/demotion/eviction、驻留峰值、restore bytes/time 和理想与实际命中差。请求级分层 token 之和等于实际跳过的 Prefill token；各 tier 分别满足容量守恒。L2/L3 为 0 时应与 PC3A 的 L1 结果一致。

## 8. 后续模块接入边界

- `F001.CB4` 使用同一 `snapshot/probe/commit/publish/release` 接口：命中 token 不占 Prefill token budget，按 chunk 实际完成的完整块增量发布；retraction 释放引用、形成真实容量，避免立即命中自身导致无进展；同 step 保持 snapshot 隔离。此编排不属于 PC3A 的完成条件。
- `F002` 消费命中前缀、实际新计算 token、逻辑 context、历史 KV 读取、新 KV 写入及 restore 信息；可扩展字节级容量与解析成本，但不得改变内容 hash、命中口径或缓存所有权。
- 本设计不实现真实 KV tensor、跨仿真持久化、跨 Engine 路由、radix tree、partial block 跨请求 copy-on-write、inclusive/write-through 策略或计算/传输重叠调度。
