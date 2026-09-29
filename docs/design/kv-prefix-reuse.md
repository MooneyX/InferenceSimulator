# KV 前缀复用与多级缓存设计

## 1. 目标与范围

本设计为推理系统仿真器增加可控的 KV 前缀复用负载，以及 L1/L2/L3 多级 KV 缓存仿真。目标是同时回答两个不同问题：

1. 工作负载在无限缓存下本来具有多少可复用前缀；
2. 给定缓存容量、层级、淘汰和调度时序后，运行时实际命中了多少前缀，以及命中来自哪一层。

首期锁定以下能力：

- 合成多个 session，每个 session 由具有父子关系的请求组成；
- 请求携带输入 token 的链式 chunk hash；
- 用户指定全局理想前缀 token 命中率；
- 生成器在 block 粒度约束下构造最接近目标的请求集合；
- 引擎维护 L1、L2、L3 的 chunk residency，并在请求实际准入时查询最长连续前缀；
- 每层使用确定性 LRU，活动 L1 block 不可淘汰；
- 区分理想命中、实际命中、分层命中、传输成本、淘汰和容量损失；
- 同一配置与 seed 的 session、token、hash、事件、淘汰和指标完全可复现。

本设计扩展 `docs/plans/feature-roadmap.md` 中 `F003` 的单 Engine、L1 prefix cache 基线。实现仍按阶段推进：先完成 session/hash 与 L1，再加入 L2/L3；不能因多级缓存尚未实现而改变 L1 的 hash、请求和指标语义。

## 2. 核心设计决策

### 2.1 命中率必须分为理想值与实际值

用户配置的是 `target_ideal_prefix_hit_rate`，表示工作负载自身具有的可复用程度，不直接控制缓存运行时结果。

- **目标理想命中率**：用户输入值；
- **构造理想命中率**：生成器在整数 block 约束下实际构造出的值；
- **实际命中率**：引擎在准入时，根据当时 L1/L2/L3 residency 得到的值。

容量不足、排队期间淘汰和层级驱逐可以使实际命中率低于构造理想命中率。生成器不得为了补偿运行时 miss 而回看或修改引擎结果，否则 workload 与 engine 将形成反馈依赖，无法复现也无法解释。

### 2.2 主要口径采用全局 token 加权平均

设请求集合为 $R$，请求 $i$ 的输入长度为 $P_i$，理想可复用 token 数为 $H_i^{ideal}$，实际命中 token 数为 $H_i^{actual}$：

$$
r_{ideal}=\frac{\sum_{i\in R}H_i^{ideal}}{\sum_{i\in R}P_i}
$$

$$
r_{actual}=\frac{\sum_{i\in R}H_i^{actual}}{\sum_{i\in R}P_i}
$$

session 根请求的 $H_i^{ideal}=0$，但其输入 token 仍进入分母。该定义反映整场仿真实际减少的 Prefill token 比例，避免将短请求和长请求等权平均。

请求级“有任意命中的请求比例”和“平均请求命中率”作为辅助指标，不替代全局 token 口径。

### 2.3 hash 是内容身份，不是命中结果

请求生成器负责构造确定的 token 内容与链式 chunk hash。请求中的 `ideal_reuse_blocks` 表示生成器有意与父请求共享的前缀；引擎只能根据 hash 和运行时 residency 判定实际命中。

不得把目标命中率直接写成请求的 `cached_tokens`，也不得根据 `session_id`、`parent_request_id` 或相同长度直接判定命中。

### 2.4 session 首期是内容继承树

session 内请求形成有根树；首期默认生成线性链，接口允许后续增加分支。每个非根请求引用一个已完成父请求，并从父请求最终逻辑上下文的起点复制若干完整 block，再生成唯一后缀。

这是用于控制前缀共享比例的合成内容继承模型。若只复制父上下文的一部分，它表示从该 block 边界分叉出的会话分支，不承诺是自然语言意义上的完整连续对话。

### 2.5 模拟客户端拥有未来请求与发送时序

子请求的输入可能包含父请求生成的 output token，因此不能在仿真开始时把所有子请求及预计到达时间交给引擎。负载生成器只生成不可变 `WorkloadPlan` 并交给模拟客户端；推理引擎不读取 session 拓扑、不保存未来请求，也不根据父子关系主动创建请求。

- `SimulationCoordinator` 持有唯一的 DES 时钟和事件队列；
- 模拟客户端根据计划为根轮次登记 `ClientSend` 事件；
- `ClientSend` 到达当前仿真时刻后，客户端才构造 `InferenceRequest` 并调用 `engine.submit(request, now_ms)`；
- 引擎只接收当前时刻真正到达的请求，并将其加入 waiting queue；请求中不携带未来到达时间；
- 引擎完成请求后产生通用 `RequestCompletion`，协调器将其交给客户端；
- 客户端收到完成结果后更新 session 历史，并为子轮次登记 `send_time = finish_time + think_time_ms` 的未来 `ClientSend`；
- `think_time_ms=0` 时沿用同刻事件全序：父请求完成与 KV release/publish 先于完成结果交付、`ClientSend`、`engine.submit` 和后续 EngineStep。

未来 `ClientSend` 事件语义上归客户端所有，物理上统一存放在协调器的事件队列中；不能让客户端和引擎各自维护独立时钟或独立事件堆。客户端维护尚未发送的计划轮次、session 历史和 in-flight 请求；引擎只维护已经实际提交的 waiting/running 请求。请求提交后不得同时留在客户端 ready 状态和引擎 waiting queue 中。

该交互是同一仿真进程内、基于不可变值对象的同步事件路由，不引入线程、协程、消息总线、真实网络或真实 `sleep()`。`SimulationCoordinator` 是全局时钟和全部未来事件的唯一所有者，Engine 只处理协调器在当前时刻交付的事件和请求。

### 2.6 多级缓存默认采用独占式 write-back 层级

首期多级策略锁定为 exclusive write-back：一个物理 KV block 同一时刻只属于一个 tier；同一 hash 仍可因同 step 重复计算而对应多个物理 block。

- 新计算的 block 进入 L1；
- L1 淘汰时向 L2 降级；
- L2 淘汰时向 L3 降级；
- L3 淘汰时彻底删除；
- L2/L3 命中后提升到 L1，原层副本移除；
- 容量为 0 的 tier 视为关闭，降级时跳过该层。

该策略使总有效容量等于各层容量之和，并避免首期引入副本一致性和 write-through 写放大。后续可增加 inclusive/write-through 策略，但不得改变 hash、probe 和指标口径。

## 3. 请求与 session 数据模型

### 3.1 计划、运行时请求与完成结果

负载生成器输出的 `WorkloadPlan` 与客户端真正提交的 `InferenceRequest` 分离。计划中的每轮至少包含：

```text
TurnPlan
  request_id: int
  session_id: int
  parent_request_id: int | None
  root_arrival_time_ms: float | None
  think_time_ms: float | None
  prompt_len: int
  output_len: int
  ideal_reuse_blocks: int
  unique_suffix_seed: int
  output_token_plan: TokenPlan
```

模拟客户端根据 `TurnPlan`、已完成父请求的 output token 和当前 session 历史构造：

```text
InferenceRequest
  request_id: int
  session_id: int
  parent_request_id: int | None
  input_token_ids: tuple[int, ...]
  output_token_plan: TokenPlan
  input_chunk_hashes: tuple[ChunkHash, ...]
  ideal_reuse_blocks: int
```

引擎完成请求后返回：

```text
RequestCompletion
  request_id: int
  output_token_ids: tuple[int, ...]
  finish_time_ms: float
```

客户端未来发送事件使用轻量唤醒值对象，不包含完整请求：

```text
ClientWakeup
  session_id: int
  turn_id: int
  send_time_ms: float
```

`ClientWakeup` 只由客户端生成并交给协调器登记。它不能进入引擎，也不能被引擎解释为已经到达的请求；完整 `InferenceRequest` 只在 wakeup 触发后由客户端构造。

约束：

- 根轮次设置 `root_arrival_time_ms`，不设置 parent 与 think time；
- 子轮次设置 parent 与 think time，不设置根到达时间；
- 子请求必须在收到父请求 `RequestCompletion` 后才能构造，不能提前读取父请求 output；
- `output_token_plan` 可以由生成器为可复现性预先确定，但客户端只能通过完成结果获得父请求的实际 `output_token_ids`；
- `input_token_ids` 长度等于计划的 `prompt_len`，完成结果 token 数等于 `output_len`；
- `input_chunk_hashes` 只包含完整 chunk；
- 完整 token 序列仅为 core 内部数据，默认不经 API 回显；
- 所有请求统一使用该 session 请求模型；单轮 workload 表示为只含一个根轮次的 session；
- 每个请求始终生成 prefix hash，并执行 cache probe、acquire、publish 与 release，不提供关闭开关。

### 3.2 workload 与模拟客户端接口

负载生成器产生不可变 session workload plan：

```text
WorkloadPlan
  turns: request_id -> TurnPlan
  root_request_ids: tuple[int, ...]
  children: parent_request_id -> tuple[request_id, ...]
```

计划只交给模拟客户端，不交给推理引擎。客户端采用最小同步接口：

```text
ClientDriver.initial_wakeups() -> tuple[ClientWakeup, ...]
ClientDriver.on_request_completed(
  completion: RequestCompletion,
  now_ms: float,
) -> tuple[ClientWakeup, ...]
ClientDriver.on_wakeup(
  wakeup: ClientWakeup,
  now_ms: float,
) -> tuple[InferenceRequest, ...]
```

`initial_wakeups` 返回根轮次的发送事件；`on_request_completed` 更新客户端状态，并为零个或多个子轮次返回未来发送事件；`on_wakeup` 只在 `now_ms == wakeup.send_time_ms` 时构造当前真正要提交的请求。线性 session 首期每次完成最多产生一个子 wakeup，每次 wakeup 最多提交一个请求，接口保留分支和批量客户端扩展能力。只有一个请求的 workload 表示为只含一个根轮次的 session。

`SimulationCoordinator` 负责登记 `ClientWakeup`、在对应时刻调用 `on_wakeup`，并立即将返回的请求传给 `engine.submit(request, now_ms)`。引擎提交接口不接受 `arrival_time`，也不保存 `ClientWakeup`；调用发生的 `now_ms` 就是实际到达时间。这样未来请求、客户端状态与引擎 waiting queue 具有明确边界。

### 3.3 token 内容生成

各类随机性使用稳定派生的独立随机流：

```text
root_arrival_rng = RNG(stable_derive(seed, "root-arrival"))
length_rng = RNG(stable_derive(seed, "turn-length"))
content_rng = RNG(stable_derive(seed, "session-content"))
topology_rng = RNG(stable_derive(seed, "session-topology"))
allocation_rng = RNG(stable_derive(seed, "hit-allocation"))
```

根 session 到达、每轮长度、session 拓扑、token 内容和命中分配互不消费对方的随机流。新增其中任一维度不得改变其他维度的结果。派生函数必须跨进程稳定，不能使用 Python 内置 `hash()`。

不同 session 的根 token 和所有唯一后缀通过 session/request 域隔离，避免非预期跨 session 命中。需要模拟跨 session 公共 system prompt 时，必须通过独立显式配置开启，不能由随机碰撞产生。

## 4. 链式 chunk hash

### 4.1 基本公式

设 block size 为 $B$，缓存 namespace 为 $N$，第 $j$ 个完整 token chunk 为 $C_j$：

```text
H_0 = Digest(N, ROOT, C_0)
H_j = Digest(N, H_(j-1), C_j)
```

`Digest` 使用 SHA-256、BLAKE2 等稳定摘要。`N` 至少包含：

- 模型和权重版本；
- tokenizer 版本；
- KV dtype；
- adapter/LoRA；
- 多模态或其他影响 KV 的输入变换；
- 用户配置的 `cache_salt`。

只 hash 当前 chunk 是错误的：相同 token chunk 位于不同历史前缀时，其 KV 不同。

### 4.2 完整 block 与尾部

- 只有完整 block 进入共享索引；
- partial tail 由活动请求在 L1 独占持有，释放后不进入 L2/L3；
- 子请求可复制父请求的 partial tail 作为内容，但只能命中其之前的完整 block；
- `kv_block_size` 同时决定 hash、容量和共享粒度，不能与 workload 的 chunk size 分别配置。

### 4.3 output KV 与多轮对话

真实多轮对话的子请求通常包含父请求的输入和输出。为使这些 token 可复用，合成 workload 必须预先生成确定的 `output_token_ids`，引擎按实际 decode 进度建立对应 KV：

- 每形成一个完整 block，step 完成后发布其 hash；
- 未完成 decode 的 token 不得提前发布；
- 父请求完成后，完整 output block 可继续驻留并被子请求命中；
- output partial tail 不跨请求共享。

这扩展了 `F003` 仅复用 prompt block 的基线，但不要求仿真真实 token 采样概率。

### 4.4 全 prompt 命中

prefix cache 保存 KV，不保存下一 token logits。需要区分：

- `matched_cache_tokens`：通过完整 block hash 找到物理 KV 的 token 数；
- `reused_prefill_tokens`：实际跳过 Prefill 计算的 token 数，也是本文主要命中率的分子。

即使所有输入 block 都匹配，请求仍至少执行最后一个输入 token 对应的计算：

$$
H_i^{reused}=\min(H_i^{matched}, P_i-1)
$$

已有物理 block 不重复分配。合成 session 为每个子请求保留至少一个唯一新 token，因此理想命中构造继续满足：

$$
H_i^{ideal}=B K_i \le B\left\lfloor\frac{P_i-1}{B}\right\rfloor
$$

引擎仍须正确处理外部 trace 中“完整输入逐 block 匹配”的情况，并分别记录 matched 与 reused token。

## 5. 理想命中率构造

### 5.1 可行容量

对每个非根请求 $i$，设父请求完成后的逻辑上下文长度为 $L_{parent(i)}$。其最多可有意复用的 block 数为：

$$
M_i=\left\lfloor\frac{\min(L_{parent(i)}, P_i-1)}{B}\right\rfloor
$$

根请求的 $M_i=0$。令：

$$
D=\sum_i P_i,\quad M=\sum_i M_i
$$

则当前 session 形状和长度下，可构造的最大理想命中率为：

$$
r_{max}=\frac{B\cdot M}{D}
$$

若用户目标大于 $r_{max}$，生成器必须返回明确的不可行错误，并给出 `r_max`。不得静默截断为最大值。上层可调整 session 数、每 session 请求数、输入长度或 output 长度后重试。

### 5.2 block 量化

任意有限 workload 的命中 token 只能以 block 为单位变化。目标总命中 block 数取：

$$
K^*=\operatorname*{argmin}_{K\in\{0,1,\ldots,M\}}
\left|\frac{B\cdot K}{D}-r_{target}\right|
$$

距离相同时选择较小的 $K$，避免构造结果超过用户目标。构造误差为：

$$
\epsilon=\left|\frac{B\cdot K^*}{D}-r_{target}\right|
$$

结果必须回显 `target`、`achieved` 和 `quantization_error`。只要目标可行，理论量化误差不超过 $B/(2D)$。

### 5.3 将目标分配到请求

为避免把所有复用集中到少数早期请求，按候选容量比例分配：

1. 对每个非根请求计算理想份额 $q_i=K^*M_i/M$；
2. 初值 $K_i=\lfloor q_i\rfloor$；
3. 按小数余量从大到小分配剩余 block；
4. 余量相同按由 `allocation_rng` 生成的稳定请求顺序打破平局；
5. 保证 $0\le K_i\le M_i$ 且 $\sum_i K_i=K^*$。

`ideal_reuse_blocks=K_i`，理想命中 token 为 $H_i^{ideal}=B K_i$。

### 5.4 构造恰好为 `K_i` 的公共前缀

对请求 $i$：

1. 复制父请求最终上下文的前 $B K_i$ 个 token；
2. 若 $K_i<M_i$，保证位置 $B K_i$ 的 token 与父请求不同；
3. 其余输入使用 request 域隔离的唯一 token 流填充；
4. 重新从根计算整条输入 hash 链。

由于 hash 包含父 hash，第一个差异 block 之后的所有 hash 都不同，因此该请求与父请求恰好共享前 $K_i$ 个完整 block。唯一内容生成还应保证它不会意外命中其他 session 或其他分支。

## 6. L1/L2/L3 缓存模型

### 6.1 层级语义

| 层级 | 抽象介质 | 可保存内容 | 命中后的动作 |
|---|---|---|---|
| L1 | GPU HBM | active、evictable 完整 block；活动 partial block | 直接建立共享引用 |
| L2 | Host DRAM | inactive 完整 block | 计入 H2D 传输并提升至 L1 |
| L3 | 本地/远程存储 | inactive 完整 block | 计入 L3 read 与 H2D 传输并提升至 L1 |

三层只模拟对调度、容量和时延有影响的元数据与字节数，不保存真实 KV tensor。

L1 容量与正在执行请求的 KV 共用，不能另建一套 prefix cache 预算。L2/L3 分别拥有独立容量。

### 6.2 block 状态

```text
KVBlock
  block_id
  content_hash
  valid_tokens
  tier: L1 | L2 | L3
  state: ACTIVE | EVICTABLE
  ref_count
  last_access_seq
  owner_generation
```

约束：

- `ACTIVE` 只允许位于 L1，且 `ref_count > 0`；
- L1 `EVICTABLE` 与所有 L2/L3 block 的 `ref_count=0`；
- partial block 不进入 `hash_index`，也不进入 L2/L3；
- `owner_generation` 防止物理 block 复用后旧引用误命中；
- 一个 hash 可以映射到多个 block，以容纳同 step 重复计算；
- 每个物理 block 同一时刻只计入一个 tier 的容量。

### 6.3 索引与目录

`HierarchicalKVCache` 至少维护：

```text
blocks: block_id -> KVBlock
request_blocks: request_id -> ordered BlockRef list
hash_index: chunk_hash -> ordered block_id collection
tier_free: tier -> free block collection
tier_lru: tier -> evictable block ordering
epoch: monotonic cache version
access_seq: monotonic committed-access sequence
```

`hash_index` 是全局目录，查询结果包含实际 tier。每层的容量、resident、free、active、evictable 和峰值分别统计。

### 6.4 最长连续前缀查询

请求准入时从输入的第一个完整 block 开始逐个查询：

1. 对同一 hash 优先选择 L1，其次 L2、L3；
2. 同层多个候选按确定性 block ID 规则选择；
3. 任一层均不存在时立即停止；
4. 不允许跳过 miss 后继续命中后续 block；
5. 返回每个命中 block 的来源 tier；
6. `probe` 只读，不刷新 LRU、不提升、不分配、不淘汰。

连续命中 block 数乘以 $B$ 得到 `matched_cache_tokens`；再按“至少计算一个输入 token”规则得到 `reused_prefill_tokens`。后文未特别说明时，`actual_hit_tokens` 指后者。

### 6.5 step 级原子事务

同一 EngineStep 的候选请求必须基于同一 `CacheSnapshot` 规划，随后以一个 `CacheTransaction` 原子提交：

```text
snapshot() -> CacheSnapshot
probe(snapshot, request_hashes) -> PrefixProbe
prepare_batch(snapshot, probes, allocations) -> CacheTransaction
commit(transaction) -> BatchCacheResult
publish(request_id, completed_hashes)
release(request_id)
```

事务需要：

- 为 L1 命中增加共享引用；
- 对 L2/L3 命中去重并安排一次物理提升；
- 为实际 miss 分配 L1 block；
- 为获得空间执行确定性降级和淘汰；
- 同一低层 block 被多个请求命中时只传输一次、建立多个 L1 引用；
- 全部成功后统一更新 LRU、引用、索引和 epoch；
- 失败时不留下部分提升、部分引用或被错误刷新的 LRU。

step 内新计算并在末尾发布的 hash 对本 step 其他请求不可见。该约束避免请求遍历顺序改变同批命中结果。

### 6.6 LRU 与层级驱逐

每层独立维护 LRU，排序键为：

```text
(last_access_seq, block_id)
```

规则：

- 只有成功提交的 acquire/hit 才刷新访问序号；
- probe、miss、未准入候选和失败事务不刷新；
- L1 active block 不进入淘汰候选；
- L1 victim 降级到下一个启用层，并作为该层最新驻留项；
- 下层空间不足时先递归驱逐其 LRU victim；
- L3 victim 被删除，并同步清理 hash 反向索引；
- lower-tier hit 提升到 L1 时原层位置释放；
- 所有选择只依赖稳定序号和 ID，不依赖 set/dict 的未定义遍历顺序。

### 6.7 容量失败

启用层级 KV cache 后，容量是硬约束：

- active block 不得为接纳新请求而淘汰；
- 无法通过驱逐 inactive block 获得足够 L1 空间时，本次准入失败；
- 若单请求即使在空 L1 中也无法容纳其活动 KV，返回明确的不可执行/OOM 结果；
- 不沿用当前 `KVCache` 的“强制准入并允许物理超限”行为；
- prefix cache 始终参与容量管理，不存在绕过 block manager 的 token 账本路径。

## 7. 时序与性能成本

### 7.1 客户端发送与请求准入流程

```text
WorkloadGenerator 生成不可变 WorkloadPlan
    ↓
ClientDriver.initial_wakeups() 返回根 ClientWakeup
    ↓
SimulationCoordinator 登记 ClientSend 事件
    ↓
全局时钟推进到 wakeup.send_time
    ↓
ClientDriver.on_wakeup() 构造 InferenceRequest
    ↓
coordinator 调用 engine.submit(request, now)
    ↓
引擎此时才取得请求所有权并加入 waiting queue
    ↓
EngineStep：snapshot/probe → 调度 → cache commit → 执行
    ↓
step 结束后 publish 已完成完整 block
    ↓
RequestDone：release 引用；完整 block 转为 L1 evictable
    ↓
RequestCompletion 经 coordinator 交付 ClientDriver
    ↓
客户端更新 output 历史并返回子 ClientWakeup
    ↓
coordinator 登记 send_time = finish_time + think_time 的 ClientSend
```

客户端完成回调只更新自身状态并安排未来 wakeup，不把未来请求或预计到达时间交给引擎。到达时刻真正发生后，客户端才通过 `on_wakeup` 构造请求并调用引擎提交接口。协调器拥有全局时间和事件堆；客户端拥有 `ClientSend` 的业务状态；引擎拥有提交后的 waiting/running 状态。

同刻事件顺序扩展为：

```text
RequestDone / KV publish-release
→ RequestCompletion 路由与 ClientDriver 回调
→ ClientSend / ClientDriver.on_wakeup
→ engine.submit / Arrival
→ EngineStep
```

因此 `think_time_ms=0` 的子请求可以命中父请求刚转为可复用状态的 KV，同时不能在父请求完成前到达。引擎 trace 中的 `Arrival` 在 `engine.submit` 时产生，其时间等于协调器当前时间，而不是预先存储的预计到达时间。

### 7.2 分层命中成本

- L1 hit：不产生 Prefill 计算和层间传输；
- L2 hit：不产生对应 Prefill 计算，产生 L2→L1 传输；
- L3 hit：不产生对应 Prefill 计算，产生 L3 read 和到 L1 的传输；
- miss：产生 Prefill 计算和 L1 KV 写入。

首期采用可解释的串行关键路径：

$$
t_{step}=t_{cache\ lookup}+t_{restore}+t_{prefill/decode}
$$

其中：

$$
t_{restore,L2}=latency_{L2}+\frac{bytes_{L2}}{BW_{L2\rightarrow L1}}
$$

$$
t_{restore,L3}=latency_{L3}+\frac{bytes_{L3}}{BW_{L3\rightarrow L2}}+
latency_{L2}+\frac{bytes_{L3}}{BW_{L2\rightarrow L1}}
$$

多个请求对同一物理 block 的共享提升只计一次传输。计算与传输重叠作为后续性能模型能力，首期不隐式假设重叠。

### 7.3 Prefill 工作量

实际 Prefill token 数为：

$$
P_i^{compute}=P_i-H_i^{actual}
$$

但现有二次经验公式不能简单把它当成一个独立短 prompt。经验模型使用增量形式：

```text
prefill_cost(context_before=actual_hit_tokens,
             new_prefill_tokens=P_i_compute)
```

解析性能模型后续显式消费 Query 长度、逻辑 context、历史 KV 读取、新 KV 写入和层间传输。

## 8. 配置归属

### 8.1 workload 配置

建议在 synthetic workload 下增加：

```text
session:
  num_sessions
  topology: linear
  think_time_ms
  target_ideal_prefix_hit_rate
  content_seed: optional
  allow_cross_session_shared_prefix: false
```

约束：

- `1 <= num_sessions <= num_requests`；
- 每个请求必须且只属于一个 session，每个 session 至少包含一个根轮次；
- 请求按 session 稳定分配，首期各 session 为线性链；
- 单轮 session 是最小合法 workload；
- `target_ideal_prefix_hit_rate` 范围为 `[0, 1]`；
- 目标可行性在请求长度和拓扑物化后校验；
- `content_seed` 未配置时由已有 seed 稳定派生。

目标命中率属于 workload 属性，不放入 engine cache 配置。

### 8.2 engine/cache 配置

建议增加：

```text
prefix_cache:
  block_size_tokens
  cache_salt
  eviction_policy: lru
  hierarchy_policy: exclusive_write_back
  l1:
    capacity_blocks
  l2:
    capacity_blocks
    read_bandwidth_bytes_per_ms
    fixed_latency_ms
  l3:
    capacity_blocks
    read_bandwidth_bytes_per_ms
    fixed_latency_ms
  kv_bytes_per_token
```

`prefix_cache` 是必需的 resolved config，不包含 `enabled` 字段；prefix hash、lookup 和 block manager 始终运行。`block_size_tokens` 是整场仿真的唯一 KV block size。若容量仍以现有 `kv_token_budget` 输入，由 resolved config 向下取整为 L1 block 数；不能同时保留两套独立 L1 容量。

L2/L3 容量为 0 时关闭对应层。未提供字节级模型时可直接配置每 block 传输耗时，作为经验后端兼容路径。

## 9. Trace 与指标

### 9.1 请求级记录

每个请求至少记录：

- `session_id`、`parent_request_id`；
- 输入/输出长度；
- `ideal_reuse_tokens`；
- `matched_cache_tokens`；
- `actual_l1_hit_tokens`、`actual_l2_hit_tokens`、`actual_l3_hit_tokens`（均为实际跳过 Prefill 的 token）；
- `actual_prefill_tokens`；
- restore bytes/time；
- 到达、排队、Prefill、Decode 和完成时间。

默认不返回 token IDs 和完整 hash 列表；调试 trace 可选择返回 hash 摘要。

### 9.2 聚合指标

必须输出：

- `target_ideal_prefix_hit_rate`；
- `constructed_ideal_prefix_hit_rate`；
- `ideal_hit_quantization_error`；
- `actual_prefix_hit_rate`；
- L1/L2/L3 各自的 token hit ratio；
- `ideal_hit_tokens - actual_hit_tokens` 的容量/时序损失；
- 实际计算与避免计算的 Prefill token；
- 有任意命中的请求比例、完全命中的请求比例；
- 各层 hit、promotion、demotion、eviction block 数；
- 各层 resident/active/evictable 峰值；
- restore bytes/time 与估计节省的 Prefill 时间；
- block 内部碎片。

分层 hit token 只按 probe 时的来源 tier 归因一次。若完整输入均匹配而必须重算最后一个 token，则从最后一个匹配 block 的来源 tier 中扣除该 token：

$$
H_i^{actual}=H_{i,L1}+H_{i,L2}+H_{i,L3}
$$

### 9.3 守恒关系

结果必须满足：

```text
actual_hit_tokens <= ideal_hit_tokens <= total_input_tokens
actual_prefill_tokens + actual_hit_tokens = total_input_tokens
L1_hit_tokens + L2_hit_tokens + L3_hit_tokens = actual_hit_tokens
free_blocks + resident_blocks = tier_capacity_blocks
active_blocks <= L1_resident_blocks
```

在启用显式跨 session 公共前缀后，`actual_hit_tokens <= ideal_hit_tokens` 仍须成立：生成器应把这类共享计入 ideal lineage，而不是视作偶然命中。

## 10. 模块边界

增加 `client` 与 `coordinator` 模块承接动态 session 行为和全局 DES 时间，并保持 Engine 对可变缓存的唯一所有权：

```text
schemas          通用不可变请求、完成结果、wakeup 与配置
  ├─ prefix_hash 稳定 namespace、token 分块与链式 hash
  ├─ workload    session 拓扑、理想命中分配与不可变 WorkloadPlan
  │    ↓
  │  client      消费计划、维护 session、产生 ClientWakeup、构造到达请求
  └─ kv_cache    L1/L2/L3 block、目录、LRU 与原子事务
       ↓
     scheduler   snapshot/probe、准入与 batch cache transaction
       ↓
     engine      实际到达队列、step、cache 生命周期与 RequestCompletion

coordinator      唯一仿真时钟与事件队列
  client ClientWakeup → ClientSend → client.on_wakeup
  client InferenceRequest → engine.submit(now)
  engine RequestCompletion → client.on_request_completed
       ↓
metrics          理想/实际/分层命中与性能收益汇总
```

边界规则：

- `workload` 只生成计划，不访问运行时 cache、Engine 或事件队列；
- `client` 独占 session 历史、父子关系、think time、未来发送状态和 in-flight 请求，不读取 cache；
- `coordinator` 独占全局仿真时钟和事件堆，负责路由 `ClientSend`、Engine 事件与完成结果；
- `engine` 不持有未来 `ClientWakeup` 或尚未到达的 `InferenceRequest`，不读取 `WorkloadPlan`，不查询 children，也不解释 think time；
- `engine.submit(request, now_ms)` 只接收当前协调器时刻真正到达的请求，并在该调用处记录 Arrival；
- `client` 不维护已提交请求的 Engine waiting/running 状态；提交后的排队和执行归 Engine；
- `kv_cache` 不感知仿真时间、客户端或事件队列；
- `scheduler` 不发布未完成 block；
- `engine` 负责 step 和 cache 生命周期，并只产出通用 `RequestCompletion`；
- `coordinator` 是 `client` 与 `engine` 的唯一连接点，模块间只交换不可变值对象；
- `perf_model` 只消费不可变 workload/cost 描述，不读取 cache 可变状态。

客户端和引擎不得分别维护可独立推进的事件堆。未来增加网络、取消、流式响应或多个服务实例时，应扩展协调器的事件类型，同时保持 Engine 只接收实际到达请求的边界。

## 11. 实施阶段

### KV1：session workload、模拟客户端与理想命中构造

- 增加 `TurnPlan`、`WorkloadPlan`、`InferenceRequest`、`RequestCompletion` 和 `ClientWakeup`；
- 保留原有负载生成器，由其生成不可变 session 计划并交给 `ClientDriver`；
- 增加 `SimulationCoordinator`，将唯一仿真时钟和事件队列从 Engine 边界中提升出来；
- 实现 `initial_wakeups`、`on_request_completed` 与 `on_wakeup`，由客户端维护未来发送状态和 session 历史；
- `ClientSend` 到时才构造请求并调用 `engine.submit(request, now_ms)`，Engine 不保存未来请求；
- 使用同一全局事件队列登记客户端发送和 Engine 执行事件，不增加真实等待；
- 使用独立随机流生成输入/输出 token；
- 实现链式 chunk hash；
- 实现目标可行性、block 量化和比例分配；
- 输出目标、构造值与量化误差。

### KV2：L1 prefix cache

- 将现有 token 账本升级为 paged block manager；
- 实现 snapshot/probe/transaction/publish/release；
- 实现 active pin、inactive residency 和确定性 LRU；
- 接入实际 Prefill 工作量和 L1 命中指标。

该阶段与 `F003` 的 PC1～PC3A 对齐。

### KV3：L2/L3 层级

- 增加全局 hash directory 与分层容量；
- 实现 exclusive write-back 的 promotion/demotion/eviction；
- 加入 restore 带宽、固定延迟和传输去重；
- 输出分层命中、resident、迁移和驱逐指标。

### KV4：continuous batching 集成

- 在 `F001.CB4` 中按 Prefill chunk 增量 publish；
- 命中 token 不占 Prefill token budget；
- retraction 时正确 release，避免立即命中自身导致无进展；
- 同 step 继续保持 snapshot 隔离。

## 12. 验收要求

### 12.1 workload 与 hash

- 同配置和 seed 生成相同 session、父子关系、根到达、每轮长度、token 和 hash；
- 根到达、长度、拓扑、内容和命中分配使用独立随机流，改变其中一个维度不扰动其他维度；
- namespace、父 hash 或 chunk token 任一变化都会改变当前及后续 hash；
- 不开启跨 session 共享时，不出现非预期跨 session 命中；
- 构造命中 block 总数严格等于 $K^*$；
- 构造误差符合 block 量化上界；
- 不可行目标返回包含 `r_max` 的明确错误。

### 12.2 模拟客户端与 session 时序

- `WorkloadPlan` 只由模拟客户端消费，引擎不查询 session children 或 think time；
- `SimulationCoordinator` 是唯一时钟和事件队列所有者，客户端和引擎不能各自推进时间；
- 根轮次只由 `ClientDriver.initial_wakeups` 注册，并在 wakeup 时刻通过 `on_wakeup` 提交；
- 子轮次只在客户端收到父请求 `RequestCompletion` 后注册，不会早于父请求完成；
- 子 `ClientWakeup.send_time_ms` 严格等于 `finish_time_ms + think_time_ms`；
- wakeup 触发前，引擎中不存在对应请求及其预计到达时间；
- `engine.submit` 只接收当前时刻实际到达的请求，引擎 waiting queue 中不存在未来请求；
- future send 通过 DES 事件调度，不调用真实 `sleep()`；
- 请求提交后只存在于引擎 waiting/running 状态，不继续停留在客户端 ready 状态；
- 零 think time 时，父请求发布/release 对同刻子请求可见；
- 父请求 output 完整 block 可被子请求命中，partial tail 不可共享；
- 分支请求只共享其声明的连续祖先前缀；
- 相同完成结果和客户端状态产生相同的后继 `ClientWakeup` 与 `InferenceRequest`。

### 12.3 cache 与 LRU

- probe 完全只读；
- 命中只返回从输入起点开始的最长连续完整 block；
- 同 step 新发布的 block 不对同批请求可见；
- active L1 block 永不淘汰；
- 每层按 `(last_access_seq, block_id)` 确定性淘汰；
- promotion/demotion/eviction 后索引和容量守恒；
- 事务失败不留下部分状态；
- L3 淘汰后对应 hash 不再可查；
- 一个 hash 对应多个物理 block 时选择规则稳定。

### 12.4 指标与性能

- 理想命中不受缓存容量和调度顺序影响；
- 实际命中由准入时 residency 决定；
- L1/L2/L3 命中归因之和等于实际命中；
- lower-tier hit 减少 Prefill，但增加对应 restore 成本；
- shared block 的物理容量只计算一次，每个请求的逻辑 context 分别计算；
- 全 prompt 可缓存时仍至少执行一个输入 token；
- 相同配置、seed 和性能参数得到相同事件、淘汰、命中和结果序列。

## 13. 非目标

首期不实现：

- hash 碰撞后的 token-by-token 二次校验；
- partial block 跨请求 copy-on-write；
- radix tree；
- 跨 Engine/实例 cache-aware routing；
- 跨仿真或 API 请求持久化 KV；
- inclusive/write-through 副本策略；
- 真实 KV tensor、GPU 地址或 DMA 并发；
- 根据计算与传输成本动态绕过低层命中；
- 自适应反向调整 workload 以强行让实际命中率等于目标值。
