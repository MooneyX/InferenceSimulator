# 待实现功能规划

## 1. 文档约定

本文用于逐项记录已经讨论并确认需要实现的功能。未经过讨论的候选功能不提前写入；后续每确认一项功能，再增加对应章节。

本文记录功能目标、模块边界、依赖关系和实施方向，不表示相关能力已经实现。开始编码前，仍需将最终确定的行为写入对应的 `docs/design/` 设计文档。

### 1.1 功能编号

每项功能使用稳定编号 `Fxxx`。后续调整标题或优先级时不修改编号，以便持续维护依赖关系。

### 1.2 依赖类型

- **硬依赖**：前置功能未完成时，后续功能不能完成集成或验收；
- **接口依赖**：只需先冻结前置功能提供的接口，双方即可并行开发；
- **可选依赖**：没有前置功能也可以运行，但具备它后能提供更完整或更准确的能力。

### 1.3 优先级规则

最终实现优先级根据依赖图决定，而不是根据功能写入文档的时间决定：

1. 硬依赖必须先于依赖它的功能完成；
2. 被多个功能依赖的公共接口优先；
3. 可并行开发的部分通过接口依赖拆开；
4. 同一依赖层内再根据用户价值、风险和实施成本排序；
5. 不允许形成循环硬依赖；出现循环时必须重新划分模块边界。

每次增加、拆分或调整功能时，必须同步更新功能清单、依赖矩阵和依赖图。

## 2. 已确认功能

| 编号 | 功能 | 状态 | 前置依赖 | 当前优先级 |
|---|---|---|---|---|
| `F001` | 完整的 token-level continuous batching | 待实现 | `F003` | P0 |
| `F002` | 可配置、可解释、可校准的性能模型 | 待实现 | `F001` | P1 |
| `F003` | 基于 token chunk hash 的 KV prefix cache 复用 | 待实现 | 无 | P0 |

这里的 `F001` 是对现有简化 continuous batching 的增强，不表示当前系统完全没有 continuous batching。

## 3. 当前依赖关系

```text
F003 KV prefix cache 复用（常开）
  │
  │ PC1～PC3A 基于当前一次性 Prefill 引擎独立完成
  ├─ 提供 probe、publish、release、snapshot 等稳定接口
  │    ▼
  │  F001 完整 continuous batching
  │    └─ 负责 chunked prefill、token budget、逐 chunk publish 和 retraction 集成
  │         │
  │         └─ 提供真实的 StepWorkload、batch shape 和 context length
  │              ▼
  └╌──────────▶ F002 可配置性能模型
                 └─ 增强 F003 的字节级容量和增量 Prefill 成本
```

### 3.1 依赖矩阵

| 功能 | 依赖功能 | 依赖类型 | 说明 |
|---|---|---|---|
| `F001` | `F003` | 硬依赖 | prefix cache 始终开启，continuous batching 必须集成已冻结的 block 接口 |
| `F002` | `F001` | 硬依赖 | 端到端集成和验收必须消费真实 `StepWorkload` |
| `F002` | `F001` | 接口依赖 | 算子公式、硬件规格和校准 profile 可先行，接口冻结后接入 |
| `F002` | `F003` | 接口依赖 | 性能成本始终消费命中前缀、实际新算 token、逻辑 context 和层级传输信息 |

依赖说明：

- `F003` 的 token/hash、block manager 和当前引擎集成没有前置功能依赖，完成 PC1、PC2 和 PC3A 即视为 `F003` 完成；
- prefix hash、lookup 和 block manager 始终运行，不提供启用开关；`F001` 必须消费 `F003` 已冻结的 block 级接口；
- chunked prefill 中的命中进度、token budget、逐 chunk 发布、`StepWorkload` 和 retraction 适配归 `F001` 负责，不作为 `F003` 完成条件；
- `F002` 的端到端集成与验收对 `F001` 构成**硬依赖**，但算子公式、硬件规格、内存公式和校准 profile 可先行；
- `F003` 不依赖 `F002`，可使用显式 `kv_block_size`、当前 `kv_token_budget` 解析出的 block 容量和经验模型独立运行；`F001` 后续迁移为 `max_total_tokens`，`F002` 再提供字节级容量与解析成本增强；
- 实施顺序为先完成 `F003`，再完成依赖其缓存接口的 `F001`，最后进行 `F002` 端到端集成。

## 4. `F001`：完整的 token-level continuous batching

### 4.1 目标

将当前按 engine step 动态准入请求、但一次完成整个 prompt 的简化实现，扩展为按 token budget 组织工作量的完整 continuous batching，为性能模型提供真实的 batch shape。

当前已经具备：

- 请求可在不同 engine step 到达；
- 新请求可在已有请求 decode 期间加入 batch；
- 完成请求退出后可以继续准入等待请求；
- batch 大小随到达和完成动态变化。

仍需新增：

- 每步 token 预算；
- 真正的 chunked prefill；
- 请求级部分 prefill 状态；
- SGLang 默认风格的 Prefill-first P/D 互斥批次；
- 每步真实 batch shape；
- KV 随实际已处理 token 增长。

首期不实现 mixed prefill/decode。同一时刻可以同时驻留 waiting prefill、partial prefill 和 running decode 请求，但一个 engine step 只执行 Prefill 或 Decode。

### 4.2 配置

调度配置直接采用 SGLang 的字段名称和语义，不再用一个 `max_batch_size` 同时表达驻留容量与执行批大小：

- `max_running_requests`：最多同时驻留的 running 请求数；waiting queue 中尚未准入的请求不计入；
- `max_queued_requests`：最多排队请求数；`None` 表示不设置该上限；
- `max_total_tokens`：KV memory pool 可容纳的总 token 数；未显式设置时最终由 `F002` 根据显存推导；
- `chunked_prefill_size`：chunked prefill 的 chunk token 上限；`-1` 表示关闭；
- `max_prefill_tokens`：单个 Prefill batch 的总 token 上限；SGLang 默认值为 `16384`；
- `prefill_max_requests`：单个 Prefill batch 的请求数上限；`None` 表示不单独限制；
- `schedule_policy`：首期使用 SGLang 默认的 `fcfs`；
- `enable_priority_scheduling`：首期采用默认值 `false`；
- `enable_mixed_chunk`：首期采用默认值 `false`；
- `prefill_decode_interval`：首期采用普通模型默认语义 `0`，即不强制插入 Decode；
- `enable_dynamic_chunking`：首期采用默认值 `false`；
- `enable_prefill_delayer`：首期采用默认值 `false`；
- `num_continuous_decode_steps`：首期采用默认值 `1`；
- `schedule_conservativeness`：Prefill admission 对未来 Decode KV 的保守程度，默认 `1.0`；
- `retraction_policy`：Decode KV 不足时的回撤顺序，首期采用默认值 `length`。

字段约束分别作用于不同集合：`max_running_requests` 限制驻留请求总数，`prefill_max_requests` 限制当前 P batch 的请求数，`max_prefill_tokens` 和 `chunked_prefill_size` 限制当前 P batch 的 token 工作量，`max_total_tokens` 限制 KV pool。单步参与执行的请求是驻留请求的子集，但不与驻留请求数等同。

现有字段按兼容层迁移：`max_batch_size -> max_running_requests`、`prefill_chunk_size -> chunked_prefill_size`、`kv_token_budget -> max_total_tokens`；通用的 `max_batched_tokens` 改为 SGLang 的 Prefill 专用 `max_prefill_tokens`。`batch_policy` 不再作为独立配置，P/D 行为由 `enable_mixed_chunk` 和 `prefill_decode_interval` 等 SGLang 配置共同决定。兼容字段只允许出现在旧输入解析层，resolved config、核心调度器和结果回显统一使用 SGLang 字段名。

具体字段归属在设计阶段确定，避免 `EngineConfig`、调度配置和运行时配置重复定义同一约束。

### 4.3 请求状态

请求状态需要从等待/运行扩展为至少包含：

- 未处理的 prompt token 数；
- 已建立 KV 的 token 数；
- 已生成 output token 数；
- 当前处于 waiting、prefilling、decoding 或 finished；
- 首次 prefill 开始和结束时间；
- 每个 step 实际处理的 token 数。

长 prompt 必须能够跨多个 engine step 完成，不能在准入后一次完成整个 prefill。

### 4.4 `StepWorkload` 契约

`F001` 负责生成不可变 `StepWorkload`，至少包含：

- `phase`：`PREFILL` 或 `DECODE`；
- 本步参与执行的请求；
- 每个请求本步处理的 prefill token 数；
- decode 请求及各自当前 context length；
- 执行 batch size；
- 本步开始时的 running 请求数和 waiting 请求数；
- prefill、decode 和总 batched tokens；
- 本步开始时的 KV 使用状态；
- 后续并行执行需要的请求到设备/rank placement。

`StepWorkload` 必须满足 P/D 互斥不变量：

```text
phase == PREFILL  => prefill_tokens > 0 && decode_tokens == 0
phase == DECODE   => prefill_tokens == 0 && decode_tokens == batch_size
```

每个 Decode 请求在普通非 speculative 模式下每步处理一个 token。`StepWorkload` 不包含性能公式算出的 duration；duration 由性能模型计算。

### 4.5 调度语义

首期采用 SGLang 默认 unified scheduler 风格的 **Prefill-first exclusive batching**：

1. 结算上一批次。未完成的 prefill 保留为 active chunk；完成 prefill 的请求转入 running decode；完成 decode 的请求释放 KV 并退出；
2. 优先尝试构造非空 `PrefillBatch`。先继续 active chunk，再按 `fcfs` 接纳 waiting 请求，并同时检查 `max_running_requests`、`prefill_max_requests`、`max_prefill_tokens`、`chunked_prefill_size` 和 `max_total_tokens`；
3. 如果成功构造 `PrefillBatch`，本步只执行 Prefill；
4. 如果不能构造 `PrefillBatch`，再更新 running decode 的资源状态并构造纯 `DecodeBatch`；
5. P、D 均不可执行时才进入 idle 或推进到下一个请求到达事件。

这里的“有 P 请求时优先”必须解释为“存在**可执行的非空 PrefillBatch**”，不能只根据 waiting queue 非空判断。被请求槽、token budget 或 KV 容量阻塞的 P 请求不得阻止 D 推进，否则 D 无法完成并释放资源时可能形成死锁。

同一时刻允许 waiting prefill、partial prefill 和 running decode 共存；互斥约束只作用于单个 engine step。Prefill 在某一步完成后最早从后续 step 进入 Decode，不能在同一步同时执行。只要后续仍能构造 P batch，新进入 running decode 的请求就继续等待。

首期 `fcfs` 只决定 waiting Prefill 请求的先后顺序。active chunk 优先继续，以避免部分 Prefill 已占用 KV 却因 `max_running_requests` 已满而无法完成。其他请求排序策略不在本功能中提前规划。

默认 `prefill_decode_interval=0`，因此在无限且持续可接纳的 Prefill 流量下，不承诺 Decode 无饥饿；这是对 SGLang 默认行为的明确复现，而不是调度器错误。有限输入且资源可推进时，所有请求仍应最终完成。

### 4.6 KV 与活性语义

- KV 只按实际已经完成 prefill 或 decode 的 token 增长；
- prefill chunk 完成后增加对应 token 的 KV；
- 每次 decode 完成后增加一个 token；
- 请求完成时解除活动引用，可复用的完整 block 转为可淘汰缓存，其余 KV 释放；
- 准入和 token 预算检查必须使用同一时刻的确定性资源快照；
- Prefill admission 应为已经驻留的 Decode 请求保留必要的后续 KV 空间，避免过度接纳造成持续 retract/preemption；
- 如果 P 因资源不足不可执行，而 D 可以推进并释放资源，必须调度 D；
- 字节级 KV 容量由 `F002` 提供后，替换或补充当前 token budget 判断。

严格按照 SGLang 建模时，Decode 下一 token 无法分配 KV 后必须执行 retraction，而不能用“首期只做保守 admission”替代：按 `retraction_policy=length` 选择请求，释放其 KV，重置为可重新调度状态并放回 waiting queue；如果仅剩的请求仍不能执行，则显式终止该请求，不能让系统停在不可推进状态。

`F001` 在 `F002` 完成前可以显式接收 `max_total_tokens` 独立运行；`F002` 完成后再根据显存、权重和运行时占用推导该值。

### 4.7 与性能模型的边界

`F001` 决定“本步做什么”，`F002` 决定“本步需要多久”：

```text
请求与资源状态
    ↓
F001 调度并生成 StepWorkload
    ↓
F002 计算 StepCostBreakdown
    ↓
引擎推进时间并更新请求、KV 状态
```

当前 prefill/decode 耗时取最大值的公式只是一种重叠近似，不等同于真实 chunked prefill，也不符合首期 P/D 互斥的执行语义。适配后的经验模型应根据 `StepWorkload.phase` 只计算当前阶段的成本。

### 4.8 实施顺序

#### CB1：状态与接口

- 增加请求级 prefill 进度；
- 定义带 `phase` 的 `StepWorkload` 及 P/D 互斥不变量；
- 调度决策与 duration 计算解耦；
- 使用现有经验模型适配新的 workload；
- 保持现有默认配置的兼容路径。

#### CB2：SGLang token budget 与 chunked prefill

- 增加 `max_running_requests`、`max_total_tokens`、`max_prefill_tokens` 和 `prefill_max_requests`；
- 增加 `chunked_prefill_size`；
- 支持长 prompt 跨多个 step；
- active chunk 优先继续；
- KV 按实际 token 增长；
- 实现 `schedule_conservativeness` 与 `retraction_policy=length` 的资源活性路径。

#### CB3：Prefill-first exclusive batching

- 每步只生成纯 `PrefillBatch` 或纯 `DecodeBatch`；
- 实现“可执行 P 优先，否则 D”的确定性选择；
- 明确 P 阻塞时回退 D 的活性规则；
- 输出每步真实 batch shape；
- 完成供 `F002` 使用的稳定接口。

#### CB4：prefix cache 调度适配

CB4 消费已经完成并冻结接口的 `F003`，归 `F001` 负责，并属于 `F001` 的必需完成条件。

- admission 时调用 `F003` 的只读 prefix probe，并以命中结果初始化请求 prefill 进度；
- 命中 token 不占用 Prefill token budget，实际未命中 token 按 `chunked_prefill_size` 推进；
- 每个 Prefill chunk 完成后，通过 `F003.publish` 发布新完成的完整 block；
- 在 `StepWorkload` 中记录 cached prefix、实际新计算 token 和逻辑 context length；
- retraction、完成和终止请求时调用 `F003.release`，并保证回撤能够形成真实可分配容量；
- 同一 step 新发布的 KV 不对该 step 其他请求可见；
- 避免“回撤后立即命中自身并重新占满”的无进展循环。

Mixed prefill/decode 不属于首期范围。后续如需模拟 vLLM 或 SGLang 的非默认 mixed chunk，可在不破坏 `StepWorkload` 基础字段的前提下新增策略和阶段类型。

### 4.9 验收要求

- 长 prompt 会跨多个 step，且已处理 token 总数守恒；
- 每个 step 不超过请求数和 token 数限制；
- 任意非空 step 的 Prefill token 和 Decode token 不同时大于零；
- 存在可执行 P 时必须生成纯 P step；P 不可执行且存在可执行 D 时必须生成纯 D step；
- Prefill 完成的请求不能在同一步 Decode；
- active chunk 不会仅因驻留请求槽已满而永久阻塞；
- 被资源阻塞的 P 不会阻止能够推进并释放资源的 D；
- 每个非空可执行 step 都有进展；
- 有限输入且资源可推进时请求最终完成；无限持续 P 流量下不要求默认策略保证 D 无饥饿；
- KV 增长和释放与实际 token 进度一致；
- 同配置、workload 和 seed 的 P/D step 序列、事件顺序及结果可复现；
- 现有不启用 chunking 的配置具有明确兼容行为；
- `StepWorkload` 可以被经验模型和新性能模型共同消费；
- cache hit 正确初始化 prefill 进度，命中 token 不重复占用 Prefill token budget；
- 完整 block 只在对应 chunk 完成后发布，retraction 与 cache residency 组合下不会形成无进展循环。

### 4.10 SGLang 与 vLLM 源码调研依据

本决策基于 `refs/` 中固定提交的源码调研：

- SGLang：`0318a8d0af86ba14a05ca093aa43fafe446da23e`；
- vLLM：`7ba3df63cbe2e3e7aca19074bed2958311f46400`。

“严格按照 SGLang”以该 SGLang 提交为行为基线，范围限定为普通 decoder-only、unified scheduler、单 GPU/首期 TP、非 speculative、非 disaggregation、非 DP-attention 的默认路径。模型特例、硬件 profile override 和实验开关不自动进入基线；它们必须作为后续独立功能显式建模，不能静默改变默认行为。

SGLang 默认 unified scheduler 先尝试构造 Prefill batch，成功则运行 Prefill，否则运行 Decode；`enable_mixed_chunk` 默认关闭，因此默认 forward batch P/D 互斥。主要证据位于：

- `refs/sglang/python/sglang/srt/managers/scheduler.py` 的 `get_next_batch_to_run` 和 mixed chunk 分支；
- `refs/sglang/python/sglang/srt/managers/schedule_batch.py` 的 `mix_with_running`；
- `refs/sglang/python/sglang/srt/arg_groups/fields/schedule.py` 的 `prefill_decode_interval` 与 `enable_mixed_chunk` 默认值。

vLLM V1 采用不同策略：scheduler 不划分独立 P/D phase，先调度所有 RUNNING 请求，再用剩余预算接纳 WAITING 请求，因此普通路径允许同一步混合 running Decode、partial Prefill 和新 Prefill。相关实现位于 `refs/vllm/vllm/v1/core/sched/scheduler.py`。本功能选择复现 SGLang 默认机制，不复现 vLLM 的 running-first mixed batching。

这里的 **P/D 批次互斥** 指单实例单 step 的执行策略，不等于 **PD 分离**。PD 分离是将 Prefill 和 Decode 放到不同实例或设备，并增加 KV 传输；它仍属于 `F002` 后续并行与通信建模范围。

## 5. `F002`：可配置性能模型

### 5.1 依赖

- **硬依赖：`F001`。** 端到端性能预测必须使用 `F001` 生成的真实 `StepWorkload` 才能完成验收；
- **接口依赖：`F001.StepWorkload`。** 接口冻结后，性能模型实现可与 continuous batching 后续工作并行；
- **可选反向增强。** `F002` 的字节级内存模型可以提高 `F001` 的 KV 准入真实性，但不是 `F001` 首期运行的前提。

### 5.2 目标

通过可配置的模型参数、硬件参数、运行时参数和并行策略，推导请求执行时间、显存占用与通信成本，使当前基于四个经验系数的估算演进为更真实、可解释、可校准的性能预测。

当前 `prefill_c1`、`prefill_c2`、`decode_d1` 和 `decode_d2` 同时吸收了模型架构、GPU 算力、显存带宽、kernel 效率和通信开销，无法解释配置变化为何影响性能。新模型不把新增参数简单换算回这四个系数，而是显式产出成本分解。

现有四系数模型继续保留，作为兼容后端和回归基线。

### 5.3 首期范围

首期支持：

- Decoder-only Transformer；
- Dense/GQA 模型；
- BF16/FP16；
- NVIDIA 同构 GPU；
- `F001` 提供的 continuous batching workload；
- paged KV cache；
- 单 GPU；
- 单 GPU 模型稳定后支持单机 TP1/2/4/8。

首期不支持：

- Pipeline Parallel；
- Expert Parallel 和复杂 MoE 路由；
- Prefill/Decode 分离；
- 跨节点和异构设备拓扑；
- KV offload、swap 和多级存储；
- speculative decoding；
- 稀疏注意力；
- 自动在线 profiling。

### 5.4 配置结构

#### 模型配置

至少包含：

- 层数、hidden size；
- attention heads、KV heads 和 head dimension；
- FFN intermediate size；
- 权重、activation 和 KV cache 精度；
- 最大上下文长度；
- 参数量自动推导及高级覆盖。

#### 硬件配置

至少包含：

- GPU 型号和设备数量；
- HBM 容量和带宽；
- 各数值精度下的峰值 FLOPS；
- GPU 间链路类型、带宽、延迟和拓扑；
- 理论规格与校准后的有效规格。

#### 并行配置

至少包含：

- Tensor Parallel degree；
- 后续扩展的 PP、DP 和 EP degree；
- 每实例设备布局；
- KV heads 在 TP 下的分片或复制策略；
- 后续的 stage 和 Prefill/Decode placement。

首期只公开已经实现的单 GPU和单机 TP 参数，避免提前暴露无语义的字段。

#### 运行时配置

至少包含：

- attention backend；
- paged KV block size；
- GPU memory utilization；
- CUDA Graph；
- kernel launch 和固定运行时开销；
- 权重与 KV 量化格式及元数据开销；
- runtime reserve、workspace 和 allocator 碎片率。

Continuous batching 的请求数、token budget 和 chunk 配置归 `F001` 管理；`F002` 只消费最终的 `StepWorkload`。

#### 性能模型配置

支持：

- `empirical`：现有四系数模型；
- `analytical`：基于 FLOPs、HBM bytes 和通信量的解析模型；
- `calibrated`：解析模型叠加指定硬件与运行时的实测校准 profile。

配置记录性能模型版本、calibration profile ID、profile 版本和适用范围。

### 5.5 计划新增的模块

| 模块 | 职责 | 主要产出 |
|---|---|---|
| 模型规格 | 校验模型结构，推导参数量、tensor shape 和算子工作量 | 不可变模型规格 |
| 硬件规格 | 描述设备算力、显存、带宽和互联拓扑 | 设备与链路规格 |
| 部署规划 | 将模型层、权重、KV 和通信映射到具体 rank | 每 rank placement |
| 内存模型 | 推导权重、KV、activation、workspace 和碎片占用 | 容量计划与内存瓶颈 |
| 算子成本模型 | 估算 GEMM、attention 和 FFN 的 FLOPs 与内存流量 | 算子成本明细 |
| 通信模型 | 估算 collective 及后续 stage/KV 传输 | 通信量与通信时间 |
| 性能模型接口 | 将 `StepWorkload` 映射为设备级成本和关键路径 | `StepCostBreakdown` |
| 校准 profile | 保存有效 FLOPS、有效带宽、固定开销和适用 shape | 校准参数 |
| 预设解析 | 在服务层将模型名、GPU 名和策略预设展开为完整配置 | resolved config |

### 5.6 `StepCostBreakdown`

至少包含：

- 总 FLOPs；
- HBM 读写字节数；
- 互联传输字节数；
- compute time；
- memory time；
- communication time；
- runtime overhead；
- 计算与通信重叠量；
- 各 rank 时间；
- critical path time；
- bottleneck 类型；
- 预测适用范围和置信度告警。

引擎只使用 critical path time 推进时钟，其他分解供指标和可视化使用。

### 5.7 内存模型

权重基础占用：

$$
M_{weights}=N_{params}\times bytes_{weight}+M_{quant\ metadata}
$$

普通 MHA/GQA 每 token 的总 KV 占用：

$$
KV_{bytes/token}=2\times L\times N_{kv-head}\times D_{head}\times bytes_{kv}
$$

可用 KV 空间：

$$
M_{kv-available}
=
M_{HBM}\times utilization
-M_{weights}
-M_{runtime}
-M_{workspace}
-M_{activation-peak}
$$

$$
KV_{token-budget}
=
\left\lfloor
\frac{M_{kv-available}}{KV_{bytes/token\ per\ rank}}
\right\rfloor
$$

每 rank 根据并行策略的分片或复制规则单独计算，并考虑 page/block 取整、allocator 元数据和碎片。解析模式将推导结果写入 resolved config 的 `max_total_tokens`；empirical 模式继续接受显式 `max_total_tokens`。旧输入字段 `kv_token_budget` 只由兼容层转换。

### 5.8 计算与访存模型

每个设备/rank 使用 Roofline 形式估算：

$$
t_{compute}=\frac{F_{step}}{P_{effective}(shape)}
$$

$$
t_{memory}=\frac{B_{HBM}}{BW_{effective}(shape)}
$$

$$
t_{device}=t_{fixed}+\max(t_{compute},t_{memory})
$$

`effective` 参数来自解析默认值或 calibration profile，不直接等同于硬件峰值。

Prefill 根据 `F001` 提供的实际 token chunk 和序列长度计算投影、FFN、attention、activation 与 KV 读写。Decode 显式计算线性层 FLOPs、batch 共享权重读取、每个请求按 context length 读取的历史 KV，以及 batch shape 对 kernel 效率的影响。

### 5.9 通信与并行模型

基础通信时间：

$$
t_{comm}=\alpha\times N_{rounds}+\frac{bytes_{transferred}}{BW_{effective}}
$$

初期计算与通信重叠模型：

$$
t_{combined}
=
t_{compute-memory}+t_{comm}
-\eta\min(t_{compute-memory},t_{comm})
$$

其中 $\eta$ 位于 0 到 1 之间，并可由 calibration profile 覆盖。

不同并行策略分别建模：

- **TP**：权重、KV 和算子分片，以及 AllReduce、AllGather 或 ReduceScatter；
- **DP**：独立服务实例和请求路由，不直接缩短单实例单步延迟；
- **PP**：stage、microbatch、stage 传输和 pipeline bubble；
- **EP**：专家 placement、token 路由和 AllToAll；
- **PD 分离**：独立 prefill/decode 队列、KV 传输和网络资源。

建议按单 GPU、单机 TP、DP、PP、EP、PD 分离、跨节点的顺序实现。

### 5.10 配置预设与可复现性

- 模型、GPU 和并行策略向普通用户提供预设；
- 高级用户可以覆盖数值字段；
- 预设目录由服务层管理；
- 服务层将名称展开为完整 resolved config；
- `core` 保持零 IO，只接收 resolved config；
- `SimResult` 回显完整 resolved config；
- 结果记录预设、性能模型和 calibration 版本。

### 5.11 API 与前端配套

配置界面增加：

- 模型、硬件和并行策略预设；
- 专家模式参数覆盖；
- 配置可行性即时校验；
- resolved config 展示；
- empirical、analytical 和 calibrated 模式切换；
- 字段单位、约束和依赖说明。

结果增加：

- 权重、KV、activation 和 workspace 显存分解；
- prefill/decode FLOPs；
- HBM 和互联传输量；
- compute、memory、communication 和 overhead 时间；
- bottleneck 类型；
- KV 容量推导；
- 各 rank 成本和 critical path；
- calibration 版本、适用范围和置信度告警。

### 5.12 校准与验证

微基准覆盖 batch size、prompt/context length、prefill/decode、模型尺寸、dtype、TP degree，以及 GEMM 和 attention shape。统计 median、P90 和最大相对误差，并验证预测瓶颈与实测瓶颈一致性。

端到端验证使用真实服务日志比较 TTFT、TPOT、throughput、makespan、batch/queue 时间序列、KV 峰值、饱和吞吐和排队拐点。

初始精度目标：

- 未校准解析模型的大多数测试点误差约 30% 以内；
- 同 GPU、同 runtime 的校准模型大多数测试点误差约 15%～20% 以内；
- 超出 calibration profile 适用范围时返回低置信度告警，不承诺上述精度。

### 5.13 实施顺序

#### PM1：经验模型适配

- 消费 `F001.StepWorkload`；
- 将当前四系数公式迁移为 empirical backend；
- 定义 `StepCostBreakdown`；
- 保持现有结果和确定性。

PM1 在 `F001` 完成 CB1、冻结接口后即可开始。

#### PM2：单 GPU 解析模型

- 增加 Model、Hardware 和 Runtime 配置；
- 实现 Dense/GQA 参数量和 KV bytes/token；
- 实现显存容量模型；
- 实现 prefill/decode FLOPs 与 HBM Roofline；
- 输出成本分解和瓶颈。

#### PM3：实测校准

- 建立 shape-aware 的有效 FLOPS 和带宽 profile；
- 标定 attention、decode、固定开销和 CUDA Graph；
- 引入 profile 版本和适用范围；
- 完成微基准与端到端误差报告。

#### PM4：单机 Tensor Parallel

- 支持 TP1/2/4/8；
- 实现权重和 KV 分片；
- 建模 collective、NVLink 和通信重叠；
- 由最慢 rank 决定 step critical path；
- 增加多 rank 成本展示。

### 5.14 验收与兼容性

- `F001` 完成后再进行 `F002` 端到端精度验收；
- 当前四系数公式保留为 empirical backend；
- empirical 模式继续接受显式 `max_total_tokens`，旧 `kv_token_budget` 只作为输入兼容别名；
- 新资源明细优先通过新增字段提供，不无故破坏现有结果字段；
- 同一 resolved config、workload 和 seed 必须得到可复现结果；
- 增加公式手算、容量守恒、placement、通信量和并行不变量测试；
- 设计与 API 变更继续遵循“先更新设计文档，再实施代码”。

## 6. `F003`：基于 token chunk hash 的 KV prefix cache 复用

### 6.1 依赖与完成边界

- **无前置功能依赖。** `F003` 使用显式 `kv_block_size`、由当前 `kv_token_budget` 解析出的 block 容量和现有经验模型，即可在一次性 Prefill 引擎上独立实现；
- **是 `F001` 的硬依赖。** `F003` 提供与调度策略无关的 block 级 `probe`、`acquire`、`publish`、`release` 和 `snapshot` 接口；chunked prefill、token budget、`StepWorkload` 和 retraction 的调用编排归 `F001`；
- **由 `F002` 可选增强。** `F002` 后续提供字节级容量、block rounding 和 cached context 下更准确的增量 Prefill 成本，不影响 `F003` 独立完成；
- **完成条件。** PC1、PC2 和 PC3A 完成并通过本节验收后，即视为 `F003` 完成；后续 `F001` 的 prefix cache 调度适配和 `F002` 的精度增强不阻塞该状态。

### 6.2 目标

为具有相同 token 前缀的请求复用已经生成的 prompt KV，减少重复 Prefill 计算，同时保持物理 KV 容量、每请求逻辑上下文和性能工作量之间的边界清晰。

核心流程为：

```text
请求 token IDs
    ↓
生成带父链和 namespace 的稳定 block hash
    ↓
在 KV cache 中查找最长连续命中前缀
    ↓
命中 block 建立共享引用，未命中部分进入当前引擎的一次性 Prefill
    ↓
Prefill step 完成后发布新 block hash
    ↓
请求结束后解除引用，缓存 block 保留到容量不足时淘汰
```

当前引擎将整个剩余 prompt 视为一个 Prefill 执行单元，但 `F003` 的缓存接口必须按完整 block 接收增量发布，不能把“一次发布整个 prompt”固化进 block manager。这样 `F001` 后续可以在每个 Prefill chunk 完成后调用同一接口，而无需修改 `F003`。

首期限定为普通 decoder-only、单 Engine、单逻辑实例、非 speculative、非 PD 分离场景。每次 `simulate` 从空 cache 开始，不在不同仿真或不同 API 请求之间持久化 KV。

### 6.3 首期范围与非目标

首期支持：

- 所有 synthetic workload 统一生成 session `WorkloadPlan`，由模拟客户端按时提交请求；
- session 输入及已完成父请求 output token 的前缀复用；
- paged KV 的完整 block 作为索引与共享粒度；
- 链式 chunk hash；
- 一个 hash 对应一个或多个物理 KV block；
- 请求到有序 block table；
- 活动引用计数；
- 请求结束后的 cache residency；
- 确定性 LRU 淘汰；
- token 级命中、实际 Prefill 和淘汰指标；
- 基于当前一次性 Prefill 引擎的端到端命中与复用。

首期不支持：

- partial block 跨请求共享；
- 非父子 session 请求之间的 generated output KV 复用；
- 跨 Engine、跨仿真、跨进程或跨设备 cache 共享；
- KV offload、swap、多级存储和远程 KV 传输；
- radix tree、copy-on-write partial block 和 cache-aware request routing；
- hash 与 lookup 的 CPU 耗时建模；
- hash 碰撞后的 token-by-token 二次校验；
- `F001` 的 chunked prefill、token budget、P/D 互斥和 retraction 编排。

最后一项由 `F001.CB4` 负责，不属于 `F003` 完成条件。首期使用足够强的稳定摘要并将碰撞视为可忽略事件；测试接口应允许注入 hasher，以验证碰撞不会破坏引用计数、索引清理和容量不变量。

### 6.4 请求内容与可复现性

仅包含 `prompt_len` 的请求不足以表达 session 继承关系或判断内容是否相同。统一 session workload 中，每个请求都必须具有确定的输入与输出 token 计划，并始终生成和查询对应 block hash。内部请求状态至少包含：

- `prompt_token_ids` 或等价的不可变输入 token 序列；
- 确定的 `output_token_plan`，完成后通过 `RequestCompletion` 交给模拟客户端；
- 按 KV block 划分的有序 `prompt_block_hashes`；
- 已命中的 prompt token 数；
- 实际完成计算的 prompt token 数；
- 已生成 output token 数；
- 当前引用的有序 KV block IDs。

统一规则：

- 所有请求必须且只属于一个 session；单轮 session 是最小合法 workload；
- workload 统一输出 `WorkloadPlan`，模拟客户端在 `ClientWakeup` 到时才构造请求并调用 `engine.submit`；
- Engine 只维护已实际到达的 waiting/running 请求，并对每个请求执行 prefix cache 生命周期；
- 不得根据长度、session ID 或父请求 ID 推断内容命中；
- 外部 API 和结果默认不回显完整 token IDs，避免放大请求与响应；
- synthetic workload 对根到达、长度、拓扑、内容和命中分配使用稳定派生的独立随机流；
- session planner 通过父子内容继承和唯一后缀控制理想前缀命中率；
- 相同 resolved config、workload 和 seed 必须生成相同 session、token、hash、客户端发送、命中、淘汰和事件序列。

### 6.5 Hash 语义

每个可缓存完整 block 的 hash 必须依赖缓存 namespace、父 block hash 和当前 block token IDs：

```text
H_0 = H(cache_namespace, ROOT, block_0_tokens)
H_i = H(cache_namespace, H_(i-1), block_i_tokens)
```

不能只 hash 当前 chunk。相同 token chunk 位于不同历史前缀时会产生不同 KV，必须通过父 hash 隔离。

`cache_namespace` 至少为以下语义预留隔离能力：

- 模型与权重版本；
- tokenizer 版本；
- KV dtype；
- adapter/LoRA；
- 用户指定 cache salt；
- 后续可能影响 KV 内容的多模态输入或运行时变换。

首期可以由固定模型配置生成 namespace，但接口中必须保留该边界。摘要使用 SHA-256、BLAKE2 等跨进程稳定算法，不使用 Python 内置 `hash()`。

`kv_block_size` 决定 `F003` 的物理分配和 hash 粒度，与调度器一次处理多少 prompt token 无关。当前引擎在一次 Prefill 中完成全部剩余 prompt，并在 step 结束时批量发布其中已完成的完整 block；未来 `F001` 的 `chunked_prefill_size` 只改变 `publish` 的调用时机和每次完成的 block 数，不改变 `F003` 的 block/hash 语义。只有完整 block 才进入共享 hash 索引。

### 6.6 KV 状态模型

现有 `request_id -> token_count` 账本需要扩展为 block manager。内部 block 至少包含：

```text
KVBlock
  block_id
  valid_tokens
  content_hash
  ref_count
  last_access_seq
  state: FREE | ACTIVE | EVICTABLE
```

KV manager 至少维护：

```text
blocks: block_id -> KVBlock
request_blocks: request_id -> ordered block_id list
hash_index: block_hash -> ordered block_id collection
free_blocks
evictable_lru
```

状态语义：

- `ref_count > 0` 的 block 为活动共享 KV，不可淘汰；
- 请求完成、回撤或释放时减少引用，不直接删除仍可缓存的完整 prompt block；
- `ref_count == 0` 且带有效 hash 的 block 转为 `EVICTABLE`；
- 容量不足时按确定性 LRU 淘汰 `EVICTABLE` block；
- block 重新分配前必须删除旧的 hash 反向映射；
- partial block 和首期不缓存的 output block 在无引用后直接释放；
- `block_id` 表示仿真器中的逻辑物理位置，不要求映射到真实 GPU 地址。

`hash_index` 允许一个 hash 对应多个 block。同一个 step 中的请求统一基于 step 开始快照查询，新产生的 KV 在该 step 结束后才发布，因此同批请求可能独立计算出相同 block；索引和淘汰逻辑必须支持这种情况，不能隐式迁移或合并正在执行的物理 KV。

### 6.7 命中规则

Prefix lookup 必须：

1. 从 prompt 第一个完整 block 开始；
2. 按父链顺序查询；
3. 返回从起点开始的最长连续命中；
4. 在第一个 miss 处停止，不能跳过中间 miss 后继续复用；
5. 返回选中的 block，但只在请求确定准入并执行 `acquire` 后建立活动引用；
6. 使用确定性规则从同 hash 的多个位置中选择 block。

Prefix cache 只保存 KV，不保存下一 token logits，因此即使整个 prompt 的 KV 都已存在，也必须保留至少一个 prompt token 的执行工作。首期可将可跳过的计算量限制为 `prompt_len - 1`；如果最后一个 token 所在的完整 block 已命中，可以继续引用该 block，但该 token 仍计入本步实际 Prefill 计算，且不重复计入物理 KV 分配。

首期只注册和复用 prompt 的完整 block。由于当前 workload 不生成具体 output token IDs，generated output KV 不进入共享 hash 链。

### 6.8 缓存接口与当前引擎接入

`F003` 对调度器提供与 Prefill 是否分块无关的 block 级接口：

```text
snapshot() -> CacheSnapshot
probe(snapshot, block_hashes) -> PrefixMatch
acquire(request_id, prefix_match, required_blocks) -> AllocationResult
publish(request_id, completed_block_hashes)
release(request_id)
```

具体命名可以在设计阶段调整，但必须保持以下边界：

- `probe` 只读，不刷新 LRU、不增加引用、不分配或淘汰 block；
- `acquire` 只在请求确定准入后执行，并原子提交命中引用、新 block 分配和必要淘汰；
- `publish` 只接收已经完成计算的完整 block，可以一次发布一个或多个 block；
- `release` 解除请求引用，可复用 prompt block 转为 `EVICTABLE`，其余 block 释放；
- `snapshot` 提供确定性的 active、evictable、free 和物理容量状态。

PC3A 的客户端与引擎接入流程为：

1. `ClientSend` 触发时，模拟客户端构造 token/hash，并通过协调器调用 `engine.submit` 进入 waiting queue，不修改 cache；
2. engine step 开始时获取同一份 `CacheSnapshot`；
3. 对候选请求只读查询最长缓存前缀；
4. 将整个剩余 prompt 作为当前的一次性 Prefill 工作量，并计算所需新增 block；
5. 请求确定准入后原子执行 `acquire`；
6. Prefill step 完成后，通过 `publish` 批量发布新完成的完整 prompt block；
7. 请求完成后调用 `release`，缓存 block 保留到后续容量不足时淘汰。

同一步中新计算的 KV 不对该 step 其他请求可见。候选请求未被调度时 cache 状态保持不变。PC3A 不负责 chunked prefill、`StepWorkload` 或 retraction；这些行为由 `F001.CB4` 使用相同接口完成。

### 6.9 容量与资源管理边界

容量判断必须区分：

- **物理 KV 占用**：实际分配的 block capacity，用于 `max_total_tokens` 或字节级显存限制；
- **有效 KV token**：block 中实际有效的 token，用于统计内部碎片；
- **逻辑 context token**：每个请求可见的完整上下文，用于 attention 工作量；
- **实际新计算 token**：本步未被 cache 跳过的 Prefill token，用于 token budget 和 Prefill 成本。

共享 block 在物理容量中只计算一次，但在每个引用请求的逻辑 context 中分别计算。不得使用共享后的物理 token 数代替 Decode 的 context length。

空间不足时，`F003.acquire` 优先淘汰无活动引用的 LRU cache block。淘汰顺序使用单调访问序号和稳定 block ID 作为 tie-break，不依赖无序容器遍历。淘汰后仍无法满足分配时返回确定的容量不足结果，且不得留下部分引用、部分分配或被错误刷新的 LRU 状态。

`F003.release` 不区分请求是正常完成、终止还是被上层调度器回撤，只负责解除引用并维护 residency。是否回撤、何时重新准入以及如何保证系统进展属于调用方策略；`F001.CB4` 负责 continuous batching 下的对应活性语义。

### 6.10 当前性能模型与后续扩展边界

PC3A 直接向当前引擎提供：

- 每个准入请求的 cached prefix token 数；
- 本次一次性 Prefill 实际新计算的 token 数；
- 请求完整逻辑 context length；
- 本步复用、新分配和淘汰的物理 block 数；
- active、evictable、free KV 状态。

当前经验模型不能把 `prompt_len - cached_tokens` 当成一个与历史无关的短 prompt。首期使用以下增量接口或等价近似：

```text
prefill_cost(context_before, new_prefill_tokens)
```

其中 `context_before` 为缓存前缀长度，`new_prefill_tokens` 为当前引擎一次计算的剩余 prompt token。该近似和现有经验系数足以完成 `F003` 的相对命中收益仿真，不依赖 `F002`。

`F001.CB4` 后续将相同数据按 step 写入 `StepWorkload`；`F002` 再根据 Query 长度、历史 KV 读取和新增 KV 写入计算解析成本。hash 和 lookup 的 CPU 开销首期视为零，后续如需建模应作为独立 runtime overhead 增加。

### 6.11 配置

prefix hash、lookup 和 block manager 默认常开，不提供启用字段。首期至少需要以下 resolved config：

- `kv_block_size`：物理 KV block 的 token 容量；
- `cache_salt`：可选 namespace 补充字段；
- `prefix_cache_eviction_policy`：首期固定或仅允许 `lru`。

`F003` 内部接收已经解析完成的 KV block 容量，不为 prefix cache 单独预留第二套容量。PC3A 由兼容层将当前 `kv_token_budget` 换算为 block 容量；`F001` 完成配置迁移后改由 `max_total_tokens` 提供同一语义。物理容量按完整 block 取整，有效 token、block capacity 和内部碎片分别统计。

合成 workload 另行提供 prompt 内容配置，不与 Engine 的 KV 策略字段混放。具体 API 字段归属在设计阶段确定。

### 6.12 Trace 与指标

Step trace 至少记录：

- cache 查询请求数和 token 数；
- hit token 数；
- 本步实际计算的 Prefill token 数；
- 复用、分配、释放和淘汰 block 数；
- step 开始和结束时 active、evictable、free block 数；
- 每个请求的 cached prefix length 和逻辑 context length。

聚合结果至少包含：

- prefix cache token hit ratio；
- 有任意命中的请求比例；
- 完整可复用前缀命中的请求比例；
- 实际计算与避免计算的 Prefill token 数；
- cache eviction block/token 数；
- active、evictable 和总 resident KV 峰值；
- block 内部碎片峰值；
- 可选的 estimated saved prefill time。

主要命中率采用 token 口径，不能只使用请求口径，否则短前缀与长前缀命中会被赋予相同权重。

### 6.13 实施顺序

#### PC1：session、输入内容与稳定 hash

- 将工作负载生成器统一改为输出 `WorkloadPlan`，每个请求必须且只属于一个 session；
- 增加 `SimulationCoordinator` 与 `ClientDriver`，由客户端在 `ClientWakeup` 到时提交请求；
- 扩展内部请求模型，使所有请求具有确定的输入与输出 token 计划；
- 为根到达、长度、拓扑、内容和命中分配建立稳定派生的独立随机流；
- 通过父子内容继承、唯一后缀和 block 量化构造目标理想命中率；
- 实现 namespace、父 hash 和完整 block token 组成的稳定链式 hash；
- 所有请求无条件执行 hash、probe、acquire、publish 和 release。

#### PC2：Paged KV block manager

- 增加 block、request block table 和 hash index；
- 增加活动引用计数、inactive residency 和确定性 LRU；
- 支持一个 hash 对应多个物理 block；
- 分离物理容量、有效 token、内部碎片和逻辑 context；
- 保证 block 重用时旧索引被清理。

#### PC3A：当前引擎端到端接入与 `F003` 完成

- 基于 engine step 开始快照进行只读 prefix probe；
- 将当前一次性 Prefill 的实际未命中 token 传给经验模型；
- 通过 `acquire` 原子提交引用、分配和淘汰；
- step 完成后批量发布新完成的完整 block hash；
- 请求完成后 unpin，可缓存 block 保留驻留；
- 增加 cache trace、命中率、淘汰、驻留和碎片指标；
- 更新服务 schema、OpenAPI 和前端配置/结果展示；
- 增加 prefix cache 的确定性和性能回归场景。

PC1、PC2 和 PC3A 通过 `6.14` 验收后，`F003` 状态即可更新为已完成。

#### PC4：可选精度增强

- `F002` 完成后接入字节级 KV 容量、block rounding 和显存碎片；
- 使用 cached context、Query 长度、历史 KV 读取和新增 KV 写入计算解析成本；
- 如有实测需求，再增加 hash 与 lookup runtime overhead。

PC4 不属于 `F003` 完成条件。原计划中的 chunked prefill、P/D 互斥、`StepWorkload` 和 retraction 集成移入 `F001.CB4`。

### 6.14 验收要求

- 相同 namespace、父前缀和 block token 产生相同 hash；任一项不同均产生不同 hash；
- prefix lookup 只返回从 prompt 起点开始的最长连续完整 block；
- `probe` 为只读操作，cache miss、候选请求未调度和容量不足均不改变引用计数、LRU 或物理占用；
- `acquire` 原子提交命中引用、新分配和淘汰，失败时不留下部分状态；
- `publish` 接受一个或多个已完成完整 block，不要求调用方一次发布整个 prompt；
- 所有请求均执行 prefix hash、probe、acquire、publish 和 release，配置与结果中不存在缓存关闭分支；
- 在当前引擎中，第一个共享前缀请求产生 miss，后续请求在 step 完成并发布 KV 后产生确定命中；
- 当前引擎中的 prefix hit 减少实际一次性 Prefill 工作量和对应经验成本，但不缩短逻辑 context；
- 同一 step 中的新 KV 不被同批其他请求命中；
- 请求完成后活动引用归零，但可缓存 prompt block 在未淘汰前仍能命中；
- 多个请求共享同一 block 时物理容量只计算一次，逻辑 context 分别计算；
- active block 不可淘汰，inactive block 按确定性 LRU 淘汰；
- block 重用后旧 hash 不再返回该位置；
- partial block 不进入共享索引；
- 即使整个 prompt 的 KV 已存在，仍至少执行一个 prompt token 以产生 logits；
- 相同 resolved config、workload 和 seed 的 token、hash、命中、淘汰、step 与结果序列可复现；
- 命中 token、实际计算 token、逻辑 context、物理 block 和内部碎片分别满足守恒不变量；
- PC1、PC2、PC3A 及以上验收全部通过后即可判定 `F003` 完成，不等待 `F001` 或 `F002`；
- 在编码前将最终数据结构、状态转移、API 字段和指标口径同步写入对应的 `docs/design/` 文档。

### 6.15 参考实现边界

PC3A 保持当前引擎的一次性 Prefill 和既有调度行为，只新增 prefix cache 对准入容量、Prefill 工作量和请求生命周期的影响。`F003` 不自行实现 SGLang Prefill-first、chunked prefill 或 retraction；这些调度行为由 `F001.CB4` 调用 `F003` 的稳定接口完成。

KV prefix cache 数据结构优先参考 vLLM 的链式 block hash、block table、引用计数和 hash-to-block 映射，但不要求逐行复刻任一运行时。首期目标是复现对容量、Prefill 工作量和延迟有影响的可观测缓存语义，不模拟真实 GPU 地址、KV tensor 内容或哈希计算过程。后续如需支持 SGLang 风格 radix cache、跨实例路由或 PD 分离 KV 传输，应作为独立功能扩展，不静默改变 `F003` 的默认行为。
