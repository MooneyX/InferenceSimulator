# 总体架构

## 模块划分

前后端分离的 monorepo，三个模块：

```
core/     仿真内核：纯 Python 包，零 IO
server/   服务层：FastAPI
web/      前端：React + TypeScript + Vite + ECharts
```

## 内核内部模块

`core` 按职责单向分解，模块间只允许向下依赖：

```
schemas      Pydantic 配置/结果与不可变内部消息（无依赖）
prefix_hash  完整块 ID 直接编码与独立随机流种子派生（无内部模块依赖）
workload     build_plan 规划长度、拓扑和理想复用；sample_length 采样（依赖 schemas、prefix_hash）
client       ClientDriver：消费计划、保存已完成请求身份与实际 KV 长度、发送时按需推导继承页 ID（依赖 schemas、prefix_hash）
perf_model   性能模型：步长纯函数（依赖 schemas）
kv_cache     TieredKVCache：唯一索引、分层容量、稳定 LRU、传输占位与完成迁入（依赖 schemas）
scheduler    F001 FCFS 调度：按 KV 页状态和未来估算选择 batch（依赖 kv_cache、schemas）
engine       单个实例的已到达请求、GPU 步与异步页传输、独占 KV 生命周期与完成结果（依赖 scheduler、kv_cache、perf_model）
events       客户端发送/回调、GPU 步完成、L3 查询与页传输完成、请求完成事件（仅依赖标准库）
router       根请求轮询、子请求严格继承父实例的路由策略（依赖 schemas）
coordinator  唯一 DES 时钟和事件队列；按实例 ID 将请求及完成交给对应 engine（依赖 client、engine、events、router）
metrics      指标汇总：区间与 cache 快照 → Metrics（依赖 schemas、kv_cache）
runtime      RuntimeConfig、AnalyticalSpec、具名 TransferLinks 的不可变值对象
capacity     prepare_runtime：从 SimConfig 一次性准备容量、计时规格与链路（依赖 runtime、schemas）
runner       顶层仿真入口 simulate(config, trace_plan=None) -> SimResult（依赖 runtime、workload、client、trace_client、coordinator、engine、metrics）
```

`runner.simulate` 是统一的完整仿真入口，包顶层导出 `SimConfig`、`TracePlan` 与 `simulate`：

- `config: SimConfig`：唯一外部配置由 `num_instances`、`workload`、`scheduler`、`inference` 组成。`inference.mode=manual` 使用显式 KV 容量与经验计时；`physical` 使用模型/硬件/存储推导容量，并选择经验或解析计时。
- `trace_plan: TracePlan | None`（仅关键字参数）：仅注入已解析的计划资源；开环/闭环、会话速率、种子和可选目标 QPS 都属于 `config.workload.trace`。Trace 缺少计划资源时拒绝运行，合成负载不接受 Trace 资源；服务层根据临时 `plan_id` 获取计划，内核不执行 IO。
- 始终返回单个 `SimResult`，其中 `resolution`、`analysis` 可为空；经验计时的 `analysis` 为 `None`。`meta.config` 完整回显输入配置，不改写成运行时容量配置，也不修改调用方输入。
- 一场仿真只调用一次 `capacity.prepare_runtime`，再用 `Engine(runtime, ...)` 将同一只读 `RuntimeConfig` 注入各实例。`RuntimeConfig` 包含内部 `engine`、`perf_model`、具名 `links`、可空 `analytical` 与 `resolution`；`AnalyticalSpec` 只共享模型/硬件与 TP/EP 规格，不持有动态状态。`TransferLinks.l2_to_l1` / `l3_to_l2` 预计算每页传输时间和固定延迟，保持原浮点运算顺序、成本公式和支持限制。
- `EngineConfig` / `PerfModelConfig` 是内部组件规格；手动配置经 `runtime_from_components` 准备运行规格，不构成第二套外部仿真配置。运行墙钟不包含配置解析；结果回显包含模型默认字段的原规范化 `SimConfig`。

合成开环、合成闭环和 Trace 仍使用各自的客户端；统一入口 `simulate` 使用 `Coordinator → start_step → 内部 _execute_step(selection 必填) → finish_step → complete` 生命周期。GPU 步完成与异步页传输分别通过事件推进，再统一汇总指标。

规则：

- 禁止跨层反向依赖（如 `scheduler` 不得引用 `engine`）。
- 模块间数据交换遵循下文「数据交换边界」。
- 每个模块可独立单测；只有 `coordinator` 感知并持有全局事件队列，`client` 与 `engine` 通过不可变事件和值对象参与调度。

### 数据交换边界

分两层边界：

- **对外边界（`core` ↔ `server`）**：可序列化的配置、计划和结果通过 `schemas` 中的 Pydantic 类型交换，`schemas` 是 HTTP 数据契约的单一事实源。服务层将请求中的 `SimConfig` 原样传给统一入口，仅根据 `workload.trace.plan_id` 另行注入不可变 `TracePlan` 资源；不存在第二套外部运行配置。
- **内核内部模块间**：只交换不可变值对象（`frozen` dataclass 或 `schemas` 类型）。内部计划与客户端/引擎消息在 `schemas` 中定义，其余值对象由提供方模块定义，只允许依赖图中的下游模块引用。
- **禁止跨模块共享可变状态**。唯一例外是依赖图中明确列出的协作者注入，须写明所有权（见下文）。

内部类型归属：

| 类型 | 定义于 | 使用方 | 说明 |
|---|---|---|---|
| `TurnPlan` / `WorkloadPlan` | `schemas`（由 `workload` 构造） | `client` | 不可变 session 拓扑、输入/输出长度与理想继承块数；不预生成块 ID |
| `ClientWakeup` | `schemas`（由 `client` 构造） | `coordinator` | 未来客户端发送事件，只含 session/turn 身份和发送时刻 |
| `InferenceRequest` | `schemas`（由 `client` 构造） | `coordinator`、`engine` | 发送时构造：输入长度、可见生成上限和继承完整页 ID；不携带实际输出目标或新页 ID 列表 |
| `RequestCompletion` | `schemas`（由 `engine` 构造） | `coordinator`、`client` | 请求 ID、完成时刻和实际生成数；客户端据此确定可继承 KV 长度 |
| `StepWorkload` | `perf_model`（由 `engine` 构造） | `perf_model`、`analytical` | 不可变的 Prefill/Decode GPU 工作量、阶段、实际申请页数及 `restore_only` 请求 ID；不携带恢复页来源或页大小，传输另由事件计时 |
| `AdmissionBatch` | `scheduler` | `engine` | FCFS 准入、恢复、失败及唯一提升页（值对象）；预算只在调度内部计算 |
| `L1Snapshot` 与分层快照 | `kv_cache` | `engine`、`metrics` | 各层页容量、驻留及峰值的不可变汇总 |
| `Event` / `EventKind` | `events` | `coordinator` | 全局事件队列载荷与同刻全序（值对象） |
| `EventFactory` | `events` | `coordinator` | 事件序号分配器，由协调器实例化并独占持有 |
| `EngineRun` | `engine` | `coordinator`、`runner` | 协调器返回的请求区间、告警、步数、分层快照及迁移/restore 计数 |

协作者注入：

- `TieredKVCache`：由 `engine` 构造并持有唯一所有权，注入 `F001Scheduler`。调度器调用 `probe` / `pin` / `cancel_pin` 保护连续队头，以 `plan` 预检包含提升需求的分层预算，但不提交同步提升。引擎通过 `reserve_transfer` 保护来源并占用目的容量，传输完成事件通过 `complete_transfer` 迁入；`ensure_free` 保留同步驱逐/降级腾位。在执行和完成边界调用 `admit` / `extend` / `publish_batch` / `finish` / `withdraw` / `snapshot`，完整页 ID 的唯一索引、各层 LRU 及迁移计数均由缓存维护。

调度事件仅用于推进仿真，不保存逐步诊断日志。`EngineRun` 只携带正常汇总使用的请求记录、分层快照与整体计数；`compute_metrics` 必须接收完整计划、三级快照和实际发送统计。

`coordinator` 只负责时间推进和事件路由，不参与块 ID 构造、调度或缓存决策。具体发送契约与同刻顺序分别见 [workload.md](workload.md) 和 [events.md](events.md)。

## 依赖规则

- `core` 不依赖任何外部 IO（无文件、无网络、无真实时钟读取），输入为配置对象，内部生成 `WorkloadPlan` 并由协调器推进模拟时钟，输出为请求区间和汇总结果对象。
- `server` 依赖 `core`，负责 HTTP 边界；`core` 不感知 `server`。
- `web` 与 `server` 仅通过 HTTP API 交互，类型由 OpenAPI 生成（见 [api.md](api.md)）。

## 工程规范

- 包管理：`uv`（`core` + `server` 为同一 workspace）。
- 可复现性为硬约束：同一随机种子的两次运行结果必须逐字节一致。
- `core` 自带基准测试：固定种子、固定规模的计时回归用例，从 M1 起常驻。
