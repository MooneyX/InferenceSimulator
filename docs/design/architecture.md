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
workload     build_plan 规划长度、拓扑和理想复用；sample_length 采样；plan_from_arrivals 适配手写 Arrival（依赖 schemas、prefix_hash）
client       ClientDriver：消费计划、保存已完成请求身份与实际 KV 长度、发送时按需推导继承页 ID（依赖 schemas、prefix_hash）
perf_model   性能模型：步长纯函数（依赖 schemas）
kv_cache     当前 L1 KV 页、索引、容量与 LRU；L2/L3 仅有结果字段预留（依赖 schemas）
scheduler    F001 FCFS 调度：按 KV 页状态和未来估算选择 batch（依赖 kv_cache、schemas）
engine       已到达请求、EngineStep、KV 生命周期与完成结果（依赖 scheduler、kv_cache、perf_model）
events       ClientSend / ClientCompletion / EngineStep / RequestDone 事件与引擎 trace 类型（依赖 schemas）
coordinator  唯一 DES 时钟和事件队列；路由 client 与 engine（依赖 client、engine、events）
metrics      指标汇总：区间与 cache 快照 → Metrics（依赖 schemas、kv_cache）
runner       顶层入口 simulate(SimConfig) -> SimResult（依赖 workload、client、coordinator、engine、metrics）
```

`runner.simulate` 是 `core` 唯一的对外调用入口，输入输出均为 `schemas` 类型。

规则：

- 禁止跨层反向依赖（如 `scheduler` 不得引用 `engine`）。
- 模块间数据交换遵循下文「数据交换边界」。
- 每个模块可独立单测；只有 `coordinator` 感知并持有全局事件队列，`client` 与 `engine` 通过不可变事件和值对象参与调度。

### 数据交换边界

分两层边界：

- **对外边界（`core` ↔ `server`）**：只通过 `schemas` 中的 Pydantic 类型交换数据。`schemas` 是对外配置和结果契约的单一事实源；同一模块另定义不对外暴露的不可变内部消息。
- **内核内部模块间**：只交换不可变值对象（`frozen` dataclass 或 `schemas` 类型）。内部计划与客户端/引擎消息在 `schemas` 中定义，其余值对象由提供方模块定义，只允许依赖图中的下游模块引用。
- **禁止跨模块共享可变状态**。唯一例外是依赖图中明确列出的协作者注入，须写明所有权（见下文）。

内部类型归属：

| 类型 | 定义于 | 使用方 | 说明 |
|---|---|---|---|
| `TurnPlan` / `WorkloadPlan` | `schemas`（由 `workload` 构造） | `client` | 不可变 session 拓扑、输入/输出长度与理想继承块数；不预生成块 ID |
| `ClientWakeup` | `schemas`（由 `client` 构造） | `coordinator` | 未来客户端发送事件，只含 session/turn 身份和发送时刻 |
| `InferenceRequest` | `schemas`（由 `client` 构造） | `coordinator`、`engine` | 发送时构造：输入长度、可见生成上限和继承完整页 ID；不携带实际输出目标或新页 ID 列表 |
| `RequestCompletion` | `schemas`（由 `engine` 构造） | `coordinator`、`client` | 请求 ID、完成时刻和实际生成数；客户端据此确定可继承 KV 长度 |
| `StepWorkload` | `perf_model`（由 `engine` 构造） | `perf_model` | 不可变的 Prefill/Decode 工作量、阶段和实际申请页数；驱动单步经验成本 |
| `AdmissionBatch` | `scheduler` | `engine` | F001 的准入、恢复、失败与预算估算（值对象） |
| `L1Snapshot` | `kv_cache` | `engine`、`metrics` | L1 页容量与驻留状态的不可变汇总 |
| `Event` / `EventKind` | `events` | `coordinator` | 全局事件队列载荷与同刻全序（值对象） |
| `EventFactory` | `events` | `coordinator` | 事件序号分配器，由协调器实例化并独占持有 |
| `TraceRecord` | `events` | `engine` | Arrival、Step 和 Done 记录；不含 ClientSend 或客户端回调记录 |
| `EventLog` | `events` | `engine` | 引擎持有的事件流容器，可关闭记录以减少开销 |
| `EngineRun` | `engine` | `coordinator`、`runner` | 协调器返回给 runner 的请求区间、告警、步数与 L1 快照 |

协作者注入：

- `L1KVCache`：由 `engine` 构造并持有所有权，注入 `F001Scheduler`。调度器调用 `probe` / `pin` / `cancel_pin` 选择连续队头；引擎在执行和完成边界调用 `admit` / `extend` / `publish_batch` / `finish` / `withdraw` / `snapshot`。完整页 ID 索引与 LRU 由缓存维护；当前仅实现 L1。

引擎可在内部记录 `ArrivalTrace`、`StepTrace`、`DoneTrace`；顶层 `simulate` 默认关闭逐步记录，HTTP API 不暴露 `EventLog`。如需未来的决策溯源，应另行定义对外 schema，不把当前内部事件流误认为已有公共响应。

`coordinator` 只负责时间推进和事件路由，不参与块 ID 构造、调度或缓存决策。具体发送契约与同刻顺序分别见 [workload.md](workload.md) 和 [events.md](events.md)。

## 依赖规则

- `core` 不依赖任何外部 IO（无文件、无网络、无真实时钟读取），输入为配置对象，内部生成 `WorkloadPlan` 并由协调器推进模拟时钟，输出为事件流与结果对象。
- `server` 依赖 `core`，负责 HTTP 边界；`core` 不感知 `server`。
- `web` 与 `server` 仅通过 HTTP API 交互，类型由 OpenAPI 生成（见 [api.md](api.md)）。

## 工程规范

- 包管理：`uv`（`core` + `server` 为同一 workspace）。
- 可复现性为硬约束：同一随机种子的两次运行结果必须逐字节一致。
- `core` 自带基准测试：固定种子、固定规模的计时回归用例，从 M1 起常驻。
