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
schemas      Pydantic 模型：Config / 请求结果 / Metrics（无依赖）
prefix_hash  稳定 token chunk hash（依赖 schemas）
workload     WorkloadGenerator + SyntheticSource：生成 WorkloadPlan（依赖 schemas、prefix_hash）
client       ClientDriver：消费计划、维护 session、按时构造请求（依赖 workload、prefix_hash）
perf_model   性能模型：步长纯函数（依赖 schemas）
kv_cache     L1/L2/L3 KV block、索引、容量与 LRU（依赖 schemas）
scheduler    FCFS 调度：batch 组成与 cache transaction 决策（依赖 kv_cache、perf_model）
engine       已到达请求、EngineStep、KV 生命周期与完成结果（依赖 scheduler、kv_cache、perf_model）
events       ClientSend / EngineStep / RequestDone 事件和值对象（依赖 schemas）
coordinator  唯一 DES 时钟和事件队列；路由 client 与 engine（依赖 client、engine、events）
metrics      指标汇总：区间与 cache 记录 → Metrics（依赖 schemas）
runner       顶层入口 simulate(SimConfig) -> SimResult（依赖 workload、client、coordinator、metrics）
```

`runner.simulate` 是 `core` 唯一的对外调用入口，输入输出均为 `schemas` 类型。

规则：

- 禁止跨层反向依赖（如 `scheduler` 不得引用 `engine`）。
- 模块间数据交换遵循下文「数据交换边界」。
- 每个模块可独立单测；只有 `coordinator` 感知并持有全局事件队列，`client` 与 `engine` 通过不可变事件和值对象参与调度。

### 数据交换边界

分两层边界：

- **对外边界（`core` ↔ `server`）**：只通过 `schemas` 中的 Pydantic 类型交换数据。`schemas` 是对外契约的单一事实源，不收纳内核内部类型。
- **内核内部模块间**：只交换不可变值对象（`frozen` dataclass 或 `schemas` 类型）。值对象由提供方模块定义并列入其 `__all__`，只允许依赖图中的下游模块引用。
- **禁止跨模块共享可变状态**。唯一例外是依赖图中明确列出的协作者注入，须写明所有权（见下文）。

内部类型归属：

| 类型 | 定义于 | 使用方 | 说明 |
|---|---|---|---|
| `TurnPlan` / `WorkloadPlan` | `workload` | `client` | 不可变 session 拓扑、内容生成规则与目标复用计划 |
| `ClientWakeup` | `client` | `coordinator` | 未来客户端发送事件，只含 session/turn 身份和发送时刻 |
| `InferenceRequest` | `client` | `coordinator`、`engine` | wakeup 到时才构造并立即提交的请求，不含未来到达时间 |
| `RequestCompletion` | `engine` | `coordinator`、`client` | 请求输出 token 与完成时刻 |
| `StepPlan` | `scheduler` | `engine` | 单步调度结果（值对象） |
| `KVStats` | `kv_cache` | `engine` | KV 统计快照（值对象），由引擎转化为 trace、warnings 与元信息 |
| `Event` / `EventKind` | `events` | `coordinator` | 全局事件队列载荷与同刻全序（值对象） |
| `EventFactory` | `events` | `coordinator` | 事件序号分配器，由协调器实例化并独占持有 |
| `TraceRecord` | `events` | `coordinator` 聚合 | 客户端发送、Arrival、Step 和 Done 记录，内核级输出 |
| `EventLog` | `events` | `coordinator` | 全局事件流容器，由协调器实例化并独占持有 |
| `EngineRun` | `engine` | `coordinator` | 引擎请求区间、告警、步数与 KV 统计，不持有全局事件队列 |
| `SimulationRun` | `coordinator` | `runner` | 合并客户端、引擎和全局事件流后的单场原始结果 |

协作者注入：

- `HierarchicalKVCache`：由 `engine` 构造并持有所有权，注入 `scheduler`。prefix hash 与 cache 始终开启；`scheduler` 调用只读 `snapshot` / `probe` 和原子 `acquire`，`engine` 在 step 与完成边界调用 `publish` / `release` / `stats`。L1/L2/L3 容量、索引和 LRU 不允许被其他模块直接修改。

事件流的对外暴露：`TraceRecord` 属于内核输出，但 MVP 不经 API 暴露，因此不进入 `schemas`。未来开放决策溯源或播放式仿真时，在 `schemas` 中新增对应的 Pydantic 类型，由服务层边界完成转换；内核内部继续使用 dataclass，避免逐步记录的开销。

### session 客户端与协调器所有权

所有 workload 统一组织为 session，单轮 session 是最小合法形式。具体边界为：

- `workload` 输出不可变 `WorkloadPlan`，只交给 `client`；
- `client` 产生 `ClientWakeup`，`coordinator` 在对应时刻调用客户端并取得真正到达的请求；
- `coordinator` 仅在当前仿真时刻调用 `engine.submit(request, now_ms)`；
- `engine` 不保存未来请求或预计到达时间，只维护已经实际到达的 waiting/running 请求；
- `engine` 返回 `RequestCompletion`，由 `coordinator` 路由给 `client`，客户端据此安排后续发送；
- 全部交互均为进程内不可变值对象，不引入 IO、线程、协程或真实等待。

`coordinator` 只负责时间推进和事件路由，不参与 session 内容构造、Engine 调度或 KV cache 决策。客户端与引擎不得各自维护可独立推进的事件堆。

## 依赖规则

- `core` 不依赖任何外部 IO（无文件、无网络、无真实时钟读取），输入为配置对象，内部生成 `WorkloadPlan` 并由协调器推进模拟时钟，输出为事件流与结果对象。
- `server` 依赖 `core`，负责 HTTP 边界；`core` 不感知 `server`。
- `web` 与 `server` 仅通过 HTTP API 交互，类型由 OpenAPI 生成（见 [api.md](api.md)）。

## 工程规范

- 包管理：`uv`（`core` + `server` 为同一 workspace）。
- 可复现性为硬约束：同一随机种子的两次运行结果必须逐字节一致。
- `core` 自带基准测试：固定种子、固定规模的计时回归用例，从 M1 起常驻。
