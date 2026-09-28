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
schemas      Pydantic 模型：Config / Arrival / 区间结果 / Metrics（无依赖）
perf_model   性能模型：步长纯函数（依赖 schemas）
workload     WorkloadSource 协议 + SyntheticSource（依赖 schemas）
kv_cache     token 预算占用与准入判断（依赖 schemas）
scheduler    FCFS 调度：batch 组成决策（依赖 kv_cache、perf_model）
engine       DES 内核：事件队列、时钟推进、EngineStep 编排（依赖以上全部）
events       事件类型定义与事件流产出（依赖 schemas）
```

规则：

- 禁止跨层反向依赖（如 `scheduler` 不得引用 `engine`）。
- 模块间数据交换遵循下文「数据交换边界」。
- 每个模块可独立单测；`engine` 之外的模块不感知事件队列的存在。

### 数据交换边界

分两层边界：

- **对外边界（`core` ↔ `server`）**：只通过 `schemas` 中的 Pydantic 类型交换数据。`schemas` 是对外契约的单一事实源，不收纳内核内部类型。
- **内核内部模块间**：只交换不可变值对象（`frozen` dataclass 或 `schemas` 类型）。值对象由提供方模块定义并列入其 `__all__`，只允许依赖图中的下游模块引用。
- **禁止跨模块共享可变状态**。唯一例外是依赖图中明确列出的协作者注入，须写明所有权（见下文）。

内部类型归属：

| 类型 | 定义于 | 使用方 | 说明 |
|---|---|---|---|
| `StepPlan` | `scheduler` | `engine` | 单步调度结果（值对象） |
| `KVStats` | `kv_cache` | `engine` | 账本统计快照（值对象），由引擎转化为 `warnings` 与元信息 |
| `Event` / `EventKind` | `events` | `engine` | 事件队列载荷与同刻全序（值对象），仅供 `engine` 使用 |
| `EventFactory` | `events` | `engine` | 事件序号分配器，由 `engine` 实例化并独占持有 |
| `TraceRecord`（`ArrivalTrace` / `StepTrace` / `DoneTrace`） | `events` | `engine` 产出 | 事件流记录（值对象），内核级输出，非 API 契约 |
| `EventLog` | `events` | `engine` | 事件流容器，由 `engine` 实例化并独占持有 |

协作者注入：

- `KVCache`：由 `engine` 构造并持有所有权，注入 `scheduler`。`scheduler` 仅调用准入相关接口（`can_admit` / `can_ever_admit` / `reserve`，准入即提交）与只读查询（`budget` / `used` / `len`）；`advance` / `release` / `stats` 仅由 `engine` 调用。

事件流的对外暴露：`TraceRecord` 属于内核输出，但 MVP 不经 API 暴露，因此不进入 `schemas`。未来开放决策溯源或播放式仿真时，在 `schemas` 中新增对应的 Pydantic 类型，由服务层边界完成转换；内核内部继续使用 dataclass，避免逐步记录的开销。

## 依赖规则

- `core` 不依赖任何外部 IO（无文件、无网络、无时钟读取），输入为配置对象与工作负载序列，输出为事件流与结果对象。
- `server` 依赖 `core`，负责 HTTP 边界；`core` 不感知 `server`。
- `web` 与 `server` 仅通过 HTTP API 交互，类型由 OpenAPI 生成（见 [api.md](api.md)）。

## 工程规范

- 包管理：`uv`（`core` + `server` 为同一 workspace）。
- 可复现性为硬约束：同一随机种子的两次运行结果必须逐字节一致。
- `core` 自带基准测试：固定种子、固定规模的计时回归用例，从 M1 起常驻。
