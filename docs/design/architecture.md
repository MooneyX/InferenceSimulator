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
- 模块间只通过 `schemas` 中的类型交换数据，不传内部对象。
- 每个模块可独立单测；`engine` 之外的模块不感知事件队列的存在。

## 依赖规则

- `core` 不依赖任何外部 IO（无文件、无网络、无时钟读取），输入为配置对象与工作负载序列，输出为事件流与结果对象。
- `server` 依赖 `core`，负责 HTTP 边界；`core` 不感知 `server`。
- `web` 与 `server` 仅通过 HTTP API 交互，类型由 OpenAPI 生成（见 [api.md](api.md)）。

## 工程规范

- 包管理：`uv`（`core` + `server` 为同一 workspace）。
- 可复现性为硬约束：同一随机种子的两次运行结果必须逐字节一致。
- `core` 自带基准测试：固定种子、固定规模的计时回归用例，从 M1 起常驻。
