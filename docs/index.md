# 文档目录

本文件是所有文档的索引。新增文档时必须在此登记。

## 设计文档

| 文档 | 内容 |
|---|---|
| [design/architecture.md](design/architecture.md) | 仓库与内核模块的边界、所有权和依赖规则 |
| [design/workload.md](design/workload.md) | 合成 session 计划、理想继承与客户端发送 |
| [design/events.md](design/events.md) | 唯一 DES 时钟、协调器事件顺序和回调 |
| [design/kernel.md](design/kernel.md) | F001 双 P/D 模式、一次性 Prefill、FCFS 预算与回撤 |
| [design/chunk-hash-encoding.md](design/chunk-hash-encoding.md) | 合成负载完整块 ID 的编码与生成时机 |
| [design/kv-prefix-reuse.md](design/kv-prefix-reuse.md) | L1/L2/L3 唯一索引、连续命中、容量、迁移与 LRU |
| [design/perf-model.md](design/perf-model.md) | 四系数经验模型、融合步长与 restore 链路计时 |
| [design/metrics.md](design/metrics.md) | 时延、吞吐、分层命中及容量迁移的计算口径 |
| [design/api.md](design/api.md) | 分层配置、HTTP 响应与 OpenAPI 类型链 |
| [design/frontend.md](design/frontend.md) | 分层配置表单、结果展示与甘特图交互 |

## 规划文档

| 文档 | 内容 |
|---|---|
| [plans/feature-roadmap.md](plans/feature-roadmap.md) | 功能状态、依赖及按模块组织的计划入口 |
| [plans/perf-model.md](plans/perf-model.md) | F002：可解释性能模型、容量、校准及单机 TP |
| [plans/model-hardware-kv-capacity.md](plans/model-hardware-kv-capacity.md) | F002 子计划：模型/硬件预置、单卡 KV 容量与支持边界 |
| [plans/expert-parallelism.md](plans/expert-parallelism.md) | F004：模型/硬件参数配置与逐步 Roofline → TP → EP 三阶段实施 |
| [plans/server-api.md](plans/server-api.md) | 各阶段服务/API 契约与类型交付 |
| [plans/frontend.md](plans/frontend.md) | 各阶段配置交互与结果展示 |

## 实现入口

| 模块 | 入口 | 状态 |
|---|---|---|
| 内核 | `simcore.simulate(SimConfig) -> SimResult` | M1 已完成 |
| 服务层 | `simserver.app:create_app`；`POST /api/simulate` | M2 已完成 |
| API 类型 | `web/openapi.json`；`web/src/api/schema.d.ts` | M2 已完成，运行 `npm run gen:api` 更新 |
| 前端应用 | `web/src/App.tsx`；`web/src/components/RequestGantt.tsx` | M3 已完成 |

运行、调用 API 与类型生成命令见仓库根目录 [README.md](../README.md)。

## 编写约定

- `design/` 一篇只讲一个当前已实现的主题：不变量、接口、所有权与必要的编码指导；不保留旧方案、讨论过程或未落地能力的细节。
- `plans/` 按负责模块记录待实施范围、依赖、阶段交付和验收；`feature-roadmap.md` 仅做状态及依赖索引，不复制设计契约。
- F001/KV3 的现行行为分别记于 [引擎](design/kernel.md)、[分层 KV](design/kv-prefix-reuse.md)、[经验模型](design/perf-model.md)、[指标](design/metrics.md)、[API](design/api.md) 和 [前端](design/frontend.md)；尚未实现的扩展留在 `plans/`，完成后将实际行为写入 `design/` 并移除已完成计划条目。
