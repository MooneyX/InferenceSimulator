# 文档目录

本文件是所有文档的索引。新增文档时必须在此登记。

## 设计文档

| 文档 | 内容 |
|---|---|
| [design/architecture.md](design/architecture.md) | 总体架构：仓库模块划分、内核内部模块分解、依赖规则 |
| [design/kernel.md](design/kernel.md) | 仿真内核设计：DES 事件模型、调度与 KV cache 语义、性能模型、工作负载接口 |
| [design/api.md](design/api.md) | 服务层契约：config schema、API 端点、metrics 定义、前后端类型同步 |
| [design/frontend.md](design/frontend.md) | 前端设计：请求甘特图视图规范与渲染约束 |
| [design/roadmap.md](design/roadmap.md) | 里程碑与验收线、MVP 明确排除项、为未来预留的扩展接缝 |

## 实现入口

| 模块 | 入口 | 状态 |
|---|---|---|
| 内核 | `simcore.simulate(SimConfig) -> SimResult` | M1 已完成 |
| 服务层 | `simserver.app:create_app`；`POST /api/simulate` | M2 已完成 |
| API 类型 | `web/openapi.json`；`web/src/api/schema.d.ts` | M2 已完成，运行 `npm run gen:api` 更新 |
| 前端应用 | `web/` | M3 规划中 |

运行、调用 API 与类型生成命令见仓库根目录 [README.md](../README.md)。

## 编写约定

- `design/` 下的设计文档只记录已锁定的设计结论与指导，不记录讨论过程与背景；其他类型的文档（如调研笔记、实验记录）按需组织，不受此限。
- 设计变更需先更新对应文档，再实施代码。
