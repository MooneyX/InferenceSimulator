# 服务与 API 配套计划

**状态：按功能阶段待实施。** `F001` 的输入、输出、错误码、指标空值和结果状态均以 [调度与引擎实施规范](scheduler-engine.md) 为准；本文只列服务层与类型生成的交付责任，不另行定义仿真行为。

| 前置模块 | 服务与 schema 交付 | 验收 |
|---|---|---|
| [F001 调度/引擎](scheduler-engine.md) | `core.schemas` 提供 `engine.max_running_requests`、`engine.max_total_tokens`、`engine.max_prefill_tokens`、`engine.page_size`（默认 64）和布尔 `engine.pd_mutually_exclusive`（默认 `true`），合成负载提供 `workload.synthetic.max_new_tokens`；回显配置及有效页容量，配置错误与请求级 abort 分别处理 | 开关默认互斥，显式 `false` 启用融合；两种模式使用相同配置和缓存契约；运行中 abort 通过结果状态报告，OpenAPI 与真实响应一致 |
| [F002 性能模型](perf-model.md) | 服务端管理预设、版本与校准 profile；展开完整 resolved config 供零 IO 内核使用；在结果中加入可解释成本和适用范围告警 | 同一预设版本与输入可复现，经验、解析与校准模式按各自契约运行 |
| [KV3 缓存](kv-cache.md) | 启用低层配置的真实约束与分层/restore 结果字段；不要把仅已解析的字段表述成已启用 | 低层关闭时维持 L1 输出，低层启用时与缓存指标相符 |

任何对外配置或路由调整均以 `core.schemas` 为事实源，重新生成并提交 `web/openapi.json`、`web/src/api/schema.d.ts`；服务层不另造同名数据模型，不把未实现的后端选项加入可选择的产品配置。UI 接入与字段说明见 [前端计划](frontend.md)。
