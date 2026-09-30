# 服务与 API 配套计划

**状态：按功能阶段待实施。** 当前同步 `POST /api/simulate`、Pydantic → OpenAPI → TS 类型链及 [现行契约](../design/api.md) 保持不变。本计划只记录未来功能对 `server/`、`schemas` 的**对外部分**和生成类型的交付；内部算法由各模块计划负责。

| 前置模块 | 服务与 schema 交付 | 验收 |
|---|---|---|
| [F001 调度/引擎](scheduler-engine.md) | 确定调度字段归属与旧配置输入兼容层；对外暴露 resolved 配置及明确的错误响应；不新增未实现的执行模式 | 旧输入能解析，新增预算/阶段参数约束可见，OpenAPI 与响应一致 |
| [F002 性能模型](perf-model.md) | 服务端管理预设、版本与校准 profile；展开完整 resolved config 供零 IO 内核使用；在结果中加入可解释成本和适用范围告警 | 同一预设版本与输入可复现，经验后端仍兼容 |
| [KV3 缓存](kv-cache.md) | 启用低层配置的真实约束与分层/restore 结果字段；不要把仅已解析的字段表述成已启用 | 低层关闭时维持 L1 输出，低层启用时与缓存指标相符 |

任何对外配置或路由调整均以 `core.schemas` 为事实源，重新生成并提交 `web/openapi.json`、`web/src/api/schema.d.ts`；服务层不另造同名数据模型，不把未实现的后端选项加入可选择的产品配置。UI 接入与字段说明见 [前端计划](frontend.md)。
