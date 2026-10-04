# 前端配套实施计划

**状态：按功能阶段待实施。** `F001` 的表单字段、默认值、完成/失败/跳过结果及指标空值以 [调度与引擎实施规范](scheduler-engine.md) 为准；本文只跟踪界面的交付责任，前端不自行推导调度或指标。

| 前置模块 | 前端工作 | 验收 |
|---|---|---|
| [F001 调度/引擎](scheduler-engine.md) | 表单提供 `max_running_requests`、`max_total_tokens`、`max_prefill_tokens`、`page_size`（仿真器默认 64）、`max_new_tokens` 与 P/D 模式开关 `pd_mutually_exclusive`（默认开启互斥，关闭时同一步融合）；展示有效 KV 容量及各字段单位；分块选项待落地再开放；展示 abort/skipped，不将它们绘成完成区间 | 开关两种取值、提交配置和结果回显与 OpenAPI 一致；两种模式都可提交合法负载，完成/失败/跳过状态展示正确 |
| [F002 性能模型](perf-model.md) | 预设选择、专家覆盖、resolved config 展示、经验/解析/校准模式切换；展示显存、FLOPs、HBM/通信、瓶颈、rank critical path 与校准适用范围 | 只展示已经实现的模式和有数据的明细，缺失/低置信度值不伪装为精确实测 |
| [KV3 缓存](kv-cache.md) | 在服务端真正支持低层时开放 L2/L3 配置及分层命中、迁移、restore 指标 | 低层关闭时不误标 L1 结果为多级复用，层级 token 与总量一致 |

请求/响应类型继续从 OpenAPI 生成；每阶段同步配置默认值、错误处理和组件测试。`F001`、`F002`、KV3 的具体实施先后及依赖由 [功能总览](feature-roadmap.md) 管理。
