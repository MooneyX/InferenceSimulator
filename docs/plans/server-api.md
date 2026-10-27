# 服务与 API 配套计划

**状态：性能模型已统一，继续跟踪 F002/F004 后续服务工作。** 当前接口不接受 `inference.timing`，OpenAPI 不再包含两种旧计时类型；结果字段类型仍允许为空，但新运行（含零 GPU 步）始终返回 `analysis` 对象；不转换旧配置。现行 API、任务状态与错误码见 [服务设计](../design/api.md)。下表保留阶段规划背景，双计时接入不再是当前能力或待办。

| 前置模块 | 服务与 schema 交付 | 验收 |
|---|---|---|
| [F004 第一阶段](expert-parallelism.md)、[F002 性能模型](perf-model.md) | 接收自定义模型/硬件规格或固定版本预置；服务端调用同一纯容量解析过程，在负载规划及引擎启动前准备唯一内部运行容量；`/api/resolve` 和 `/api/simulate` 均接收 `SimConfig`，分别返回 `ResolutionReport` 与单个 `SimResult`，回显原规范化输入并展示显存账单、来源；接入经验计时和按步 Roofline 解析计时 | 同输入与规格快照可复现；无有效 KV 布局或显存不足明确拒绝；旧外部配置与专用运行路由移除，迁移后等价输入保持原计时；解析步成本有可核对分项 |
| [F004 第二、三阶段](expert-parallelism.md) | 在第一阶段解析入口扩展逐 rank TP/EP 放置、容量与通信成本契约；每阶段同步对外 schema | TP 与 EP 按阶段分别验收，不提前开放未经验证的模型或后端组合 |

任何对外配置或路由调整均以 `core.schemas` 为事实源，重新生成并提交 `web/openapi.json`、`web/src/api/schema.d.ts`；服务层不另造同名数据模型，不把未实现的后端选项加入可选择的产品配置。UI 接入与字段说明见 [前端计划](frontend.md)。
