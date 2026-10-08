# 功能状态与模块计划索引

本文登记实施状态、依赖和未实施计划入口。已实现行为以 [设计文档索引](../index.md) 及代码和测试为准；模块计划仅描述尚未交付的部分，不是现有 API 或性能保证。功能编号 `Fxxx` 保持稳定。

## 功能状态与依赖

| 编号 | 能力 | 当前边界 | 依赖与后续计划 |
|---|---|---|---|
| `F003` | 完整块 ID 的前缀复用 | **已实现** L1 单池连续复用；不包括低层与真实内容哈希 | [身份](../design/chunk-hash-encoding.md)、[L1](../design/kv-prefix-reuse.md)、[负载](../design/workload.md) |
| `F001` | 单实例连续调度主干 | **已实现** 两种 P/D 模式、FCFS 预算、回撤、三类结果及服务/页面接入；不含 chunked prefill | [引擎设计](../design/kernel.md)、[API](../design/api.md)、[前端](../design/frontend.md) |
| `F002` | 可配置、可解释和可校准的性能模型 | **部分实现** 四系数 `StepWorkload` 纯步/融合计时；成本分解、解析、校准、单机 TP 未实现 | [现行经验模型](../design/perf-model.md)、[剩余计划](perf-model.md) |
| `KV3` | 多级 KV 缓存 | **未实现**；L2/L3 只有结果字段零值预留，**无输入配置或迁移** | [KV 模块计划](kv-cache.md)，适配现行 [L1](../design/kv-prefix-reuse.md) |

```text
F003 L1（已完成） ──▶ F001 调度/引擎（已完成） ──▶ F002 后续成本分解、解析与校准
                                      └────────▶ KV3 多级缓存（适配现行 L1）
```

M1 内核、M2 服务/OpenAPI、M3 页面及 F001 两种 P/D 模式已接通；F002 当前只接通 `StepWorkload` 的四系数经验计时。服务器与前端未完成的各阶段交付分别见 [服务/API 计划](server-api.md) 和 [前端计划](frontend.md)。单 GPU、单 Engine 当前不含真实 KV tensor、跨实例持久化或 trace replay；实时单步、任务持久化和调度溯源 UI 也未实现。

## 其余调度扩展（未实现）

- **Chunked prefill**：长 prompt 跨步、active chunk 优先与逐 chunk 发布，需新增分块阶段及调度边界用例；目前一次 Prefill 算完整个剩余上下文。
- **真实共享与 radix tree**：真实 token/内容身份、命名空间、跨独立请求共享及树节点分裂；现有按请求 ID 编码的新页不代表真实公共前缀。
- **高级执行**：CPU/GPU overlap、speculative、SWA/Mamba、PD 拆分等均不作为当前可选配置；多级缓存与解析耗时分别见 KV3 和 F002 计划。

新增功能时更新本表、依赖图及对应设计文档；已完成的实现以 `design/` 为准。
