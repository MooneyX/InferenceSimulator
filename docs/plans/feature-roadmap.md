# 功能状态与模块计划索引

本文登记实施状态、依赖和未实施计划入口。已实现行为以 [设计文档索引](../index.md) 及代码和测试为准；模块计划仅描述尚未交付的部分，不是现有 API 或性能保证。功能编号 `Fxxx` 保持稳定。

## 功能状态与依赖

| 编号 | 能力 | 当前边界 | 依赖与后续计划 |
|---|---|---|---|
| `F003` | 完整块 ID 的前缀复用 | **已实现** 跨层连续复用；不包括真实内容哈希 | [身份](../design/chunk-hash-encoding.md)、[分层 KV](../design/kv-prefix-reuse.md)、[负载](../design/workload.md) |
| `F001` | 单实例连续调度主干 | **已实现** 两种 P/D 模式、FCFS 预算、回撤、三类结果及服务/页面接入；不含 chunked prefill | [引擎设计](../design/kernel.md)、[API](../design/api.md)、[前端](../design/frontend.md) |
| `F002` | 可配置、可解释和可校准的性能模型 | **部分实现** 四系数计时及单卡 Dense/GQA 逐步 Roofline 估计；形状校准、解析 MoE 与单机 TP 未实现 | [现行经验模型](../design/perf-model.md)、[剩余计划](perf-model.md)、[模型/硬件预置与 KV 容量子计划](model-hardware-kv-capacity.md) |
| `F004` | 单实例 TP + EP | **第一阶段已接入** 单 GPU 自定义规格、三级容量解析与逐步 Roofline；TP 逐 rank 和 EP/DeepEP 待实现；DPA 不在首期 | [三阶段计划](expert-parallelism.md)、[单卡容量](model-hardware-kv-capacity.md)、[解析计时](perf-model.md) |
| `KV3` | 多级 KV 缓存 | **已实现** 单实例 L1/L2/L3 分层准入、逐级迁移与 restore 计时；低层默认关闭 | [分层 KV](../design/kv-prefix-reuse.md)、[计时](../design/perf-model.md)、[指标](../design/metrics.md) |

```text
F003 完整块 ID（已完成） ──▶ F001 调度/引擎（已完成） ──▶ F002 通用成本分解、解析与校准
                                           ├────────▶ KV3 分层缓存（已完成）
                                           └────────▶ F004 单实例 TP + EP（待实现，共用 F002 成本接口）
```

M1 内核、M2 服务/OpenAPI、M3 页面及 F001 两种 P/D 模式已接通；F002 同时接通四系数计时与单 GPU Dense/GQA 未校准解析计时。服务器与前端未完成的各阶段交付分别见 [服务/API 计划](server-api.md) 和 [前端计划](frontend.md)。单 GPU、单 Engine 当前不含真实 KV tensor、跨实例持久化或 trace replay；实时单步、任务持久化和调度溯源 UI 也未实现。

## 其余调度扩展（未实现）

- **Chunked prefill**：长 prompt 跨步、active chunk 优先与逐 chunk 发布，需新增分块阶段及调度边界用例；目前一次 Prefill 算完整个剩余上下文。
- **真实共享与 radix tree**：真实 token/内容身份、命名空间、跨独立请求共享及树节点分裂；现有按请求 ID 编码的新页不代表真实公共前缀。
- **高级执行**：CPU/GPU overlap、speculative、SWA/Mamba、PD 拆分等均不作为当前可选配置；已实现的分层缓存见 [KV 设计](../design/kv-prefix-reuse.md)，解析耗时仍见 [F002 计划](perf-model.md)。

新增功能时更新本表、依赖图及对应设计文档；已完成的实现以 `design/` 为准。
