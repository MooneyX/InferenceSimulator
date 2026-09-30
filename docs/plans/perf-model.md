# 性能模型实施计划（F002）

**状态：待实现。** 当前只有四系数纯函数经验模型（[现行设计](../design/perf-model.md)）。本计划负责 `perf_model` 及模型、设备、内存、算子、通信与校准模块；端到端集成必须消费 [F001 调度计划](scheduler-engine.md) 产出的 `StepWorkload`，但接口冻结后可先行实现公式与 profile。现有四系数模型作为兼容后端保留。

## 目标与首期范围

从可配置的模型结构、硬件、运行时和并行策略推导单步时间、显存占用和瓶颈，给出可解释、可校准的成本分解。首期支持 decoder-only Dense/GQA、BF16/FP16、同构 NVIDIA GPU、单 GPU；单 GPU 模型稳定后支持单机 TP1/2/4/8。PP、DP、EP、PD 分离、跨节点、异构设备、KV offload、多级存储、speculative decoding 与自动在线 profiling **不属于本次验收**，不得以规划字段冒充已可用模式。

## 待冻结的配置和接口

- 模型规格：层数、hidden/KV heads、head dimension、FFN、权重/activation/KV 精度、最大上下文，推导参数量和算子 shape。
- 设备规格：HBM 容量/带宽、各精度有效 FLOPS、单机互联的带宽/延迟；区分理论峰值和校准后的有效值。
- 运行时与部署：paged KV block size、memory utilization、runtime/workspace/activation reserve、allocator 碎片、CUDA Graph 和固定开销；TP 时记录每 rank 权重、KV、算子 placement。连续批处理请求/token/chunk 上限仍归 F001。
- 模型选择：`empirical`（现行兼容）、`analytical`（解析）、`calibrated`（解析+指定版本 profile）。预设由服务层展开为完整 resolved config；`core` 只接收显式参数，保持零 IO；结果记录预设、模型与校准版本。
- `StepCostBreakdown`：逐 rank FLOPs、HBM/互联 bytes、compute/memory/communication/overhead、重叠量、critical path、瓶颈与适用范围告警。引擎只使用 critical path 时间推进时钟，其他明细用于指标及界面。

## 成本与容量算法

单 GPU 的权重和 KV 基础量为：

```text
weights_bytes = params * bytes_per_weight + quant_metadata
kv_bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_kv
kv_available = HBM * utilization - weights - runtime - workspace - activation_peak
token_budget_per_rank = floor(kv_available / kv_bytes_per_token_per_rank)
```

容量还须按每 rank 的分片/复制规则和 paged block 取整，并计入 allocator 元数据与碎片。解析模式把推导值写入 resolved `max_total_tokens`；empirical 模式接受显式预算，旧 `kv_token_budget` 仅在兼容层转换。计算与访存初期采用 Roofline：

```text
t_compute = step_FLOPs / effective_FLOPS(shape)
t_memory = HBM_bytes / effective_HBM_BW(shape)
t_device = t_fixed + max(t_compute, t_memory)
```

Prefill 使用实际 chunk、历史上下文、算子与新 KV 读写；Decode 使用每请求逻辑 context、batch 权重读取、attention KV 读取和 kernel shape。`effective` 来源于适用 shape 的默认/校准数据，不直接等于理论峰值。单机 TP 增加 collective 次数、传输 bytes 和互联带宽，基础通信项为 `latency*rounds + bytes/bandwidth`；重叠系数 `eta` 在 `[0,1]` 内，总成本初期为 `t_device+t_comm-eta*min(t_device,t_comm)`，最慢 rank 决定 critical path。DP/PP/EP/PD 的公式不在此首期提前冻结。

## 实施阶段与交付

1. **PM1：经验后端适配。** F001 冻结 `StepWorkload` 后用它驱动四系数后端，定义 `StepCostBreakdown`，维护确定性与兼容回归。
2. **PM2：单 GPU 解析。** 新增模型/硬件/运行时解析配置、Dense/GQA 参数与 KV bytes/token、容量预算、Prefill/Decode FLOPs、HBM Roofline 与瓶颈解释。
3. **PM3：校准。** 按模型、GPU、dtype、runtime 和 shape 建有效算力/带宽及固定开销 profile，记录版本/适用范围；微基准与端到端误差报告覆盖 Prefill/Decode、batch、context 和饱和负载；超出 profile 范围给低置信度告警。原计划的大多数测试点约 30%（未校准）和 15%～20%（同设备同 runtime 校准）是待验证目标，不是当前精度承诺。
4. **PM4：单机 TP。** TP1/2/4/8 的权重/KV 分片、通信、rank placement 和关键路径，补充多 rank 明细与展示；不隐式加入跨节点拓扑。

## 验收与外部交付边界

F001 完成后才做端到端精度验收；各阶段均用手算形状/算子成本、容量守恒、placement、通信量、确定性和兼容模式回归校验。接口及预设解析由 [服务计划](server-api.md)、配置与结果展示由 [前端计划](frontend.md) 跟踪；新明细优先新增字段，不无故改变已有时延与请求区间字段。
