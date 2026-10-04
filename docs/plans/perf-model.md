# 性能模型实施计划（F002）

**状态：待实现。** F001 的四系数成本、`StepWorkload` 和 P/D 融合时长以 [调度与引擎实施规范](scheduler-engine.md) 为唯一依据；本计划仅扩展后续模型、设备、内存、算子、通信与校准模块，不为 F001 增加实现前提。

## 目标与首期范围

从可配置的模型结构、硬件、运行时和并行策略推导单步时间、显存占用和瓶颈，给出可解释、可校准的成本分解。首期支持 decoder-only Dense/GQA、BF16/FP16、同构 NVIDIA GPU、单 GPU；单 GPU 模型稳定后支持单机 TP1/2/4/8。PP、DP、EP、PD 分离、跨节点、异构设备、KV offload、多级存储、speculative decoding 与自动在线 profiling **不属于本次验收**，不得以规划字段冒充已可用模式。

## 待冻结的配置和接口

- 模型规格：层数、hidden/KV heads、head dimension、FFN、权重/activation/KV 精度、最大上下文，推导参数量和算子 shape。
- 设备规格：HBM 容量/带宽、各精度有效 FLOPS、单机互联的带宽/延迟；区分理论峰值和校准后的有效值。
- 运行时与部署：读取 F001 唯一的 `engine.page_size`（不另设 paged KV block size）、memory utilization、runtime/workspace/activation reserve、allocator 碎片、CUDA Graph 和固定开销；TP 时记录每 rank 权重、KV、算子 placement。连续批处理请求/token/chunk 上限仍归 F001。
- 模型选择：`empirical`（四系数经验）、`analytical`（解析）、`calibrated`（解析+指定版本 profile）。预设由服务层展开为完整 resolved config；`core` 只接收显式参数，保持零 IO；结果记录预设、模型与校准版本。
- `StepCostBreakdown`：逐 rank FLOPs、HBM/互联 bytes、compute/memory/communication/overhead、重叠量、critical path、瓶颈与适用范围告警。引擎只使用 critical path 时间推进时钟，其他明细用于指标及界面。

## 成本与容量算法

单 GPU 的权重和 KV 基础量为：

```text
weights_bytes = params * bytes_per_weight + quant_metadata
kv_bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_kv
kv_available = HBM * utilization - weights - runtime - workspace - activation_peak
token_budget_per_rank = floor(kv_available / kv_bytes_per_token_per_rank)
```

容量推导须考虑每 rank 的分片/复制规则、allocator 元数据与碎片。F001 的显式 `max_total_tokens` 输入必须是 `page_size` 的正整数倍，不自动取整；未来解析模式仅在生成配置前将硬件推导值向下对齐到完整页（不足一页时报错）。若同时给出已校验的用户上限和已对齐的硬件容量，取二者较小值作为唯一的 resolved `max_total_tokens`；进入 `EngineConfig` 时该值仍必须为整页且不得另设有效容量字段。计算与访存初期采用 Roofline：

```text
t_compute = step_FLOPs / effective_FLOPS(shape)
t_memory = HBM_bytes / effective_HBM_BW(shape)
t_device = t_fixed + max(t_compute, t_memory)
```

Prefill 使用实际输入、历史上下文、算子与新 KV 读写；Decode 使用每请求逻辑 context、batch 权重读取、attention KV 读取和 kernel shape。`StepWorkload` 的纯阶段按该阶段耗时推进时钟；融合阶段分别计算 P/D 子工作量，经验后端取 `max(t_prefill, t_decode)`，解析/校准模式需要显式定义共享设备上的融合关键路径，不能把两项无条件相加。`effective` 来源于适用 shape 的默认/校准数据，不直接等于理论峰值。单机 TP 增加 collective 次数、传输 bytes 和互联带宽，基础通信项为 `latency*rounds + bytes/bandwidth`；重叠系数 `eta` 在 `[0,1]` 内，总成本初期为 `t_device+t_comm-eta*min(t_device,t_comm)`，最慢 rank 决定 critical path。DP/PP/EP/PD 的公式不在此首期提前冻结。

## 实施阶段与交付

1. **PM1：经验模型。** 用 `StepWorkload` 驱动四系数成本计算，纯阶段按对应成本计时，融合阶段取 P/D 两项成本的最大值；定义 `StepCostBreakdown`，验收两种 P/D 模式的确定性与阶段时间。
2. **PM2：单 GPU 解析。** 新增模型/硬件/运行时解析配置、Dense/GQA 参数与 KV bytes/token、容量预算、Prefill/Decode FLOPs、HBM Roofline 与瓶颈解释。
3. **PM3：校准。** 按模型、GPU、dtype、runtime 和 shape 建有效算力/带宽及固定开销 profile，记录版本/适用范围；微基准与端到端误差报告覆盖 Prefill/Decode、batch、context 和饱和负载；超出 profile 范围给低置信度告警；微基准与端到端误差报告须分别覆盖互斥及融合执行的适用范围。
4. **PM4：单机 TP。** TP1/2/4/8 的权重/KV 分片、通信、rank placement 和关键路径，补充多 rank 明细与展示；不隐式加入跨节点拓扑。

## 验收与外部交付边界

F001 完成后才做端到端精度验收；各阶段用手算形状/算子成本、容量守恒、placement、通信量及两种 P/D 模式的确定性校验。接口及预设解析由 [服务计划](server-api.md)、配置与结果展示由 [前端计划](frontend.md) 跟踪；结果字段与实际时延、请求区间一致。
