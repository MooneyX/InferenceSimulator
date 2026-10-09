# 性能模型实施计划（F002）

**状态：部分实现。** 四系数及单 GPU Dense/GQA 逐步 Roofline 估计已接通，见 [现行计时设计](../design/perf-model.md)；本计划继续跟踪逐阶段/逐 rank 成本分解、未覆盖算子、形状校准、TP/EP 通信和实测误差，不为 F001 增加实现前提。

## 目标与首期范围

从可配置的模型结构、硬件、运行时和并行策略推导单步时间、显存占用和瓶颈，给出可解释、可校准的成本分解。第一阶段与 [F004 三阶段计划](expert-parallelism.md) 共用模型/硬件配置和单 GPU Roofline 步成本接口，先支持 decoder-only Dense/GQA、BF16/FP16、同构 NVIDIA GPU；通用单机 TP1/2/4/8 的性能建模为 F004 第二阶段提供基础。MoE 单实例 TP + EP 的组合验收由 F004 管理；本计划不单独定义 MoE 并行拓扑或 EP 公式。PP、DP、PD 分离、跨节点、异构设备、KV offload、多级存储、speculative decoding 与自动在线 profiling **不属于本次验收**，不得以规划字段冒充已可用模式。

模型/硬件预置及单卡 KV 容量的分阶段边界见 [模型与硬件容量子计划](model-hardware-kv-capacity.md)：可浏览的预置不等于可计算或可运行；非 NVIDIA 型号在该子计划中只作为待核验的容量资料，不扩展本计划首期解析性能模型的硬件范围。

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

模型结构、权重/KV 字节数、设备的有效性能和固定开销在仿真前解析为静态输入；每个仿真步再根据 `StepWorkload` 的实际 batch、复用长度与上下文生成相应算子工作量并应用 Roofline。上述 `max` 仅用于同一可重叠的计算/访存阶段；顺序依赖的层和通信按关键路径合成，不将整场工作量预先压成四个经验系数。未经校准的输出标为解析估计。

Prefill 使用实际输入、历史上下文、算子与新 KV 读写；Decode 使用每请求逻辑 context、batch 权重读取、attention KV 读取和 kernel shape。`StepWorkload` 的纯阶段按该阶段耗时推进时钟；融合阶段分别计算 P/D 子工作量，经验后端取 `max(t_prefill, t_decode)`，解析/校准模式需要显式定义共享设备上的融合关键路径，不能把两项无条件相加。`effective` 来源于适用 shape 的默认/校准数据，不直接等于理论峰值。单机 TP 增加 collective 次数、传输 bytes 和互联带宽，基础通信项为 `latency*rounds + bytes/bandwidth`；重叠系数 `eta` 在 `[0,1]` 内，总成本初期为 `t_device+t_comm-eta*min(t_device,t_comm)`，最慢 rank 决定 critical path。DP/PP/EP/PD 的公式不在此首期提前冻结。

## 实施阶段与交付

1. **PM1 剩余：可解释成本分解。** 在已接通的 `StepWorkload` 四系数计时基础上定义 `StepCostBreakdown` 与对应结果明细；目前只有步时长，没有逐阶段/逐 rank 的成本分解输出。
2. **PM2：单 GPU 解析（基础已接通）。** 当前已支持 Dense/GQA 规格、三级 KV 容量和 Prefill/Decode 的投影、注意力、FFN 逐步 Roofline 估计及总量汇总；后续补词表投影、更多算子与可校准的逐阶段瓶颈明细。
3. **PM3：校准。** 按模型、GPU、dtype、runtime 和 shape 建有效算力/带宽及固定开销 profile，记录版本/适用范围；微基准与端到端误差报告覆盖 Prefill/Decode、batch、context 和饱和负载；超出 profile 范围给低置信度告警；微基准与端到端误差报告须分别覆盖互斥及融合执行的适用范围。
4. **PM4：单机 TP（Dense 基线已接通）。** 单实例同构 TP1/2/4/8 已支持逐卡权重/KV 容量、均匀本地 Roofline 和每层两次 Ring 等效同步计时；下一步校准 collective 形态、真实链路效率、非均匀 rank 工作量及逐 rank/逐阶段成本明细，不隐式加入跨节点拓扑。

## 验收与外部交付边界

端到端精度验收在解析/校准模型接通后进行；各阶段用手算形状/算子成本、容量守恒、placement、通信量及两种 P/D 模式的确定性校验。接口及预设解析由 [服务计划](server-api.md)、配置与结果展示由 [前端计划](frontend.md) 跟踪；结果字段与实际时延、请求区间一致。
