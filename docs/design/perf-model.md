# 当前经验性能模型与 KV 恢复计时

`core/src/simcore/perf_model.py` 使用 `prefill_c1`、`prefill_c2`、`decode_d1`、`decode_d2`（单位 ms）估算 GPU 工作量的纯函数成本；可选 KV 上行链路参数由引擎用于独立异步传输计时。它估算**仿真时间**，不是实测 GPU 时间或已标定的硬件模型。

## 单项成本

对每个首次或回撤恢复 Prefill 项，新计算的 token 数 `n >= 1`、此前已复用前缀长度 `c >= 0`：

```text
prefill_cost(c, n) = prefill_c1*n + prefill_c2*((c+n)^2 - c^2)
decode_step_time(T) = decode_d1 + decode_d2*T
```

首次准入时 `c=actual_hit_tokens`、`n=prompt_len-c`；即使完整输入块都命中也至少重算最后一个输入 token。回撤恢复到 `P+g-1` 时按实际命中与重算长度再次计费。二次项包含历史上下文增量成本；`T` 只统计**本步 Decode 请求**各自处理上一输出后逻辑上下文长度 `P+g` 之和，不能因为共享物理 KV 而缩短。

## 当前单步工作量与时长

`Engine.start_step` 选批，内部 `_execute_step`（`selection` 必填）构造不可变 `StepWorkload`：GPU 阶段只有 `PREFILL`、`DECODE`、`MIXED`。Prefill/Decode 项按请求 ID 排序；还记录新申请页数、批内请求及 `restore_only` 请求 ID（混合步中只恢复准入、不新采样输出的请求），不携带恢复页来源或页大小。传输是独立异步任务，不构造纯 `RESTORE` GPU 步；两种 P/D 模式等待回载时均可推进已有 Decode，新恢复请求本步不参加 Decode。

`workload_time` 对 Prefill 项顺序求和 `P_cost=Σ(prefill_c1*n+prefill_c2*n*(2*c+n))`；Decode 集合非空时算 `D_cost=decode_d1+decode_d2*T`，否则为 0。GPU 步长仍为 `max(P_cost,D_cost)`，**不相加**，不再附加 KV 传输时长。

## 异步页恢复计时

启用 L3 的正常路径先进行 L3 查询；连续 L3 命中段的 token 数严格大于 `l3_prefetch_min_tokens` 时发起 L3→L2 预取，选批通过预算后再经 L2→L1 回载，完成后才执行准入；未预取的 L3 页不直接供 GPU 复用。每个实例的每条链路按页串行，两条链路和 GPU 可独立推进。令 `S=page_size*kv_bytes_per_token` 字节，每个页传输任务分别计算：

```text
L3→L2 service_ms = l3_latency_ms + S/l3_to_l2_bandwidth_bytes_per_ms
L2→L1 service_ms = l2_latency_ms + S/l2_to_l1_bandwidth_bytes_per_ms
transfer_start = max(任务发起时刻, 同链路上一任务完成时刻)
transfer_end = transfer_start + service_ms
restore_bytes = Σ(已完成上行任务的 S)
restore_time_ms = Σ(已完成上行任务的 service_ms)
```

固定延迟是**每页每条链路一次**，不是每 GPU 步一次；L3 页先实际迁入 L2，再回载 L1，两段完成各计一页上行量。L3 查询另由 `l3_lookup_latency_ms` 推进事件，链路排队和 lookup 不计入 `restore_time_ms`；逐级驱逐/降级仍同步，不计写回传输耗时。恢复总耗时是任务服务量，不等于 makespan 增量，也不能直接加到 GPU 步长或全场设备时间上。低层关闭时新增成本全为零，原有两种 P/D 计时保持不变。此模型不代表真实设备并行或已校准耗时。

## 单 GPU 与同构 TP 解析计时

`POST /api/simulate` 使用 `inference.mode=physical`、`inference.timing.mode=analytical` 时，在 Dense 全注意力 GQA/MQA、BF16/FP16、单 GPU 条件下，以已解析的 `ModelSpec` 和 `HardwareSpec` 使用 `analytical.step_cost` 替代上述四系数整步计算。投影、因果注意力、FFN 各阶段分别取 `kernel_overhead_ms + max(FLOPs/effective_FLOPS, HBM_bytes/effective_HBM_BW)`，逐层顺序相加；融合步汇总本步 Prefill 和 Decode 的工作量、权重每层计一次。KV 恢复不产生 GPU 工作量或算子成本，按前述异步页传输事件独立推进，不附加到解析 GPU 步长。自动模式把物理链路带宽解析为 token/ms，每页每链路按固定延迟加页 token 数/速率计服务时长；`restore_bytes` 仍使用派生的 KV bytes/token 供指标展示。单个 `SimResult` 中的 `analysis` 汇总执行步数、计算量、HBM 访问与解析耗时；该单卡模型不包括 embedding/输出词表投影、形状校准或 TP/EP 通信，不得解释为真实设备测量值。`inference.timing.mode=empirical` 时，同一入口仍执行本文件前述经验计时。同构 TP>1 仅支持全注意力 Dense BF16/FP16 的解析模式：Query 头与 FFN 按 rank 均分，KV 头不足 TP 数时在不同 rank 复制。每 rank 对 QKV、注意力、输出投影和 FFN 分别按本地 FLOPs/HBM 做 Roofline；同组工作均匀，取任意一张卡的阶段耗时。每层注意力输出与 FFN 输出各计一次等效 Ring AllReduce，不模拟内部通信事件或计算/通信重叠。对每次通信，令 `S=本步新计算 token 数×hidden_size×2 bytes`（Decode 每请求计 1 token），`N=tp_size`，`B=单 rank 有效互联 bytes/ms`，则 `t=collective_latency_ms+2×(N-1)/N×S/B`；这不是对 SGLang 实际 collective 调用的逐算子复刻。仅恢复准入不产生 TP collective，L2/L3 页传输在独立事件链路上推进，不追加到设备步长。解析结果的 `flops`、`hbm_bytes`、`compute_ms` 与 `memory_ms` 均是同构单 rank 口径；`analysis.device_ms` 含 TP 通信，`communication_ms` 为其中的同步时间，`communication_bytes` 为单 rank 总传输量，`collectives` 为次数；不将通信时间再次加到 `device_ms`。TP1 的原单 GPU 公式及旧经验模式不变，TP>1 经验四系数不运行。

## 按层数组合的注意力解析粗估

`ModelSpec.attention_groups` 可选：按 `gqa`、`mla`、`mla_dsa` 组合填写层数和对应结构参数，层数之和必须等于总层数；MLA+DSA 的 `full_indexer_layers` 表示重算索引的层数，其余复用，所有层仍各自执行主注意力。自定义组按压缩 KV BF16/FP16、索引 Key FP8（每 128 元素另有 4 字节 scale）、普通 TP 下缓存复制与等成本页假设计算容量；开启 L2 时为 DSA 每层预留 index-K。按组替换普通 GQA 注意力投影、KV 读写和稀疏选位阶段，FFN/MoE 以及 TP/EP 平均通信沿用现有模型；MLA 组可通过 `output_gate` 启用按头分片的输出门控投影，与输出投影合并为一个 Roofline 阶段；默认关闭。`num_shared_experts` 与 `shared_expert_intermediate_size` 配置每个 MoE 层始终参与计算、逐 rank 复制的独立共享专家，其计算和权重读取单列，不新增 TP/EP 通信；默认不启用。门控和共享专家也计入自定义模型的驻留权重与总/激活参数。辅助容量由 `norms`（每层及最终归一化）、`router`（打分矩阵与选位修正向量）、可选 `ihc`（每层两处和最终 head）、组内 `qk_norm`/`norm_dtype`、`index_norm`/`index_norm_dtype` 及 `sink_dtype` 显式配置；空 dtype 跟随主权重，sink 的空 dtype 表示关闭，辅助精度支持 BF16/FP16/FP32。RMSNorm 只含 scale，LayerNorm 含 scale 与 bias；索引 norm 只计 full indexer 层，sink 每层每 Query head 一个参数并按头分片，其余上述辅助项逐 rank 复制。按组件直接累加逻辑参数与驻留字节，不再先减旧 GQA 账本或按模型名选择容量公式。iHC、sink、归一化和 Router 精度配置仅作用于容量，不新增或校正逐步性能开销；不支持层顺序敏感的复用、滑动窗口、线性状态或 FP8 KV。未核对具体推理后端与硬件效率，结果仅为条件化粗估，不可当成任意同名注意力模型的精确显存或延迟。

DSA 主注意力每层另按有效候选数计一次 INT32 选位索引读取，即 `4 × sparse_pairs` 字节，不乘 Query/index 头数。依据 SGLang 的 `sgl_kernel/flash_mla.py` 中 `indices: [s_q, h_kv, topk], int32` 接口，以及 `dsa_backend.py` 中单 KV 头的调用（本地参考见 `refs/sglang/python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py`）。这是有效索引流量粗估，不包含 top-k 输出写入、排序、padding 扫描或真实缓存命中，不能解释为精确 HBM 访问。

## HY4-Preview BF16 MLA/DSA 条件化粗估

HY4 预设以官方检查点 revision `705d81ee51566a186d645b74c974d642ef2828fe` 为初始参考，提供完整注意力组、输出门控和独立共享专家配置，参数不锁定；用户可直接修改主干、注意力与辅助参数，界面编辑后标为自定义模型并清除参考修订号。预设和自定义组使用同一条通用计算路径，不按 HY4 名称或布局分派性能公式。权重与 KV 容量均使用通用组件账本，FP32 router、iHC、sink、索引 LayerNorm 及 BF16 block/最终/低秩 RMSNorm 由预设显式填写；复制为自定义模型保留完整容量与性能配置。固定 revision 的 `model.safetensors.index.json` 确认存在 `model.norm.weight`（`model-00099-of-00131.safetensors`），SGLang `hunyuan_v4.py` 创建并在最终 iHC head 后使用它。原账本遗漏此最终 RMSNorm，现补上 6,144 个逻辑参数及每 rank 12,288 字节 BF16 权重；未将索引目录核验表述为全部 tensor 形状/精度审计。78 层中 21 层 full indexer 执行选位，其余复用索引；所有层仍处理每个 query 的至多 2048 个候选。解析模式按一组 step 聚合查询与历史位置对数，full 层另计索引 Key（FP8 + scale）读写、投影与打分，所有层计压缩 KV 读写、稀疏 MLA、注意力门控；独立共享专家的计算亦单列。复用下文 TP/EP 平均负载通信公式，不含 MTP、算子融合、top-k 排序和 iHC 的逐算子成本，也未做形状/硬件校准。结果用于趋势对比，不能视为 SGLang 真实延迟。

## 平均负载 MoE/EP 解析计时（未经校准）

`TP=EP=GPU 数` 且 BF16/FP16 时，用本步新 Prefill token 数加 Decode 请求数作为同一批 MoE 输入；每层保留注意力输出的 TP 同步，Dense 层继续计 FFN 及第二次 TP 同步，MoE 层则改计 gate、均匀放置的本地专家计算、dispatch、combine，**不重复计 Dense FFN 与第二次 TP 同步**。专家计算每 rank 的平均任务数为 `tokens * experts_per_token / ep_size`，权重 HBM 按本 rank 全部驻留专家保守计入；不存在逐 token 路由或热点专家的尾延迟预测。`MIXED` 一次合并批次收费，只恢复准入的请求不计专家计算/通信；KV 恢复由异步页传输独立计时，不产生纯恢复 GPU 步。

设 `E=num_experts`、`R=ep_size`、`K=experts_per_token`、`H=hidden_size`、`T=本步 token 数`，均匀无放回路由下某目标 rank 至少被选一次的概率为 `p=1-∏_{i=0}^{K-1}(E-E/R-i)/(E-i)`。每 rank 的平均远端专家选择数为 `M=T*K*(R-1)/R²`；`normal` 每 rank 平均远端向量数为 `T*(R-1)/R*p`，`low_latency` 按 `M` 个选择计。激活字节宽度与本版计算精度一致（BF16/FP16：2 bytes/元素）；dispatch 与 combine 每次远端选择分别额外计 8 bytes 逻辑元数据（ID/索引 4 bytes、权重 4 bytes）。这是未校准的**逻辑流量**，不等同于 DeepEP 的实际链路流量或内核布局。

含 Prefill 的步采用 `normal`，纯 Decode 采用 `low_latency`；两模式共用单 rank `effective_interconnect_gb_s` 和 `ep_communication_latency_ms`，dispatch/combine 各按 `固定开销 + 每 rank 逻辑字节/有效带宽` 计一次、各 MoE 层顺序加入步关键路径。不做通信与计算重叠、专家复制或 EPLB。TP 的 `collective_latency_ms` 独立计费；分析结果的 `communication_ms` 为 TP+EP 时间之和，`ep_dispatch_ms`、`ep_combine_ms`、`tp_communication_ms` 是其中分项，`collectives` 仅计 TP，`ep_exchanges` 单列。`device_ms` 已包含全部 TP/EP 通信，不得再把通信分项加到时长上；`restore_ms` 单列已完成 KV 传输任务的服务时长之和，不附加到 GPU 步长，也不代表 makespan 增量。页面试算使用单 rank 有效互联 600 GB/s、TP 集合通信固定延迟 0.01 ms/次、EP dispatch/combine 固定开销 0.02 ms/次，均未经校准，可手动修改；1800 GB/s 仅为 B300 的单卡双向标称值，不直接代入有效带宽。精度与进一步校准见 [F002 计划](../plans/perf-model.md) 和 [F004 计划](../plans/expert-parallelism.md)。
