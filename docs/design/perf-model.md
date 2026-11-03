# 性能模型与 KV 恢复计时

所有仿真通过 `core/src/simcore/perf_model.py` 的 `step_cost` 使用同一性能模型：根据模型结构、硬件有效性能和实际步工作量估算逐阶段计算、访存与通信时间。同一模块定义不可变工作量、`StepCost` 及全部成本计算函数；`PerfModelConfig` 仅保留 KV 字节数、链路速率和延迟；L3→L2 预取由引擎独立异步计时，L2→L1 回载与本步计算逐层流水并计入步长。公开配置不再接受 `inference.timing`，不保留经验四系数或兼容转换。这里估算的是**仿真时间**，不是实测 GPU 时间或已标定的硬件性能。

## 单项工作量

对每个首次或回撤恢复 Prefill 项，新计算的 token 数 `n >= 1`、此前已复用前缀长度 `c >= 0`。首次准入时 `c=actual_hit_tokens`、`n=prompt_len-c`；即使完整输入块都命中也至少重算最后一个输入 token。回撤恢复到 `P+g-1` 时按实际命中与重算长度再次计费。Decode 使用各请求处理上一输出后的逻辑上下文长度 `P+g`，不能因为共享物理 KV 而缩短。

## 当前单步工作量与时长

`Engine.start_step` 选批，内部 `_execute_step`（`selection` 必填）构造不可变 `StepWorkload`：GPU 阶段只有 `PREFILL`、`DECODE`、`MIXED`。Prefill/Decode 项按请求 ID 排序；还记录新申请页数、批内请求及 `restore_only` 请求 ID（混合步中只恢复准入、不新采样输出的请求），不携带恢复页来源或页大小。不构造纯 `RESTORE` GPU 步；新恢复请求本步不参加 Decode。

`perf_model.step_cost` 汇总本步 Prefill/Decode 工作量，对每个计算阶段使用 `kernel_overhead_ms + max(FLOPs/effective_FLOPS, HBM_bytes/effective_HBM_BW)`，再依次累加层、算子阶段与同步通信。融合步共享该批的权重读取；不是分别计算 P/D 后取最大值。本批有 L2→L1 回载时，按下文逐层流水把 `step_cost` 的计算时长换算为步长。结果字段类型仍允许为空，但新运行（含零 GPU 步）始终提供 `analysis` 对象。

## L3→L2 异步预取计时

启用 L3 的正常路径先进行 L3 查询；连续 L3 命中段的 token 数严格大于 `l3_prefetch_min_tokens` 时发起 L3→L2 预取，全部完成后请求才就绪；未预取的 L3 页不直接供 GPU 复用。每个实例的 L3→L2 链路按页串行，与 GPU 独立推进。令 `S=page_size*kv_bytes_per_token` 字节，每个页传输任务分别计算：

```text
L3→L2 service_ms = l3_latency_ms + S/l3_to_l2_bandwidth_bytes_per_ms
transfer_start = max(任务发起时刻, 同链路上一任务完成时刻)
transfer_end = transfer_start + service_ms
```

固定延迟是**每页一次**。只有本地连续前缀不足且仍有可缓存输入页时才查询 L3，由顶层 `shared_l3.lookup_latency_ms` 推进一次查询事件，链路排队和 lookup 不计入 `restore_time_ms`；逐级驱逐/降级仍同步，不计写回传输耗时。

## L2→L1 逐层流水回载

参照 SGLang HiCache：调度器选批时即为 L2 命中页分配 L1 目标页，请求立即进入本步，不在调度层等待回载；回载在独立拷贝流上逐层发出，前向第 i 层注意力只等第 i 层 KV 到齐。仿真在选批后对本批唯一提升页依次 `reserve_transfer` + `complete_transfer`，同步完成页迁移，不产生传输事件；回载耗时只通过步长体现。令 `p` 为本批唯一提升页数、`L=num_layers`、`T=page_size*kv_bytes_per_token/l2_to_l1_bandwidth_bytes_per_ms`（每页整组传输时长），`compute_ms` 为 `step_cost` 的计算时长：

```text
t = l2_latency_ms + p*T/L          # 每层回载：每层一次固定延迟，字节按层均分
c = compute_ms / L                 # 单层计算取均值（含同批 Decode）
F_0 = 0; F_i = max(F_{i-1}, i*t) + c
step_ms = F_L = max(t + L*c, L*t + c)
restore_time_ms += L*t
restore_exposed_ms += step_ms - compute_ms
```

回载自步起点开始（保守口径，不模拟 overlap scheduler 提前一步发起）；本步结束前回载必已完成，同实例下一步无需再排队。计算较长时只暴露首层回载 `t`，回载较长时步长约为全部回载加末层计算。回载慢会拉长整步，同批 Decode 一并变慢。只有恢复准入、没有计算的步 `c=0`，步长为 `L*t`，不计入 GPU 步数。

简化：各层计算与 KV 字节均按层均分，MoE 的 Dense/MoE 层与异构注意力组不区分层序；`l2_to_l1_gb_s` 视为整实例有效带宽；不模拟多批在途、`load_back_threshold` 回退重算或 IO 后端差异；Chunked Prefill 未实现。

`restore_bytes` 累计已完成的 L3→L2 页与 L2→L1 提升页字节；`restore_time_ms` 与 `analysis.restore_ms` 为两条链路的服务时长之和，不等于 makespan 增量。`analysis.device_ms` 只计 `step_cost` 计算时长，步长中被回载拉长的部分单列为 `analysis.restore_exposed_ms`，两者之和为 GPU 步长总和。低层关闭时新增成本全为零。此模型不代表真实设备并行或已校准耗时。

## 单 GPU 与同构 TP 解析计时

`POST /api/simulate` 始终使用 `perf_model.step_cost`。在 Dense 全注意力 GQA/MQA、BF16/FP16、单 GPU 条件下，以已解析的 `ModelSpec` 和 `HardwareSpec` 估算各阶段成本。投影、因果注意力、FFN 各阶段分别取 `kernel_overhead_ms + max(FLOPs/effective_FLOPS, HBM_bytes/effective_HBM_BW)`，逐层顺序相加；融合步汇总本步 Prefill 和 Decode 的工作量、权重每层计一次。KV 恢复不产生 GPU 工作量或算子成本；L3→L2 预取独立推进，L2→L1 回载按前述逐层流水换算步长。容量解析把物理链路带宽转换为 token/ms；`restore_bytes` 仍使用派生的 KV bytes/token 供指标展示。单个 `SimResult` 中的 `analysis` 汇总执行步数、计算量、HBM 访问与解析耗时；该单卡模型不包括 embedding/输出词表投影、形状校准或 TP/EP 通信，不得解释为真实设备测量值。全注意力 Dense BF16/FP16 的同构 TP>1 也沿用同一性能模型：Query 头与 FFN 按 rank 均分，KV 头不足 TP 数时在不同 rank 复制。每 rank 对 QKV、注意力、输出投影和 FFN 分别按本地 FLOPs/HBM 做 Roofline；同组工作均匀，取任意一张卡的阶段耗时。每层注意力输出与 FFN 输出各计一次等效 Ring AllReduce，不模拟内部通信事件或计算/通信重叠。对每次通信，令 `S=本步新计算 token 数×hidden_size×2 bytes`（Decode 每请求计 1 token），`N=tp_size`，`B=单 rank 有效互联 bytes/ms`，则 `t=collective_latency_ms+2×(N-1)/N×S/B`；这不是对 SGLang 实际 collective 调用的逐算子复刻。仅恢复准入不产生 TP collective；L2→L1 回载按逐层流水调整步长，不计入 `device_ms`。解析结果的 `flops`、`hbm_bytes`、`compute_ms` 与 `memory_ms` 均是同构单 rank 口径；`analysis.device_ms` 含 TP 通信，`communication_ms` 为其中的同步时间，`communication_bytes` 为单 rank 总传输量，`collectives` 为次数；不将通信时间再次加到 `device_ms`。TP1 保持原单 GPU 成本公式；TP>1 须补齐有效互联带宽与通信固定延迟。

## 按层数组合的注意力解析粗估

`ModelSpec.attention_groups` 可选：按 `gqa`、`mla`、`mla_dsa` 组合填写层数和对应结构参数，层数之和必须等于总层数；MLA+DSA 的 `full_indexer_layers` 表示重算索引的层数，其余复用，所有层仍各自执行主注意力。自定义组按压缩 KV BF16/FP16、索引 Key FP8（每 128 元素另有 4 字节 scale）、普通 TP 下缓存复制与等成本页假设计算容量；开启 L2 时为 DSA 每层预留 index-K。按组替换普通 GQA 注意力投影、KV 读写和稀疏选位阶段，FFN/MoE 以及 TP/EP 平均通信沿用现有模型；MLA 组可通过 `output_gate` 启用按头分片的输出门控投影，与输出投影合并为一个 Roofline 阶段；默认关闭。`num_shared_experts` 与 `shared_expert_intermediate_size` 配置每个 MoE 层始终参与计算、逐 rank 复制的独立共享专家，其计算和权重读取单列，不新增 TP/EP 通信；默认不启用。门控和共享专家也计入自定义模型的驻留权重与总/激活参数。辅助容量由 `norms`（每层及最终归一化）、`router`（打分矩阵与选位修正向量）、可选 `ihc`（每层两处和最终 head）、组内 `qk_norm`/`norm_dtype`、`index_norm`/`index_norm_dtype` 及 `sink_dtype` 显式配置；空 dtype 跟随主权重，sink 的空 dtype 表示关闭，辅助精度支持 BF16/FP16/FP32。RMSNorm 只含 scale，LayerNorm 含 scale 与 bias；索引 norm 只计 full indexer 层，sink 每层每 Query head 一个参数并按头分片，其余上述辅助项逐 rank 复制。按组件直接累加逻辑参数与驻留字节，不再先减旧 GQA 账本或按模型名选择容量公式。iHC、sink、归一化和 Router 精度配置仅作用于容量，不新增或校正逐步性能开销；不支持层顺序敏感的复用、滑动窗口、线性状态或 FP8 KV。未核对具体推理后端与硬件效率，结果仅为条件化粗估，不可当成任意同名注意力模型的精确显存或延迟。

DSA 主注意力每层另按有效候选数计一次 INT32 选位索引读取，即 `4 × sparse_pairs` 字节，不乘 Query/index 头数。依据 SGLang 的 `sgl_kernel/flash_mla.py` 中 `indices: [s_q, h_kv, topk], int32` 接口，以及 `dsa_backend.py` 中单 KV 头的调用（本地参考见 `refs/sglang/python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py`）。这是有效索引流量粗估，不包含 top-k 输出写入、排序、padding 扫描或真实缓存命中，不能解释为精确 HBM 访问。

## HY4-Preview BF16 MLA/DSA 条件化粗估

HY4 预设以官方检查点 revision `705d81ee51566a186d645b74c974d642ef2828fe` 为初始参考，提供完整注意力组、输出门控和独立共享专家配置，参数不锁定；用户可直接修改主干、注意力与辅助参数，界面编辑后标为自定义模型并清除参考修订号。预设和自定义组使用同一条通用计算路径，不按 HY4 名称或布局分派性能公式。权重与 KV 容量均使用通用组件账本，FP32 router、iHC、sink、索引 LayerNorm 及 BF16 block/最终/低秩 RMSNorm 由预设显式填写；复制为自定义模型保留完整容量与性能配置。固定 revision 的 `model.safetensors.index.json` 确认存在 `model.norm.weight`（`model-00099-of-00131.safetensors`），SGLang `hunyuan_v4.py` 创建并在最终 iHC head 后使用它。原账本遗漏此最终 RMSNorm，现补上 6,144 个逻辑参数及每 rank 12,288 字节 BF16 权重；未将索引目录核验表述为全部 tensor 形状/精度审计。78 层中 21 层 full indexer 执行选位，其余复用索引；所有层仍处理每个 query 的至多 2048 个候选。性能模型按一组 step 聚合查询与历史位置对数，full 层另计索引 Key（FP8 + scale）读写、投影与打分，所有层计压缩 KV 读写、稀疏 MLA、注意力门控；独立共享专家的计算亦单列。复用下文 TP/EP 平均负载通信公式，不含 MTP、算子融合、top-k 排序和 iHC 的逐算子成本，也未做形状/硬件校准。结果用于趋势对比，不能视为 SGLang 真实延迟。

## 平均负载 MoE/EP 解析计时（未经校准）

`TP=EP=GPU 数` 且 BF16/FP16 时，用本步新 Prefill token 数加 Decode 请求数作为同一批 MoE 输入；每层保留注意力输出的 TP 同步，Dense 层继续计 FFN 及第二次 TP 同步，MoE 层则改计 gate、均匀放置的本地专家计算、dispatch、combine，**不重复计 Dense FFN 与第二次 TP 同步**。专家计算每 rank 的平均任务数为 `tokens * experts_per_token / ep_size`，权重 HBM 按本 rank 全部驻留专家保守计入；不存在逐 token 路由或热点专家的尾延迟预测。`MIXED` 一次合并批次收费，只恢复准入的请求不计专家计算/通信；KV 恢复不产生纯恢复 GPU 步，L2→L1 回载按逐层流水调整步长。

设 `E=num_experts`、`R=ep_size`、`K=experts_per_token`、`H=hidden_size`、`T=本步 token 数`，均匀无放回路由下某目标 rank 至少被选一次的概率为 `p=1-∏_{i=0}^{K-1}(E-E/R-i)/(E-i)`。每 rank 的平均远端专家选择数为 `M=T*K*(R-1)/R²`；`normal` 每 rank 平均远端向量数为 `T*(R-1)/R*p`，`low_latency` 按 `M` 个选择计。激活字节宽度与本版计算精度一致（BF16/FP16：2 bytes/元素）；dispatch 与 combine 每次远端选择分别额外计 8 bytes 逻辑元数据（ID/索引 4 bytes、权重 4 bytes）。这是未校准的**逻辑流量**，不等同于 DeepEP 的实际链路流量或内核布局。

含 Prefill 的步采用 `normal`，纯 Decode 采用 `low_latency`；两模式共用单 rank `effective_interconnect_gb_s` 和 `ep_communication_latency_ms`，dispatch/combine 各按 `固定开销 + 每 rank 逻辑字节/有效带宽` 计一次、各 MoE 层顺序加入步关键路径。不做通信与计算重叠、专家复制或 EPLB。TP 的 `collective_latency_ms` 独立计费；分析结果的 `communication_ms` 为 TP+EP 时间之和，`ep_dispatch_ms`、`ep_combine_ms`、`tp_communication_ms` 是其中分项，`collectives` 仅计 TP，`ep_exchanges` 单列。`device_ms` 已包含全部 TP/EP 通信，不得再把通信分项加到时长上；`restore_ms` 单列 KV 传输服务时长之和，不代表 makespan 增量；回载拉长步长的部分见 `restore_exposed_ms`。页面试算使用单 rank 有效互联 600 GB/s、TP 集合通信固定延迟 0.01 ms/次、EP dispatch/combine 固定开销 0.02 ms/次，均未经校准，可手动修改；1800 GB/s 仅为 B300 的单卡双向标称值，不直接代入有效带宽。精度与进一步校准见 [F002 计划](../plans/perf-model.md) 和 [F004 计划](../plans/expert-parallelism.md)。
