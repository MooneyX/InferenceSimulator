# 性能模型与 KV 恢复计时

所有仿真通过 `core/src/simcore/perf_model.py` 的 `step_cost` 使用同一性能模型：根据模型结构、硬件有效性能和实际步工作量估算逐阶段计算、访存与通信时间。同一模块定义不可变工作量、`StepCost` 及全部成本计算函数；`PerfModelConfig` 仅保留 KV 字节数、链路速率和延迟；L3→L2 预取由引擎独立异步计时，L2→L1 回载与本步计算逐层流水并计入步长。公开配置不再接受 `inference.timing`，不保留经验四系数或兼容转换。这里估算的是**仿真时间**，不是实测 GPU 时间或已标定的硬件性能。

## 单项工作量

对每个首次或回撤恢复 Prefill 项，新计算的 token 数 `n >= 1`、此前已复用前缀长度 `c >= 0`。首次准入时 `c=actual_hit_tokens`、`n=prompt_len-c`；即使完整输入块都命中也至少重算最后一个输入 token。回撤恢复到 `P+g-1` 时按实际命中与重算长度再次计费。Decode 使用各请求处理上一输出后的逻辑上下文长度 `P+g`，不能因为共享物理 KV 而缩短。

## 当前单步工作量与时长

`Engine.start_step` 选批，内部 `_execute_step`（`selection` 必填）构造不可变 `StepWorkload`：GPU 阶段只有 `PREFILL`、`DECODE`、`MIXED`。Prefill/Decode 项按请求 ID 排序；还记录新申请页数、批内请求及 `restore_only` 请求 ID（混合步中只恢复准入、不新采样输出的请求），不携带恢复页来源或页大小。不构造纯 `RESTORE` GPU 步；新恢复请求本步不参加 Decode。

`perf_model.step_cost` 汇总本步 Prefill/Decode 工作量，按模型层序（`ModelSpec.layers`）对每类同构层（注意力组 × Dense/MoE × 是否完整计算 DSA 索引器）计一次单层成本：每个计算阶段使用 `kernel_overhead_ms + max(FLOPs/effective_FLOPS, HBM_bytes/effective_HBM_BW)`，再加该层串行的 TP/EP 同步通信，最后按层数汇总。`StepCost.layer_ms` 按层序给出（连续层数, 单层耗时），合计等于 `duration_ms`。融合步共享该批的权重读取；不是分别计算 P/D 后取最大值。本批有 L2→L1 回载时，按下文逐层流水把各层计算换算为步长。结果字段类型仍允许为空，但新运行（含零 GPU 步）始终提供 `analysis` 对象。

## KV 来源（`kv_source.KvSource`）

L2→L1 回载与 L3 读取都通过通用 KV 来源计时：每个引擎为每条链路持有一个来源，按提交顺序先来先服务串行读取，每组读取计一次固定延迟，返回各组到齐的绝对时刻，并记录链路空闲时刻 `free_ms`。DRAM 与 SSD 式存储不分类型实现，只以参数区分：

- **L2→L1（本地 DRAM）**：确定性，`l2_latency_ms` + 字节 / `l2_to_l1_gb_s`。
- **L3 读取**（三层结构读到 L2，两层结构直接读到 L1）：`l3_latency_ms` + 字节 / `l3_read_gb_s`，并叠加 SSD 式波动（`L3_FLUCTUATION`，暂无配置入口）：
  - 两状态马尔可夫切换：降速状态稳态占比 2%、平均持续 50 ms（各段时长服从指数分布），降速时带宽 ×0.3、发起读取另加 1 ms；一次读取跨越状态切换时分段按各段带宽传输。
  - 单次读取带宽乘对数正态系数（均值 1、变异系数 2%）。
  - 种子由 `workload.seed`、实例 ID 与链路名派生，同一配置结果可复现。空闲和固定延迟期间用两状态链的转移概率直接跳至目标时刻，不逐个推进空档中的状态切换；实际传输仍分段积分。

波动参数为**未经实测校准的占位值**，依据的是公开资料的定性特征：企业盘混合负载下 IOPS 的逐秒变异系数约 1%（单次抖动很小）；GC/写入干扰造成成段慢读，读延迟 p99.99 达数十毫秒、盘接近写满时更高（如 arXiv 2610.02676 对 Samsung PM9D3a 的测量）。没有找到大块顺序读或 KV 负载下 SSD 带宽随时间变化的分布数据；温控降速未建模。降速频率与写回流量、盘利用率的关联未建模。

## L3→L2 异步预取计时

启用 L3 的正常路径先进行 L3 查询；连续 L3 命中段的 token 数严格大于 `l3_prefetch_min_tokens` 时发起 L3→L2 预取，全部完成后请求才就绪（对应 SGLang `prefetch_stop_policy=wait_complete`）；未预取的 L3 页不直接供 GPU 复用。参照 SGLang（commit `2b36c718`）`HiCacheController._page_transfer`（`managers/cache_controller.py`）每 `STORAGE_BATCH_SIZE=128` 页一次存储读取、由单个 IO 线程串行处理各预取操作：

- 每个请求本实例需新读的连续前缀页按顺序每 `STORAGE_BATCH_PAGES=128` 页一批提交 L3 来源，每批一次固定延迟与一个 `L3_L2_DONE` 事件；一页包含全部层，不逐层读取。
- 已在本实例在途的页直接等待其所在批次；共享 L2 中其他实例在途的页等待其完成时刻，不重复读取。
- 来源先来先服务：`batch_start = max(提交时刻, 链路空闲时刻)`，`service_ms = batch_end - batch_start`。

只有本地连续前缀不足且仍有可缓存输入页时才查询 L3，由顶层 `shared_l3.lookup_latency_ms` 推进一次查询事件；SGLang 在读取前还有一次 `batch_exists` 命中查询，本项目只建模为该固定查询延迟。链路排队和 lookup 不计入 `restore_time_ms`；逐级驱逐/降级仍同步，不计写回传输耗时。

## L2→L1 逐层流水回载

参照 SGLang HiCache（commit `2b36c718`）：调度器选批时即为 L2 命中页分配 L1 目标页，请求立即进入本步，不在调度层等待回载；`start_loading` 将整批回载合并，在独立拷贝流上逐层发出（`mem_cache/l2_transfer.py` 每层完成记录事件），前向第 i 层注意力只 `wait_until(i)`。仿真在选批后对本批唯一提升页依次 `reserve_transfer` + `complete_transfer`，同步完成页迁移，不产生传输事件；回载耗时只通过步长体现。令 `p` 为本批唯一提升页数，`b_i` 为整组每页 KV 在第 i 层的字节数（按层序，由各层注意力组的 KV 布局决定，合计为每页字节数），`c_i` 为 `StepCost.layer_ms` 给出的第 i 层计算耗时：

```text
R_1..R_L = L2 来源.submit(步起点, [p*b_1, ..., p*b_L])  # 逐层串行，每层一次 l2_latency_ms
F_0 = 步起点; F_i = max(F_{i-1}, R_i) + c_i
step_ms = F_L - 步起点
restore_time_ms += R_L - 回载开始
restore_exposed_ms += step_ms - sum(c_i)
```

各层同构时等价于闭式 `max(t + L*c, L*t + c)`；异构层序（如 Dense 首层、DSA 完整/复用索引层交替、GQA 与 MLA 混合）按实际顺序递推，不再取层均值。回载自步起点开始（保守口径，不模拟 overlap scheduler 提前一步发起）；本步结束前回载必已完成，同实例下一步开始时链路空闲。回载慢会拉长整步，同批 Decode 一并变慢。只有恢复准入、没有计算的步 `c_i=0`，步长为全部回载时间，不计入 GPU 步数。

简化：`l2_to_l1_gb_s` 视为整实例有效带宽；不模拟多批在途、`load_back_threshold` 回退重算、prefill CUDA graph 等待末层或 IO 后端差异；Chunked Prefill 未实现。

## 两层结构 L3→L1 直接读取

L2 容量为 0 且启用 L3 时（行为见 [kv-prefix-reuse.md](kv-prefix-reuse.md)），选批后本批唯一 L3 命中页自步起点按前缀顺序每 128 页一批提交 L3 来源。每批包含全部层，首层计算也须等末批到齐，因此不与计算逐层流水：

```text
E_1..E_n = L3 来源.submit(步起点, [每批页数 × 每页字节])  # 每批一次 l3_latency_ms，带波动
step_ms = (E_n - 步起点) + sum(c_i)
restore_time_ms += E_n - 读取开始
restore_exposed_ms += E_n - 步起点
```

对应 SGLang LMCache 接入（commit `2b36c718`）`_start_external_load` 准入时分配 GPU 页并 `submit_load`、`prepare_load_on_stream` 让前向计算流等待读取完成。读取慢会拉长整步，同批 Decode 一并变慢；只有恢复准入的步步长为全部读取时间。每批完成时产生 `L3_L1_DONE` 事件，仅释放源 L3 页保护，不唤醒调度。L1 淘汰写入 L3 仍同步且不计写回耗时，SSD 读写不争用同一链路，结果偏乐观。

`restore_bytes` 累计已完成的 L3 读取页与 L2→L1 提升页字节；`restore_time_ms` 与 `analysis.restore_ms` 为各链路的服务时长之和，不等于 makespan 增量。`analysis.device_ms` 只计 `step_cost` 计算时长，步长中被回载拉长的部分单列为 `analysis.restore_exposed_ms`，两者之和为 GPU 步长总和。低层关闭时新增成本全为零。此模型不代表真实设备并行或已校准耗时。

## 单 GPU 与同构 TP 解析计时

`POST /api/simulate` 始终使用 `perf_model.step_cost`。在 Dense 全注意力 GQA/MQA、BF16/FP16、单 GPU 条件下，以已解析的层序结构（`build_layer_plan(ModelSpec)`，运行前构建一次）和 `HardwareSpec` 估算各阶段成本。投影、因果注意力、FFN 各阶段分别取 `kernel_overhead_ms + max(FLOPs/effective_FLOPS, HBM_bytes/effective_HBM_BW)`，逐层顺序相加；融合步汇总本步 Prefill 和 Decode 的工作量、权重每层计一次。KV 恢复不产生 GPU 工作量或算子成本；L3→L2 预取独立推进，L2→L1 回载按前述逐层流水换算步长。容量解析把物理链路带宽转换为 token/ms；`restore_bytes` 仍使用派生的 KV bytes/token 供指标展示。单个 `SimResult` 中的 `analysis` 汇总执行步数、计算量、HBM 访问与解析耗时；该单卡模型不包括 embedding/输出词表投影、形状校准或 TP/EP 通信，不得解释为真实设备测量值。全注意力 Dense BF16/FP16 的同构 TP>1 也沿用同一性能模型：Query 头与 FFN 按 rank 均分，KV 头不足 TP 数时在不同 rank 复制。每 rank 对 QKV、注意力、输出投影和 FFN 分别按本地 FLOPs/HBM 做 Roofline；同组工作均匀，取任意一张卡的阶段耗时。每层注意力输出与 FFN 输出各计一次等效 Ring AllReduce，不模拟内部通信事件或计算/通信重叠。对每次通信，令 `S=本步新计算 token 数×hidden_size×2 bytes`（Decode 每请求计 1 token），`N=tp_size`，`B=单 rank 有效互联 bytes/ms`，则 `t=collective_latency_ms+2×(N-1)/N×S/B`；这不是对 SGLang 实际 collective 调用的逐算子复刻。仅恢复准入不产生 TP collective；L2→L1 回载按逐层流水调整步长，不计入 `device_ms`。解析结果的 `flops`、`hbm_bytes`、`compute_ms` 与 `memory_ms` 均是同构单 rank 口径；`analysis.device_ms` 含 TP 通信，`communication_ms` 为其中的同步时间，`communication_bytes` 为单 rank 总传输量，`collectives` 为次数；不将通信时间再次加到 `device_ms`。TP1 保持原单 GPU 成本公式；TP>1 须补齐有效互联带宽与通信固定延迟。

## 层序与注意力组合的解析粗估

`ModelSpec.layers` 是层数与层类型的唯一来源：按层号顺序列出连续层段，每段给出层数 `count`、引用的注意力组下标 `attention_group`、FFN 类型 `ffn`（`dense`/`moe`）及 MLA+DSA 层是否完整计算索引器 `full_indexer`。`num_layers`、`moe_layers` 只是由层序推导的只读属性。校验：引用的组须存在、每组至少被一层引用、仅 DSA 层可完整计算索引器、每个 DSA 组的首层须完整计算。未配置 `attention_groups` 时以顶层 GQA 字段作为唯一注意力组（下标 0）。预置模型层序取自 HF `config.json`：HY4-preview 的 `mlp_layer_types`（仅第 0 层 Dense）与 `indexer_types`（第 0、1 层 full，之后 shared×3 + full 循环，共 21 个 full 层）；GLM-5 的 `first_k_dense_replace=3`、`moe_layer_freq=1`（同 SGLang `deepseek_v2.py` 的 `layer_id >= first_k_dense_replace` 判定），各层均完整计算索引器；Qwen3 MoE 的 `decoder_sparse_step=1`、`mlp_only_layers=[]`。

`ModelSpec.attention_groups` 可选：按 `gqa`、`mla`、`mla_dsa` 填写各组结构参数，所在层由层序引用；复用索引的 DSA 层不计索引器，所有层仍各自执行主注意力。自定义组按压缩 KV BF16/FP16、索引 Key FP8（每 128 元素另有 4 字节 scale）、普通 TP 下缓存复制与等成本页假设计算容量；开启 L2 或 L3 时为 DSA 每层预留 index-K。按组替换普通 GQA 注意力投影、KV 读写和稀疏选位阶段，FFN/MoE 以及 TP/EP 平均通信沿用现有模型；MLA 组可通过 `output_gate` 启用按头分片的输出门控投影，与输出投影合并为一个 Roofline 阶段；默认关闭。`num_shared_experts` 与 `shared_expert_intermediate_size` 配置每个 MoE 层始终参与计算、逐 rank 复制的独立共享专家，其计算和权重读取单列，不新增 TP/EP 通信；默认不启用。门控和共享专家也计入自定义模型的驻留权重与总/激活参数。辅助容量由 `norms`（每层及最终归一化）、`router`（打分矩阵与选位修正向量）、可选 `ihc`（每层两处和最终 head）、组内 `qk_norm`/`norm_dtype`、`index_norm`/`index_norm_dtype` 及 `sink_dtype` 显式配置；空 dtype 跟随主权重，sink 的空 dtype 表示关闭，辅助精度支持 BF16/FP16/FP32。RMSNorm 只含 scale，LayerNorm 含 scale 与 bias；索引 norm 只计 full indexer 层，sink 每层每 Query head 一个参数并按头分片，其余上述辅助项逐 rank 复制。按组件直接累加逻辑参数与驻留字节，不再先减旧 GQA 账本或按模型名选择容量公式。iHC、sink、归一化和 Router 精度配置仅作用于容量，不新增或校正逐步性能开销；层序只影响逐层计时与回载流水，不建模跨层依赖的其他效应；不支持滑动窗口、线性状态或 FP8 KV。未核对具体推理后端与硬件效率，结果仅为条件化粗估，不可当成任意同名注意力模型的精确显存或延迟。

DSA 主注意力每层另按有效候选数计一次 INT32 选位索引读取，即 `4 × sparse_pairs` 字节，不乘 Query/index 头数。依据 SGLang 的 `sgl_kernel/flash_mla.py` 中 `indices: [s_q, h_kv, topk], int32` 接口，以及 `dsa_backend.py` 中单 KV 头的调用（本地参考见 `refs/sglang/python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py`）。这是有效索引流量粗估，不包含 top-k 输出写入、排序、padding 扫描或真实缓存命中，不能解释为精确 HBM 访问。

## HY4-Preview BF16 MLA/DSA 条件化粗估

HY4 预设以官方检查点 revision `705d81ee51566a186d645b74c974d642ef2828fe` 为初始参考，提供完整注意力组、输出门控和独立共享专家配置，参数不锁定；用户可直接修改主干、注意力与辅助参数，界面编辑后标为自定义模型并清除参考修订号。预设和自定义组使用同一条通用计算路径，不按 HY4 名称或布局分派性能公式。权重与 KV 容量均使用通用组件账本，FP32 router、iHC、sink、索引 LayerNorm 及 BF16 block/最终/低秩 RMSNorm 由预设显式填写；复制为自定义模型保留完整容量与性能配置。固定 revision 的 `model.safetensors.index.json` 确认存在 `model.norm.weight`（`model-00099-of-00131.safetensors`），SGLang `hunyuan_v4.py` 创建并在最终 iHC head 后使用它。原账本遗漏此最终 RMSNorm，现补上 6,144 个逻辑参数及每 rank 12,288 字节 BF16 权重；未将索引目录核验表述为全部 tensor 形状/精度审计。78 层中 21 层 full indexer 执行选位，其余复用索引；所有层仍处理每个 query 的至多 2048 个候选。性能模型按一组 step 聚合查询与历史位置对数，full 层另计索引 Key（FP8 + scale）读写、投影与打分，所有层计压缩 KV 读写、稀疏 MLA、注意力门控；独立共享专家的计算亦单列。复用下文 TP/EP 平均负载通信公式，不含 MTP、算子融合、top-k 排序和 iHC 的逐算子成本，也未做形状/硬件校准。结果用于趋势对比，不能视为 SGLang 真实延迟。

## 平均负载 MoE/EP 解析计时（未经校准）

`TP=EP=GPU 数` 且 BF16/FP16 时，用本步新 Prefill token 数加 Decode 请求数作为同一批 MoE 输入；每层保留注意力输出的 TP 同步，Dense 层继续计 FFN 及第二次 TP 同步，MoE 层则改计 gate、均匀放置的本地专家计算、dispatch、combine，**不重复计 Dense FFN 与第二次 TP 同步**。专家计算每 rank 的平均任务数为 `tokens * experts_per_token / ep_size`，权重 HBM 按本 rank 全部驻留专家保守计入；不存在逐 token 路由或热点专家的尾延迟预测。`MIXED` 一次合并批次收费，只恢复准入的请求不计专家计算/通信；KV 恢复不产生纯恢复 GPU 步，L2→L1 回载按逐层流水调整步长。

设 `E=num_experts`、`R=ep_size`、`K=experts_per_token`、`H=hidden_size`、`T=本步 token 数`，均匀无放回路由下某目标 rank 至少被选一次的概率为 `p=1-∏_{i=0}^{K-1}(E-E/R-i)/(E-i)`。每 rank 的平均远端专家选择数为 `M=T*K*(R-1)/R²`；`normal` 每 rank 平均远端向量数为 `T*(R-1)/R*p`，`low_latency` 按 `M` 个选择计。激活字节宽度与本版计算精度一致（BF16/FP16：2 bytes/元素）；dispatch 与 combine 每次远端选择分别额外计 8 bytes 逻辑元数据（ID/索引 4 bytes、权重 4 bytes）。这是未校准的**逻辑流量**，不等同于 DeepEP 的实际链路流量或内核布局。

含 Prefill 的步采用 `normal`，纯 Decode 采用 `low_latency`；两模式共用单 rank `effective_interconnect_gb_s` 和 `ep_communication_latency_ms`，dispatch/combine 各按 `固定开销 + 每 rank 逻辑字节/有效带宽` 计一次、各 MoE 层顺序加入步关键路径。不做通信与计算重叠、专家复制或 EPLB。TP 的 `collective_latency_ms` 独立计费；分析结果的 `communication_ms` 为 TP+EP 时间之和，`ep_dispatch_ms`、`ep_combine_ms`、`tp_communication_ms` 是其中分项，`collectives` 仅计 TP，`ep_exchanges` 单列。`device_ms` 已包含全部 TP/EP 通信，不得再把通信分项加到时长上；`restore_ms` 单列 KV 传输服务时长之和，不代表 makespan 增量；回载拉长步长的部分见 `restore_exposed_ms`。页面试算使用单 rank 有效互联 600 GB/s、TP 集合通信固定延迟 0.01 ms/次、EP dispatch/combine 固定开销 0.02 ms/次，均未经校准，可手动修改；1800 GB/s 仅为 B300 的单卡双向标称值，不直接代入有效带宽。精度与进一步校准见 [F002 计划](../plans/perf-model.md) 和 [F004 计划](../plans/expert-parallelism.md)。
