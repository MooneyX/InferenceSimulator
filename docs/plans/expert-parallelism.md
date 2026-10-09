# 单实例 TP + EP 实施计划（F004）

**状态：单卡 Dense/GQA 基线和同构 Dense TP 逐步解析计时已接入；MoE/EP 待实现。** 按“模型与硬件参数配置 → TP → EP”三阶段交付，各阶段独立验收。第一阶段仅提供自定义规格、Qwen3-8B 与 Qwen3-235B-A22B 参考结构、B300 标称显存、三级容量解析及未经校准的单卡 Roofline；预置未附有效性能或真实驻留量，也不等于已验证的真实部署。其余预置证据与校准仍见 [容量计划](model-hardware-kv-capacity.md) 和 [F002 性能计划](perf-model.md)。

## 已确定的范围与拓扑

- 最终目标是单实例、单机同构 GPU、`tp_size = ep_size`、`attn_dp_size = 1`；以 `TP8 + EP8`、8 张 B300 作为待核验场景，首个 EP 通信后端为 DeepEP。模型结构、实际权重和 KV 布局未核实时，不标为可运行或已校准。
- 保持一个 `Engine`、一条实例内 FCFS 请求队列、一个全局仿真时钟和现有请求生命周期。参与的 GPU 协作执行同一个步骤，不为每个 GPU 建立独立请求队列或 `Engine`。
- TP 表达注意力及其他非专家部分的逐 rank 权重、KV 放置和计算/通信；EP 在同一组 rank 上表达各 MoE 层的专家放置、token 分发、专家计算及结果汇集。共享权重与专家权重均计入逐 rank 显存账单。
- 暂不实现 DPA（`attn_dp_size > 1`）、多实例、跨实例路由或 PD 分离。旧配置保留显式 token 容量及四系数计时，输入与种子相同时保持原行为。

## 分阶段交付

1. **模型与硬件参数配置（单 GPU 基线）**：前端增加自定义模型、单卡硬件、精度、权重驻留、显存预留及计时模式的输入入口，展示来源与解析结果；预置与自定义输入走同一服务端解析过程。服务端在负载规划和 `Engine` 创建前，将经验证的模型 KV 页字节成本与单卡显存预算解析为整页对齐的唯一 `engine.max_total_tokens`；内核仍以 token 和现有 `page_size` 管理逻辑 KV 页，旧手填模式保留。解析计时模式预先准备模型/硬件静态参数，在仿真每一步按实际 Prefill/Decode 工作量用 Roofline 估算计算与访存成本，并按执行依赖形成步长；不把设备峰值直接换算成四个经验系数。先以可核实的单 GPU、Dense 全注意力 GQA/MQA 模型验收容量和步成本；此阶段不宣称大型 MoE 已可运行。
2. **TP（Dense 解析模式已接入）**：`ResolvedRequest.tp_size` 须等于实例 `hardware.gpu_count`，首期仅对 Dense 全注意力 BF16/FP16 计算逐 rank 账单。由词表、嵌入共享标记与矩阵维度推导可分片权重；归一化及参数总量未归属差额按每卡复制保守计入。Query 头与 FFN 维度须能均分；KV 头数足够时均分，不足时按头整倍复制。L1 输入为每卡含权重预算，按最紧 rank 可容纳整页数给出逻辑上限；L2/L3 输入是整组物理预算，按各 rank 的 KV 页合计字节数取整。`TP>1` 在解析计时模式可运行：按本地 FLOPs/HBM 计算 Roofline，每个 Dense 层的注意力输出和 FFN 输出各收一次等效 Ring 集合通信费用，不考虑计算/通信重叠；四系数经验模式仅可预览容量，不可运行。专家总权重尚未放置时，MoE 预置仍不可预览为完整部署。
3. **EP**：在同一组 rank 上加入各 MoE 层的全部驻留专家权重与专家放置、token 路由及 rank 负载；以 DeepEP 为首个 dispatch/combine 通信后端，将专家计算与通信纳入每步关键路径。逐 rank 显存账单须同时包含 TP 与 EP 占用，再解析容量；完成后才验证 `TP8 + EP8 + DPA1` 的目标模型组合。

## 编码接入点

1. `core/src/simcore/schemas.py`：区分手填/解析计时模式及显式模型、硬件、部署规格；字段、单位及可运行组合按阶段冻结。旧 `SimConfig` 不提供新字段时继续原路径。
2. `server/src/simserver/routes.py`、`core/src/simcore/runner.py`：服务端解析输入并返回显存账单、有效配置及不可用原因；内核保持零 IO，消费已解析配置。必须在 `build_plan` 和单个 `Engine` 创建前确定唯一的 `engine.max_total_tokens`；`core/src/simcore/coordinator.py` 保持单实例时钟与事件所有权。
3. `core/src/simcore/scheduler.py`、`core/src/simcore/kv_cache.py`：保留逻辑页、前缀复用、准入与完成语义，不按 GPU 数直接扩大 token 池。启用 L2/L3 时，恢复传输的 KV 字节口径须与解析容量一致。
4. `core/src/simcore/engine.py`、`core/src/simcore/perf_model.py`：复用 `StepWorkload` 的实际 Prefill/Decode 请求与上下文作为解析成本输入；将静态模型/设备参数与动态步工作量分开。TP/EP 共享逐 rank 成本接口，EP 逐层计费；不得在包含完整步成本的四系数结果上重复叠加通信费。
5. 对外契约实现时同步 `web/openapi.json`、`web/src/api/schema.d.ts`、表单、配置 JSON 导入导出及结果展示；前端不另算容量或性能公式，错误按缺失规格、显存不足和不支持的布局区分。

## 待冻结与验收

- **第一阶段**：固定页成本适用范围、模型/设备规格及来源、有效算力/带宽的参数来源、Roofline 算子分项及融合步依赖、预置/自定义输入和结果契约；用手算容量与单步 Prefill/Decode 案例验证，展示未经校准的估计性质。未知 KV 布局或缺失权重预算不得返回精确可运行容量。
- **第二阶段后续校准**：当前 Dense TP 使用同构 rank、均匀工作量、Ring 两次等效同步/层和零计算通信重叠的近似模型；按目标 SGLang 执行路径校准通信模式、集体操作延迟与实际带宽，并补词表投影及形状相关算子开销。MoE 的专家放置和成本归第三阶段。
- **第三阶段**：核对 MoE 层及全部驻留专家权重、DeepEP 通信量与阶段依赖，验证同种子结果确定性、专家偏斜、逐 rank 显存和完成/失败/跳过计数。具体校准误差标准和目标模型部署可用性另按证据冻结。
