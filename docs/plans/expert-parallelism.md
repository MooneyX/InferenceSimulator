# 单实例 TP + EP 实施计划（F004）

**状态：单卡 Dense/GQA、同构 Dense TP 及平均负载 MoE/EP 的容量与逐步解析计时已接入；真实部署与性能校准待核验。** 按“模型与硬件参数配置 → TP → EP”三阶段交付。Qwen3-8B、Qwen3-235B-A22B 与 B300 仅为参考结构和标称规格；预置未附目标设备的有效性能或真实驻留量，不等于已验证的部署。其余证据与校准见 [容量计划](model-hardware-kv-capacity.md) 和 [F002 性能计划](perf-model.md)。

## 已确定的范围与拓扑

- 最终目标是单实例、单机同构 GPU、`tp_size = ep_size`、`attn_dp_size = 1`；以 `TP8 + EP8`、8 张 B300 作为待核验场景，首个 EP 通信后端为 DeepEP。模型结构、实际权重和 KV 布局未核实时，不标为可运行或已校准。
- 保持一个 `Engine`、一条实例内 FCFS 请求队列、一个全局仿真时钟和现有请求生命周期。参与的 GPU 协作执行同一个步骤，不为每个 GPU 建立独立请求队列或 `Engine`。
- TP 表达注意力及其他非专家部分的逐 rank 权重、KV 放置和计算/通信；EP 在同一组 rank 上表达各 MoE 层的专家放置、token 分发、专家计算及结果汇集。共享权重与专家权重均计入逐 rank 显存账单。
- 暂不实现 DPA（`attn_dp_size > 1`）、多实例、跨实例路由或 PD 分离。统一 `SimConfig.inference.mode=manual` 保留显式 token 容量及四系数计时，迁移后的等价输入与种子保持原行为；不保留旧外部配置结构。

## 分阶段交付

1. **模型与硬件参数配置（单 GPU 基线）**：前端增加自定义模型、单卡硬件、精度、权重驻留、显存预留及计时模式的输入入口，展示来源与解析结果；预置与自定义输入走同一服务端解析过程。服务端在负载规划和 `Engine` 创建前，将经验证的模型 KV 页字节成本与单卡显存预算解析为整页对齐的唯一内部运行容量；内核仍以 token 和唯一 `inference.page_size` 管理逻辑 KV 页，手动分支使用显式 `kv` 容量和四系数，不保留旧外部配置。解析计时模式预先准备模型/硬件静态参数，在仿真每一步按实际 Prefill/Decode 工作量用 Roofline 估算计算与访存成本，并按执行依赖形成步长；不把设备峰值直接换算成四个经验系数。先以可核实的单 GPU、Dense 全注意力 GQA/MQA 模型验收容量和步成本；此阶段不宣称大型 MoE 已可运行。
2. **TP（Dense 解析模式已接入）**：`SimConfig.inference.tp_size` 须等于实例 `inference.hardware.gpu_count`，首期仅对 Dense 全注意力 BF16/FP16 计算逐 rank 账单。由词表、嵌入共享标记与矩阵维度推导可分片权重；归一化按每卡复制计入；总参数与权重占用改为由结构估算，不再录入名义总量的未归属差额。Query 头与 FFN 维度须能均分；KV 头数足够时均分，不足时按头整倍复制。L1 输入为每卡含权重预算，按最紧 rank 可容纳整页数给出逻辑上限；L2/L3 输入是整组物理预算，按各 rank 的 KV 页合计字节数取整。`TP>1` 在解析计时模式可运行：按本地 FLOPs/HBM 计算 Roofline，每个 Dense 层的注意力输出和 FFN 输出各收一次等效 Ring 集合通信费用，不考虑计算/通信重叠；四系数经验模式仅可预览容量，不可运行。MoE 的全部专家驻留及 EP 成本已在第三阶段按平均负载近似接入，但不是经过验证的真实部署。
3. **EP（平均负载版已接入）**：同一组 rank 均分各 MoE 层的全部驻留专家权重，逐 rank 账单同时计 TP/EP 占用并解析容量；每步按实际新 Prefill/Decode token 数估计 gate、专家计算及 DeepEP 风格的 dispatch/combine 逻辑流量。含 Prefill 的步按 `normal` 不同目标 rank 去重，纯 Decode 按 `low_latency` 有效专家选择数计；两模式共用互联带宽和独立于 TP 的 EP 单次通信延迟。只支持 BF16/FP16、无专家复制、无逐 token 路由、无通信/计算重叠；逐 rank 平均负载不预测热点专家或尾延迟。`TP8 + EP8 + DPA1` 是待核验场景，不宣称已校准。

## 编码接入点

1. `core/src/simcore/schemas.py`：唯一 `SimConfig` 由实例数、负载、调度和 `inference` 判别分支组成；手动分支传 KV/经验计时，物理分支传模型、硬件、存储、TP/EP 和计时规格，不保留旧外部配置结构。
2. `server/src/simserver/routes.py`、`core/src/simcore/runner.py`：预览和运行都接收同一 `SimConfig`；预览返回 `ResolutionReport`，唯一仿真入口返回含可空 `resolution` / `analysis` 的 `SimResult`。内核保持零 IO，在 `build_plan` 和 `Engine(runtime, ...)` 创建前通过 `capacity.prepare_runtime` 一次性确定内部运行容量；`meta.config` 回显原规范化输入而非推导配置；`core/src/simcore/coordinator.py` 保持时钟与事件所有权。
3. `core/src/simcore/scheduler.py`、`core/src/simcore/kv_cache.py`：保留逻辑页、前缀复用、准入与完成语义，不按 GPU 数直接扩大 token 池。启用 L2/L3 时，恢复传输的 KV 字节口径须与解析容量一致。
4. `core/src/simcore/engine.py`、`core/src/simcore/perf_model.py`：复用 `StepWorkload` 的实际 Prefill/Decode 请求与上下文作为解析成本输入；将静态模型/设备参数与动态步工作量分开。TP/EP 共享逐 rank 成本接口，EP 逐层计费；不得在包含完整步成本的四系数结果上重复叠加通信费。
5. 对外契约实现时同步 `web/openapi.json`、`web/src/api/schema.d.ts`、表单、配置 JSON 导入导出及结果展示；前端不另算容量或性能公式，错误按缺失规格、显存不足和不支持的布局区分。

## 待冻结与验收

- **第一阶段**：固定页成本适用范围、模型/设备规格及来源、有效算力/带宽的参数来源、Roofline 算子分项及融合步依赖、预置/自定义输入和结果契约；用手算容量与单步 Prefill/Decode 案例验证，展示未经校准的估计性质。未知 KV 布局或缺失权重预算不得返回精确可运行容量。
- **第二阶段后续校准**：当前 Dense TP 使用同构 rank、均匀工作量、Ring 两次等效同步/层和零计算通信重叠的近似模型；按目标 SGLang 执行路径校准通信模式、集体操作延迟与实际带宽，并补词表投影及形状相关算子开销。MoE 的专家放置和成本归第三阶段。
- **第三阶段**：核对 MoE 层及全部驻留专家权重、DeepEP 通信量与阶段依赖，验证同种子结果确定性、专家偏斜、逐 rank 显存和完成/失败/跳过计数。具体校准误差标准和目标模型部署可用性另按证据冻结。
