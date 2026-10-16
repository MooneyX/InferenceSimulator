# 模型与硬件预置、KV 容量实施计划（F002 子计划）

**状态：GQA/MQA、按层数填写的 GQA/MLA/MLA+DSA 组合及 BF16 HY4-Preview（SGLang 抽象口径）支持条件化三级容量与未校准解析试算；顺序敏感布局、实测驻留权重及性能校准待实现。资料核查基准日：2026-10-12。** 本计划记录 [F002 性能模型](perf-model.md) 的后续模型/设备预置及显存容量工作；当前已实现行为以 [API](../design/api.md)、[L1 缓存](../design/kv-prefix-reuse.md) 为准。

## 1. 目标与边界

- 让用户选择指定版本的模型、具体硬件形态与部署精度，查看可追溯的显存账单、KV 字节/页、可容纳页数及最终生效的 L1 token 容量；可编辑参数并区分官方规格、部署假设和用户输入。
- **预置可浏览 ≠ 可计算 ≠ 可在当前仿真器中运行。** 每个组合分别标记 `catalog_only`（仅目录）、`capacity_ready`（KV/权重/单卡显存口径齐备且逐项标注证据或用户假设）、`runnable`（容量可解析且满足当前单卡拓扑，可提交仿真）；缺任一关键值时不返回虚假的精确容量，也不静默用别的型号代替。已计算容量仍只是估算，不代表真实硬件适配、运行时性能或实际服务可用。
- 首期落实单 GPU、单 Engine、文本 **Dense 全注意力 GQA/MQA** KV 容量及权重预算，并作为 [F004 第一阶段](expert-parallelism.md) 的规格/容量基线；普通 MoE 即使 KV 是标准 GQA，也须在补齐**总驻留权重**和运行时预算后作为后续容量扩展，不使用每 token 激活参数量替代。非 NVIDIA 硬件可作为**容量预算输入**，不等于已有对应运行时或校准模型。物理输入仍可选择原四系数经验计时；解析计时使用 [F002 Roofline 计划](perf-model.md) 根据有效硬件参数在仿真每步计算 Prefill/Decode 成本，不从卡名或峰值 FLOPS 直接推导四个经验系数。多卡 TP/EP 的逐 rank 容量由 F004 后续阶段交付。
- 自定义 MLA/DSA 组合只按用户显式填写的固定缓存布局作条件化粗估；要将 DeepSeek、Hy4 等具体大模型标记为经过核验的容量仍须有**针对对应版本和推理实现的缓存布局公式/测试**，不能把 `num_key_value_heads` 套进普通 GQA 公式。对 KV 字节/token 随上下文、层、实现变化的模型，不能压成单一固定 token 容量喂给现有 L1 页池；需要独立的可变页成本设计，未完成前只展示目录和不可计算原因。
- 超出单卡的模型显示“当前单卡模式放不下”，不将多卡显存求和假装为单实例容量，也不暗示 TP/PP、权重卸载、KV offload、多模态视觉缓存已实现。

## 2. 首批预置目录与证据状态

预置记录稳定 ID、显示名、版本/修订号、来源 URL 与核查日期；“最新”只用于**目录更新规则**，运行或导出必须锁定具体版本和规格快照，不随上游发布悄然变化。下表为**候选目录**，不是已验证可运行矩阵。

| 模型预置 | 可核验配置/结构 | 首期容量状态 |
|---|---|---|
| DeepSeek-V4.1-Flash（截至基准日的最新官方发布） | [官方新闻索引](https://www.deepseek.com/en/news/)；[模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json) 显示跨层来源、分层压缩/滑动窗口及索引结构 | `catalog_only`：不可按普通 GQA 或简化 MLA 推断真实 KV；等待部署实现与逐层缓存口径验证。 |
| Tencent Hy4 Preview | [固定修订](https://huggingface.co/tencent/Hy4-preview/tree/705d81ee51566a186d645b74c974d642ef2828fe) 含 MLA、DSA、MoE；SGLang 默认 TP/EP 与 index-K 分配作为模型假设 | BF16 主干条件化试算：TP/EP 逐卡结构权重估算、固定 MLA + DSA 页成本与未校准性能粗估；不计 MTP，权重未逐 tensor 校对 safetensors header，运行时显存与后端性能不保证。 |
| Qwen3.5-397B-A17B | [官方配置](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json) 含线性注意力与全注意力混合、MoE | `catalog_only`：线性状态不能套全注意力 KV 页公式。 |
| GLM-5 | [模型配置](https://huggingface.co/zai-org/GLM-5/raw/main/config.json) 含 DSA/低秩 KV 字段 | `catalog_only`：先核验推理缓存布局。 |
| Kimi-K2.5 | [模型配置](https://huggingface.co/moonshotai/Kimi-K2.5/raw/main/config.json) 含 MLA 字段 | `catalog_only`：先核验推理缓存布局。 |
| Qwen3-32B、Qwen2.5-72B-Instruct | [Qwen3-32B 配置](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)、[Qwen2.5-72B 配置](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/raw/main/config.json) 可作为全注意力 GQA 的对照 | 结构可核验；**须再补齐权重部署占用和设备预算**才能标为 `capacity_ready`，是否 `runnable` 由组合计算决定。 |

| 硬件预置 | 规格来源与待核查点 | 录入规则 |
|---|---|---|
| NVIDIA H20 | 明确板卡具体版本、单卡显存及可用显存，不能从同系列 H100 推导 | 未有可靠的对应板卡来源前，显存数值保持未知；允许用户提供显存并标记为用户假设。 |
| NVIDIA B300 | [NVIDIA DGX B300 指南](https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html) 列 **8 × 288 GB**；288 GB 为**每 GPU**，非 2.3 TB 整机 | 记录单卡 288 GB 的标称值，容量仅按单卡计算；扣留量另设。 |
| 昇腾 950PR、950DT | 处理器名不必然等于唯一板卡 SKU；显存容量依产品形态/修订而定 | 分别建立产品记录；取得可追溯的**单卡**规格前不填写硬编码数值，不把两款互换。 |
| 昆仑芯 P800 | 核对实际交付板卡、单卡显存与固件/部署口径 | 缺一手/可复核资料时仅目录展示或使用注明来源的用户输入；不从超节点总显存倒推。 |
| P900（厂商/形态待核验） | 现有检索同时将其用于“超节点”等不同描述，尚不能确认唯一单卡 SKU | 原样保留用户指定名称作为**待确认目录项**；不预填厂商/显存，不提供可计算单卡结果，核实后拆分单卡/超节点条目。 |

发布前对每条数值单独登记单位（GB 为十进制、GiB 为二进制）、来源、型号变体、版本及核查日；单纯搜索摘要或营销峰值不能作为生产级容量常量。目录更新要有回归样例和版本迁移说明；不支持的条目可以被选择查看，但运行按钮给出具体不可用原因。

## 3. 规格数据与校验

- **模型**：`architecture`/`kv_layout`、注意力层清单、层数、KV 头数、头维度、最大上下文、KV dtype、权重部署占用 bytes（包括量化元数据、embedding、专家和必要的辅助权重）；允许按权重文件清单或用户测量值录入。若仅有总参数量与精度，`params_total × bytes_per_weight + metadata` 标记为**粗估**，MoE 不使用激活参数替代驻留总量。多模态额外内存不明时禁止给出“完整模型可部署”的结论。
- **硬件**：厂商、芯片、板卡 SKU/修订、`memory_bytes_per_device`、来源和适用精度；不能把系统/机柜合计容量当单卡显存。有效算力、有效 HBM 带宽及其适用精度/形状作为第一阶段解析性能模式的独立输入，不影响容量计算；标称峰值不可冒充有效值。
- **部署**：单卡数固定为 1；权重格式、KV dtype、显存可用比例 `u ∈ (0,1]`、运行时/workspace/activation 峰值/其他预留 bytes、allocator 页元数据或碎片安全预留、可选的已校验用户 token 上限。预置规格与用户覆盖值分别保留，冲突时明确以覆盖值解析，标记“用户假设”；不允许空缺关键字段回退为零。
- 模型最大上下文是**单请求**限制；L1 `max_total_tokens` 是所有运行请求和可复用缓存页的**共享池**，不是单请求上下文长度。请求数、prompt/output 分布、前缀命中和页碎片影响可运行并发，不能从总池大小直接推导保证并发数。

## 4. 单卡容量解析与公式

仅当已验证为**各层全注意力且等宽 KV**的 Dense/GQA/MQA 文本配置时使用：

```text
kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_bytes_per_element
kv_bytes_per_page  = kv_bytes_per_token * inference.page_size
usable_bytes       = floor(memory_bytes_per_device * utilization)
kv_budget_bytes    = usable_bytes - weights_resident_bytes
                   - runtime_reserve_bytes - workspace_reserve_bytes
                   - activation_peak_bytes - allocator_reserve_bytes - other_reserve_bytes
hardware_pages     = floor(kv_budget_bytes / kv_bytes_per_page)
hardware_tokens    = hardware_pages * inference.page_size
resolved_tokens    = hardware_tokens
```

所有中间计算以**整数 bytes**记录和核对；负余额、非正每页字节数、推导不足一页都报显式错误。当前无 `user_cap_tokens` 手填入口；按物理预算向下对齐整页，L1 不足一页时拒绝运行。只把唯一 `resolved_tokens` 写进内部 `RuntimeConfig.engine.max_total_tokens`，不改写外部 `SimConfig`，不再增加另一套生效容量或 paged-KV block size。每页碎片和前缀物理共享由现有 L1 模拟处理；显存预算只规定页池**上限**，不把逻辑上下文 token 数误认为物理驻留页数。与已实现的 [分层 KV 设计](../design/kv-prefix-reuse.md) 共用同一 KV bytes/token 口径；后续自动解析容量时不得独立填写与 L2/L3 传输参数矛盾的字节数。

验证例：Qwen3-32B 配置为 64 层、8 个 KV 头、头维 128，BF16 时 `2 × 64 × 8 × 128 × 2 = 262144` bytes/token，即 **256 KiB/token**；这是 KV 数据字节，不含权重、运行时或预留，不能由此单独宣布哪张卡可运行。对 MLA/DSA、压缩、混合线性注意力及滑窗先提供计算器分支的**专用设计与金标准测试**；在当前固定 token 页池不能表达真实存储时，拒绝自动容量解析，不能用标准公式得出看似精确的结论。

## 5. 服务、内核与页面契约

1. **服务端解析入口。** `GET /api/presets` 提供参考规格；后续目录状态、证据与来源扩展按本计划验收。`POST /api/resolve` 与运行接收同一个 `SimConfig`，模型/硬件/存储及页大小位于 `inference.mode=physical`，返回 `ResolutionReport`（容量账单、运行 token 容量、可运行性与原因），不返回另一套仿真输入。内核在 `Client` 与 `Engine(runtime, ...)` 创建之前调用一次 `capacity.prepare_runtime`；纯解析逻辑只消费显式规格，不读取服务端目录或执行 IO。
2. **仿真模式区分。** 唯一 `POST /api/simulate` 接收统一 `SimConfig`：物理输入根据模型/硬件/存储推导内部运行容量并使用选定计时后端，返回单个 `SimResult`。`meta.config` 回显包含字段默认值的原规范化输入，`resolution` 展示实际容量与账单，`analysis` 展示解析计时汇总；不能将临时预览报告重新当作运行配置，也不能由此推断规格已获真实部署认证。非法规格、不支持布局或显存不足返回具体 422；请求级 KV 失败仍归运行结果。
3. **前端配置与展示。** 在同一 `SimConfig` 表单编辑物理推理输入和独立调度、负载字段；展示服务端的显存账单、KV bytes/token、完整页数、最终运行容量与已实现的步骤成本分项。前端不维护另一套容量或 Roofline 公式；不可计算的配置给出原因并禁止提交。JSON 导入导出只保留统一输入规格与计时/回放设置，不嵌入解析报告；Trace 导出清空临时 `plan_id`，导入后重新上传独立计划资源。后续证据目录的版本/来源展示不得冒充已认证部署。
4. 改动对外 schema/路由后同步 `web/openapi.json` 和 `web/src/api/schema.d.ts`，更新 [服务计划](server-api.md)、[前端计划](frontend.md) 的交付与校验；只把真正落地行为迁入 `docs/design/`，不把目录选项当作已有仿真模式。

## 6. 交付顺序与验收

1. **证据目录**：按型号/修订建立版本、来源和单位；六项指定硬件与 DeepSeek-V4.1-Flash、Hy4 Preview 等候选均在目录中；未核实 P900/显存的状态和原因正确显示，禁止凭空填数。
2. **单卡 GQA 容量**：以 Qwen3-32B 等全注意力模型作手算金标准；验证十进制 GB/GiB、不同 KV/权重精度、权重预算、页边界、用户上限及显存不足；原显式配置数值与结果逐字段回归。
3. **集成与可复现**：目录版本固定、自定义规格与覆盖合并、服务端解析/运行同公式、服务错误、API 生成类型、结果中的有效配置和来源、前端导入导出及旧调用兼容；经验计时不因硬件名字本身变化（容量/调度变化可间接影响结果），解析计时仅用有来源的模型结构和有效设备参数，并按 [F002 计划](perf-model.md) 的逐步 Roofline 案例验收。
4. **高级结构与拓扑后续单列**：对 DeepSeek-V4.1-Flash、Hy4、GLM-5、Kimi-K2.5、Qwen3.5 的缓存布局分别核验实现/精度/不同长度后再解锁容量；MoE 多卡部署的逐 rank TP 放置、专家权重及容量约束按 [F004 单实例 TP + EP 计划](expert-parallelism.md) 分阶段验收，不以 `card_count × 单卡容量` 替代。带宽/算力计时校准也须单独验收。

**验收底线：**目录完整但不伪造未知规格；任何页面宣称的“可容纳 KV 数量”均能追溯到明确模型版本、单卡板卡版本、dtype、权重与预留假设、页大小以及同一份服务端解析结果。