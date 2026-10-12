# 前端设计

## 页面结构

单页应用，自上而下三个区域：**输入（配置表单）→ 运行入口 → 输出（指标 + 甘特图）**。

视觉与布局风格参考 [refs/frontend-style/](../../refs/frontend-style/)（深色主题 CSS 变量、卡片式布局、表单控件与结果网格样式）；仅借鉴风格，不照搬其多 Tab 结构——MVP 只有一个页面、一个视图。

## 输入：配置表单

表单只暴露当前可编辑字段；其余 Config schema 字段由前端默认值填充或由服务端解析，不与表单控件一一对应：

- **负载（workload）**：`mode=synthetic` 时编辑 `workload.synthetic`，可选 `arrival_mode=open_loop`（默认）或 `closed_loop`，共用总请求数、根输入长度、子轮新增内容长度、输出长度、生成上限和种子。子轮输入均为父轮输入 + 父轮已生成输出 + 新增内容，理想命中率由完整页继承自然产生。开环填写 `arrival_rate`（期望请求 QPS），无就绪子轮时新建会话；闭环**不输入请求 QPS**，填写 `session.num_sessions`、`session_rate`（泊松根会话/秒）和 `session.think_time_ms`（子轮独立指数思考间隔均值）。跨 session 不共享公共前缀。`mode=trace` 时编辑 `workload.trace` 的开环/闭环、会话速率、种子与可选目标 QPS，通过上传/扫描/下载计划或直接上传计划获得临时 `plan_id`。
- **调度（scheduler）**：编辑 `max_running_requests`、`max_prefill_tokens`（默认 16384）和 `pd_mutually_exclusive`（默认 `true`，Prefill/Decode 分步；关闭后可在同一步融合）。调度参数独立于推理配置方式。
- **推理（inference）**：`mode=manual` 时编辑 `page_size`（默认 64）、`kv` 中的 L1/L2/L3 token 容量与 KV 上行链路参数，以及 `timing.mode=empirical` 的四系数；`mode=physical` 时编辑同一个 `page_size`、`model`、`hardware`、`storage`、`tp_size`、`ep_size` 与 `timing`（解析或经验）。容量由服务端推导；完整页 ID 判等及 L1 cache 路径始终开启，不提供关闭开关。

规则：

- 所有字段给出默认值，提交前校验范围与必填项；L1 至少一页，L2/L3 均须为 `page_size` 的非负整数倍，启用 L3 要求启用 L2；启用低层时对应 KV 字节/token 和链路带宽须为正，延迟非负。未启用的链路可保持空值，配置非法时表单提示，服务端同样返回 422。不提供 chunked prefill 选项。
- 手填表单值序列化为 config JSON，作为 `POST /api/simulate` 的请求体；字段类型来自 `openapi-typescript` 生成的 TS 类型，禁止手写。
- **自动配置区（Dense TP／平均负载 MoE EP）**：编辑模型结构、权重/KV 精度、单卡显存、有效算力与带宽，以及三级物理容量；不输入可由结构估算的总参数、激活参数、专家参数或实际驻留字节，服务端在预览中只读展示估算值。TP 必须等于本实例 GPU 数；MoE 还须 EP=TP、填写专家规格。`/api/resolve` 按逐 rank 权重（含全部驻留专家）和 KV 展示容量；缺通信参数时仍可预览，但统一运行按钮禁用。解析计时在通信参数齐备时运行 Dense TP 或 MoE EP，经验四系数不支持多卡 TP/EP。B300 的 1800 GB/s 是单卡双向标称值，不代入有效带宽；界面默认 600 GB/s 单 rank 有效互联、0.01 ms/次 TP 同步和 0.02 ms/次 EP dispatch/combine，均是可修改的未经校准试算值。TP、EP 与 GPU 卡数仍分别输入。物理与手动运行均提交同一 `SimConfig` 到 `/api/simulate`；`/api/resolve` 仅接收同一配置并预览 `ResolutionReport`，运行不复用预览中的 token 容量。导入导出只接受统一 `SimConfig` JSON，保留完整物理规格；Trace 导出将临时 `plan_id` 清为 `null` 并保留回放设置，导入后提示重新上传独立计划资源。运行期间表单不可修改；前端不计算容量和性能公式。

## 输出：指标与告警

- **指标卡片**：展示响应提供的时延、实际发送 QPS、完成吞吐、根请求占比、平均输入输出长度、构造/实际及子轮前缀命中、Prefill 工作量及 L1/L2/L3 分层命中、容量/驻留/驱逐/峰值、提升/降级与 restore 字节/时长；总命中不误标为 L1 命中。自动解析模式区分整组权重与逐卡 TP 分片、复制、EP 专家权重/KV 账单，展示总 FLOPs、HBM bytes、TP 同步和 EP dispatch/combine 时间与单 rank 逻辑流量；通信已含在设备时间内，恢复时间单列。结果注明均匀路由、无重叠、未经校准；不称为 GPU 实测。开环 `arrival_rate` 是期望请求速率，闭环 `session_rate` 是期望新会话速率；实际请求 QPS 与完成吞吐分别展示，不混为目标。字段及计算口径见 [metrics.md](metrics.md)。
- **状态与原因**：展示 `meta.num_completed/num_aborted/num_skipped`；分别列出 `aborted_requests`、`skipped_requests` 的请求 ID 及原因。运行中的请求级容量失败属于 200 结果，不作为整场 HTTP 422；开环模式不会预建未来子轮，失败会话不再续写，下一节拍可以创建新根；显式闭环模式中预建的后代才会标记跳过。
- **告警列表**：`warnings` 非空时逐条展示，为空不渲染；全失败时空指标按响应空值展示，不伪造时延或吞吐。
- **甘特图**：只使用 `requests` 中的已完成区间；若没有成功请求，显示无法绘制的空态而不绘制失败/跳过记录。见下节。

## 唯一视图：请求甘特图

MVP 只有一个解释性视图——请求生命周期甘特图。

- 每请求一行，y 轴为请求 ID（按到达时间排序），x 轴为仿真时钟（ms）。
- 三段着色：
  - 排队（灰）：arrival → prefill_start
  - prefill（蓝）：prefill_start → prefill_end
  - decode（绿）：prefill_end → finish，合并为一个区间，不逐步渲染
- hover tooltip：TTFT、decode 时长、各阶段时间戳。
- 交互：缩放与平移。

## 约束

- 甘特图采用 canvas 渲染，区间交由 ECharts 管理，缩放平移由 dataZoom 承载。
- 技术栈：React + TypeScript + Vite + ECharts。
- 前端不自行计算任何指标，所有推理配置与负载模式均消费 `POST /api/simulate` 返回的单个 `SimResult`，从其可空 `resolution`、`analysis` 读取容量与解析明细（见 [api.md](api.md)）。
- ECharts 仅注册甘特图使用的模块；甘特图在仿真结果返回后异步加载，不阻塞配置页首屏。

## 运行状态与验收

- 请求运行期间禁用重复提交；新请求会清理旧结果与错误。
- `422` 的 schema 校验错误若包含字段详情则逐项展示；理想目标不可行等计划构造错误的字符串 `detail` 显示为普通错误。运行中的请求级 OOM 展示在 200 结果的失败列表中；`501`、网络错误和无效响应展示统一错误状态。
- Vitest 覆盖配置、客户端、指标、告警与甘特图数据和组件；Playwright 使用真实 FastAPI 验收 1,000 请求 Canvas 渲染，并检查 390px 视口无横向溢出。
