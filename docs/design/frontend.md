# 前端设计

## 页面结构

单页应用，提供普通仿真配置、独立参数组合扫描、后台任务列表，以及单场指标/甘特图和多扫描组合折线图。

视觉与布局风格参考 [refs/frontend-style/](../../refs/frontend-style/)（深色主题 CSS 变量、卡片式布局、表单控件与结果网格样式）；仅借鉴风格，不照搬其多 Tab 结构——MVP 只有一个页面、一个视图。

## 输入：配置表单

表单只暴露当前可编辑字段；其余 Config schema 字段由前端默认值填充或由服务端解析，不与表单控件一一对应：

- **负载（workload）**：合成/Trace 共用父层 `mode`、`arrival_mode`、`seed`、`session_rate` 和 `target_qps`；网页显式默认 `arrival_mode=open_loop`、`seed=0`，API 省略时默认闭环。`mode=synthetic` 的 `workload.synthetic` 只保存总请求数、根输入长度、子轮新增内容长度、输出长度和 session 配置；子轮输入为父轮输入 + 父轮实际输出 + 新增内容，理想命中率由完整页继承自然产生。合成开环必填 `workload.target_qps`（期望请求 QPS），禁止 `session_rate`，无就绪子轮时新建会话；闭环**不输入请求 QPS**，填写 `synthetic.session.num_sessions`、`workload.session_rate`（泊松根会话/秒）和 `synthetic.session.think_time_ms`（子轮独立指数思考间隔均值）。跨 session 不共享公共前缀。Trace 两种模式均填写父层会话速率，开环还可选填父层目标 QPS；`workload.trace` 只保存通过上传/扫描/下载计划或直接上传计划获得的可选临时 `plan_id`。生成上限不属于负载，位于推理配置区。
- **调度（scheduler）**：编辑 `max_running_requests`、`max_prefill_tokens`（默认 16384）和 `pd_mutually_exclusive`（默认 `true`，Prefill/Decode 分步；关闭后可在同一步融合）。调度参数独立于计时方式。
- **推理（inference）**：唯一物理配置提供 `max_output_tokens`（默认 65536，正整数），合成/Trace 共用这个引擎上限；每请求可见上限由客户端按实际输出向上取二次幂并封顶，超限请求失败、不截短实际输出。编辑 `page_size`（默认 64）、`model`、`hardware`、`storage`、`tp_size`、`ep_size` 与 `timing`（解析或经验四系数）；独立「共享 L3」区编辑顶层 `shared_l3` 的全场预算和查询延迟。容量由服务端推导；完整页 ID 判等及 L1 cache 路径始终开启，不提供关闭开关。

规则：

- 所有字段给出默认值，提交前校验范围与必填项；L1 扣除权重和预留后至少保留一页，L2/L3 预算由服务端按每组 KV 页字节向下取整；启用 L3 要求启用 L2，启用低层时须配置正的接入带宽且延迟非负。未启用的链路可保持空值，配置非法时表单提示或服务端返回 422。不提供 chunked prefill 选项。
- 手填表单值序列化为 config JSON，作为 `POST /api/simulate` 的请求体；字段类型来自 `openapi-typescript` 生成的 TS 类型，禁止手写。
- **自动配置区（Dense TP／平均负载 MoE EP）**：编辑模型结构、权重/KV 精度、单卡显存、有效算力与带宽，以及三级物理容量；不输入可由结构估算的总参数、激活参数、专家参数或实际驻留字节，服务端在预览中只读展示估算值。TP 必须等于本实例 GPU 数；MoE 还须 EP=TP、填写专家规格。`/api/resolve` 按逐 rank 权重（含全部驻留专家）和 KV 展示容量；缺通信参数时仍可预览，但统一运行按钮禁用。解析计时在通信参数齐备时运行 Dense TP 或 MoE EP，经验四系数不支持多卡 TP/EP。B300 的 1800 GB/s 是单卡双向标称值，不代入有效带宽；界面默认 600 GB/s 单 rank 有效互联、0.01 ms/次 TP 同步和 0.02 ms/次 EP dispatch/combine，均是可修改的未经校准试算值。TP、EP 与 GPU 卡数仍分别输入。物理配置运行提交 `SimConfig` 到 `/api/simulate`；`/api/resolve` 仅接收同一配置并预览 `ResolutionReport`，运行不复用预览中的 token 容量。导入导出只接受统一 `SimConfig` JSON，保留完整物理规格；Trace 导出将临时 `plan_id` 清为 `null` 并保留回放设置，导入后提示重新上传独立计划资源。后台运行期间普通表单仍可编辑及提交新任务；前端不计算容量和性能公式。

## 参数组合扫描

扫描区独立选择合成/Trace 负载及开环/闭环到达模式；Trace 直接上传回放计划 JSON。参数目录按当前模式筛选，输入中文名或字段路径搜索横轴、颜色、形状和线型参数，每个已选择维度设置离散档位或范围；GPU/TP/必要时 EP 使用一个共同参数。普通仿真和扫描不互相禁用输入。完成的扫描可跨批次勾选组合，重新分配维度、切换指标并导出内嵌脚本和点级数据的离线 HTML；导出页支持参数值筛选及 CSV 下载。

## 输出：指标与告警

- **指标卡片**：展示响应提供的时延、实际发送 QPS、完成吞吐、根请求占比、平均输入输出长度、构造/实际及子轮前缀命中、Prefill 工作量及 L1/L2/L3 分层命中、容量/驻留/驱逐/峰值、提升/降级与 restore 字节/时长；总命中不误标为 L1 命中。自动解析模式区分整组权重与逐卡 TP 分片、复制、EP 专家权重/KV 账单，展示总 FLOPs、HBM bytes、TP 同步和 EP dispatch/combine 时间与单 rank 逻辑流量；通信已含在设备时间内，恢复时间单列。全场展示共享 L3 的容量、驻留和实际峰值；逐实例隐藏其为 `null` 的池统计，但保留该实例 L3 命中量。L1/L2 多实例峰值标为各实例峰值之和的上界，不套用到共享 L3。结果注明均匀路由、无重叠、未经校准；不称为 GPU 实测。合成开环 `workload.target_qps` 是期望请求速率，闭环 `workload.session_rate` 是期望新会话速率；实际请求 QPS 与完成吞吐分别展示，不混为目标。字段及计算口径见 [metrics.md](metrics.md)。
- **状态与原因**：展示 `meta.num_completed/num_aborted/num_skipped`；分别列出 `aborted_requests`、`skipped_requests` 的请求 ID 及原因。运行中的请求级容量失败属于 200 结果，不作为整场 HTTP 422；开环模式不会预建未来子轮，失败会话不再续写，下一节拍可以创建新根；显式闭环模式中预建的后代才会标记跳过。
- **告警列表**：`warnings` 非空时逐条展示，为空不渲染；全失败时空指标按响应空值展示，不伪造时延或吞吐。
- **甘特图**：只使用 `requests` 中的已完成区间；若没有成功请求，显示无法绘制的空态而不绘制失败/跳过记录。见下节。

## 唯一视图：请求甘特图

单场仿真仍使用请求生命周期甘特图；参数组合扫描另用指标折线图，不把点级结果拼接为请求甘特图。

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
- 前端不自行计算仿真指标；`POST /api/simulate` 提交返回任务 ID，通过 SSE/断线轮询更新状态，完成后从 `GET /api/jobs/{id}/result` 读取 `SimResult`；扫描从 `GET /api/jobs/{id}/points` 读取各点指标（见 [api.md](api.md)）。
- ECharts 甘特图与扫描折线图按需加载；离线报告内嵌库源码，不依赖 CDN。

## 运行状态与验收

- 提交完成后表单保持可编辑，新任务不清除旧结果；任务列表可查看、取消、跨页面刷新重新获取同一服务进程中的状态。SSE 更新完成状态并获取完整结果；断线时低频轮询兜底。运行中取消会终止对应计算进程，任务失败时展示独立错误信息。
- `422` 的配置校验字段错误逐项展示；运行时不可解析配置及不支持的计时组合在任务状态中展示，已发送请求的容量失败保留在 200 结果的失败列表中。
- Vitest 覆盖配置、客户端、指标、告警与甘特图数据和组件；Playwright 使用真实 FastAPI 验收 1,000 请求 Canvas 渲染，并检查 390px 视口无横向溢出。
