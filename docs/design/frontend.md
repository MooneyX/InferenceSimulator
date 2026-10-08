# 前端设计

## 页面结构

单页应用，自上而下三个区域：**输入（配置表单）→ 运行入口 → 输出（指标 + 甘特图）**。

视觉与布局风格参考 [refs/frontend-style/](../../refs/frontend-style/)（深色主题 CSS 变量、卡片式布局、表单控件与结果网格样式）；仅借鉴风格，不照搬其多 Tab 结构——MVP 只有一个页面、一个视图。

## 输入：配置表单

表单只暴露当前可编辑字段；其余 Config schema 字段由前端默认值填充或由服务端解析，不与表单控件一一对应：

- **负载（workload）**：编辑 `num_requests`（总轮次数）、`arrival_rate`（所有轮次的目标请求 QPS，非根 session/秒）、`prompt_len` / `output_len`（LengthDist：fixed / poisson / lognormal 及其参数）、`max_new_tokens`（每请求生成上限，默认 128）、`seed`，以及 `session.num_sessions`、`session.think_time_ms`（子轮间指数分布的平均思考时间，0 为无等待）、`session.target_ideal_prefix_hit_rate`。`session.topology` 固定为 `linear`；`session.content_seed` 与跨会话共享开关均已移除，表单及提交 JSON 不包含这些字段；跨 session 公共前缀不支持，同 session 父子轮次仍可继承完整块。所有负载均按 session 计划提交；根 session 速率自动换算为 `arrival_rate * num_sessions / num_requests`，有限场不保证实测到达率等于目标；`mode` 固定 synthetic，trace 分支不在前端暴露。
- **引擎（engine）**：编辑 `max_running_requests`、`max_total_tokens`、`max_prefill_tokens`（默认 16384）、`page_size`（默认 64）和 `pd_mutually_exclusive`（默认 `true`，Prefill/Decode 分步；关闭后可在同一步融合）；直接展示 KV token 容量及由 `max_total_tokens/page_size` 派生的 L1 页数。完整页 ID 判等及 L1 cache 路径始终开启，不提供关闭开关。
- **性能模型（perf_model）**：`prefill_c1` / `prefill_c2`、`decode_d1` / `decode_d2`。

规则：

- 所有字段给出默认值，提交前校验范围与必填项；`max_total_tokens` 必须至少一页且是 `page_size` 的正整数倍，非法时表单提示，服务端同样返回 422。不提供 chunked prefill 或 L2/L3 的表单选项。
- 表单值序列化为 config JSON，作为 `POST /api/simulate` 的请求体；字段类型来自 `openapi-typescript` 生成的 TS 类型，禁止手写。

## 输出：指标与告警

- **指标卡片**：展示响应提供的时延、实测吞吐、目标/构造/实际前缀命中、Prefill 工作量与 L1 容量/驱逐/驻留峰值；`arrival_rate` 是输入的目标到达 QPS，不把完成吞吐误标为目标达成的到达率。字段及计算口径见 [metrics.md](metrics.md)。
- **状态与原因**：展示 `meta.num_completed/num_aborted/num_skipped`；分别列出 `aborted_requests`、`skipped_requests` 的请求 ID 及原因。运行中的请求级容量失败属于 200 结果，不作为整场 HTTP 422；父轮失败导致未发送后代跳过。
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
- 前端不自行计算任何指标，全部消费 `POST /api/simulate` 响应（见 [api.md](api.md)）。
- ECharts 仅注册甘特图使用的模块；甘特图在仿真结果返回后异步加载，不阻塞配置页首屏。

## 运行状态与验收

- 请求运行期间禁用重复提交；新请求会清理旧结果与错误。
- `422` 的 schema 校验错误若包含字段详情则逐项展示；理想目标不可行等计划构造错误的字符串 `detail` 显示为普通错误。运行中的请求级 OOM 展示在 200 结果的失败列表中；`501`、网络错误和无效响应展示统一错误状态。
- Vitest 覆盖配置、客户端、指标、告警与甘特图数据和组件；Playwright 使用真实 FastAPI 验收 1,000 请求 Canvas 渲染，并检查 390px 视口无横向溢出。
