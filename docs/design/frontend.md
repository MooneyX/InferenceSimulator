# 前端设计

## 页面结构

单页应用，自上而下三个区域：**输入（配置表单）→ 运行入口 → 输出（指标 + 甘特图）**。

视觉与布局风格参考 [refs/frontend-style/](../../refs/frontend-style/)（深色主题 CSS 变量、卡片式布局、表单控件与结果网格样式）；仅借鉴风格，不照搬其多 Tab 结构——MVP 只有一个页面、一个视图。

## 输入：配置表单

表单只暴露当前可编辑字段；其余 Config schema 字段由前端默认值填充或由服务端解析，不与表单控件一一对应：

- **负载（workload）**：编辑 `num_requests`（总轮次数）、`arrival_rate`（所有轮次的目标请求 QPS，非根 session/秒）、`prompt_len` / `output_len`（LengthDist：fixed / poisson / lognormal 及其参数）、`seed`，以及 `session.num_sessions`、`session.think_time_ms`、`session.target_ideal_prefix_hit_rate`。`session.topology` 固定为 `linear`；`session.content_seed` 已移除，跨会话共享开关固定 `false` 且不展示。所有负载均按 session 计划提交；根 session 速率自动换算为 `arrival_rate * num_sessions / num_requests`，有限场不保证实测到达率等于目标；`mode` 固定 synthetic，trace 分支不在前端暴露。
- **引擎（engine）**：编辑 `max_batch_size`、`kv_token_budget` 和 `prefix_cache.block_size_tokens`；L1 容量由预算和块大小换算。`cache_salt` 及模型维度保持默认且不进入合成块 ID，L2/L3 不在当前表单或引擎中启用。完整块 ID 判等及 L1 cache 路径始终开启，不提供关闭开关。
- **性能模型（perf_model）**：`prefill_c1` / `prefill_c2`、`decode_d1` / `decode_d2`。

规则：

- 所有字段给出默认值，提交前由前端做基础校验（范围、必填），最终以服务端 Pydantic 校验为准。
- 表单值序列化为 config JSON，作为 `POST /api/simulate` 的请求体；字段类型来自 `openapi-typescript` 生成的 TS 类型，禁止手写。

## 输出：指标与告警

- **指标卡片**：TTFT（均值 / P50 / P99）、TPOT、实测端到端完成吞吐、makespan，以及目标/构造/实际前缀命中率、Prefill 工作量与 L1 容量/驱逐/驻留峰值，以结果网格展示；`arrival_rate` 是输入的目标到达 QPS，不把完成吞吐误标为目标达成的到达率。
- **告警列表**：`warnings` 逐条展示（如 KV 超限事件），为空则不渲染。
- **甘特图**：见下节。

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

- 甘特图采用 canvas 渲染，仅渲染可视区域附近的区间，缩放平移由 dataZoom 承载。
- 技术栈：React + TypeScript + Vite + ECharts。
- 前端不自行计算任何指标，全部消费 `POST /api/simulate` 响应（见 [api.md](api.md)）。
- ECharts 仅注册甘特图使用的模块；甘特图在仿真结果返回后异步加载，不阻塞配置页首屏。

## 运行状态与验收

- 请求运行期间禁用重复提交；新请求会清理旧结果与错误。
- `422` 展示字段级校验详情，`501`、网络错误和无效响应展示统一错误状态。
- Vitest 覆盖配置、客户端、指标、告警与甘特图数据和组件；Playwright 使用真实 FastAPI 验收 1,000 请求 Canvas 渲染，并检查 390px 视口无横向溢出。
