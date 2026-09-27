# 前端设计

## 页面结构

单页应用，自上而下三个区域：**输入（配置表单）→ 运行入口 → 输出（指标 + 甘特图）**。

视觉与布局风格参考 [refs/frontend-style/](../../refs/frontend-style/)（深色主题 CSS 变量、卡片式布局、表单控件与结果网格样式）；仅借鉴风格，不照搬其多 Tab 结构——MVP 只有一个页面、一个视图。

## 输入：配置表单

表单字段与 [api.md](api.md) 的 Config schema 一一对应，分组展示：

- **负载（workload）**：`num_requests`、`arrival_rate`、`prompt_len` / `output_len`（LengthDist：fixed / poisson / lognormal 及其参数）、`seed`。`mode` 固定 synthetic，trace 分支不在前端暴露。
- **引擎（engine）**：`max_batch_size`、`kv_token_budget`。
- **性能模型（perf_model）**：`prefill_c1` / `prefill_c2`、`decode_d1` / `decode_d2`。

规则：

- 所有字段给出默认值，提交前由前端做基础校验（范围、必填），最终以服务端 Pydantic 校验为准。
- 表单值序列化为 config JSON，作为 `POST /api/simulate` 的请求体；字段类型来自 `openapi-typescript` 生成的 TS 类型，禁止手写。

## 输出：指标与告警

- **指标卡片**：TTFT（均值 / P50 / P99）、TPOT、端到端吞吐、makespan，以结果网格（参考样式的 `result-grid`）展示。
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
