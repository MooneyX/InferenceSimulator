# 参考资料索引

本文件是 `refs/` 目录的索引。新增参考资料时必须在此登记。

| 路径 | 内容 |
|---|---|
| [frontend-style/](frontend-style/) | 前端视觉与布局风格参考（旧版 KV Cache 建模平台前端） |
| [sglang/](sglang/) | SGLang 官方源码浅克隆；调研 Prefill-first、P/D 互斥、mixed chunk 和 Decode retraction |
| [vllm/](vllm/) | vLLM 官方源码浅克隆；对比 V1 running-first、mixed batching、chunked prefill 和 preemption |

## frontend-style/

| 文件 | 简介 |
|---|---|
| [index.html](frontend-style/index.html) | 页面结构参考：顶部栏 + 侧边导航 + 卡片式内容区；参数表单（`form-row` / `preset-tag`）、结果网格（`result-grid`）、图表容器、公式框等组织方式 |
| [styles.css](frontend-style/styles.css) | 样式参考：深色主题 CSS 变量（`--bg` / `--surface` / `--accent` 等）、卡片、表单控件、按钮、结果卡片、响应式断点 |

> 用途：新前端（React + TypeScript + Vite + ECharts，见 [../docs/design/frontend.md](../docs/design/frontend.md)）复用其**视觉风格与布局范式**，不照搬其多 Tab 结构与业务逻辑。
