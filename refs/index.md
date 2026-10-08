# 参考资料索引

本文件是 `refs/` 目录的索引。新增参考资料时必须在此登记。

| 路径 | 内容 |
|---|---|
| [KVCacheModeling/](KVCacheModeling/) | KV Cache 建模平台源码参考（Git 子模块）；包括参数导入导出的前端交互、页面布局和样式 |
| [sglang/](sglang/) | SGLang 官方源码浅克隆；调研 Prefill-first、P/D 互斥、mixed chunk 和 Decode retraction |
| [vllm/](vllm/) | vLLM 官方源码浅克隆；对比 V1 running-first、mixed batching、chunked prefill 和 preemption |

## KVCacheModeling/

| 文件 | 简介 |
|---|---|
| [index.html](KVCacheModeling/index.html) | 页面结构与参数导入导出入口参考 |
| [src/ui/parameters.js](KVCacheModeling/src/ui/parameters.js) | 参数 JSON 收集、复制与导入交互参考 |
| [src/ui/styles.css](KVCacheModeling/src/ui/styles.css) | 深色主题、卡片、表单控件、按钮和响应式样式参考 |

> 用途：新前端（React + TypeScript + Vite + ECharts，见 [../docs/design/frontend.md](../docs/design/frontend.md)）借鉴其交互与布局范式，不照搬其业务逻辑。
