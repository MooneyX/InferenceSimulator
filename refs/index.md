# 参考资料索引

本文件是 `refs/` 目录的索引。新增参考资料时必须在此登记。

| 路径 | 内容 |
|---|---|
| `cc-traces/` | AgentX 请求轨迹数据集（本地下载，待接入 replay）；包含 JSONL 轨迹和可视化报告，不随仓库分发 |
| [KVCacheModeling/](KVCacheModeling/) | KV Cache 建模平台源码参考（Git 子模块）；包括参数导入导出的前端交互、页面布局和样式 |
| [sglang/](sglang/) | SGLang 官方源码浅克隆；调研 Prefill-first、P/D 互斥、mixed chunk 和 Decode retraction |
| [vllm/](vllm/) | vLLM 官方源码浅克隆；对比 V1 running-first、mixed batching、chunked prefill 和 preemption |

## cc-traces/

| 文件 | 简介 |
|---|---|
| `traces.jsonl` | 请求轨迹；每条记录包含请求列表、到达时间、输入/输出 token 长度及块哈希 ID 等字段 |
| `agentx_report.html` | AgentX Trace Explorer 可视化报告 |

> 本地下载的数据文件不纳入普通 Git 历史；接入 replay 前需确认数据来源、许可、版本和校验值，并提供可复现的获取方式。

## KVCacheModeling/

| 文件 | 简介 |
|---|---|
| [index.html](KVCacheModeling/index.html) | 页面结构与参数导入导出入口参考 |
| [src/ui/parameters.js](KVCacheModeling/src/ui/parameters.js) | 参数 JSON 收集、复制与导入交互参考 |
| [src/ui/styles.css](KVCacheModeling/src/ui/styles.css) | 深色主题、卡片、表单控件、按钮和响应式样式参考 |

> 用途：新前端（React + TypeScript + Vite + ECharts，见 [../docs/design/frontend.md](../docs/design/frontend.md)）借鉴其交互与布局范式，不照搬其业务逻辑。
