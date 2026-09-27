# InferenceSimulator

LLM Serving 系统级推理仿真器。离散事件仿真内核 + 前后端分离架构，目标是可解释性与仿真高性能。

当前处于 MVP 阶段，详见文档目录：[docs/index.md](docs/index.md)。

## 仓库结构（规划）

```
core/     # 仿真内核：纯 Python 包，零 IO
server/   # FastAPI 服务层
web/      # React + TS 前端
docs/     # 文档
```
