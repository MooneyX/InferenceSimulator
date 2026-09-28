# InferenceSimulator

LLM Serving 系统级推理仿真器。离散事件仿真内核 + 前后端分离架构，目标是可解释性与仿真高性能。

当前处于 MVP 阶段，详见文档目录：[docs/index.md](docs/index.md)。

## 仓库结构

```
core/     # 仿真内核 simcore：纯 Python 包，零 IO（M1 已完成）
server/   # FastAPI 服务层（M2，规划中）
web/      # React + TS 前端（M3，规划中）
docs/     # 文档
```

## 开发

```powershell
uv sync                                   # 安装依赖（根级 .venv）
uv run pytest                             # 单元测试
uv run pytest core/benchmarks -s          # 基准测试（验收线：1k 请求 < 1s、同种子逐字节一致）
uv run ruff check . ; uv run ruff format --check .
uv run python -m mypy
```

## 使用内核

```python
from simcore import SimConfig, simulate

result = simulate(
    SimConfig.model_validate(
        {
            "workload": {
                "mode": "synthetic",
                "synthetic": {
                    "num_requests": 1000,
                    "arrival_rate": 50.0,
                    "seed": 0,
                    "prompt_len": {"dist": "lognormal", "mean": 512, "sigma": 1.0},
                    "output_len": {"dist": "poisson", "mean": 128},
                },
            },
            "engine": {"max_batch_size": 32, "kv_token_budget": 65536},
            "perf_model": {
                "prefill_c1": 0.05,
                "prefill_c2": 1e-5,
                "decode_d1": 8.0,
                "decode_d2": 0.002,
            },
        }
    )
)
print(result.metrics)
```
