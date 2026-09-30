# InferenceSimulator

LLM Serving 系统级推理仿真器。离散事件仿真内核 + 前后端分离架构，目标是可解释性与仿真高性能。

当前处于 MVP 阶段，详见文档目录：[docs/index.md](docs/index.md)。

## 仓库结构

```
core/     # 仿真内核 simcore：纯 Python 包，零 IO（M1 已完成）
server/   # FastAPI 服务层：同步仿真 API 与 OpenAPI 导出（M2 已完成）
web/      # React + TypeScript + Vite + ECharts 单页前端（M3 已完成）
docs/     # 设计文档
```

## 开发

```powershell
uv sync                                   # 安装依赖（根级 .venv）
uv run pytest                             # 单元测试
uv run pytest core/benchmarks -s          # 原有基准验收线：1k 请求 < 1s、同种子逐字节一致
uv run python core/benchmarks/bench_chunk_identity.py --requests 1000 --sessions 250 --block-size 16 --repeats 3
uv run ruff check core server ; uv run ruff format --check core server
uv run python -m mypy

cd web
npm ci                                   # 安装前端依赖（Node.js 22）
npm test                                 # Vitest 单元与组件测试
npm run typecheck                        # 前端与浏览器测试类型检查
npm run build                            # Vite 生产构建
npm run check:types                      # 检查生成的 TS 类型是否最新
npm run test:e2e                         # Playwright：真实后端 1k 请求与移动端验收
```

块身份基准分别报告程序执行耗时与 `tracemalloc` 跟踪的 Python 峰值分配（不是进程 RSS 或仿真吞吐）。与迁移前提交 `542b8d4` 对照时，用同一脚本和固定参数，仅切换 `simcore` 源码路径；确认两次输出的 step、命中、Prefill 与淘汰计数一致：

```powershell
git worktree add --detach .baseline-chunk-id 542b8d4
$env:PYTHONPATH = (Resolve-Path .baseline-chunk-id/core/src).Path
uv run python core/benchmarks/bench_chunk_identity.py --requests 1000 --sessions 250 --block-size 16 --repeats 3
Remove-Item Env:PYTHONPATH
uv run python core/benchmarks/bench_chunk_identity.py --requests 1000 --sessions 250 --block-size 16 --repeats 3
git worktree remove .baseline-chunk-id
```

`--block-size 8` 可复现第二组；以输出的 `simcore` 路径确认版本。`docs/plans/feature-roadmap.md` 第 6.16 节记录了一次实测，数据不构成跨机器性能保证。

## 启动服务

```powershell
uv run uvicorn simserver.app:create_app --factory --reload
```

服务默认监听 `http://127.0.0.1:8000`，交互式 API 文档位于 `/docs`，OpenAPI 文档位于 `/openapi.json`。

另开一个终端启动前端：

```powershell
cd web
npm run dev
```

前端默认监听 `http://127.0.0.1:5173`，开发服务器会将 `/api` 代理到 FastAPI。页面支持完整 synthetic 配置、指标与告警展示，以及可缩放平移的 Canvas 请求甘特图。

调用 `POST /api/simulate`：

```powershell
$body = '{"workload":{"mode":"synthetic","synthetic":{"num_requests":1000,"arrival_rate":50.0,"prompt_len":{"dist":"lognormal","mean":512,"sigma":1.0},"output_len":{"dist":"poisson","mean":128},"seed":0}},"engine":{"max_batch_size":32,"kv_token_budget":65536},"perf_model":{"prefill_c1":0.05,"prefill_c2":0.00001,"decode_d1":8.0,"decode_d2":0.002}}'
$body | Set-Content -Encoding ascii simulate-request.json
curl.exe -X POST http://127.0.0.1:8000/api/simulate `
  -H "Content-Type: application/json" --data-binary '@simulate-request.json'
Remove-Item simulate-request.json
```

响应包含配置与时钟元信息 `meta`、汇总指标 `metrics`、每请求生命周期区间 `requests` 和 `warnings`。无效配置返回 `422`；尚未实现的 `workload.mode=trace` 返回 `501`。

## 同步前端类型

`core` 的 Pydantic 模型是 API schema 的单一事实源。修改模型或路由后重新生成并提交两个生成物：

```powershell
cd web
npm run gen:api
```

该命令依次更新 `web/openapi.json` 与 `web/src/api/schema.d.ts`；禁止手写生成的 TS 类型。

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
