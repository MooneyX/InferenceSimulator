# InferenceSimulator

LLM Serving 系统级推理仿真器。离散事件仿真内核 + 前后端分离架构，目标是可解释性与仿真高性能。

当前处于 MVP 阶段，详见文档目录：[docs/index.md](docs/index.md)。

## 仓库结构

```
core/     # 仿真内核 simcore：纯 Python 包，零 IO（M1 已完成）
server/   # FastAPI 服务层：同步仿真 API 与 OpenAPI 导出（M2 已完成）
web/      # React + TypeScript + Vite + ECharts 单页前端（M3 已完成）
docs/     # 现行设计、模块实施计划与文档索引
```

## 开发

```powershell
uv sync                                   # 安装依赖（根级 .venv）
uv run pytest                             # 单元测试
uv run pytest core/benchmarks/bench_engine.py -s  # 100/1k/10k，两种 P/D 模式的耗时、步数和 KV 峰值
uv run python core/benchmarks/bench_chunk_identity.py --requests 1000 --sessions 250 --block-size 16 --repeats 3
uv run ruff check core server ; uv run ruff format --check core server
uv run python -m mypy

cd web
npm ci                                   # 安装前端依赖（Node.js 22）
npm test                                 # Vitest 单元与组件测试
npm run typecheck                        # 前端与浏览器测试类型检查
npm run build                            # Vite 生产构建
npm run check:types                      # 检查生成的 TS 类型是否最新
npm run test:e2e                         # Playwright：真实后端六组规模/模式响应与移动端验收
```

块身份基准报告程序执行耗时与 `tracemalloc` 跟踪的 Python 峰值分配，并非进程 RSS 或仿真预测的吞吐；数据不构成跨机器性能保证。

## 启动服务

Windows 本地一键更新部署：双击项目根目录的 `deploy-local.cmd`，或在 PowerShell 中运行 `./deploy-local.cmd`。脚本只会停止本项目占用的 8000/5173 端口进程，按锁文件同步 Python/前端依赖，重新启动服务并验证页面及仿真接口；不执行 `git pull`，使用当前工作区代码。

手动启动后端：

```powershell
uv run uvicorn simserver.app:create_app --factory --reload
```

服务默认监听 `http://127.0.0.1:8000`，交互式 API 文档位于 `/docs`，OpenAPI 文档位于 `/openapi.json`。

另开一个终端启动前端：

```powershell
cd web
npm run dev
```

前端默认监听 `http://127.0.0.1:5173`，开发服务器会将 `/api` 代理到 FastAPI。页面默认以目标 QPS 作为期望速率，按泊松过程投放全部请求：优先续写已完成会话，否则创建新会话；根输入、子轮新增内容及输出分别采样。理想命中率是构造结果，完成吞吐不保证等于发送 QPS。也可切换闭环会话模式：沿用根输入＋每轮新增内容的长度构造，配置会话数、新会话投放频率和平均思考间隔，不输入目标请求 QPS；结果显示实测请求 QPS 与完成吞吐。页面支持设置实例数与每实例独占的硬件/引擎配置（自动模式总 GPU 数 = 实例数 × 每实例 GPU 数），根请求轮询、子请求回到父实例；可切换全场与逐实例指标和甘特图。页面支持 P/D 模式切换、完成/失败/跳过状态及指标展示；Canvas 甘特图仅绘制已完成请求。

调用 `POST /api/simulate`：

```powershell
$body = '{"num_instances":1,"workload":{"mode":"synthetic","synthetic":{"num_requests":1000,"arrival_rate":50.0,"arrival_mode":"open_loop","prompt_len":{"dist":"lognormal","mean":512,"sigma":1.0},"turn_increment_len":{"dist":"poisson","mean":128},"output_len":{"dist":"poisson","mean":128},"seed":0}},"engine":{"max_running_requests":32,"max_total_tokens":65536,"page_size":64,"pd_mutually_exclusive":true},"perf_model":{"prefill_c1":0.05,"prefill_c2":0.00001,"decode_d1":8.0,"decode_d2":0.002}}'
$body | Set-Content -Encoding ascii simulate-request.json
curl.exe -X POST http://127.0.0.1:8000/api/simulate `
  -H "Content-Type: application/json" --data-binary '@simulate-request.json'
Remove-Item simulate-request.json
```

响应包含 `meta`、全场 `metrics`、逐实例 `instances`、仅成功请求的 `requests`、`aborted_requests`、`skipped_requests` 和 `warnings`；各请求含 `instance_id`。无效配置返回 `422`；运行中请求失败以成功响应中的 `aborted_requests` 返回；尚未实现的 `workload.mode=trace` 返回 `501`。

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
                    "arrival_mode": "open_loop",
                    "seed": 0,
                    "prompt_len": {"dist": "lognormal", "mean": 512, "sigma": 1.0},
                    "turn_increment_len": {"dist": "poisson", "mean": 128},
                    "output_len": {"dist": "poisson", "mean": 128},
                },
            },
            "engine": {"max_running_requests": 32, "max_total_tokens": 65536, "page_size": 64},
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
