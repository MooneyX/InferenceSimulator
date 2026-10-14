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
$body = '{"num_instances":1,"workload":{"mode":"synthetic","arrival_mode":"open_loop","target_qps":50.0,"seed":0,"synthetic":{"num_requests":1000,"prompt_len":{"dist":"lognormal","mean":512,"sigma":1.0},"turn_increment_len":{"dist":"poisson","mean":128},"output_len":{"dist":"poisson","mean":128}}},"scheduler":{"max_running_requests":32,"max_prefill_tokens":16384,"pd_mutually_exclusive":true},"inference":{"mode":"manual","page_size":64,"kv":{"max_total_tokens":65536},"timing":{"mode":"empirical","prefill_c1":0.05,"prefill_c2":0.00001,"decode_d1":8.0,"decode_d2":0.002}}}'
$body | Set-Content -Encoding ascii simulate-request.json
curl.exe -X POST http://127.0.0.1:8000/api/simulate `
  -H "Content-Type: application/json" --data-binary '@simulate-request.json'
Remove-Item simulate-request.json
```

响应是单个 `SimResult`，包含 `meta`、全场 `metrics`、逐实例 `instances`、仅成功请求的 `requests`、`aborted_requests`、`skipped_requests`、`warnings` 及可为空的 `resolution`、`analysis`；各请求含 `instance_id`。`meta.config` 完整回显规范化后的输入 `SimConfig`（包括模型字段默认值），不以推导容量改写输入。无效配置返回 `422`；运行中请求失败以成功响应中的 `aborted_requests` 返回。手动/物理配置、合成/Trace 负载均使用同一 `POST /api/simulate`；Trace 通过 `workload.trace.plan_id` 引用已上传的计划。

## Trace 数据集 replay

页面的「配置方式」（自动解析或手动配置）与「负载模式」（合成负载或数据集 Trace 回放）可独立选择。选择 Trace 后，可上传原始 JSONL、等待后台扫描、预览并选择会话后下载请求计划，也可直接上传既有计划 JSON；准备就绪后使用同一个「运行仿真」按钮执行。上传完整数据集需要服务端磁盘空间；更推荐先在数据所在机器本地执行：

```powershell
uv run python -m simserver.trace_processing refs/cc-traces/traces.jsonl --output trace-summary.json
uv run python -m simserver.trace_processing refs/cc-traces/traces.jsonl --sessions "002001296e8a8c38ad9d7cc436d691afc602" --output trace-plan.json
```

本地摘要/计划与服务端共用同一处理器。计划保留输入块身份、请求长度、原始偏移和同分组时间推断的父子边；回放时设置会话/秒、随机种子及开环或闭环模式；开环还可勾选「按目标 QPS 等比例缩放到达间隔」，填写目标请求/秒。缩放保留原始突发与空闲比例，并不保证均匀投放或完成吞吐。手动配置使用填写的容量和四系数，自动解析根据模型/硬件计算容量，并支持解析或经验计时及已有的 TP/EP 约束。trace 原始模型字段不参与计算；参数 JSON 与 API 使用同一 `SimConfig` 结构，Trace 与合成共用 `workload` 中的 `arrival_mode`、`session_rate`、`seed` 和 `target_qps`；`workload.trace` 只保存可选 `plan_id`。`plan_id` 是服务进程临时资源引用；参数导出将其清为 `null`，保留回放设置。导入后需重新上传独立请求计划资源并更新 ID，不能复用旧临时引用。本地数据集保持 Git 忽略。文件限制和临时任务生命周期见 [服务层契约](docs/design/api.md#trace-文件与计划接口)。

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
            "num_instances": 1,
            "workload": {
                "mode": "synthetic",
                "arrival_mode": "open_loop",
                "target_qps": 50.0,
                "seed": 0,
                "synthetic": {
                    "num_requests": 1000,
                    "prompt_len": {"dist": "lognormal", "mean": 512, "sigma": 1.0},
                    "turn_increment_len": {"dist": "poisson", "mean": 128},
                    "output_len": {"dist": "poisson", "mean": 128},
                },
            },
            "scheduler": {
                "max_running_requests": 32,
                "max_prefill_tokens": 16384,
                "pd_mutually_exclusive": True,
            },
            "inference": {
                "mode": "manual",
                "page_size": 64,
                "kv": {"max_total_tokens": 65536},
                "timing": {
                    "mode": "empirical",
                    "prefill_c1": 0.05,
                    "prefill_c2": 1e-5,
                    "decode_d1": 8.0,
                    "decode_d2": 0.002,
                },
            },
        }
    )
)
print(result.metrics)
```

`simulate` 始终返回单个 `SimResult`；经验计时的 `result.analysis` 为 `None`。物理模式直接在同一个 `SimConfig.inference` 中传入 `mode="physical"`、`page_size`、完整 `model` / `hardware` / `storage`、`tp_size`、`ep_size` 和 `timing`；解析计时为 `{"mode":"analytical"}`，经验计时为带四系数的 `{"mode":"empirical", ...}`。可先用同一个配置调用 `POST /api/resolve` 预览 `ResolutionReport`，但运行不接收或复用预览报告，仍只提交原始 `SimConfig`。

Trace 回放将负载设为 `{"mode":"trace","arrival_mode":"open_loop","session_rate":1,"seed":0,"trace":{}}`，再调用 `simulate(config, trace_plan=plan)`，其中 `plan` 是 `TracePlan`。可选 `target_qps` 仅适用于开环；回放设置属于配置，`trace_plan` 仅是资源注入。内核不读取文件或服务端 `plan_id`；服务层根据 `workload.trace.plan_id` 获取计划后调用同一入口。Trace 请求长度、块身份和依赖语义不变。合成与 Trace 共用 `inference.max_output_tokens`（手动/物理默认均为 65536）；客户端的每请求 `max_new_tokens` 是 `max(1,output_len)` 向上取二次幂后以该上限封顶的值，不是用户负载字段。采样/原始输出不按 KV 容量或生成上限截短；超过引擎上限返回请求级 `output_exceeds_engine_limit`，Trace 零输出仍为 `invalid_trace_output`。
