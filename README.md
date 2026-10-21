# InferenceSimulator

LLM Serving 系统级推理仿真器。离散事件仿真内核 + 前后端分离架构，目标是可解释性与仿真高性能。

当前处于 MVP 阶段，详见文档目录：[docs/index.md](docs/index.md)。

## 仓库结构

```
core/     # 仿真内核 simcore：纯 Python 包，零 IO（M1 已完成）
server/   # FastAPI 服务层：后台仿真与扫描任务、缓存及 OpenAPI 导出
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

前端默认监听 `http://127.0.0.1:5173`，开发服务器会将 `/api` 代理到 FastAPI。页面默认以目标 QPS 作为期望速率，按泊松过程投放全部请求：优先续写已完成会话，否则创建新会话；根输入、子轮新增内容及输出分别采样。理想命中率是构造结果，完成吞吐不保证等于发送 QPS。也可切换闭环会话模式：沿用根输入＋每轮新增内容的长度构造，配置会话数、新会话投放频率和平均思考间隔，不输入目标请求 QPS；结果显示实测请求 QPS 与完成吞吐。页面支持设置实例数与每实例独占的硬件/引擎配置（总 GPU 数 = 实例数 × 每实例 GPU 数），各实例独占 L1/L2，一场内共享唯一 L3，容量在「共享 L3」区按总量配置（0 为关闭，不乘实例数量）；根请求轮询、子请求回到父实例。可切换全场与逐实例指标和甘特图；共享 L3 的容量、驻留和峰值仅在全场指标展示。页面支持 P/D 模式切换、完成/失败/跳过状态及指标展示；Canvas 甘特图仅绘制已完成请求。

调用 `POST /api/simulate`：

```powershell
$body = '{"num_instances":1,"workload":{"mode":"synthetic","arrival_mode":"open_loop","target_qps":50.0,"seed":0,"synthetic":{"num_requests":1000,"prompt_len":{"dist":"lognormal","mean":512,"sigma":1.0},"turn_increment_len":{"dist":"poisson","mean":128},"output_len":{"dist":"poisson","mean":128}}},"scheduler":{"max_running_requests":32,"max_prefill_tokens":16384,"pd_mutually_exclusive":true},"inference":{"page_size":64,"model":{"name":"toy","num_layers":2,"hidden_size":16,"num_attention_heads":2,"num_kv_heads":1,"head_dim":8,"ffn_intermediate_size":32,"vocab_size":16},"hardware":{"name":"toy","memory_bytes_per_gpu":10000000,"effective_fp16_tflops":1,"effective_hbm_gb_s":1},"storage":{"l1_available_gb":0.004204768},"timing":{"mode":"empirical","prefill_c1":0.05,"prefill_c2":0.00001,"decode_d1":8.0,"decode_d2":0.002}}}'
$body | Set-Content -Encoding ascii simulate-request.json
curl.exe -X POST http://127.0.0.1:8000/api/simulate `
  -H "Content-Type: application/json" --data-binary '@simulate-request.json'
Remove-Item simulate-request.json
```

`POST /api/simulate` 立即返回 `202` 与 `{"id":"任务ID"}`，不再同步返回整场结果。通过 `GET /api/jobs` 或 `GET /api/jobs/{id}` 查询状态，`GET /api/jobs/events` 接收 SSE 主动通知；完成后 `GET /api/jobs/{id}/result` 返回 `SimResult`，也可 `DELETE /api/jobs/{id}` 取消排队或运行中的任务。浏览器断线时页面回退为低频查询。`SimResult` 包含 `meta`、全场 `metrics`、逐实例 `instances`、仅成功请求的 `requests`、`aborted_requests`、`skipped_requests`、`warnings`、物理容量 `resolution` 和可为空的 `analysis`。`meta.config` 完整回显规范化输入。提交阶段无效配置返回 `422`；容量或计时不支持等运行时错误在任务状态中记为 `failed`，请求级失败在成功结果的 `aborted_requests` 中。合成/Trace 均通过该接口运行；Trace 使用已上传计划的 `plan_id`。旧版输入结构仍不支持。

## 仿真结果缓存

服务端自动使用 SQLite 保存完整结果，默认位置为启动工作目录下的 `.simulation-cache/results.sqlite3`；本地部署从项目根启动，重启后仍可复用。键包含补齐默认值的配置、随机种子、规范化 Trace 计划的 SHA-256 及内核/服务源代码、Python/Pydantic 版本指纹。同内容计划重复上传可命中，响应仍回显本次 `plan_id`；Trace 临时 ID 过期或服务重启后须先重新上传计划。

页面在任务完成后显示结果缓存命中状态、缓存查找/读取耗时和原始仿真计算耗时。「强制重新计算」使用当前表单参数提交重算任务并更新缓存；「清空结果缓存」清空本地缓存，不删除已显示或正在运行的任务结果。缓存头在 `GET /api/jobs/{id}/result` 返回；提交 API 为 `POST /api/simulate?force=true`，清理为 `DELETE /api/simulation-cache`（204）。

缓存默认最多保存 512 MiB 的压缩结果及输入记录，超限按最久未使用记录清理；单条超限仍正常返回但不保存。启动前可通过 `SIMULATOR_CACHE_PATH` 指定数据库路径、`SIMULATOR_CACHE_MAX_BYTES` 指定容量字节数，通过 `SIMULATOR_WORKERS` 设置计算进程数（默认 8）。所有普通仿真与扫描点共用严格 FIFO 队列，有空闲 worker 即分发；取消运行中点会终止并补建其 worker。同参数已完成结果共享 SQLite 缓存，但不同 worker 上同时进行的同配置计算不会合并。计算队列仍只在单个后端进程内运行；任务列表、已完成单场的完整 JSON 及扫描点另存于同目录的 `jobs.sqlite3`（可通过 `SIMULATOR_JOBS_PATH` 单独配置），服务重启后保留，不受结果缓存清空和容量淘汰影响。任务记录代码指纹并标注与当前代码是否一致；旧版本仍可查看和导出，单场可直接下载 JSON，扫描可导出离线 HTML。重启时未完成的任务转为 `interrupted`，已完成扫描点仍可查看和导出，不自动续跑；未产生结果的单场不能导出。首次执行 `deploy-local.cmd` 时会在停止旧版后端前归档尚在运行服务内存中的任务；旧任务无法追溯代码指纹，显示为“代码版本未知”。若旧服务已经停止，丢失的内存任务不能从缓存完整恢复。Trace 上传计划仍受临时目录生命周期约束；历史归档不会自动过期，须为数据库预留磁盘空间并自行备份。

## 参数组合扫描

页面的扫描区有独立的「负载模式」与「请求到达模式」；Trace 扫描直接上传回放计划 JSON，使用上传回执中的块大小限制页大小。选择一个必需的横轴，并可为颜色、节点形状和线样式选择三个不同参数；参数选择器支持输入中文名称或字段路径搜索，档位可用逗号列表或 `起点:终点:步长`。参数目录由 `GET /api/scans/parameters` 返回，按照合成/Trace 和开环/闭环筛选，不包含模型配置或 GPU 预设名称。GPU 卡数与 TP（MoE 时同时包含 EP）是一个共同扫描参数，硬件/存储预算按页面显示单位填写。`POST /api/scans` 提交 `{ "base": SimConfig, "axes": { "x": { "key": "参数路径", "values": [1, 2] } } }`；扫描点和普通任务共用 `GET /api/jobs/{id}`、SSE 与取消接口；结果通过 `GET /api/jobs/{id}/points` 查询点级指标、实际配置和失败原因。组合数不设软件上限，但实际执行受 CPU、内存与磁盘容量限制。扫描任务完成后可勾选多批次组合、重分配图表维度、切换指标并导出包含 ECharts 源码和数据的离线可操作单文件 HTML。

## Trace 数据集 replay

页面根据模型、硬件与存储预算自动解析容量；「负载模式」可选择合成负载或数据集 Trace 回放。选择 Trace 后，可上传原始 JSONL、等待后台扫描、预览并选择会话后下载请求计划，也可直接上传既有计划 JSON；准备就绪后使用同一个「运行仿真」按钮执行。上传完整数据集需要服务端磁盘空间；更推荐先在数据所在机器本地执行：

```powershell
uv run python -m simserver.trace_processing refs/cc-traces/traces.jsonl --output trace-summary.json
uv run python -m simserver.trace_processing refs/cc-traces/traces.jsonl --sessions "002001296e8a8c38ad9d7cc436d691afc602" --output trace-plan.json
```

本地摘要/计划与服务端共用同一处理器。计划保留输入块身份、请求长度、原始偏移和同分组时间推断的父子边；回放时设置会话/秒、随机种子及开环或闭环模式；开环还可勾选「按目标 QPS 等比例缩放到达间隔」，填写目标请求/秒。缩放保留原始突发与空闲比例，并不保证均匀投放或完成吞吐。容量根据模型、硬件与存储预算解析，可选解析或经验计时，并保持已有的 TP/EP 约束。trace 原始模型字段不参与计算；参数 JSON 与 API 使用同一 `SimConfig` 结构，Trace 与合成共用 `workload` 中的 `arrival_mode`、`session_rate`、`seed` 和 `target_qps`；`workload.trace` 只保存可选 `plan_id`。`plan_id` 是服务进程临时资源引用；参数导出将其清为 `null`，保留回放设置。导入后需重新上传独立请求计划资源并更新 ID，不能复用旧临时引用。本地数据集保持 Git 忽略。文件限制和临时任务生命周期见 [服务层契约](docs/design/api.md#trace-文件与计划接口)。

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
                "page_size": 64,
                "model": {"name": "toy", "num_layers": 2, "hidden_size": 16,
                          "num_attention_heads": 2, "num_kv_heads": 1, "head_dim": 8,
                          "ffn_intermediate_size": 32, "vocab_size": 16},
                "hardware": {"name": "toy", "memory_bytes_per_gpu": 10_000_000,
                             "effective_fp16_tflops": 1, "effective_hbm_gb_s": 1},
                "storage": {"l1_available_gb": 0.004204768},
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

Trace 回放将负载设为 `{"mode":"trace","arrival_mode":"open_loop","session_rate":1,"seed":0,"trace":{}}`，再调用 `simulate(config, trace_plan=plan)`，其中 `plan` 是 `TracePlan`。可选 `target_qps` 仅适用于开环；回放设置属于配置，`trace_plan` 仅是资源注入。内核不读取文件或服务端 `plan_id`；服务层根据 `workload.trace.plan_id` 获取计划后调用同一入口。Trace 请求长度、块身份和依赖语义不变。合成与 Trace 共用 `inference.max_output_tokens`（默认 65536）；客户端的每请求 `max_new_tokens` 是 `max(1,output_len)` 向上取二次幂后以该上限封顶的值，不是用户负载字段。采样/原始输出不按 KV 容量或生成上限截短；超过引擎上限返回请求级 `output_exceeds_engine_limit`，Trace 零输出仍为 `invalid_trace_output`。
