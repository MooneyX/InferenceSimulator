# 服务层契约

## Config schema

唯一外部配置为 `SimConfig`；物理推理配置支持经验或解析计时，可与合成/Trace 负载组合。旧手动配置 JSON 与 API 请求返回 422，不提供自动迁移。

```yaml
num_instances: int             # 必填，正整数；每实例独占 L1/L2，共享 L3
shared_l3:                     # 可选，null 或容量 0 表示关闭；全场最多一个池
  available_gb: float          # 非负十进制 GB 总预算
  lookup_latency_ms: float     # 默认 0，非负
workload:
  mode: synthetic | trace      # 默认 synthetic
  arrival_mode: closed_loop | open_loop  # 默认 closed_loop；网页显式 open_loop
  seed: int                    # 默认 0，非负整数
  session_rate: float | null   # 合成闭环和 Trace 所有模式必填，正数；合成开环禁止
  target_qps: float | null     # 合成开环必填；Trace 开环可选；闭环禁止
  synthetic:                   # 仅 mode=synthetic
    num_requests: int          # 必填，所有 session 的总轮次数
    prompt_len: LengthDist     # 必填，两种模式仅根轮输入分布
    turn_increment_len: LengthDist  # 必填，子轮新增内容分布
    output_len: LengthDist     # 必填，实际输出目标的采样分布
    session:                   # 可选，仅 closed_loop 模式使用
      num_sessions: int        # 省略时等于 num_requests
      topology: linear
      think_time_ms: float     # 子轮思考时间的指数分布均值（ms），默认 0
  trace:                       # 仅 mode=trace；计划内容是独立资源
    plan_id: str | null        # 可选；HTTP 运行引用已上传计划；内核另注入 TracePlan
scheduler:
  max_running_requests: int    # 必填，正整数
  max_prefill_tokens: int      # 默认 16384
  l3_prefetch_min_tokens: int  # 默认 256；L3 连续命中 token 数须严格超过此值
  pd_mutually_exclusive: bool  # 默认 true；false 启用同一步融合
inference:
  page_size: int               # 默认 64
  max_output_tokens: int       # 默认 65536，正整数；引擎输出上限
  model: ModelSpec             # 必填；结构、权重与 KV 精度
  hardware: HardwareSpec       # 必填；每实例的 GPU/有效性能
  storage: StorageSpec         # 必填；每卡 L1/每实例 L2 预算与接入带宽
  tp_size: int                 # 默认 1
  ep_size: int                 # 默认 1
  timing:                     # 可选，默认解析计时
    mode: analytical | empirical
    prefill_c1: float          # 仅经验计时
    prefill_c2: float          # 仅经验计时，默认 0
    decode_d1: float           # 仅经验计时
    decode_d2: float           # 仅经验计时
```

`inference` 接收完整 `model: ModelSpec`、`hardware: HardwareSpec`、`storage: StorageSpec`、`tp_size`、`ep_size` 和 `timing`，不接收手填 `kv`。解析计时只传 `{"mode":"analytical"}`；经验计时传 `{"mode":"empirical","prefill_c1":...,"prefill_c2":...,"decode_d1":...,"decode_d2":...}`。L3 容量及查询延迟只接受顶层 `shared_l3`；旧手动容量及链路字段均返回 422，不保留兼容入口。旧版物理 JSON 也必须删去 `inference.mode` 和 `shared_l3.mode`；保留 `workload.mode` 与 `inference.timing.mode`，多余字段在 API 返回 422、前端导入报错。

### LengthDist（长度分布，可判别联合）

```yaml
dist: fixed | poisson | lognormal
value: int      # 仅 fixed：固定 token 数
mean: float     # 仅 poisson / lognormal：分布均值
sigma: float    # 仅 lognormal：形状参数，越大尾越长
```

合成开环在 `workload` 公共层只传 `target_qps=q`（期望请求/秒），不传 `session_rate`，全部 N 次投放各采样均值 `1000/q` ms 的指数间隔（含首请求）。闭环只传 `session_rate`（期望新会话/秒），根请求以该频率泊松到达；子轮由实际完成时刻和 `think_time_ms` 决定，最终请求 QPS 在结果中实测；未指定 `num_sessions` 时等于 `num_requests`。运行时 `max_total_tokens` 从每卡 L1 预算扣除权重和预留后向下取整页推导，至少保留一页；不能直接填写 token 容量。合成采样输出与 Trace 原始输出视为已发生结果，不按 KV 容量或输出上限截短。每请求可见 `max_new_tokens=min(next_power_of_two(max(1,output_len)), inference.max_output_tokens)`，真实输出目标随内部 `InferenceRequest.output_len` 传递，但当前调度仍只用生成上限与进度估算；超过引擎上限时返回请求级 `output_exceeds_engine_limit`。公共投放字段只位于 `workload`，旧合成子层 `arrival_mode`、`seed`、`session_rate`、`arrival_rate`、`max_new_tokens` 及 Trace 子层 `mode`、`seed`、`session_rate`、`target_qps` 均按未知字段返回 422。采样、到达及继承见 [workload.md](workload.md)。跨 session 公共前缀不支持，同 session 父子轮仍可继承完整页；`session.content_seed` 和 `allow_cross_session_shared_prefix` 不在 schema 中，显式提交 `true`/`false` 等未知字段均返回 422，`meta.config` 不回显。`SimConfig` 不提供独立 `prefix_cache` 或未实现的校准模式；模型、硬件与存储都属于唯一的 `inference` 输入。低层预算为零时保持原有 L1 行为；启用 L2 要求 L2→L1 带宽，启用 L3 还要求有 L2 及 L3→L2 带宽。运行规格从模型推导 KV 字节/token，预算须至少容纳一个完整页；非法请求返回 422。传输只按页大小和链路计时，不接入真实硬件。

## API

后台仿真端点 `POST /api/simulate` 接受完整 `SimConfig`，立即返回 `202 {"id":"任务ID"}`。`GET /api/jobs`、`GET /api/jobs/{id}` 返回任务状态（`queued/running/completed/failed/cancelled`）和完成/失败点计数；`GET /api/jobs/events` 按 SSE 推送快照与状态变化，断开后可重新查询；`DELETE /api/jobs/{id}` 移除排队点并终止其运行中 worker。完成后 `GET /api/jobs/{id}/result` 返回 200 和整场结果，未完成或失败返回 409。请求必须包含 `num_instances >= 1`；`scheduler` / `inference` 表示单实例配置，`inference.hardware.gpu_count` 也表示每实例卡数，总卡数为两者相乘。所有实例独占硬件和 L1/L2；`shared_l3` 为全场总预算。仿真内核保持零 IO。结果即使存在请求级失败仍为 200；三个结果列表均按请求 ID 升序，数量之和等于计划请求总数：

```yaml
meta:              # config 回显；num_requests / num_completed / num_aborted /
                   # num_skipped / num_engine_steps；sim_start / sim_end（ms）
metrics:           # 全场指标及无成功请求时的空值，见 metrics.md
instances:         # 按 instance_id 排序的逐实例指标、完成/失败/跳过和引擎步数
requests:          # 仅成功请求；含 id、instance_id、session_id、parent_request_id、
                   # arrival / prefill_start / prefill_end / finish、输入输出长度及命中/重算量
aborted_requests:  # 已发送但失败：id、instance_id、arrival、failed_at、reason、
                   # generated_tokens、retry_count
skipped_requests:  # 父轮未完成、从未发送：id、父轮 instance_id、parent_request_id、reason
warnings:          # 非致命告警；不代替请求级失败状态
resolution:        # ResolutionReport；物理容量与可运行性报告
analysis:          # AnalyticalSummary | null；经验计时为 null
```

`aborted_requests.reason` 包括 `input_exceeds_kv_capacity`、`decode_kv_exhausted`，以及合成/Trace 通用的 `output_exceeds_engine_limit`；Trace 零输出为 `invalid_trace_output`。`skipped_requests.reason` 为 `parent_not_completed`。未发送后代没有 arrival；`requests` 才供甘特图绘制。

- 提交请求体是 `SimConfig`；提交响应为任务收据，结果接口响应仍是完整 `SimResult`。`meta.config` 是补齐默认值的原规范化输入，完整保留模型/硬件/存储与回放设置，不替换成推导后的运行时配置。
- `meta.wall_clock_ms` 仅度量运行阶段，不包含配置解析；它不是仿真时钟或 GPU 预测时长，也不属于确定性比较字段。除墙钟字段外，相同配置、计划与种子的仿真结果可复现；响应按内核 `SimResult` 序列化。

### 结果缓存

- 配置先完成校验和默认值补齐；Trace 先加载并校验计划。以稳定 JSON 配置、规范化计划内容的 SHA-256、缓存格式及源代码/运行时版本指纹生成缓存键，保留数组顺序，用计划哈希替换临时 `plan_id`。缓存命中仍回显本次完整配置。
- 默认使用 SQLite 持久保存压缩完整结果；路径及容量见 README。任务调度与扫描摘要仅在服务进程内存中，不跨服务重启保留；已完成同参数结果可复用，并行 worker 之间不合并正在进行的计算；计算时不占用数据库事务。请求级失败的完整结果可缓存。
- `POST /api/simulate?force=true` 提交跳过已完成缓存的重算任务，默认 `force=false`。结果接口响应头 `X-Simulation-Cache` 为 `hit` / `miss` / `shared`，`X-Simulation-Cache-Lookup-Ms` 为查找/读取耗时，`meta.wall_clock_ms` 保留原计算耗时。
- `DELETE /api/simulation-cache` 返回 204；跨进程共享缓存代际防止清空前开始的计算重新写入。单条超容量仍返回但不保存，容量超限按最久未使用记录清理。
- 不改变 Trace 临时资源的有效期；失效计划仍返回 404，重新上传相同计划后可复用缓存。

### 错误响应

| 状态码 | 场景 | 响应体 |
|---|---|---|
| `422` | 提交的配置格式、范围或模式与载荷不匹配；无效的扫描参数/档位 | FastAPI 字段错误列表，或 `{"detail": str}` |
| `404` | 提交时 Trace 计划 ID 未知或已过期；查询未知任务 | `{"detail": str}` |
| `409` | 任务尚未完成、已失败或已取消，无法获取完整结果 | `{"detail": str}` |

每点运行时容量解析或计时模式错误记录为 `failed` 任务/扫描点及错误文本，不作为提交 HTTP 422/501 返回。

运行中的单请求容量不足不归入 HTTP 422，而是正常结果的 `aborted_requests`；其他请求可继续执行。开环模式不预建未来子轮，失败时下一节拍可创建新根；闭环模式的预建后代才进入 `skipped_requests`。

错误响应体属于 HTTP 边界，由服务层定义并登记进 OpenAPI，不进入 `core.schemas`。

`metrics` 包含 TTFT、TPOT、完成吞吐、makespan、L1/L2/L3 命中与驻留、驱逐/提升/降级及恢复字节和时间；`requests` 的分层命中只记录首次准入。计算与空值口径统一见 [metrics.md](metrics.md)。

## 参数组合扫描与后台调度

`GET /api/scans/parameters` 返回可选参数的键、名称、数值/整数/布尔类型和适用的合成/Trace、开环/闭环组合。不包含模型配置字段、GPU 预设名及派生 KV 容量；`inference.parallelism` 代表必须同步写入的每实例 GPU 卡数与 TP 数（MoE 时 EP 数，Dense 固定 EP1）。`POST /api/scans` 以 `{ "base": SimConfig, "axes": { "x": { "key": "num_instances", "values": [1, 2] }, "color": { "key": "scheduler.pd_mutually_exclusive", "values": [true, false] } } }` 提交扫描，`shape` 与 `line` 为可选的独立参数轴。提交必须有横轴，其余轴不重复，参数键由服务端白名单及负载模式检查；每点构建完整配置并重新执行模型校验/容量解析，不可运行点独立失败，不中断其他点。

`GET /api/jobs/{id}/points` 返回已完成的点级指标、实际 `SimConfig`、同一配置/Trace 内容/引擎指纹的去重签名或失败原因。所有普通任务和扫描点共享单条严格 FIFO 队列；先提交扫描的全部点在后提交普通任务之前领取空闲 worker，不预留容量或按任务类型抢占。使用自行管理的计算进程，`SIMULATOR_WORKERS` 默认 8；取消整个扫描移除排队点、终止正在运行该扫描的 worker 并补建进程。扫描点数量不设软件上限，调度器以组合迭代器保存队列；任务状态与扫描摘要留在单一服务进程内存中，不单独持久化或自动设置期限，服务重启后不恢复。已有完整仿真结果仍遵循 `ResultCache` 容量限制，Trace 上传资源保持原临时存储策略。

浏览器可以勾选多项已完成扫描、切换指标与四种参数角色，将相同输入身份的点去重；对缺失的组合不插值、不对未显示参数差异求平均。导出 HTML 在单文件中内嵌 ECharts、可筛选的点级数据和图表脚本，无网络时仍能选择批次、重新透视、缩放和下载数据 CSV。

## Trace 文件与计划接口

原始 `traces.jsonl` 可本地处理，也可通过 `POST /api/traces` 用 `application/octet-stream` 直接上传原始字节（无需 multipart，最大 3 GiB）；返回 202 和任务 ID。`GET /api/traces/{id}` 查 `processing/ready/failed`，`GET /api/traces/{id}/sessions?offset=0&limit=50` 分页查看实际扫描摘要。`POST /api/traces/{id}/plans` 接受 `{"sessions":["原始会话ID"]}`，下载 JSON 计划，最大 64 MiB；`POST /api/replay-plans` 直接上传本地导出的计划字节并返回计划 ID、请求数、会话数和块大小。这些端点只管理计划资源，保留原有上传、扫描、预览及下载路径。

回放将同一个 `SimConfig.workload` 设为 `{"mode":"trace","arrival_mode":"open_loop","session_rate":1,"seed":0,"trace":{"plan_id":"已上传计划ID"}}`，向唯一 `POST /api/simulate` 提交完整配置。提交时服务层根据 ID 加载 `TracePlan` 并在任务中持有其快照，worker 调用 `simulate(config, trace_plan=plan)`；回放设置不在独立包装体中。任务完成后的结果接口返回 `SimResult`，解析计时保留 TP/EP 通信与 `analysis`，经验计时为 `analysis=null`。

公共层 `workload.target_qps` 只允许在 `workload.arrival_mode=open_loop` 时设置；先由 `session_rate` 生成会话起点，再按整个所选计划的首末发送跨度等比例缩放所有请求到达时间。单请求或所有请求同刻到达时任务失败；闭环携带该参数、非正数及非有限数在提交阶段返回 422。不设置时沿用原开环时间；缩放不保证均匀投放或完成吞吐。

原始文件与计划暂存于服务进程的临时目录，闲置 24 小时后在后续访问时清理；服务进程退出后失效，当前版本不支持跨实例共享任务或断点续传。文件超限为 413，解析/校验失败为 422，处理中尝试导出为 409，未知或过期 ID 为 404。上传计划的块 ID 必须是会话局部一致的前缀身份；原始输入块参与运行时连续缓存匹配，不包含原始输出块身份。计划内容不嵌入 `SimConfig`；HTTP 运行使用 `workload.trace.plan_id` 引用独立资源。参数 JSON 导出将临时 `plan_id` 清为 `null`，保留开环/闭环、会话速率、种子与可选目标 QPS，导入后提示重新上传计划。

## 自动解析与 TP/EP 容量预览

- `GET /api/presets` 仅返回 `models` 和 `hardware` 目录，不返回搭配场景列表。模型包含 Qwen3-8B、Qwen3-235B-A22B 和固定修订的 HY4-Preview 主干参考结构以及 B300 单卡显存、互联带宽标称值；不提供已测有效算力或模型修订认证，目录条目不等于已验证部署。B300 的 1800 GB/s 是单卡双向标称值；页面以 600 GB/s 作为未经校准的单 rank 有效带宽试算值，非实测值。
- `POST /api/resolve` 接受与运行完全相同的 `SimConfig`，返回 `ResolutionReport`，不产生另一套可提交的配置。`ModelSpec`、`HardwareSpec`、`StorageSpec`、`timing`、`tp_size`、`ep_size` 均位于 `inference`。TP 数须等于 GPU 数；Dense 使用 EP1，MoE 首版须 EP=TP；支持全注意力 GQA/MQA、按层数组合的 GQA/MLA/MLA+DSA BF16/FP16 粗估，以及以完整组件配置表示的 HY4-Preview 参考预设，参数可修改，不以名称或修订号锁定结构，容量和性能均复用通用路径。组数不限但层数之和须等于模型层数；组只描述注意力，FFN/MoE 按全模型独立配置，不支持需逐层顺序或可变页成本的注意力。MoE 须填写专家结构，各 MoE 层规格相同；所有驻留专家均匀放置到 EP ranks。L1 输入为**每卡**含权重可用预算，按 rank 扣除非专家、全部驻留专家权重与预留后取最紧整页容量；L2 输入为**单实例整组**仅放 KV 的预算；L3 输入为顶层 `shared_l3.available_gb` 的**全场总预算**。两者均按一个 TP 组保存一份逻辑页的字节数换算，L3 不乘实例数量。模型词表大小与结构维度必填；总参数、每 token 激活参数和全部专家参数由结构估算，BF16/FP16 全部驻留权重按参数矩阵和逐卡复制/分片关系计入，不再接受总参数或实际驻留字节覆盖。报告包含参数估算、每 rank TP 分片/复制权重、EP 专家权重、KV 字节与页数；可用 `norms`、`router`、`ihc` 和注意力组辅助字段配置 BF16/FP16/FP32 权重，类型及形状决定逻辑参数数，精度与放置决定驻留字节。辅助容量配置不扩展性能算子模型，未建模权重仍不计入，结果不能解释为真实部署占用。缺失结构、显存不足或不支持拓扑返回 422。缺少 TP/EP 通信计时参数仍可预览容量，但 `runnable=false`；单卡支持经验或解析计时，多卡 TP/EP 仅支持通信参数齐备时的解析计时。
- `ResolutionReport` 包含 `capacity: CapacityReport`、`max_total_tokens`、`l2_capacity_tokens`、`l3_capacity_tokens`、`runnable` 与可空 `reason`；报告只描述容量与可运行性，不替换原输入，不可作为 `simulate` 的参数。
- 后台 worker 对每场配置独立准备物理运行规格后执行；支持 Dense TP 与平均负载 MoE/EP 的解析计时，缺通信参数或 TP/EP>1 经验四系数模式使任务失败。任务结果接口返回单个 `SimResult`：`resolution` 给出逐 rank 容量与运行容量，`analysis` 分列 TP 同步、EP dispatch/combine 的时间与逻辑流量；这些费用已计入 `device_ms`，KV 恢复时间独立。422 `detail` 可为字段错误列表或业务错误字符串，OpenAPI 同时登记两种格式。前端不复用上次容量预览的 token 数作为运行配置；结构估算参数和权重占用未经部署核验，旧配置 JSON 结构会被拒绝。
- 合成/Trace 负载均使用唯一 `/api/simulate`。四系数仅在 `inference.timing.mode=empirical` 使用，解析模式的逐步估计见 [性能模型](perf-model.md)。

## 类型同步

schema 单一事实源为 `core` 中的 Pydantic 模型；FastAPI 自动生成 OpenAPI，前端用 `openapi-typescript` 生成 TS 类型。禁止双侧手写类型定义。

生成链路：

```
simcore.schemas → FastAPI OpenAPI → web/openapi.json → openapi-typescript → web/src/api/schema.d.ts
```

- 两个生成物均入库，在 `web/` 下执行 `npm run gen:api` 一次性重新生成；改动 `schemas` 或路由后必须重新生成并随同提交。
- 新鲜度校验：Python 测试比对 `web/openapi.json` 与当前模型；`npm run check:types` 比对 TS 类型与 `openapi.json`。
- 请求与响应共用同一组 schema（关闭 FastAPI 的 input/output 拆分），每个内核模型在 TS 中只有一个同名类型。
- 字段在 TS 中是否必填以生成的 OpenAPI 契约为准，不由是否存在 Python 默认值统一推断；例如 `InferenceConfig.timing` 使用默认工厂，生成类型中可选。前端导入缺省字段按 schema 默认值补齐，不使用页面未经校准的试算默认值覆盖输入；判别字段（如 `mode`、`dist`）仍按生成契约提供，以支持联合类型收窄。
