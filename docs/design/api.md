# 服务层契约

## Config schema

唯一外部配置为 `SimConfig`；手动/物理推理配置与合成/Trace 负载可独立组合，不再使用单独的解析请求或 Trace 仿真包装体。

```yaml
num_instances: int             # 必填，正整数；其余参数描述每实例独占配置
workload:
  mode: synthetic | trace
  synthetic:                   # 仅 mode=synthetic
    num_requests: int           # 所有 session 的总轮次数
    arrival_rate: float | null  # 仅 open_loop 必填，目标请求 QPS
    session_rate: float | null  # 仅 session 必填，新会话泊松投放频率（会话/秒）
    arrival_mode: session | open_loop  # 省略为 session；网页默认 open_loop，可切换
    prompt_len: LengthDist      # 两种模式仅根轮输入分布
    turn_increment_len: LengthDist  # 两种模式必填，子轮新增内容分布
    output_len: LengthDist
    max_new_tokens: int         # 每请求可见生成上限，默认 128
    seed: int
    session:                   # 仅 session 模式使用
      num_sessions: int        # 省略时等于 num_requests
      topology: linear
      think_time_ms: float     # 子轮思考时间的指数分布均值（ms），默认 0
  trace:                       # 仅 mode=trace；设置属于配置，计划内容是独立资源
    mode: open_loop | session
    session_rate: float        # 新会话/秒，正数
    seed: int                  # 非负整数
    target_qps: float | null   # 可选正数，仅 open_loop
    plan_id: str | null        # HTTP 运行引用已上传计划；内核另注入 TracePlan
scheduler:
  max_running_requests: int    # 必填，正整数
  max_prefill_tokens: int      # 默认 16384
  l3_prefetch_min_tokens: int  # 默认 256；L3 连续命中 token 数须严格超过此值
  pd_mutually_exclusive: bool  # 默认 true；false 启用同一步融合
inference:                     # 按 mode 判别，两个分支共用唯一 page_size
  mode: manual | physical
  page_size: int               # 默认 64
  # manual 分支：
  kv:
    max_total_tokens: int      # 必填，实际 L1 KV token 容量
    l2_capacity_tokens: int    # 默认 0，关闭 L2；须为整页倍数
    l3_capacity_tokens: int    # 默认 0，关闭 L3；须为整页倍数，要求 L2>0
    kv_bytes_per_token: int | null # 默认 null；启用 L2 时为正整数
    l2_to_l1_bandwidth_bytes_per_ms: float | null # 启用 L2 时为正
    l3_to_l2_bandwidth_bytes_per_ms: float | null # 启用 L3 时为正
    l2_latency_ms: float       # 默认 0，非负
    l3_latency_ms: float       # 默认 0，非负
    l3_lookup_latency_ms: float # 默认 0，非负
  timing:
    mode: empirical            # manual 只支持经验计时
    prefill_c1: float
    prefill_c2: float           # 默认 0
    decode_d1: float
    decode_d2: float
```

`inference.mode=physical` 不接收手填 `kv`，而是接收完整 `model: ModelSpec`、`hardware: HardwareSpec`、`storage: StorageSpec`、`tp_size`、`ep_size` 和 `timing`。解析计时只传 `{"mode":"analytical"}`；经验计时传 `{"mode":"empirical","prefill_c1":...,"prefill_c2":...,"decode_d1":...,"decode_d2":...}`。两种推理分支都从 `inference.page_size` 读取唯一页大小。

### LengthDist（长度分布，可判别联合）

```yaml
dist: fixed | poisson | lognormal
value: int      # 仅 fixed：固定 token 数
mean: float     # 仅 poisson / lognormal：分布均值
sigma: float    # 仅 lognormal：形状参数，越大尾越长
```

开环只传 `arrival_rate=q`（期望请求/秒），全部 N 次投放各采样均值 `1000/q` ms 的指数间隔（含首请求）。闭环只传 `session_rate`（期望新会话/秒），根请求以该频率泊松到达；子轮由实际完成时刻和 `think_time_ms` 决定，最终请求 QPS 在结果中实测；未指定 `num_sessions` 时等于 `num_requests`。`max_total_tokens` 必须至少为一页、且为 `page_size` 的整数倍，直接表示实际 L1 KV 池容量；页数仅由二者相除得到，不另设取整或可写容量。合成负载按 `max_new_tokens` 与输入大小的安全余量截断实际输出目标；手工 `Arrival.max_new_tokens` 默认 128。采样、到达及继承见 [workload.md](workload.md)。跨 session 公共前缀不支持，同 session 父子轮仍可继承完整页；`session.content_seed` 和 `allow_cross_session_shared_prefix` 不在 schema 中，显式提交 `true`/`false` 等未知字段均返回 422，`meta.config` 不回显。`SimConfig` 不提供独立 `prefix_cache` 或未实现的校准模式；模型/硬件/存储均属于 `inference.mode=physical` 分支，手填 token 容量和链路参数属于 `inference.mode=manual` 的 `kv`。低层容量为零时保持原有 L1 行为；启用 L2 要求正的 KV 字节/token 和 L2→L1 带宽，启用 L3 还要求 L3→L2 带宽。配置的跨字段约束在 `SimConfig` 与直接构造 `Engine`/`run_engine` 时共用校验，非法请求返回 422。传输只按页大小和链路计时，不接入真实硬件。

## API

同步仿真端点：

```
POST /api/simulate
```

- 请求必须包含 `num_instances >= 1`；`scheduler` / `inference` 表示单实例配置，物理模式的 `inference.hardware.gpu_count` 也表示每实例卡数，总卡数为两者相乘。所有实例独占硬件与 KV。
- 同步执行整场仿真，无任务 ID、无持久化。
- 成功响应即使存在请求级失败仍为 200；三个结果列表均按请求 ID 升序，数量之和等于计划请求总数：

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
resolution:        # ResolutionReport | null；物理容量与可运行性报告
analysis:          # AnalyticalSummary | null；经验计时为 null
```

`aborted_requests.reason` 为 `input_exceeds_kv_capacity`、`decode_kv_exhausted`，或 trace 专用的 `invalid_trace_output`、`output_exceeds_kv_capacity`；`skipped_requests.reason` 为 `parent_not_completed`。未发送后代没有 arrival；`requests` 才供甘特图绘制。

- 请求体即 `SimConfig`，响应体即单个 `SimResult`，服务层不另行定义数据类型。`meta.config` 是包含字段默认值的原规范化输入，完整保留模型/硬件/存储与回放设置，不替换成推导后的运行时配置，不原地修改输入。
- `meta.wall_clock_ms` 仅度量运行阶段，不包含配置解析；它不是仿真时钟或 GPU 预测时长，也不属于确定性比较字段。除墙钟字段外，相同配置、计划与种子的仿真结果可复现；响应按内核 `SimResult` 序列化。

### 错误响应

| 状态码 | 场景 | 响应体 |
|---|---|---|
| `422` | config 校验失败（类型、范围、未知字段、缺少子轮增量分布、非整页容量、`mode` 与载荷不匹配）或配置业务错误 | 配置错误为 FastAPI 默认 `HTTPValidationError`；业务错误为 `{"detail": str}` |
| `404` | Trace 计划 ID 未知或已过期 | `{"detail": str}` |
| `501` | 不支持的计时组合，如 TP/EP>1 使用经验四系数 | `{"detail": str}` |

运行中的单请求容量不足不归入 HTTP 422，而是正常结果的 `aborted_requests`；其他请求可继续执行。开环模式不预建未来子轮，失败时下一节拍可创建新根；闭环模式的预建后代才进入 `skipped_requests`。

错误响应体属于 HTTP 边界，由服务层定义并登记进 OpenAPI，不进入 `core.schemas`。

`metrics` 包含 TTFT、TPOT、完成吞吐、makespan、L1/L2/L3 命中与驻留、驱逐/提升/降级及恢复字节和时间；`requests` 的分层命中只记录首次准入。计算与空值口径统一见 [metrics.md](metrics.md)。

## Trace 文件与计划接口

原始 `traces.jsonl` 可本地处理，也可通过 `POST /api/traces` 用 `application/octet-stream` 直接上传原始字节（无需 multipart，最大 3 GiB）；返回 202 和任务 ID。`GET /api/traces/{id}` 查 `processing/ready/failed`，`GET /api/traces/{id}/sessions?offset=0&limit=50` 分页查看实际扫描摘要。`POST /api/traces/{id}/plans` 接受 `{"sessions":["原始会话ID"]}`，下载 JSON 计划，最大 64 MiB；`POST /api/replay-plans` 直接上传本地导出的计划字节并返回计划 ID。这些端点只管理计划资源，保留原有上传、扫描、预览及下载路径。

回放将同一个 `SimConfig.workload` 设为 `{"mode":"trace","trace":{"mode":"open_loop","session_rate":1,"seed":0,"plan_id":"已上传计划ID"}}`，向唯一 `POST /api/simulate` 提交完整配置。服务层根据 ID 获取 `TracePlan`，调用 `simulate(config, trace_plan=plan)`；回放设置不再放在独立请求包装体中。手动/物理模式均返回单个 `SimResult`，解析计时保留 TP/EP 通信与 `analysis`，经验计时为 `analysis=null`。

`target_qps` 只允许在 `mode=open_loop` 时设置；先由 `session_rate` 生成会话起点，再按整个所选计划的首末发送跨度等比例缩放所有请求到达时间。单请求或所有请求同刻到达时返回 422；闭环携带该参数、非正数及非有限数也返回 422。不设置时沿用原开环时间；缩放不保证均匀投放或完成吞吐。

原始文件与计划暂存于服务进程的临时目录，闲置 24 小时后在后续访问时清理；服务进程退出后失效，当前版本不支持跨实例共享任务或断点续传。文件超限为 413，解析/校验失败为 422，处理中尝试导出为 409，未知或过期 ID 为 404。上传计划的块 ID 必须是会话局部一致的前缀身份；原始输入块参与运行时连续缓存匹配，不包含原始输出块身份。计划内容不嵌入 `SimConfig`；HTTP 运行使用 `workload.trace.plan_id` 引用独立资源。参数 JSON 导出将临时 `plan_id` 清为 `null`，保留开环/闭环、会话速率、种子与可选目标 QPS，导入后提示重新上传计划。

## 自动解析与 TP/EP 容量预览

- `GET /api/presets` 返回 Qwen3-8B、Qwen3-235B-A22B 和固定修订的 HY4-Preview 主干参考结构以及 B300 单卡显存、互联带宽标称值；不提供已测有效算力或模型修订认证，目录条目不等于已验证部署。B300 的 1800 GB/s 是单卡双向标称值；页面以 600 GB/s 作为未经校准的单 rank 有效带宽试算值，非实测值。
- `POST /api/resolve` 接受与运行完全相同的 `SimConfig`，返回 `ResolutionReport`，不产生另一套可提交的配置。物理模式的 `ModelSpec`、`HardwareSpec`、`StorageSpec`、`timing`、`tp_size`、`ep_size` 均位于 `inference`。TP 数须等于 GPU 数；Dense 使用 EP1，MoE 首版须 EP=TP；支持全注意力 GQA/MQA、按层数组合的 GQA/MLA/MLA+DSA BF16/FP16 粗估，以及以完整组件配置表示的 HY4-Preview 参考预设，参数可修改，不以名称或修订号锁定结构，容量和性能均复用通用路径。组数不限但层数之和须等于模型层数；组只描述注意力，FFN/MoE 按全模型独立配置，不支持需逐层顺序或可变页成本的注意力。MoE 须填写专家结构，各 MoE 层规格相同；所有驻留专家均匀放置到 EP ranks。L1 输入为**每卡**含权重可用预算，按 rank 扣除非专家、全部驻留专家权重与预留后取最紧整页容量；L2/L3 输入为**整组**仅放 KV 的物理容量。模型词表大小与结构维度必填；总参数、每 token 激活参数和全部专家参数由结构估算，BF16/FP16 全部驻留权重按参数矩阵和逐卡复制/分片关系计入，不再接受总参数或实际驻留字节覆盖。报告包含参数估算、每 rank TP 分片/复制权重、EP 专家权重、KV 字节与页数；可用 `norms`、`router`、`ihc` 和注意力组辅助字段配置 BF16/FP16/FP32 权重，类型及形状决定逻辑参数数，精度与放置决定驻留字节。辅助容量配置不扩展性能算子模型，未建模权重仍不计入，结果不能解释为真实部署占用。缺失结构、显存不足或不支持拓扑返回 422。缺少 TP/EP 通信计时参数仍可预览容量，但 `runnable=false`；齐备且为解析计时时才可运行。
- `ResolutionReport` 包含 `capacity: CapacityReport`、`max_total_tokens`、`l2_capacity_tokens`、`l3_capacity_tokens`、`runnable` 与可空 `reason`；报告只描述容量与可运行性，不替换原输入，不可作为 `simulate` 的参数。
- `POST /api/simulate` 在内核内一次性准备物理运行规格后执行；支持 Dense TP 与平均负载 MoE/EP 的解析计时，缺通信参数返回 422，TP/EP>1 经验四系数模式返回 501。成功返回单个 `SimResult`：`resolution` 给出逐 rank 容量与运行容量，`analysis` 分列 TP 同步、EP dispatch/combine 的时间与逻辑流量；这些费用已计入 `device_ms`，KV 恢复时间独立。422 `detail` 可为字段错误列表或业务错误字符串，OpenAPI 同时登记两种格式。前端不复用上次容量预览的 token 数作为运行配置；结构估算参数和权重占用未经部署核验，旧配置 JSON 结构会被拒绝。
- 合成/Trace 负载与手动/物理配置均使用唯一 `/api/simulate`。四系数仅在 `inference.timing.mode=empirical` 使用，解析模式的逐步估计见 [性能模型](perf-model.md)。

## 类型同步

schema 单一事实源为 `core` 中的 Pydantic 模型；FastAPI 自动生成 OpenAPI，前端用 `openapi-typescript` 生成 TS 类型。禁止双侧手写类型定义。

生成链路：

```
simcore.schemas → FastAPI OpenAPI → web/openapi.json → openapi-typescript → web/src/api/schema.d.ts
```

- 两个生成物均入库，在 `web/` 下执行 `npm run gen:api` 一次性重新生成；改动 `schemas` 或路由后必须重新生成并随同提交。
- 新鲜度校验：Python 测试比对 `web/openapi.json` 与当前模型；`npm run check:types` 比对 TS 类型与 `openapi.json`。
- 请求与响应共用同一组 schema（关闭 FastAPI 的 input/output 拆分），每个内核模型在 TS 中只有一个同名类型。
- 字段在 TS 中是否必填以生成的 OpenAPI 契约为准，不由是否存在 Python 默认值统一推断；例如 `PhysicalInferenceConfig.timing` 使用默认工厂，生成类型中可选。前端导入缺省字段按 schema 默认值补齐，不使用页面未经校准的试算默认值覆盖输入；判别字段（如 `mode`、`dist`）仍按生成契约提供，以支持联合类型收窄。
