# 服务层契约

## Config schema

```yaml
workload:
  mode: synthetic | trace        # 普通 simulate 仅支持 synthetic；trace 使用计划专用端点
  synthetic:
    num_requests: int            # 所有 session 的总轮次数
    arrival_rate: float | null   # 仅 open_loop 必填，目标请求 QPS
    session_rate: float | null   # 仅 session 必填，新会话泊松投放频率（会话/秒）
    arrival_mode: session | open_loop  # 省略为 session；网页默认 open_loop，可切换
    prompt_len: LengthDist       # 两种模式仅根轮输入分布
    turn_increment_len: LengthDist  # 两种模式必填，子轮新增内容分布
    output_len: LengthDist
    max_new_tokens: int          # 每请求可见生成上限，默认 128
    seed: int
    session:                     # 仅 session 模式使用；open_loop 不设置这些可选参数
      num_sessions: int          # 省略时等于 num_requests
      topology: linear
      think_time_ms: float       # 子轮思考时间的指数分布均值（ms），默认 0
engine:
  max_running_requests: int      # 必填，正整数
  max_total_tokens: int          # 必填，实际 KV token 容量
  max_prefill_tokens: int        # 默认 16384
  page_size: int                 # 默认 64
  pd_mutually_exclusive: bool    # 默认 true；false 启用同一步融合
  l2_capacity_tokens: int       # 默认 0，关闭 L2；须为整页倍数
  l3_capacity_tokens: int       # 默认 0，关闭 L3；须为整页倍数，要求 L2>0
perf_model:
  prefill_c1: float
  prefill_c2: float              # 默认 0
  decode_d1: float
  decode_d2: float
  kv_bytes_per_token: int | null # 默认 null；启用 L2 时为正整数
  l2_to_l1_bandwidth_bytes_per_ms: float | null # 默认 null，启用 L2 时为正
  l3_to_l2_bandwidth_bytes_per_ms: float | null # 默认 null，启用 L3 时为正
  l2_latency_ms: float          # 默认 0，非负
  l3_latency_ms: float          # 默认 0，非负
```

### LengthDist（长度分布，可判别联合）

```yaml
dist: fixed | poisson | lognormal
value: int      # 仅 fixed：固定 token 数
mean: float     # 仅 poisson / lognormal：分布均值
sigma: float    # 仅 lognormal：形状参数，越大尾越长
```

开环只传 `arrival_rate=q`（期望请求/秒），全部 N 次投放各采样均值 `1000/q` ms 的指数间隔（含首请求）。闭环只传 `session_rate`（期望新会话/秒），根请求以该频率泊松到达；子轮由实际完成时刻和 `think_time_ms` 决定，最终请求 QPS 在结果中实测；未指定 `num_sessions` 时等于 `num_requests`。`max_total_tokens` 必须至少为一页、且为 `page_size` 的整数倍，直接表示实际 L1 KV 池容量；页数仅由二者相除得到，不另设取整或可写容量。合成负载按 `max_new_tokens` 与输入大小的安全余量截断实际输出目标；手工 `Arrival.max_new_tokens` 默认 128。采样、到达及继承见 [workload.md](workload.md)。跨 session 公共前缀不支持，同 session 父子轮仍可继承完整页；`session.content_seed` 和 `allow_cross_session_shared_prefix` 不在 schema 中，显式提交 `true`/`false` 等未知字段均返回 422，`meta.config` 不回显。旧 `SimConfig` 未提供独立 `prefix_cache`、硬件或校准模式；自动规格单独走下文 `ResolvedRequest`。低层容量为零时保持原有 L1 行为；启用 L2 要求正的 KV 字节/token 和 L2→L1 带宽，启用 L3 还要求 L3→L2 带宽。配置的跨字段约束在 `SimConfig` 与直接构造 `Engine`/`run_engine` 时共用校验，非法请求返回 422。传输只按页大小和链路计时，不接入真实硬件。

## API

同步仿真端点：

```
POST /api/simulate
```

- 请求必须包含 `num_instances >= 1`；`engine` / `perf_model` 表示单实例配置，自动解析的 `hardware.gpu_count` 也表示每实例卡数，总卡数为两者相乘。所有实例独占硬件与 KV。
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
```

`aborted_requests.reason` 为 `input_exceeds_kv_capacity`、`decode_kv_exhausted`，或 trace 专用的 `invalid_trace_output`、`output_exceeds_kv_capacity`；`skipped_requests.reason` 为 `parent_not_completed`。未发送后代没有 arrival；`requests` 才供甘特图绘制。

- 请求体即 `SimConfig`，响应体即 `SimResult`，服务层不另行定义数据类型。
- 响应体字节与内核 `SimResult` 的 JSON 序列化一致：同种子两次请求的响应逐字节相同。

### 错误响应

| 状态码 | 场景 | 响应体 |
|---|---|---|
| `422` | config 校验失败（类型、范围、未知字段、缺少子轮增量分布、非整页容量、`mode` 与载荷不匹配）或配置业务错误 | 配置错误为 FastAPI 默认 `HTTPValidationError`；业务错误为 `{"detail": str}` |
| `501` | 普通 `/api/simulate` 直接传 `workload.mode = trace`（应改用计划入口） | `{"detail": str}` |

运行中的单请求容量不足不归入 HTTP 422，而是正常结果的 `aborted_requests`；其他请求可继续执行。开环模式不预建未来子轮，失败时下一节拍可创建新根；闭环模式的预建后代才进入 `skipped_requests`。

错误响应体属于 HTTP 边界，由服务层定义并登记进 OpenAPI，不进入 `core.schemas`。

`metrics` 包含 TTFT、TPOT、完成吞吐、makespan、L1/L2/L3 命中与驻留、驱逐/提升/降级及恢复字节和时间；`requests` 的分层命中只记录首次准入。计算与空值口径统一见 [metrics.md](metrics.md)。

## Trace 文件与计划接口

原始 `traces.jsonl` 可本地处理，也可通过 `POST /api/traces` 用 `application/octet-stream` 直接上传原始字节（无需 multipart，最大 3 GiB）；返回 202 和任务 ID。`GET /api/traces/{id}` 查 `processing/ready/failed`，`GET /api/traces/{id}/sessions?offset=0&limit=50` 分页查看实际扫描摘要。`POST /api/traces/{id}/plans` 接受 `{"sessions":["原始会话ID"]}`，下载 JSON 计划，最大 64 MiB；`POST /api/replay-plans` 直接上传本地导出的计划字节并返回计划 ID；`POST /api/replay-plans/{id}/simulate` 接受 `{ "config": SimConfig, "mode": "open_loop" | "session", "session_rate": 会话每秒, "seed": 非负整数 }`，同步返回 `SimResult`。此处 `config` 使用手动模式引擎和性能参数，synthetic 负载字段不参与计划运行；返回 `meta.config.workload.mode=trace`。

原始文件与计划暂存于服务进程的临时目录，闲置 24 小时后在后续访问时清理；服务进程退出后失效，当前版本不支持跨实例共享任务或断点续传。文件超限为 413，解析/校验失败为 422，处理中尝试导出为 409，未知或过期 ID 为 404。上传计划的块 ID 必须是会话局部一致的前缀身份；原始输入块参与运行时连续缓存匹配，不包含原始输出块身份。计划不能直接传入普通 `/api/simulate`。

## 自动解析与 TP/EP 容量预览

- `GET /api/presets` 返回 Qwen3-8B、Qwen3-235B-A22B 的参考结构以及 B300 单卡显存、互联带宽标称值；不提供已测有效算力或模型修订认证，目录条目不等于已验证部署。B300 的 1800 GB/s 是单卡双向标称值；页面以 600 GB/s 作为未经校准的单 rank 有效带宽试算值，非实测值。
- `POST /api/resolve` 接受 `SimConfig` 加显式 `ModelSpec`、`HardwareSpec`、`StorageSpec`、`timing`、`tp_size`、`ep_size`。TP 数须等于 GPU 数；Dense 使用 EP1，MoE 首版须 EP=TP；仅支持全注意力 GQA/MQA 与 BF16/FP16。MoE 须填写专家结构，各 MoE 层规格相同；所有驻留专家均匀放置到 EP ranks。L1 输入为**每卡**含权重可用预算，按 rank 扣除非专家、全部驻留专家权重与预留后取最紧整页容量；L2/L3 输入为**整组**仅放 KV 的物理容量。模型词表大小与结构维度必填；总参数、每 token 激活参数和全部专家参数由结构估算，BF16/FP16 全部驻留权重按参数矩阵和逐卡复制/分片关系计入，不再接受总参数或实际驻留字节覆盖。报告包含参数估算、每 rank TP 分片/复制权重、EP 专家权重、KV 字节与页数；未建模辅助权重不计入，结果不能解释为真实部署占用。缺失结构、显存不足或不支持拓扑返回 422。缺少 TP/EP 通信计时参数仍可预览容量，但 `runnable=false`；齐备且为解析计时时才可运行。
- `POST /api/simulate/resolved` **再次解析**物理输入后执行；支持 Dense TP 与平均负载 MoE/EP 的解析计时，缺通信参数返回 422，TP/EP>1 经验四系数模式返回 501。成功返回 `ResolvedRun`：`resolution` 回显逐 rank 容量和有效 `SimConfig`，`analysis` 分列 TP 同步、EP dispatch/combine 的时间与逻辑流量；这些费用已计入 `device_ms`，KV 恢复时间独立。自动接口的 422 `detail` 可为字段错误列表或业务错误字符串，OpenAPI 同时登记两种格式。前端不复用上次容量预览的 token 数作为运行配置；结构估算参数和权重占用未经部署核验，旧自动配置 JSON 中已移除的参数输入会被拒绝。
- 手填 `/api/simulate` 与自动解析请求使用相同的负载模式；四系数仅在 `timing=empirical` 使用，解析模式的逐步估计见 [性能模型](perf-model.md)。

## 类型同步

schema 单一事实源为 `core` 中的 Pydantic 模型；FastAPI 自动生成 OpenAPI，前端用 `openapi-typescript` 生成 TS 类型。禁止双侧手写类型定义。

生成链路：

```
simcore.schemas → FastAPI OpenAPI → web/openapi.json → openapi-typescript → web/src/api/schema.d.ts
```

- 两个生成物均入库，在 `web/` 下执行 `npm run gen:api` 一次性重新生成；改动 `schemas` 或路由后必须重新生成并随同提交。
- 新鲜度校验：Python 测试比对 `web/openapi.json` 与当前模型；`npm run check:types` 比对 TS 类型与 `openapi.json`。
- 请求与响应共用同一组 schema（关闭 FastAPI 的 input/output 拆分），每个内核模型在 TS 中只有一个同名类型。
- 有默认值的字段（如 `dist`、`mode`、`prefill_c2`）在 TS 中为必填：响应中恒存在；请求侧前端总是发送完整配置（表单字段均有默认值）。`dist` 必填也使 `LengthDist` 可在 TS 中按判别字段收窄。
