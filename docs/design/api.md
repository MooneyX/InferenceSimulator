# 服务层契约

## Config schema

```yaml
workload:
  mode: synthetic | trace        # trace 请求返回 501
  synthetic:
    num_requests: int            # 所有 session 的总轮次数
    arrival_rate: float          # 所有轮次目标 QPS（请求/秒）
    prompt_len: LengthDist
    output_len: LengthDist
    max_new_tokens: int          # 每请求可见生成上限，默认 128
    seed: int
    session:
      num_sessions: int          # 省略时等于 num_requests
      topology: linear
      think_time_ms: float       # 子轮思考时间的指数分布均值（ms），默认 0
      target_ideal_prefix_hit_rate: float
engine:
  max_running_requests: int      # 必填，正整数
  max_total_tokens: int          # 必填，实际 KV token 容量
  max_prefill_tokens: int        # 默认 16384
  page_size: int                 # 默认 64
  pd_mutually_exclusive: bool    # 默认 true；false 启用同一步融合
perf_model:
  prefill_c1: float
  prefill_c2: float              # 默认 0
  decode_d1: float
  decode_d2: float
```

### LengthDist（长度分布，可判别联合）

```yaml
dist: fixed | poisson | lognormal
value: int      # 仅 fixed：固定 token 数
mean: float     # 仅 poisson / lognormal：分布均值
sigma: float    # 仅 lognormal：形状参数，越大尾越长
```

`arrival_rate` 是所有轮次的目标请求 QPS，`num_sessions` 缺省时等于 `num_requests`；不表示完成吞吐。`max_total_tokens` 必须至少为一页、且为 `page_size` 的整数倍，直接表示实际 L1 KV 池容量；页数仅由二者相除得到，不另设取整或可写容量。合成负载按 `max_new_tokens` 与输入大小的安全余量截断实际输出目标；手工 `Arrival.max_new_tokens` 默认 128。采样、到达及继承见 [workload.md](workload.md)。跨 session 公共前缀不支持，同 session 父子轮仍可继承完整页；`session.content_seed` 和 `allow_cross_session_shared_prefix` 不在 schema 中，显式提交 `true`/`false` 等未知字段均返回 422，`meta.config` 不回显。现行输入**没有** `prefix_cache`、L2/L3、KV bytes/token、硬件或校准模式配置；仅支持 L1，结果中的低层字段为零值预留。

## API

MVP 唯一端点：

```
POST /api/simulate
```

- 同步执行整场仿真，无任务 ID、无持久化。
- 成功响应即使存在请求级失败仍为 200；三个结果列表均按请求 ID 升序，数量之和等于计划请求总数：

```yaml
meta:              # config 回显；num_requests / num_completed / num_aborted /
                   # num_skipped / num_engine_steps；sim_start / sim_end（ms）
metrics:           # 汇总指标及无成功请求时的空值，见 metrics.md
requests:          # 仅成功请求；含 id、session_id、parent_request_id、
                   # arrival / prefill_start / prefill_end / finish、输入输出长度及命中/重算量
aborted_requests:  # 已发送但失败：id、arrival、failed_at、reason、
                   # generated_tokens、retry_count
skipped_requests:  # 父轮未完成、从未发送：id、parent_request_id、reason
warnings:          # 非致命告警；不代替请求级失败状态
```

`aborted_requests.reason` 为 `input_exceeds_kv_capacity` 或 `decode_kv_exhausted`；`skipped_requests.reason` 为 `parent_not_completed`。未发送后代没有 arrival；`requests` 才供甘特图绘制。

- 请求体即 `SimConfig`，响应体即 `SimResult`，服务层不另行定义数据类型。
- 响应体字节与内核 `SimResult` 的 JSON 序列化一致：同种子两次请求的响应逐字节相同。

### 错误响应

| 状态码 | 场景 | 响应体 |
|---|---|---|
| `422` | config 校验失败（类型、范围、未知字段、非整页容量、`mode` 与载荷不匹配）或理想命中目标不可行等计划构造错误 | 配置错误为 FastAPI 默认 `HTTPValidationError`；计划构造错误为 `{"detail": str}` |
| `501` | `workload.mode = trace`（预留接缝，MVP 未实现） | `{"detail": str}` |

运行中的单请求容量不足不归入 HTTP 422，而是正常结果的 `aborted_requests`；其他请求可继续执行，失败父轮的后代进入 `skipped_requests`。

错误响应体属于 HTTP 边界，由服务层定义并登记进 OpenAPI，不进入 `core.schemas`。

`metrics` 包含 TTFT、TPOT、完成吞吐、makespan 及 L1 缓存命中/容量字段；计算与空值口径统一见 [metrics.md](metrics.md)。

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
