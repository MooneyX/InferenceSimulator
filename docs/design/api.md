# 服务层契约

## Config schema

```yaml
workload:
  mode: synthetic | trace        # trace 分支 MVP 返回未实现错误
  synthetic:
    num_requests: int            # 所有 session 的总轮次数
    arrival_rate: float          # 所有轮次的目标请求 QPS（请求/秒），根到达率由 S/N 换算
    prompt_len: LengthDist       # 每轮输入长度，见下
    output_len: LengthDist       # 每轮输出长度
    seed: int
    session:
      num_sessions: int          # 根请求数，1 <= num_sessions <= num_requests
      topology: linear
      think_time_ms: float
      target_ideal_prefix_hit_rate: float
engine:
  max_batch_size: int
  kv_token_budget: int           # 解析为 L1 block 容量
  prefix_cache:                  # 输入可省略；解析后始终存在，无 enabled 字段
    block_size_tokens: int
    cache_salt: str
    eviction_policy: lru
    hierarchy_policy: exclusive_write_back
    l1:
      capacity_blocks: int
    l2:
      capacity_blocks: int
      read_bandwidth_bytes_per_ms: float
      fixed_latency_ms: float
    l3:
      capacity_blocks: int
      read_bandwidth_bytes_per_ms: float
      fixed_latency_ms: float
    kv_bytes_per_token: int
    model_id: str
    tokenizer_id: str
    kv_dtype: str
    adapter_id: str
    multimodal_transform_id: str
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

输入配置中的 `arrival_rate` 表示所有轮次的目标请求 QPS，`num_sessions` 缺省时等于 `num_requests`；不表示完成吞吐。采样、到达计划及理想前缀继承量的内部算法见 [workload.md](workload.md)。跨 session 公共前缀不支持，配置中没有共享开关；同 session 父子轮次仍可继承完整块。`session.content_seed` 亦不在 schema 中，显式提交这些额外字段（包括布尔值 `true` 或 `false`）均因未知字段返回 422，响应 `meta.config` 不回显。`l1.capacity_blocks` 由 `floor(kv_token_budget/block_size_tokens)` 解析；显式输入不一致时报校验错误。L2/L3、`kv_bytes_per_token` 等字段已有 schema，但现行引擎仅使用 L1；低层配置非零也不意味着低层已启用。

## API

MVP 唯一端点：

```
POST /api/simulate
```

- 同步执行整场仿真，无任务 ID、无持久化。
- 响应：

```yaml
meta:      # 配置回显、仿真时钟范围等
metrics:   # 汇总指标；计算口径见 metrics.md
requests:  # 每个 session 轮次一条区间记录，供甘特图消费
  - id: int
    session_id: int
    parent_request_id: int | null
    arrival: float
    prefill_start: float
    prefill_end: float           # = decode_start
    finish: float
warnings:  # 成功结果中的非致命告警；无法容纳的请求返回 422 而非 warnings
```

- 请求体即 `SimConfig`，响应体即 `SimResult`，服务层不另行定义数据类型。
- 响应体字节与内核 `SimResult` 的 JSON 序列化一致：同种子两次请求的响应逐字节相同。

### 错误响应

| 状态码 | 场景 | 响应体 |
|---|---|---|
| `422` | config 校验失败（类型、范围、未知字段、`mode` 与载荷不匹配）；理想命中目标不可行；或单请求超过 L1 可执行容量（OOM） | 配置错误为 FastAPI 默认 `HTTPValidationError`；计划与 OOM 错误为 `{"detail": str}` |
| `501` | `workload.mode = trace`（预留接缝，MVP 未实现） | `{"detail": str}` |

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
