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
      content_seed: int | null
      allow_cross_session_shared_prefix: false
engine:
  max_batch_size: int
  kv_token_budget: int           # 解析为 L1 block 容量
  prefix_cache:                  # 必填且始终开启，无 enabled 字段
    block_size_tokens: int
    cache_salt: str | null
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

规则：采样结果向上取整并钳制到 ≥ 1 token。`lognormal` 用于表达长尾负载（少量超长 prompt/output 对 KV 预算与调度的冲击）。

所有 synthetic workload 都生成 `WorkloadPlan` 并经模拟客户端提交。每个请求必须且只属于一个 session；单轮 session 用于表达没有后续轮次的最小 workload。`num_requests=N` 为总轮数、`num_sessions=S` 为根数，用户输入的 `arrival_rate=q` 是所有轮次的目标请求 QPS（正有限数）；计划先把 `N` 轮均衡分配给 `S` 个 session，并用 `lambda_root=q*S/N`（session/秒）的泊松过程采样根到达。子请求不预定到达时刻，实际到达由父请求完成时刻与 `think_time_ms` 决定。该换算仅给出持续注入时的名义长期平均速率，有限场实测 QPS 不保证等于 `q`；默认 `S=N` 时保持旧单轮到达率口径。

## API

MVP 唯一端点：

```
POST /api/simulate
```

- 同步执行整场仿真，无任务 ID、无持久化。
- 响应：

```yaml
meta:      # 配置回显、仿真时钟范围等
metrics:   # 见下
requests:  # 每个 session 轮次一条区间记录，供甘特图消费
  - id: int
    session_id: int
    parent_request_id: int | null
    arrival: float
    prefill_start: float
    prefill_end: float           # = decode_start
    finish: float
warnings:  # 如 KV 超限事件
```

- 请求体即 `SimConfig`，响应体即 `SimResult`，服务层不另行定义数据类型。
- 响应体字节与内核 `SimResult` 的 JSON 序列化一致：同种子两次请求的响应逐字节相同。

### 错误响应

| 状态码 | 场景 | 响应体 |
|---|---|---|
| `422` | config 校验失败（类型、范围、未知字段、`mode` 与载荷不匹配） | FastAPI 默认 `HTTPValidationError` |
| `501` | `workload.mode = trace`（预留接缝，MVP 未实现） | `{"detail": str}` |

错误响应体属于 HTTP 边界，由服务层定义并登记进 OpenAPI，不进入 `core.schemas`。

## Metrics 定义

- TTFT（= prefill_end − arrival）：均值 / P50 / P99
- TPOT（decode 阶段平均每 token 耗时）
- 实测端到端完成吞吐（总轮次数 / makespan），与输入的目标到达 QPS 不同
- makespan（末请求完成时刻 − 首根请求到达时刻）

计算口径：

- 分位数：线性插值（与 numpy `percentile` 默认 `linear` 方法一致），即对升序样本取位置 `(n−1)·q` 并在相邻两点间插值。
- TPOT：逐请求 `(finish − prefill_end) / (output_len − 1)`，再对请求取算术平均；`output_len = 1` 的请求无 decode 阶段，不计入。全部请求均为 `output_len = 1` 时为 `null`。
- 吞吐单位为 请求/秒（makespan 为 ms，换算后相除）；makespan 为 0 时吞吐无定义，为 `null`。
- 时间类指标单位均为 ms。

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
