# 服务层契约

## Config schema

```yaml
workload:
  mode: synthetic | trace        # trace 分支 MVP 返回未实现错误
  synthetic:
    num_requests: int
    arrival_rate: float          # 泊松到达率（请求/秒）
    prompt_len: LengthDist       # 见下
    output_len: LengthDist
    seed: int
engine:
  max_batch_size: int
  kv_token_budget: int           # KV cache token 预算
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
requests:  # 每请求一条区间记录，供甘特图消费
  - id: int
    arrival: float
    prefill_start: float
    prefill_end: float           # = decode_start
    finish: float
warnings:  # 如 KV 超限事件
```

## Metrics 定义

- TTFT（= prefill_end − arrival）：均值 / P50 / P99
- TPOT（decode 阶段平均每 token 耗时）
- 端到端吞吐（请求数 / makespan）
- makespan（末请求完成时刻 − 首请求到达时刻）

## 类型同步

schema 单一事实源为 `core` 中的 Pydantic 模型；FastAPI 自动生成 OpenAPI，前端用 `openapi-typescript` 生成 TS 类型。禁止双侧手写类型定义。
