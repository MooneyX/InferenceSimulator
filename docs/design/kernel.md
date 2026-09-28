# 仿真内核设计

## 仿真层级

LLM Serving 系统级：请求到达 → 调度 → prefill/decode → continuous batching → KV cache 约束。单 GPU。

## DES 事件模型：引擎步骨架

离散事件仿真，事件队列仅承载三类事件：

- `Arrival`：请求到达
- `EngineStep`：引擎迭代步。batch 非空时自再调度下一步；batch 为空时不压事件，时钟直接跳跃到下一个 `Arrival`
- `RequestDone`：请求完成结算

`EngineStep` 内执行：调度决策（准入）→ 推进 batch 内所有请求的 decode → 结算完成请求。一次事件推进整个 batch，使堆操作次数 ≈ 引擎迭代数而非请求步数。

**约束**：性能模型的步长计算必须是纯函数（无全局状态依赖），为后续内核替换（如 Rust）预留。

### 请求生命周期

- 准入步：执行 prefill，`prefill_start` = 步起点，`prefill_end` = 步终点，同时产出首 token。
- 其后每个 decode 步产出 1 token，全部 `output_len` 个 token 产出后完成；`output_len = 1` 的请求在准入步结束即完成。
- batch = 全部 KV 驻留请求。完成请求在步末释放，故每步开始时驻留请求均处于 decode 阶段。

## 调度与 KV cache 语义

- 调度策略：FCFS，受 `max_batch_size` 与 KV 预算双重约束。
- KV 占用：token 预算制（`kv_token_budget`）。prefill 占用 prompt 长度额度；decode 每步逐请求 +1 token。
- 预算耗尽：禁止新请求准入；在跑请求继续执行，不做 OOM stall。配置应保证不超限，超限事件记入 `warnings`。
- 严格 FCFS：队头放不下时停止本步准入，不跳过队头去准入后续较小请求。
- 超大请求强制准入：prompt 长度超过整个 `kv_token_budget` 的请求永远无法正常准入。为避免队头死锁，当 batch 为空且队头为此类请求时强制准入，并记入 `warnings`。

## 性能模型

系数均来自 config，单位 ms：

- prefill 耗时 = `prefill_c1 · prompt_tokens + prefill_c2 · prompt_tokens²`（`prefill_c2` 默认 0）
- decode 单步耗时 = `decode_d1 + decode_d2 · batch_total_tokens`

### 融合批步长

一个 `EngineStep` 内可同时存在新准入请求的 prefill 与在跑请求的 decode。步长取两者最大值：

```
step_time = max(Σ prefill 耗时（本步准入的全部请求）, decode 单步耗时)
```

语义：prefill 与 decode 在同一次迭代中融合执行（chunked-prefill 式），而非串行相加。`batch_total_tokens` 计入本步准入请求的 prompt token。

除排队外不建模其他延迟来源（无网络、无采样开销）。

## 工作负载接口

```python
class WorkloadSource(Protocol):
    def __iter__(self) -> Iterator[Arrival]: ...  # 按 sim_time 升序
```

- 生成器只向事件队列供给 `Arrival` 序列，永不接触引擎。
- MVP 唯一实现：`SyntheticSource`——泊松到达 + prompt/output 长度各自按 `LengthDist` 配置采样（fixed / poisson / lognormal，见 [api.md](api.md)）+ 固定种子。
- 仿真启动时将到达事件全部物化压入队列（未来大规模 trace 可改惰性供给，接口不变）。
- `TraceSource`（真实 trace replay）仅预留接口与 config 字段，MVP 不实现、不设计细节。
