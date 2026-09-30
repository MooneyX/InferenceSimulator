# 当前经验性能模型

`core/src/simcore/perf_model.py` 使用配置中的 `prefill_c1`、`prefill_c2`、`decode_d1`、`decode_d2`（单位 ms）作为纯函数输入。它估算**仿真时间**，不是实测 GPU 时间或已标定的硬件模型。

## 单项成本

对于新计算的输入 token 数 `n >= 1` 和已复用前缀长度 `c >= 0`：

```text
prefill_cost(c, n) = prefill_c1*n + prefill_c2*((c+n)^2 - c^2)
prefill_time(P) = prefill_cost(0, P)
decode_step_time(T) = decode_d1 + decode_d2*T
```

缓存命中时 `c=actual_hit_tokens`、`n=prompt_len-c`；即使整个输入块都命中也至少重算最后一个输入 token。二次项保留历史上下文的增量成本，不能将未命中后缀当作独立的短 prompt。`T` 为当前批请求的**逻辑**上下文 token 总量，共享物理 KV 不会缩短 Decode 的逻辑上下文。

## 当前步长

当前引擎可在同一个 step 准入新请求并推进已有请求的 Decode。对准入请求顺序累加 `prefill_cost`；仅当步首已有 Decode 工作时计算 `decode_step_time(T)`，否则 Decode 项为 0。步长取两项的最大值，而不是求和；空步不执行。调度器给性能函数传实际复用前缀、新计算 token 数和逻辑上下文，并把所得时长交给引擎与协调器推进时钟。

这个 `max(prefill, decode)` 是**当前的重叠计时近似**：每个新请求仍在一个 step 内完成其全部剩余 Prefill，不表示已经实现 chunked prefill 或 Prefill/Decode 分离。未来的分阶段 `StepWorkload` 和解析模型分别由 [调度计划](../plans/scheduler-engine.md) 与 [性能模型计划](../plans/perf-model.md) 负责，不能写入当前公式。
