# 当前经验性能模型

`core/src/simcore/perf_model.py` 使用配置中的 `prefill_c1`、`prefill_c2`、`decode_d1`、`decode_d2`（单位 ms）作为纯函数输入。它估算**仿真时间**，不是实测 GPU 时间或已标定的硬件模型。

## 单项成本

对每个首次或回撤恢复 Prefill 项，新计算的 token 数 `n >= 1`、此前已复用前缀长度 `c >= 0`：

```text
prefill_cost(c, n) = prefill_c1*n + prefill_c2*((c+n)^2 - c^2)
decode_step_time(T) = decode_d1 + decode_d2*T
```

首次准入时 `c=actual_hit_tokens`、`n=prompt_len-c`；即使完整输入块都命中也至少重算最后一个输入 token。回撤恢复到 `P+g-1` 时按实际命中与重算长度再次计费。二次项包含历史上下文增量成本；`T` 只统计**本步 Decode 请求**各自处理上一输出后逻辑上下文长度 `P+g` 之和，不能因为共享物理 KV 而缩短。

## 当前单步工作量与时长

`Engine.step` 构造不可变 `StepWorkload`：阶段为 `PREFILL`、`DECODE` 或 `MIXED`；Prefill 项按请求 ID 排序，含 `request_id/reused_tokens/new_tokens/context_after`，Decode 项含 `request_id/context_after_decode`；还记录本步实际新申请页数及两类请求总数。阶段与非空集合须一致，两集合不得重叠。默认 `pd_mutually_exclusive=true` 仅有纯 Prefill/Decode 步；设为 `false` 才可同时计两类工作量，新准入请求本步不参与 Decode。

`workload_time` 对 Prefill 项顺序求和 `P_cost=Σ(prefill_c1*n+prefill_c2*n*(2*c+n))`；Decode 集合非空时算 `D_cost=decode_d1+decode_d2*T`，否则为 0。纯步取对应阶段成本，`MIXED` 取 `max(P_cost,D_cost)`，**不相加**。零系数仍推进有限工作量的状态，空工作量不执行计算步。此 `max` 是经验模型对融合执行的计时近似，不代表真实设备并行、分块 Prefill 或已校准耗时。解析/校准及 `StepCostBreakdown` 尚未实现，见 [F002 计划](../plans/perf-model.md)。
