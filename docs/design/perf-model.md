# 当前经验性能模型与 KV 恢复计时

`core/src/simcore/perf_model.py` 使用 `prefill_c1`、`prefill_c2`、`decode_d1`、`decode_d2`（单位 ms）及可选的 KV 上行链路参数作为纯函数输入。它估算**仿真时间**，不是实测 GPU 时间或已标定的硬件模型。

## 单项成本

对每个首次或回撤恢复 Prefill 项，新计算的 token 数 `n >= 1`、此前已复用前缀长度 `c >= 0`：

```text
prefill_cost(c, n) = prefill_c1*n + prefill_c2*((c+n)^2 - c^2)
decode_step_time(T) = decode_d1 + decode_d2*T
```

首次准入时 `c=actual_hit_tokens`、`n=prompt_len-c`；即使完整输入块都命中也至少重算最后一个输入 token。回撤恢复到 `P+g-1` 时按实际命中与重算长度再次计费。二次项包含历史上下文增量成本；`T` 只统计**本步 Decode 请求**各自处理上一输出后逻辑上下文长度 `P+g` 之和，不能因为共享物理 KV 而缩短。

## 当前单步工作量与时长

`Engine.step` 构造不可变 `StepWorkload`：阶段为 `PREFILL`、`DECODE`、`MIXED`，或只有上行传输的 `RESTORE`。Prefill/Decode 项按请求 ID 排序；还记录新申请页数、批内请求及唯一恢复页来源。默认 `pd_mutually_exclusive=true` 的恢复项按 Prefill 侧执行，不与 Decode 同步；融合模式可同时恢复和 Decode，新恢复请求本步不参加 Decode。

`workload_time` 对 Prefill 项顺序求和 `P_cost=Σ(prefill_c1*n+prefill_c2*n*(2*c+n))`；Decode 集合非空时算 `D_cost=decode_d1+decode_d2*T`，否则为 0。计算成本取 `max(P_cost,D_cost)`，**不相加**。令 `n2/n3` 分别为本步唯一从 L2/L3 提升的页数、`S=page_size*kv_bytes_per_token` 字节，则：

```text
restore_bytes = (n2 + 2*n3) * S
restore_time_ms = [n2+n3>0]*l2_latency_ms + (n2+n3)*S/l2_to_l1_bandwidth_bytes_per_ms
                + [n3>0]*l3_latency_ms + n3*S/l3_to_l2_bandwidth_bytes_per_ms
step_time_ms = max(P_cost, D_cost) + restore_time_ms
```

链路固定延迟每步每条被使用的链路只计一次；L3 页经 L3→L2→L1 两链路，不在 L2 永久驻留。`RESTORE` 纯传输步的计算耗时为零；lookup 与逐级降级只计元数据及次数，不计写回传输耗时。低层关闭时新增成本全为零，原有两种 P/D 计时保持不变。此模型不代表真实设备并行或已校准耗时。

## 单 GPU 解析计时

`POST /api/simulate/resolved` 在 Dense 全注意力 GQA/MQA、BF16/FP16、单 GPU 条件下，以已解析的 `ModelSpec` 和 `HardwareSpec` 使用 `analytical.step_cost` 替代上述四系数整步计算。投影、因果注意力、FFN 各阶段分别取 `kernel_overhead_ms + max(FLOPs/effective_FLOPS, HBM_bytes/effective_HBM_BW)`，逐层顺序相加；融合步汇总本步 Prefill 和 Decode 的工作量、权重每层计一次。纯恢复步不产生算子成本，低层恢复仍按相同的 KV 字节数和链路计费并另加到步长。自动模式把物理链路带宽解析为 token/ms，`restore_time_tokens` 按迁移页的 token 数和两段链路时延计费；`restore_bytes` 仍使用派生的 KV bytes/token 供指标展示。结果包装中的 `analysis` 汇总执行步数、计算量、HBM 访问与解析耗时；该模型不包括 embedding/输出词表投影、形状校准或 TP/EP 通信，不得解释为真实设备测量值。未选择解析模式的旧 API 仍执行本文件前述经验计时。TP、EP 及进一步校准见 [F002 计划](../plans/perf-model.md)。
