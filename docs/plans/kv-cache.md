# KV 缓存模块实施计划（KV3）

**状态：待实现。** F001 的全局 `chunk_id` 缓存索引、请求级 `K/F`、容量、发布和回撤规则由 [调度与引擎实施规范](scheduler-engine.md) 唯一定义；本计划仅扩展后续的 L2/L3 容量、迁移与 restore，不作为 F001 的额外实现条件。

## 目标与接口边界

沿用 `WorkloadPlan`、合成完整块 ID 与唯一的 `engine.page_size`；KV3 在全局 `chunk_id → 缓存项` 中增加实际层次，沿用请求级 `K/F`、引用计数和按需分配接口，两种 P/D 模式共用同一层级缓存。在缓存内部实现 L1/L2/L3 独占式 write-back。容量为零的层关闭；层级连续启用（`L3>0` 必须 `L2>0`），否则在配置层拒绝。新完整页进入 L1，L1 驱逐时向 L2 降级，L2 驱逐时向 L3 降级，最终层驱逐时删除；低层命中提升至 L1 并移除原层驻留。同一缓存项同刻只归属一层，每层独立稳定 LRU，删除时清理全局索引；只模拟所在层及所占容量，不记录真实 GPU/主机地址。

## 分模块工作

1. **`schemas` 与 cache 模型**：启用 L2/L3 容量、带宽、固定延迟和 KV bytes/token 的实际约束；补齐分层驻留、命中来源和迁移记录，同时维持 L1 预算的唯一来源与块容量守恒。
2. **`kv_cache`**：全局索引记录完整 `chunk_id` 的实际 tier 与引用数；按输入 ID 的逻辑顺序做 L1→L2→L3 连续查找。逐请求接纳时将待提升的低层命中计入本批 L1 预算并保护来源，实际载入时按 `chunk_id` 去重；同一低层缓存项被多请求引用只计一次传输和驻留，分别增加引用。失败不得留下部分提升、错误 LRU 或旧索引。降级块在迁入层视作最近使用，层满递归驱逐；无需为仿真构造物理 block ID。
3. **`engine` 与 `perf_model`**：引擎保持完成时的发布/释放时序，向成本函数传唯一迁移 `chunk_id` 集合及各层 bytes；restore 首期与计算串行，不假设重叠。restore 只经过连续启用的层级；禁用层不发生传输或计费，启用的传输链路带宽须大于零：

   ```text
   L2_restore = L2_latency + L2_bytes / L2_to_L1_bandwidth
   L3_restore = L3_latency + L3_bytes / L3_to_L2_bandwidth
                + L2_latency + L3_bytes / L2_to_L1_bandwidth
   ```

   `L2=0、L3>0` 在配置层拒绝，避免无定义的 L3→L1 链路和零带宽相除。

   `step_time = lookup_time + restore_time + compute_time`；身份查找 CPU 成本若不建模，`lookup_time` 为零。`compute_time` 由 F001 的 `StepWorkload` 和性能后端给出：纯 Prefill/Decode 步按各自阶段计时，融合步采用 `max(prefill_time, decode_time)`；两种模式都按实际 KV 分配和迁移计费。
4. **`metrics`、服务和界面**：分层 token 命中率、promotion/demotion/eviction、驻留峰值、restore bytes/time 和理想/实际差额；tier 只归因于 probe 时的来源，最后一个输入 token 重算时从相应层命中量扣除。外部 schema/界面工作在 [服务计划](server-api.md)、[前端计划](frontend.md) 跟踪。

## 验收

同批首次计算同 ID 的临时私有页最终去重为一个缓存项、多请求同批命中、禁用层不计费（特别是 L2=0、L3>0 必须明确拒绝）、跨层迁移和失败后的资源清理都保持稳定；每个缓存项只在一层驻留，每层的 `free + resident = capacity` 成立，未来输出估算额度不得记作已占容量；分层实际命中 token 之和等于跳过的 Prefill token。L2/L3 容量均为零时，两种 P/D 模式的结果分别与单层 L1 一致；非零层级需由实际迁移与 restore 路径验证。
