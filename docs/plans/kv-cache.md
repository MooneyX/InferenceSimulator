# KV 缓存模块实施计划（KV3）

**状态：待实现。** 当前 [L1 缓存设计](../design/kv-prefix-reuse.md) 已接通；配置中存在 `l2/l3` 容量、读带宽、固定延迟、`kv_bytes_per_token` 以及内部 tier 字段，但 Engine 只创建 L1，尚无跨层迁移、restore 或实际低层命中。非零低层配置不是能力证明。F001 的 chunked prefill/retraction **调用编排**由 [调度计划](scheduler-engine.md) 负责，不属于 KV3 的实现范围。

## 目标与接口边界

保留 `WorkloadPlan`、合成完整块 ID、现有 `snapshot/probe/prepare_batch/commit/publish/advance/release` 的 block 级语义；在缓存内部实现 L1/L2/L3 独占式 write-back。容量为零的层关闭；本期要求层级连续启用（`L3>0` 必须 `L2>0`），否则在配置层拒绝。新块进入 L1，L1 淘汰块在启用 L2 时降至 L2，L2 淘汰块在启用 L3 时降至 L3，最终层淘汰时删除。低层 hit 提升至 L1、移除原层位置；同一个物理块在同刻只属于一层，每层独立稳定 LRU，删除时清理反向索引。不能把迁移描述为当前引擎行为。

## 分模块工作

1. **`schemas` 与 cache 模型**：启用 L2/L3 容量、带宽、固定延迟和 KV bytes/token 的实际约束；补齐分层驻留、命中来源和迁移记录，同时维持 L1 预算的唯一来源与块容量守恒。
2. **`kv_cache`**：全局目录记录块实际 tier；按连续输入 ID 做 L1→L2→L3 查找，同层仍稳定选取物理 block ID；每个候选基于同一 snapshot，批事务中对低层命中预留 L1 空间并原子地去重提升。同一低层块被多请求引用时只传输一次，建立多份 L1 引用；失败不得留下部分提升、错误 LRU 或旧索引。降级被迁入层视作最近使用，层满递归驱逐。
3. **`engine` 与 `perf_model`**：引擎保持完成时的发布/释放时序，向成本函数传唯一迁移物理块和各层 bytes；restore 首期与计算串行，不假设重叠。restore 只经过连续启用的层级；禁用层不发生传输或计费，启用的传输链路带宽须大于零：

   ```text
   L2_restore = L2_latency + L2_bytes / L2_to_L1_bandwidth
   L3_restore = L3_latency + L3_bytes / L3_to_L2_bandwidth
                + L2_latency + L3_bytes / L2_to_L1_bandwidth
   ```

   `L2=0、L3>0` 在配置层拒绝，避免无定义的 L3→L1 链路和零带宽相除。

   `step_time = lookup_time + restore_time + compute_time`；身份查找 CPU 成本若不建模，`lookup_time` 为零。`compute_time` 由当时的阶段工作量和性能后端给出：F001 上线后使用 P/D 互斥的 `StepWorkload`，不把旧融合式 max 当作新调度语义。
4. **`metrics`、服务和界面**：分层 token 命中率、promotion/demotion/eviction、驻留峰值、restore bytes/time 和理想/实际差额；tier 只归因于 probe 时的来源，最后一个输入 token 重算时从相应层命中量扣除。外部 schema/界面工作在 [服务计划](server-api.md)、[前端计划](frontend.md) 跟踪。

## 验收

同 ID 的多物理块、多请求同批命中、禁用层不计费（特别是 L2=0、L3>0 必须明确拒绝）、跨层迁移与原子回滚都保持稳定；每个物理块仅占一层，每层的 `free + resident (+ future reservation) = capacity` 成立，分层实际命中 token 之和等于跳过的 Prefill token。L2/L3 容量均为零时，与当前 L1 语义和结果保持一致；不能以预留字段存在或模拟的非零配置代替功能验收。
