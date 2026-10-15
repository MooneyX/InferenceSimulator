# 当前引擎与调度

每个 Engine 独占请求 KV 管理器和 L1/L2 资源池，所有实例共享本场唯一 L3 资源池，使用 FCFS 连续调度、一次性 Prefill；多个同构实例在同一全局时钟上并行步进。根请求轮询分配，子请求严格送回父轮实例，过载也不改道；缓存亲和不保证实际命中。低层容量默认为零，不支持 chunked prefill。全局事件顺序见 [events.md](events.md)，块身份及物理容量见 [chunk-hash-encoding.md](chunk-hash-encoding.md) 与 [kv-prefix-reuse.md](kv-prefix-reuse.md)。

## 请求与两种执行模式

- 已发送请求按 `(发送时刻, 请求 ID)` 排队。`InferenceRequest` 携带实际输出目标 `O=output_len`；引擎与调度器持有同一请求，但当前调度只能依据输入长 `P`、生成上限 `M`、已生成数 `g` 和 KV 状态，不以 `O` 作为选批、预算或回撤依据。预知输出长度的调度方案及对比暂不实现。合成采样或 Trace 原始输出 `O` 都视为已发生的结果，不按 KV 容量或生成上限截短；客户端在发送时构造 `M=min(next_power_of_two(max(1,O)), inference.max_output_tokens)`，并将 `O` 与 `M` 一起放入请求提交。正输出超过引擎上限时记请求级 `output_exceeds_engine_limit`；Trace 零输出仍为 `invalid_trace_output`，两者不拒绝整场仿真。
- `pd_mutually_exclusive=true`（默认）就绪准入批次的 Prefill 和只恢复项优先于 Decode；设为 `false` 时，在选定准入批次后准备步首已有请求的 Decode，可产生 `MIXED`；新准入或恢复的请求不参加本步 Decode。GPU 工作量只有 `PREFILL`、`DECODE`、`MIXED`；L3 查询、预取和 L2 回载由独立异步事件推进，不生成纯 `RESTORE` GPU 步。两模式等待回载时均可推进已有请求的 Decode；本步刚准入或恢复的请求不参与本步 Decode，恢复准入不采样新 token。两模式共用 FCFS、KV 分配、回撤及结果规则。正常执行统一经 `Coordinator → start_step → 内部 _execute_step(selection 必填) → finish_step → complete`，没有公共 `Engine.step` 同步入口。计时见 [perf-model.md](perf-model.md)。
- 首次 Prefill 一步算完尚未命中的输入，步末采样首 token；`O=1` 时直接完成。之后每个 Decode 步建立上次采样的一个输出 token 的 KV，再采样下一 token。因此生成 `O` 个输出时只建立 `P+O-1` 个 KV token；首次 `prefill_start/end` 与首 token 时刻在回撤重算后保持不变。完成请求在同刻的 `RequestDone` 事件才真正发布剩余完整页并释放 KV。

## 准入预算与回撤

`F001Scheduler` 逐 FCFS 队头做跨层连续前缀 `probe` 和临时 `pin`；失败撤销引用，不越过队头。候选需要的实际新计算页 token 数 `I=ceil((目标长度-命中长度)/B)*B`，可用额度 `A=(L1.free+L1.evictable)*B`。选批即时预算包含候选的 `I+B`（每人一页安全余量）和本批**唯一低层命中页**提升所需 L1 空间；总预算再加候选剩余生成上限和运行请求按动态 `ratio` 折算的未来需求（均至多 4096 token）。低层来源页受 pin 保护不可用于降级。候选通过预算后仍用缓存 `plan` 预检全批提升与新计算所需容量；预检失败撤销当前 pin，不改变 LRU 和迁移计数。预检不提交同步提升，实际回载经 `reserve_transfer` 占位、`complete_transfer` 在完成事件迁入；驱逐/降级由 `ensure_free` 同步腾位。运行请求数不得超过 `max_running_requests`。`max_prefill_tokens` 仅限制后续候选的新计算量，首个 Prefill 候选可超过该计算上限；提升页不算新计算 token。未来需求仍按可见生成上限、4096 token 封顶与 `ratio` 估算，不读取真实输出目标、不分配物理页。空闲且尚未选中候选时，若即时预算（含一页增长余量与提升空间）能容纳，则不因未来估算偏大永久拒绝；不可运行证明仅检查即时物理需求，不包含未来项，实际 Decode 耗尽仍按请求级失败处理。

`ratio` 初值 `0.7`，无回撤且执行 Decode 时逐步下降至 `0.098`；回撤后据剩余运行请求的生成进度重估，闲置时重置。执行前先按需驱逐无引用的旧页；Decode 缺页时按 `(已生成数升序, 原输入长降序, ID 升序)` 回撤，若仅余单请求仍无法推进则标记 `decode_kv_exhausted`。回撤释放私有 KV、保留已发布缓存和输出进度，重入队列后只重算到 `P+g-1`，不重采样。恢复全命中且无需跨层回载时，可在调度边界直接转为运行态；若需回载则等待独立页传输完成后再恢复准入，不采样新 token。混合步的 `restore_only` 请求 ID 只标记此类恢复准入，不新增计算或输出；只有恢复没有 Prefill/Decode 计算时不生成 GPU 步，就绪请求在后续 Decode 才继续生成。无计算工作时跳到下一查询、传输或客户端事件；请求级失败与跳过后代的对外结果见 [api.md](api.md)。

当前模拟分层 KV 上行页传输，不模拟请求网络传输、真实 GPU 并行或采样开销。
