# 当前引擎与调度

单 Engine、单 L1 KV 池使用 FCFS 连续调度、一次性 Prefill；不支持 chunked prefill。全局事件顺序见 [events.md](events.md)，块身份及物理容量见 [chunk-hash-encoding.md](chunk-hash-encoding.md) 与 [kv-prefix-reuse.md](kv-prefix-reuse.md)。

## 请求与两种执行模式

- 已发送请求按 `(发送时刻, 请求 ID)` 排队。`Engine` 持有实际输出目标 `O`，调度器只看到输入长 `P`、可见生成上限 `M` 和已生成数 `g`，不预知未来输出。计划先采样输出并截断到 `M=min(配置上限, max_total_tokens-ceil(P/B)*B-B)`；`M<1` 时在发送时记 `input_exceeds_kv_capacity`，而非拒绝整场仿真。
- `pd_mutually_exclusive=true`（默认）先尝试 Prefill；没有可执行 Prefill 才推进步首运行请求的 Decode，单步只有 `PREFILL` 或 `DECODE`。设为 `false` 时先确认步首 Decode 集合，再准入新请求，可产生 `MIXED`；本步刚准入的请求不参与本步 Decode。两模式共用 FCFS、KV 分配、回撤及结果规则。纯步和融合步的计时见 [perf-model.md](perf-model.md)。
- 首次 Prefill 一步算完尚未命中的输入，步末采样首 token；`O=1` 时直接完成。之后每个 Decode 步建立上次采样的一个输出 token 的 KV，再采样下一 token。因此生成 `O` 个输出时只建立 `P+O-1` 个 KV token；首次 `prefill_start/end` 与首 token 时刻在回撤重算后保持不变。完成请求在同刻的 `RequestDone` 事件才真正发布剩余完整页并释放 KV。

## 准入预算与回撤

`F001Scheduler` 逐 FCFS 队头做连续前缀 `probe` 和临时 `pin`；失败撤销引用，不越过队头。候选需要的实际新页 token 数 `I=ceil((目标长度-命中长度)/B)*B`，可用额度 `A=(free_pages+evictable_pages)*B`。本步即时预算包含已确定 Decode 新页需求、已选候选的 `I+B`（每人一页安全余量）；总预算再加新候选的剩余生成上限和运行请求按动态 `ratio` 折算的未来需求（均至多 4096 token）。加入候选后即时及总预算均不得超过保护命中页后的 `A`，运行请求数不得超过 `max_running_requests`。`max_prefill_tokens` 限制后续候选的新增计算额度，首个 Prefill 候选可超过该计算上限，但仍须通过 KV 预算。未来需求只是估算，不分配物理页。

`ratio` 初值 `0.7`，无回撤且执行 Decode 时逐步下降至 `0.098`；回撤后据剩余运行请求的生成进度重估，闲置时重置。执行前先按需驱逐无引用的旧页；Decode 缺页时按 `(已生成数升序, 原输入长降序, ID 升序)` 回撤，若仅余单请求仍无法推进则标记 `decode_kv_exhausted`。回撤释放私有 KV、保留已发布缓存和输出进度，重入队列后只重算到 `P+g-1`，不重采样。恢复全命中时可在调度边界直接转为运行态，下一步才 Decode。没有可执行 Prefill/Decode 时不安排空计算步；请求级失败与跳过后代的对外结果见 [api.md](api.md)。

当前不模拟网络传输、真实 GPU 并行或采样开销。
