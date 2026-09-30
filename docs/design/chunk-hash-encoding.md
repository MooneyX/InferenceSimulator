# 块身份直接编码：取消 token 物化与链式 SHA-256

本文档锁定合成负载中 chunk ID 的生成方式：chunk ID 是仅供判等的不透明结构化字符串，由请求 ID 与块序号直接编码，不再物化 token id 序列，不再做链式 SHA-256。缓存仍仅索引完整块；不足 `B` 的尾块须按逻辑长度及物理占用处理，不得误作可复用块。适用前提是每场仿真的 `request_id` 唯一、每场使用独立 cache，且不跨 session 复用（`allow_cross_session_shared_prefix=false`）。本文档修订 `kv-prefix-reuse.md` 中 PC1-A 的 `prefix_hash.py` 约定、token 内容构造约定，以及 PC3A-A 中 `RequestCompletion` 返回实际 `output_token_ids` 的约定；上述前提下其余缓存、调度、时序与指标语义不变。

## 1. 结论

- 完整块 ID 的消费者是 `HierarchicalKVCache` 的 `hash_index` 判等与 `probe` 的前缀连续匹配；token id 在当前合成仿真中无其他语义。因此身份编码直接产出字符串，token id 序列（`input_token_ids`、`output_token_plan`、`output_token_ids`）从数据契约中删除。
- 编码**不链式**：新块身份不依赖前驱块的 ID。共享仅源于同 session 子轮次从块 0 连续复制父请求的完整块前缀；其余块以唯一 `request_id` 隔离。在此合成构造下，非预期共享被排除，前缀命中语义与原链式构造一致；该等价性不适用于任意真实内容。
- 块身份编码**不经过 SHA-256**：身份由结构化字符串直接表达，可读、可解析，测试中可直接看出来源请求。`stable_derive` 的摘要用途（其他随机流派生种子）保留，不属于块身份编码。
- `build_plan` 仅规划长度、拓扑与理想复用块数，不预生成输出块 ID。客户端在发送时形成完整输入块 ID，收到完成消息后再构造已生成的完整输出块 ID，供后继请求继承；引擎独立地在输出块实际填满时按同一规则生成 ID 并及时发布到缓存。

## 2. 编码格式

```text
chunk:{request_id}:{block_index}
```

- `request_id`：生成该**新块**的请求 ID；要求在同一 cache 生命周期内唯一。使用固定分隔符与十进制非负整数规范表示，避免拼接歧义。仅由 ID 和序号编码的**新块**在此范围内严格唯一，无需依赖 64 位派生值的无碰撞假设。
- `block_index`：块在该请求连续逻辑上下文（input + output）中的 0 基序号；完整块 i 覆盖 token 区间 `[i*B, (i+1)*B)`，输出不从块 0 重新编号。输入尾部的不足块可以由输出补齐；最终仍不足 `B` 的尾块保留有效 token 数和物理占用，但不生成可缓存 ID，也不进入 `hash_index`。
- 子轮次的前 `K_i` 个完整继承块**整体拷贝父请求的字符串**，不以子请求 ID 重新编码；从块 `K_i` 起按子请求 ID 和对应块序号编码。即使父块本身继承自更早的祖先，也继续拷贝其原字符串。父请求未填满的尾块不继承；不同请求的新块不会相等，继承块则刻意相等。
- 不支持跨 session 公共前缀：`allow_cross_session_shared_prefix=true` 在此方案下须明确拒绝，不能静默退化为不共享；如需重新支持，应另行定义相应身份来源与理想命中规则。
- `namespace` 不进入编码：单场内模型/tokenizer/salt 等维度固定，隔离边界由独立 cache 实例承载。未来多场或多 namespace 共享同一 cache 时，必须给身份增加隔离域，不能直接复用上述格式。

hash 对内核其余部分保持不透明字符串；禁止按 session ID、请求 ID、长度或命中率直接判定 cache hit——请求 ID 只用于新块身份**生成**，命中**判定**仍只能经 `hash_index` 判等。

## 3. 数据契约变更

```text
TurnPlan(..., prompt_len, output_len, ideal_reuse_blocks)
                                      # 删除 unique_suffix_seed / output_token_plan；不存输出块 ID
InferenceRequest(request_id, session_id, parent_request_id,
                 prompt_len, output_len, input_chunk_ids,
                 ideal_reuse_blocks)  # 删除 input_token_ids / output_token_plan；仅携带完整输入块 ID
RequestCompletion(request_id, finish_time_ms)      # 删除 output_token_ids
```

- 记 `P=prompt_len`、`O=output_len`、`B=block_size_tokens`、`I=P//B`、`N=(P+O)//B`。`input_chunk_ids` 恰含 `I` 个完整输入块 ID，供 `probe`、准入校验和 prefill 发布使用；请求不携带、计划也不预生成输出块 ID。运行期间填满的输出块数为 `N-I`，序号为 `I` 到 `N-1`（包括由输入尾部与输出拼成的块），并不一定等于 `O//B`。
- 输入末尾不足 `B` 时，物理 KV 仍分配 `ceil(P/B)` 块，其中尾块持有 `P%B` 个有效 token；输出先填充此块，填满才生成并发布序号 `I` 的 ID。完成时若 `(P+O)%B != 0`，最终尾块保留有效 token 数并在释放时回收，不生成可复用 ID。物理容量和未来增长仍按 `ceil((P+O)/B)` 规划，不能用 `N` 代替物理块数。例如 `B=4, P=3, O=3`：输入 ID 数为 0，产出第 1 个输出 token 时形成序号 0 的完整块，最后 2 个 token 是无可复用 ID 的尾块。
- 输入/输出长度改为显式字段，所有 `len(input_token_ids)` / `len(output_token_plan)` 用法改为读字段。`unique_suffix_seed` 从 `TurnPlan` 移除；`session.content_seed` 从配置契约与对外 API 删除，不再派生合成内容种子，也不兼容传入此字段的旧请求。不同仿真即使有相同请求 ID，也不会共享 cache。
- 对外 API 删除 `session.content_seed`，结果模型仍不回显 token/hash；`allow_cross_session_shared_prefix=true` 需在计划构造时明确报不支持。`api.md`、`frontend.md` 和 `kv-prefix-reuse.md` 中保留该字段的旧约定均以本文为准，实施时同步更新配置模型、API schema 与前端。

## 4. 生成时机与各方职责

### `workload.py`

- `build_plan` 只生成拓扑、长度、理想复用块数与发送计划，不预生成任何 output chunk ID；`K_i=ideal_reuse_blocks` 的上限仍由父请求的完整逻辑块数 `floor((P_parent+O_parent)/B)` 与子请求的可复用完整输入块数决定，命中块总量恰为 `K*` 的口径不变。
- 旧长度入口 `plan_from_arrivals` 转换为独立根 session（`K=0`）的约定不变；须保持输入 `request_id` 在该场内唯一。

### `client.py`

- `ClientDriver` 不再持有 token 序列，改为保存已完成请求的**完整块 ID 上下文**。`on_wakeup` 对根请求编码 `encode(request_id, i)`（`0 <= i < P//B`）；对子请求复制已完成父上下文的前 `K_i` 个 ID，再编码自身的 `i=K_i..P//B-1`。只将所得 `input_chunk_ids` 及长度交给 engine；父请求未填满的尾块不复制，子请求未填满的输入尾块也不生成 ID。
- `on_request_completed` 校验该请求在途，使用已发送请求的 `P`、计划 `O` 和 `B`，**此时才**构造 `output_chunk_ids = tuple(encode(request_id, i) for i in range(P//B, (P+O)//B))`，并保存 `finished_context_ids = input_chunk_ids + output_chunk_ids` 供子轮次使用；最终不足 `B` 的尾块不进入该序列。只有完成回调后才登记子轮次 `finish_time_ms + think_time_ms` 的发送事件。
- 当前合成引擎严格产出计划的 `output_len` 个 token；`RequestCompletion` 只传 ID 与完成时间，此时客户端用已知长度重建输出块身份。若未来真实输出长度或内容可能不同，须扩展完成消息及身份协议，不能据计划长度臆造输出块。

### `engine.py` / `prefix_hash.py`

- `prefix_hash.py` 的 `chunk_hashes`、`next_chunk_hash`、`cache_namespace` 删除，替换为共享的 `encode(request_id, block_index)`；`stable_derive` 保留用于其他随机流。engine 删除 `_last_hash` 与步内链式 SHA-256，但仍须在完整输出块形成时实时编码 ID，不能等待客户端收到完成消息才发布。
- 发布规则与时序不变：prefill 末批量发布 `request.input_chunk_ids[probe.matched_blocks:]`；每产出一个 output token 先推进 KV，当已产出 `t` 个且 `(P+t)%B==0` 时，在该步末发布 `encode(request_id, (P+t)//B-1)`。首个 output token 仍在 prefill 步产生；输入尾块与输出跨界填满时按同一序号发布。结束时不足 `B` 的尾块仅保留有效 token 与容量占用，在 `release` 时回收，绝不发布。发布位置由 `HierarchicalKVCache` 的既有账本维护。
- 引擎在线发布的每个输出块 ID，与客户端完成回调后按 `i=P//B..(P+O)//B-1` 重建的对应项逐字节一致；两方只依赖相同的 `request_id`、`P`、`B` 及已生成数量，不交换预生成的输出 ID。完成时已发布的块经释放后可供子请求命中；同一步仍维持快照隔离。

### `kv_cache.py` / `scheduler.py`

- 缓存接口与语义不变：`probe` 前缀连续匹配、遇 miss 停止、同 ID 取最小稳定 block ID；`publish` 仅接受完整块；准入校验「`input_chunk_ids` 数等于 `P//B`」保留。调度器传递请求的输入 ID 给 `probe` 与 `AdmissionAllocation`，同时按原有 `ceil` 口径预留输入尾块及输出增长所需物理容量。

## 5. 边界与预留

- 重复 ID 健壮性测试不再注入 hasher，改为直接构造相同字符串；测试应覆盖父/祖先继承、跨 session 无共享、`P%B` 与 `O%B` 的各种组合、输入尾块由输出填满、完成时最终尾块无 ID、客户端重建与引擎在线发布一致，以及 `think_time_ms=0` 时子请求可见父块。
- 该身份表示合成负载的预设内容关系，**不是真实 token 内容哈希**。未来 trace replay 若读取真实内容，可由真实链式 hash 构造身份；若输出长度或内容可能偏离计划，还需扩展完成消息及 `client`/`engine` 的身份协议，不能只替换编码函数。`kv_cache`/`scheduler` 仍以不透明字符串判等。
- 性能意图：消除大整数 token 元组的内存占用与步内 SHA-256/序列化开销；完整输出块形成时进行一次短字符串编码，批量发布成本仍随块数增长；最终尾块依然占用物理块，不能当作零开销。
- 本变更在上述适用前提内不改变：`ideal_reuse_blocks` 分配、命中率口径（`matched`/`actual`/`ideal`）、发布时序、同 step 可见性、LRU 与容量守恒、可复现性（同配置与 seed 逐字节一致）。
