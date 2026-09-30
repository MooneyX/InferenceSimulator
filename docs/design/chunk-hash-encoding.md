# 合成负载的完整块身份

本文只定义当前合成负载的完整块 ID。它是供缓存判等的不透明结构化字符串，**不是真实 token 内容的哈希**；仅在单场唯一请求 ID、独立 cache、同 session 父子继承的范围内有效。计划只保存长度、拓扑和理想继承量；请求不物化 token ID 序列。计划与客户端接口见 [workload.md](workload.md)，物理缓存规则见 [kv-prefix-reuse.md](kv-prefix-reuse.md)。

## 编码格式

```text
chunk:{request_id}:{block_index}
```

- `request_id`：生成该**新块**的请求 ID；要求在同一 cache 生命周期内唯一。使用固定分隔符与十进制非负整数规范表示，避免拼接歧义。仅由 ID 和序号编码的**新块**在此范围内严格唯一，无需依赖 64 位派生值的无碰撞假设。
- `block_index`：块在该请求连续逻辑上下文（input + output）中的 0 基序号；完整块 i 覆盖 token 区间 `[i*B, (i+1)*B)`，输出不从块 0 重新编号。输入尾部的不足块可以由输出补齐；最终仍不足 `B` 的尾块保留有效 token 数和物理占用，但不生成可缓存 ID，也不进入 `hash_index`。
- 子轮次的前 `K_i` 个完整继承块**整体拷贝父请求的字符串**，不以子请求 ID 重新编码；从块 `K_i` 起按子请求 ID 和对应块序号编码。即使父块本身继承自更早的祖先，也继续拷贝其原字符串。父请求未填满的尾块不继承；不同请求的新块不会相等，继承块则刻意相等。
- 不支持跨 session 公共前缀：`allow_cross_session_shared_prefix=true` 在此方案下须明确拒绝，不能静默退化为不共享；如需重新支持，应另行定义相应身份来源与理想命中规则。
- `namespace` 不进入编码：单场内模型/tokenizer/salt 等维度固定，隔离边界由独立 cache 实例承载。未来多场或多 namespace 共享同一 cache 时，必须给身份增加隔离域，不能直接复用上述格式。

hash 对内核其余部分保持不透明字符串；禁止按 session ID、请求 ID、长度或命中率直接判定 cache hit——请求 ID 只用于新块身份**生成**，命中**判定**仍只能经 `hash_index` 判等。

## 数据契约

`TurnPlan` 保存长度与理想继承块数；`InferenceRequest` 保存显式的 `prompt_len`、`output_len` 和完整的 `input_chunk_ids`；`RequestCompletion` 只保存请求 ID 与完成时间。计划和请求都不携带输出块 ID。

- 记 `P=prompt_len`、`O=output_len`、`B=block_size_tokens`、`I=P//B`、`N=(P+O)//B`。`input_chunk_ids` 恰含 `I` 个完整输入块 ID，供 `probe`、准入校验和 prefill 发布使用；请求不携带、计划也不预生成输出块 ID。运行期间填满的输出块数为 `N-I`，序号为 `I` 到 `N-1`（包括由输入尾部与输出拼成的块），并不一定等于 `O//B`。
- 输入末尾不足 `B` 时，物理 KV 仍分配 `ceil(P/B)` 块，其中尾块持有 `P%B` 个有效 token；输出先填充此块，填满才生成并发布序号 `I` 的 ID。完成时若 `(P+O)%B != 0`，最终尾块保留有效 token 数并在释放时回收，不生成可复用 ID。物理容量和未来增长仍按 `ceil((P+O)/B)` 规划，不能用 `N` 代替物理块数。例如 `B=4, P=3, O=3`：输入 ID 数为 0，产出第 1 个输出 token 时形成序号 0 的完整块，最后 2 个 token 是无可复用 ID 的尾块。
- 输入/输出长度是显式字段；配置中不接受 `session.content_seed`，每场使用新 cache，不同仿真即使请求 ID 相同也不会共享 KV。结果不回显内部块 ID；`allow_cross_session_shared_prefix=true` 在计划构造时明确报不支持。

## 生成与发布时机

- `workload.build_plan` 不预生成块 ID。`ClientDriver.on_wakeup` 复制已完成父上下文的前 `K_i` 个完整块 ID，再对其余输入块调用 `encode(request_id, i)`；根请求所有完整输入块由自己编码。`on_request_completed` 使用已发送请求的输入块、已知 `P`、`O`、`B`，在完成回调后重建序号 `P//B` 到 `(P+O)//B-1` 的输出块，供子轮次继承。
- 引擎对本步算完的完整输入块在 Prefill 步末发布；每个 output token 先计入物理 KV，若已产出 `t` 个且 `(P+t)%B==0`，步末发布 `encode(request_id, (P+t)//B-1)`。输入尾块可由输出补齐，最终不足块不发布。引擎在线发布与客户端完成后重建使用同一编码规则，不交换预生成的输出 ID。
- `HierarchicalKVCache` 仅以块 ID 做连续前缀判等；`scheduler` 通过物理容量另行处理未满块与未来输出增长。相同块 ID 可用于测试重复索引健壮性，但生成新块时请求 ID 必须唯一。若今后真实输出长度或内容可能偏离计划，需要同时重新定义完成消息和客户端—引擎身份协议，不能仅替换编码函数。
