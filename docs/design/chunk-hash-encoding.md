# 合成负载的完整块身份

本文只定义当前合成负载的完整块 ID。它是供缓存判等的不透明结构化字符串，**不是真实 token 内容的哈希**；仅在单场唯一请求 ID、独立 cache、同 session 父子继承的范围内有效。计划只保存长度、拓扑和理想继承量；请求不物化 token ID 序列。计划与客户端接口见 [workload.md](workload.md)，物理缓存规则见 [kv-prefix-reuse.md](kv-prefix-reuse.md)。

## 编码格式

```text
chunk:{request_id}:{block_index}
```

- `request_id`：生成该**新块**的请求 ID；要求在同一 cache 生命周期内唯一。使用固定分隔符与十进制非负整数规范表示，避免拼接歧义。仅由 ID 和序号编码的**新块**在此范围内严格唯一，无需依赖 64 位派生值的无碰撞假设。
- `block_index`：块在该请求连续逻辑上下文（input + output）中的 0 基序号；完整块 i 覆盖 token 区间 `[i*B, (i+1)*B)`，输出不从块 0 重新编号。输入尾部的不足块可以由输出补齐；最终仍不足 `B` 的尾块保留有效 token 数和物理占用，但不生成可缓存 ID，也不进入 `hash_index`。
- 子轮次的前 `K_i` 个完整继承块（全部可继承完整页）**整体拷贝父请求的字符串**，不以子请求 ID 重新编码；从块 `K_i` 起按子请求 ID 和对应块序号编码。即使父块本身继承自更早的祖先，也继续拷贝其原字符串。父请求未填满的尾块不继承；不同请求的新块不会相等，继承块则刻意相等。
- 不支持跨 session 公共前缀：不同根请求的新块 ID 相互隔离，只有同 session 的父子轮次显式继承完整块 ID；配置中不提供跨 session 共享开关，未知字段由 schema 校验拒绝。如需重新支持，应另行定义相应身份来源与理想命中规则。
- `namespace` 不进入编码：单场内模型/tokenizer/salt 等维度固定，隔离边界由独立 cache 实例承载。未来多场或多 namespace 共享同一 cache 时，必须给身份增加隔离域，不能直接复用上述格式。

hash 对内核其余部分保持不透明字符串；禁止按 session ID、请求 ID、长度或命中率直接判定 cache hit——请求 ID 只用于新块身份**生成**，命中**判定**仍只能经 `hash_index` 判等。

## 数据契约

`TurnPlan` 保存输入长度、采样后截断的目标输出长度、可见生成上限与理想继承块数；客户端在投放时将目标输出长度单独交给引擎，调度器不读取它。`InferenceRequest` 携带 `prompt_len`、`max_new_tokens`、`inherited_prefix_ids`（仅继承的完整页 ID）及请求身份，不携带目标 `output_len` 或整条 `input_chunk_ids`。`RequestCompletion` 携带请求 ID、完成时间和实际 `generated_tokens`。新页始终通过 `logical_chunk_id(request_id, index, inherited_prefix_ids)` 按需推导，无需物化整条身份序列。

- 记 `P=prompt_len`、实际生成数 `O=generated_tokens`、`B=page_size`。首次 Prefill 只建立 `P` 个 KV token 并采样第一个输出；每次后续 Decode 才建立上一输出 token。完成时建立长度 **`K=P+O-1`**，完整可继承页数为 `floor(K/B)`；逻辑 KV 总占页数为 `ceil(K/B)`，不足一页的私有尾页只占一页并在完成时回收。子轮最多继承 `min(floor((P_parent+O_parent-1)/B), floor((P_child-1)/B))` 页。
- 例如 `B=4, P=3, O=3`：Prefill 时尚无完整输入页；第一次 Decode 建立输出的第一个 token，形成序号 0 的完整页；第二次 Decode 后 `K=5`，剩余 1 token 是私有尾页。第三个（最后采样的）输出 token 不建立 KV。完成时才发布 Decode 形成的完整页，然后回收尾页。
- 配置不接受 `session.content_seed` 或跨 session 共享开关，未知字段由 schema 校验拒绝。每场新建缓存，不同仿真即使请求 ID 相同也不共享 KV；结果不回显块 ID。

## 生成与发布时机

- `workload.build_plan` 不预生成 ID；`Client.on_wakeup` 从已完成父轮的继承前缀和父轮请求 ID 按页序号推导所需的前缀 ID，只发送这段 `inherited_prefix_ids`。根轮的该元组为空，新输入/输出页由引擎用当前请求 ID 和页序号按需编码。完成回调保存实际建立的 `P+generated_tokens-1` 长度，供子轮发送时再推导可继承页，而不是复制整条父轮块序列。
- `TieredKVCache` 由引擎在 `finish_step`（GPU 步完成事件）按请求 ID 发布 Prefill 已计算完整页，步首不公开；Decode 中新填满的完整页保持私有，`RequestDone` 调用引擎 `complete` 后由缓存 `finish` 发布，回撤则丢弃未发布私有页。输入尾页可由后续已建立输出 token 补齐，但未建立的末次采样 token 不形成页。连续前缀命中只比较已发布的完整页 ID；新计算私有页按实际计算需求分配，异步恢复页另经传输占位和完成迁入。
