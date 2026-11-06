# 完整块身份

本文定义合成负载与 Trace 的完整块 ID。它是供缓存判等的**整数**，**不是真实 token 内容的哈希**；仅在本场缓存池内有效。计划只保存长度、拓扑和理想继承量；请求不物化 token ID 序列。计划与客户端接口见 [workload.md](workload.md)，物理缓存规则见 [kv-prefix-reuse.md](kv-prefix-reuse.md)。

## 编码格式

```text
chunk_id = (owner << 32) | offset        # prefix_hash.chunk_base(owner) + offset
```

- `owner`：归属域编号，不同域的 ID 不相交。`offset` 是域内偏移，同一域内尽量连续，便于按 `id >> 32` / `id & 0xFFFFFFFF` 还原归属与位置。
- 同一场仿真只有一种负载来源，合成与 Trace 的域编号不需要互相区分。
- `namespace` 不进入编码：单场内模型/tokenizer/salt 等维度固定，隔离边界由每场重新创建的缓存池承载。未来多场或多 namespace 共享同一 cache 时，必须给身份增加隔离域。
- ID 对内核其余部分保持不透明；禁止按 session ID、请求 ID、长度或命中率直接判定 cache hit，命中只能经各池的条目索引判等。

### 合成负载：会话域 + 页序号

- `owner = session_id`，第 `i` 个完整页的 ID 为 `chunk_base(session_id) + i`；请求只携带 `chunk_base`，`chunk_ids` 为空。
- 依据：合成会话是线性单链，子轮输入 = 父轮输入 + 父轮实际输出 + 新增量，同一会话的上下文只追加不改写，因此任意轮次的第 `i` 页都覆盖同一段 token `[i*B, (i+1)*B)`。父子轮自然共享前 `K_i` 页的 ID，无需传递继承前缀。
- 父轮只发布 `floor((P+O-1)/B)` 个完整页，而 `K_i` 恰为该数（或受子轮输入限制更小），子轮自第 `K_i` 页起的 ID 从未被任何祖先发布，不会产生虚假命中。同一会话任一时刻至多一个在途请求。
- 不支持跨 session 公共前缀：不同会话位于不同域；配置中不提供跨 session 共享开关，未知字段由 schema 校验拒绝。

### Trace：会话域内前缀树重编号 + 请求尾页域

- 输入完整页（`TraceTurn.hash_ids`）使用 `owner = session_id`。上传校验已保证同一会话内同一 `hash_id` 总对应相同的 `(位置, 前驱)`，会话内全部输入构成一棵前缀树。`Client` 在准备计划时调用 `prefix_tree_order` 一次：先序遍历该树，每个分叉点先编号被最多输入经过的子节点（重子优先），得到域内偏移。多数输入的块 ID 因此连续，分叉只让走非主分支的输入断开。
- 重编号只在加载时执行：某块首次出现后其后续块必然也是首次出现，二分定位新块起点后只为新块建节点，开销约为“新块数 + 请求数 × log(块数)”，不进入仿真热路径。原始 `hash_id` 的数值不参与缓存判等，跨会话同数字 `hash_id` 不会互相命中。
- 输入尾部不足一页的 token 与输出形成的完整页没有原始身份，使用 `owner = len(sessions) + request_id` 的请求私有域，偏移即页序号；不同请求的这些页互不相等，与原有语义一致。
- 请求携带整段输入页 ID `chunk_ids` 与尾页域 `chunk_base`：第 `i` 页 ID 为 `chunk_ids[i]`（`i < len(chunk_ids)`），否则为 `chunk_base + i`。`Client` 只按会话保存“原始 `hash_id` → 块 ID”映射，发送时再组装 `chunk_ids`；同一 ID 在各请求间共用一个 int 对象，在途请求之外不额外持有逐块元组。

## 数据契约

`TurnPlan` 保存输入长度、未截短的采样输出长度、生成上限与理想继承块数；生成上限按输出长度向上取二次幂并以 `inference.max_output_tokens` 封顶。`InferenceRequest` 携带 `prompt_len`、`max_new_tokens`、`output_len`、`chunk_ids`、`chunk_base` 及请求身份，不携带整条 token 序列。调度器虽持有真实输出长度，当前仍只以生成上限与进度估算未来需求，不使用真实长度作调度依据。成功完成即达到 `output_len`；协调器结算引擎后只向客户端通知请求 ID 与完成时间，无须单独回传生成数。`TieredKVCache._chunk_id` 按上述规则 O(1) 取页 ID，不格式化字符串。

- 记 `P=prompt_len`、成功完成的实际生成数 `O=output_len`、`B=page_size`。首次 Prefill 只建立 `P` 个 KV token 并采样第一个输出；每次后续 Decode 才建立上一输出 token。完成时建立长度 **`K=P+O-1`**，完整可继承页数为 `floor(K/B)`；逻辑 KV 总占页数为 `ceil(K/B)`，不足一页的私有尾页只占一页并在完成时回收。子轮最多继承 `min(floor((P_parent+O_parent-1)/B), floor((P_child-1)/B))` 页。
- 例如 `B=4, P=3, O=3`：Prefill 时尚无完整输入页；第一次 Decode 建立输出的第一个 token，形成序号 0 的完整页；第二次 Decode 后 `K=5`，剩余 1 token 是私有尾页。第三个（最后采样的）输出 token 不建立 KV。完成时才发布 Decode 形成的完整页，然后回收尾页。
- 配置不接受 `session.content_seed` 或跨 session 共享开关，未知字段由 schema 校验拒绝。每场新建缓存，不同仿真不共享 KV；结果不回显块 ID。

## 生成与发布时机

- `workload.build_synthetic_closed_loop_plan` 不预生成 ID；`Client.on_wakeup` 只为合成请求填入会话域 `chunk_base`，为 Trace 请求按会话映射组装 `chunk_ids` 并填入请求尾页域。
- `TieredKVCache` 由引擎在 `finish_step`（GPU 步完成事件）按请求 ID 发布 Prefill 已计算完整页，步首不公开；Decode 中新填满的完整页保持私有，`RequestDone` 调用引擎 `complete` 后由缓存 `finish` 发布，回撤则丢弃未发布私有页。输入尾页可由后续已建立输出 token 补齐，但未建立的末次采样 token 不形成页。连续前缀命中只比较已发布的完整页 ID；新计算私有页按实际计算需求分配，异步恢复页另经传输占位和完成迁入。

## 与 SGLang 的差异

SGLang 的前缀缓存按 token 内容在 radix tree 上匹配；本项目不模拟 token 内容，以上整数 ID 是仿真抽象，只决定“哪些页内容相同”。编号方式（连续与否）不改变命中关系，也不对应 SGLang 的任何内部数据结构。
