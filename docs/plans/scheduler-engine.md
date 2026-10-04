# 调度与引擎实施计划（F001）

**状态：待实现。本文是 F001 最终实现的唯一行为规范。** 配置、公式、事件顺序与边界均以本文为准；其他文档仅记录相关模块的交付分工，不需查阅外部项目或其他实现才能完成本期功能。F001 一次性交付配置、单步模型、L1 缓存、请求结果、API 和前端，两种 P/D 模式共用同一套准入、分配与释放规则。

## 范围和术语

范围为单实例、单 KV 池、普通 decoder-only 请求、FCFS（按发送时刻及请求 ID 稳定排序）、L1 前缀复用、单时钟顺序执行。`engine.pd_mutually_exclusive=true` 为默认的 Prefill 优先、单步 P/D 互斥；`false` 为同一步处理新请求 Prefill 与步首已有请求 Decode 的融合模式。两种模式的区别仅在单步工作量组合、容量检查顺序及计时；没有设备并行或异步结果处理。无 chunked prefill：每次 Prefill 完成当前请求尚未算出的整段上下文；首个候选可超过本批计算 token 上限，但不能突破物理容量判断。

页是仿真中的 KV 容量单位，不代表真实 GPU 地址；`B=engine.page_size` 个 token 占一个模拟页。KV 池容量为 `C=floor(max_total_tokens/B)*B` token，即 `C/B` 页。`free` 是未占用容量，`evictable` 是无活动引用、可逐出的已驻留完整页容量；`A=free+evictable` 是准入可用量，**不是**已经空闲的容量。生成 token 数与已建立 KV token 数不同：采样结果须到后续 Decode 才建立 KV。每次调度只依赖已发送请求和当前缓存状态，不可读取未来请求或预采样的最终输出长度。

首期通过 `chunk:{request_id}:{block_index}` 标识新完整页；继承页保留祖先 ID，不同根请求的新页不共享。每页索引 `i` 覆盖逻辑 token 位置 `[iB,(i+1)B)`；仅已实际计算且填满的页进入共享索引。真实 token 内容、跨独立请求共享、radix tree、chunked prefill、CPU/GPU overlap、多级缓存和推测解码不在本期。

## 状态、配置与单步契约

- 请求状态分为 `waiting/running/finished/aborted/skipped`。内部 `ClientDriver` 发送请求时只需提交不可由本请求 ID 推导的**继承前缀完整页 ID**（按位置排列、不可变）；引擎持有这一份输入身份直到完成或失败。逻辑输入的后续新页，以及所有已生成输出的完整页 ID，都由 `(request_id, 页序号)` 按需构造，不额外保存整条输入或输出的 ID 列表，也不要求客户端在回撤后重新提交。请求另记录 ID、发送时间、逻辑输入长度 `P`、可见生成上限 `M`、已生成输出数 `g`、首次 Prefill/首 token 时间和回撤次数。对逻辑页序号 `i`，若 `i` 小于继承前缀页数则读取该位置继承 ID，否则生成 `chunk:{request_id}:{i}`；这也是 probe、发布、释放和回撤重算使用的唯一身份规则。KV 管理器按请求 ID 维护已建立长度 `K` 与已发布且被该请求引用的连续前沿 `F`；私有页数和尾页使用量由 `K/F` 推导，不维护请求级 GPU 物理地址页表。页占用按整数页计，输入和输出长度按 token 计。
- `StepWorkload` 是不可变的计费快照：`phase=PREFILL|DECODE|MIXED`、按请求 ID 稳定排序的 Prefill `(request_id, reused_tokens, new_tokens, context_after)`、Decode `(request_id, context_after_decode)`、本步新分配页数、批内总请求数。`PREFILL` 要求仅 Prefill 集合非空，`DECODE` 要求仅 Decode 集合非空，两者均非空才为 `MIXED`；`pd_mutually_exclusive=true` 时 `MIXED` 非法。新 Prefill 请求不在本步 Decode，`output_len=1` 的请求在含其 Prefill 的步末完成。
- 本期沿用四系数经验成本，单位均为 ms：`P_cost=Σ[c1*n + c2*n*(2*h+n)]`，其中每个 Prefill 请求复用 `h` 个 token、新计算 `n≥1` 个 token，按请求 ID 顺序累加；`D_cost=d1+d2*T`，其中 `T=Σ_(q∈Decode)(P_q+g_q)`：`g_q` 是本步执行前已采样的输出数，等于本步处理上一输出 token 后的逻辑上下文长度。纯步取对应成本，`MIXED` 取 `max(P_cost,D_cost)`，不将两项相加。`T` 只统计 Decode 请求，Prefill 请求的成本由 `P_cost` 计入。空步不得推进时钟；两项系数都为零时仍须按有限工作量推进状态。

### 配置命名与输入契约

| 首期字段 | 归属与含义 | 决议 |
|---|---|---|
| `engine.max_running_requests` | 同时驻留的运行请求数上限，不是单批 Prefill 请求数 | 正整数，显式提供 |
| `engine.max_total_tokens` | 单实例 KV 池的 token 容量上限，不是整卡实时 HBM 余量 | 必填正整数；有效容量按 `page_size` 向下取整 |
| `engine.max_prefill_tokens` | 单次 Prefill 批新计算输入的 token 额度，与 KV 池容量分开 | 正整数，默认 16384；无分块时批中首请求可超额 |
| `engine.page_size` | 每个物理 KV 页的 token 数，首期也用作完整块 ID 的身份粒度 | 正整数，仿真器默认 64；可显式设为 1 或其他受支持粒度 |
| `engine.pd_mutually_exclusive` | P/D 执行方式 | 布尔值，默认 `true`：Prefill 与 Decode 分步；`false`：同一步允许 Prefill 新请求并推进步首运行请求 Decode |
| `workload.synthetic.max_new_tokens` | 合成请求的生成上限，调度器可知 | 正整数，默认 128；与抽样得到的实际 `output_len` 分开 |

`engine` 是容量、页大小和 P/D 开关的唯一配置源；`workload.synthetic.max_new_tokens` 是每个合成请求可见的生成上限，手工 `Arrival.max_new_tokens` 为可选正整数（默认 128）。`C=floor(max_total_tokens/page_size)*page_size` 必须 ≥ `page_size`；响应回显输入配置与 `effective_max_total_tokens=C`，不另外接受可编辑的物理页数。`page_size=64` 为仿真器默认，页大小可以设为 1；使用已发布的完整页 ID 逐页查连续前缀，匹配长度只能是页大小的整数倍。令输入长度为 `P`，至多匹配 `floor((P-1)/B)*B` 个 token；剩余不足一页的输入仍需计算。首期内部固定 FCFS、`length` 回撤策略、无等待队列/单批请求数额外上限，不暴露 chunked prefill、异步重叠、层级缓存和未实现的 GPU 模式。

在计划构造阶段固定随机种子采样 `output_len` 得到目标输出长度 `O_sample≥1`；对每个请求先求可执行上限 `M=min(配置 max_new_tokens, C-ceil(P/B)*B-B)`，其中最后一个 `B` 是准入的一页安全余量。若 `M<1`，在该请求**实际发送时**将其记为 `aborted`（原因 `input_exceeds_kv_capacity`，失败时刻为发送时刻）；否则实际生成长度 `O=min(O_sample,M)`，不以缓存命中提高 `M`。调度器只读取 `P`、`M`、已经生成的 `g` 和已发布的缓存；`O_sample/O` 仅由引擎用于判断何时完成，绝不进入准入估算。生成上限很大时，未来需求估算最多取剩余上限的 4096 token，但实际输出仍可超过 4096。未配置模型最大上下文长度与提前 EOS/stop；没有硬件数据时 `max_total_tokens` 必须由输入显式给出。

## 缓存状态与请求生命周期

- 全局 L1 用 `chunk_id → 缓存项` 索引已发布的完整页；缓存项记录层次（首期恒为 `L1`）、活动引用数、最后访问序号和一页容量，不记录实际 GPU 地址。结果发布前，同批算出的相同 `chunk_id` 分属不同请求的**私有未发布页**，各占一页；结果处理按请求 ID 稳定顺序发布，首份建立缓存项，后续重复页改为引用同一项并立即归还其私有页。索引中每个已发布 ID 最终仅对应一个缓存项。未满页以及已填满但尚未发布的页不进入全局索引；它们计入所属请求的私有页数和全局驻留页数。未来多级缓存时缓存项的层次字段再用于迁移，层次本身不能代替引用数、容量和访问顺序。
- 每个驻留请求仅记录 `K` 和**已发布前沿** `F`：逻辑位置 `[0,F)` 的完整页均对应全局缓存项且被该请求引用，`F` 是 `B` 的整数倍、`0≤F≤K`；`[F,K)` 对应请求私有页，数量为 `ceil(K/B)-F/B`；有私有尾页时，其有效 token 数为 `K%B`（余数为 0 表示尾页已满 `B` 个 token）。请求在调度执行期间通过不可变输入身份和 `(request_id, 页序号)` 计算任意完整页 ID，查询全局索引取得共享引用；无需保存完整的页地址列表或已生成输出 ID 列表。`F` 以外的私有完整页可以在 Decode 时逐个形成，但仅在完成时发布；回撤时直接回收。容量守恒为 `free_pages + resident_pages = C/B`，其中 `resident_pages=全局索引中缓存项数+所有请求私有页数`；发布去重时释放重复私有页并同时更新两个计数，估算的未来输出不计入物理驻留。
- `probe` 顺序查请求输入的完整页 ID，到第一个未发布的页为止；首次输入最多复用 `floor((P-1)/B)*B` token，回撤恢复的查询上限见下文。候选准入时临时增加命中缓存项的引用，令零引用项从 `evictable` 中移出；失败撤销增加，成功则作为请求在 `[0,F)` 的引用。多个请求引用同一缓存项仍只占一页。只读查询和失败候选不更新访问顺序；成功获得引用时递增全局序号并刷新访问顺序，新项首次发布时也递增序号。最后引用释放时保留原序号；淘汰按 `(last_access_seq, chunk_id)` 升序，使用有序候选结构或带失效校验的最小堆，只在确需腾空间时访问。每次引用从零变正或从正变零时同步更新 `evictable` 计数，常规查找和容量读数不扫描整个缓存。
- 新请求 Prefill 从命中 `h` 到输入末尾 `P`，提交 `P-h` 个 KV token、采样首 token，`g=1`；若实际生成长度 `O=1`，步末完成。随后每个 Decode 步处理此前采样而尚未建立 KV 的一个 token：处理前 `K=P+g-1`，处理后 `K=P+g`，再采样令 `g+=1`；`g=O` 时完成，**不为最后一次采样的 token 建 KV**。因此完成时 `K=P+O-1`。未满尾页占用一页但不进入全局索引。
- Prefill 结果处理时，将 `[F,floor(K/B)*B)` 的完整页按逻辑顺序发布/去重，令 `F=floor(K/B)*B`，运行请求继续持有 `[0,F)` 的引用；新请求命中前缀时 `F=h`。Decode 填满的完整页保持私有直到请求完成；完成时发布剩余完整页，再按从小到大的页序号解除 `[0,F)` 的全部引用（ID 可按需推导），完整页零引用后驻留为可淘汰缓存，最后的未满私有页直接回收。回撤时解除 `[0,F)` 的引用并回收 `[F,K)` 的私有页，已发布缓存项是否驻留由引用数与淘汰规则决定。本步发布的页仅在下一步可供其他请求命中。
- 回撤时保留 `g`、不可变输入身份和首次首 token 时间；KV 管理器按上段规则解除 `[0,F)` 的引用、回收私有页并清空该请求的 `K/F`，不发布未完成 KV。请求重新进入 waiting；再次准入时的重算目标为原 prompt 加上已生成的前 `g-1` 个输出 token，即 `R=P+g-1`。由不可变输入身份和请求 ID/页序号按需推导 `[0,R)` 的完整页 ID，复用已发布前缀，恢复 Prefill 只补算未命中的 token 到 `K=R`，不新增输出、不重置首 token 时间；第 `g` 个已采样输出留待下一 Decode 步建立 KV。重算结束后按发布规则更新 `F`；若全部命中且无需重算，可在调度边界直接恢复 running。父轮只在真正完成时通知客户端。父轮可供子轮继承的完整 KV 页数为 `floor((P+O-1)/B)`；子轮可继承数还须 ≤ `floor((P_child-1)/B)`。`workload.py` 按该上限规划可达的理想继承率，`ClientDriver` 构造子轮输入时从父轮已发布的逻辑身份推导所需页 ID 并发送给引擎；父轮最后采样但未建立 KV 的 token 不产生可继承页。

## 准入容量算法与动态比例

所有容量计算都以 **token 容量** 为单位并将新物理页向上取整到 `B`；`A=free+evictable` 每尝试一个候选都从当前物理状态读取。令当前驻留请求集为 `Q`，本步参与 Decode 的步首请求集为 `D⊆Q`；`D_new=B*Σ_(q∈D)[K_q 是 B 的倍数]`（请求尾页未满时再写一个 token 不需要新页）。互斥模式尝试 Prefill 时 `D` 为空但 `Q` 仍可能非空；融合模式在尝试 Prefill 前确定 `D`，其物理申请尚未执行，但 `D_new` 已计入本步预算。对当前候选 `r`：逻辑重算/输入长度 `L=P`，回撤恢复时 `L=R=P+g-1`；命中长度 `h` 是已发布的完整页 token 数，首轮至多 `floor((P-1)/B)*B`，恢复时至多 `floor(R/B)*B`。`n=L-h`，新增物理页需求 `I=ceil(n/B)*B`；若恢复时 `n=0`，在调度边界取得共享页引用并返回 running，不构造空 Prefill 步；它至早参与下一引擎步的 Decode，不参与已确定的本步 `D`。每个接纳候选额外在**预算**中计一页 `B` 安全余量，不立刻分配这页。

新候选的未来估算 `F_new=min(max(M-g,0),4096)`；已经运行的请求 `q` 的未来估算 `F_run(q)=min(max(M_q-g_q,0),4096)*ratio`。令已接纳候选集为 `S`，本步累计即时预算 `current_offset=D_new+Σ_(s∈S)(I_s+B)`；总预算 `total_offset=current_offset+Σ_(s∈S)F_new(s)+Σ_(q∈Q)F_run(q)`。尝试候选时，先临时保护其命中页并重新计算 `A`，以**加入该候选后的**两个 offset 判定：`current_offset ≤ A` 且 `total_offset ≤ A`；此外必须满足 `|Q|+|S|+1 ≤ max_running_requests`。命中页即使被多个候选共享也只占一份物理容量。若本步已有 Prefill 候选，计算预算要求 `ceil(n/B)*B < max_prefill_tokens - |D| - Σ_(s∈S)ceil(n_s/B)*B`；第一个 Prefill 候选免去这项计算预算检查，仍须通过两项 KV 检查。接纳成功留下对命中页的引用和新的 offset，失败释放该候选临时引用，停止本轮 FCFS 扫描、不跳过队头；若批中已有人则照常执行。

`ratio` 是 `[0,1]` 的调度状态，初始 `0.7`，下限 `0.098`，每个**没有回撤且实际执行 Decode 的步**更新为 `max(0.098,ratio-(0.7-0.098)/600)`；只做 Prefill 时不衰减。若本步回撤后仍有 `k` 个运行请求，记其已生成输出数之和为 `G`、上限之和为 `H`，更新为 `min(1,(G+20*k)/(H+1))`；运行集合清空并闲置时重置为 `0.7`。相同模式、相同事件顺序下该比例必须确定性复现。它只影响 **已有运行请求**的未来估算，不乘到新候选上，也不在任何时刻预分配未来输出页。

若队头在当前有运行请求时仅因容量不足而无法准入，则让运行请求继续 Decode，等容量变化后再重试；若无运行请求且候选在 `A=C`、无保护空池条件下仍不满足本节两项预算，则标记该请求 `aborted`（原因 `input_exceeds_kv_capacity`）并继续后续队列。无论模式如何，不能发生既没有可执行 Prefill、也没有可执行 Decode，却反复在同一状态安排步骤的无限循环。

## 单步调度、分配与结果顺序

1. 对同一仿真时间的事件，先按请求 ID 顺序处理完成、释放、客户端回调及由此产生的子轮发送，再处理其他发送，最后调度一步；同一时刻最多安排一个引擎步。服务时间由 `StepWorkload` 的成本给出；计算期间到达的请求先入 waiting，至该步结束后再参与调度。
2. 默认互斥模式先按上节算法尝试非空 Prefill 批；若没有可执行的候选，才对步首 running 检查本步 `D_new`、按需逐出可淘汰页并在不足时回撤，最后执行 Decode。融合模式先确认步首 running 的本步 Decode 可执行集合及 `D_new`，然后按同一准入规则组 Prefill 批；若两类都非空执行 `MIXED`，只有一类则执行纯步。**不能**在已选新候选后更改本步 Decode 集合而不重新核算预算；被回撤者不得同一步重新准入。
3. 执行前实际申请的只有 `D_new+ΣI_s`，先按最旧可淘汰完整页逐出到空闲容量足够，再增加请求私有页数及全局驻留计数；已有未满尾页的 Decode 直接填充而无需新页。同批请求各自计算的相同完整页在发布前分别计入私有页，结果处理时按全局 `chunk_id` 去重并立即归还重复页。完成本步计算后再采样、发布、释放；该步新页仅对下一步可见。融合模式在 `StepWorkload` 中把两类请求分开记账，不重复计入费用或容量。
4. Decode 缺空间时先逐出无引用缓存，再按 `(已生成输出数升序、原始输入长度降序、请求 ID 升序)` 逐个回撤直到剩余 `D_new` 可分配；已回撤请求的活动引用和自有页立即释放，已发布而无引用的页仍按缓存规则驻留。只剩一个请求仍无法处理其下一个 token 时，将其 `aborted`（原因 `decode_kv_exhausted`），不无限重试。回撤请求下一步重入 waiting，恢复 Prefill 使用同一容量检查；多次回撤计入重试次数与已重复计算量。

## 实施阶段与交付

1. **F1：配置与请求数据。** 在 `schemas.py` 实现本页配置表、`Arrival.max_new_tokens`、页容量派生和结果状态；`workload.py` 用固定种子采样目标输出并按 `M` 截断实际输出；`client.py` 只向调度层传可见上限与逻辑身份，真实完成长度由引擎持有。所有入口从相同配置派生 `B/C/M`，无多份可写容量字段。本阶段交付配置拒绝/默认值/可见性单测。
2. **F2：L1 管理和预算。** `kv_cache.py` 实现全局 `chunk_id → 缓存项`、每请求的 `K/F`、从逻辑身份推导页 ID、连续前缀查找、引用增减、可淘汰候选结构、私有页容量计数及发布去重；不实现模拟器不需要的 GPU 物理地址或每请求页表。`scheduler.py` 按本页的 `A/current_offset/total_offset/ratio` 公式逐候选接纳，并处理首候选计算上限例外。不能在常规准入复制缓存全貌或排序所有驻留页；容量和引用不变量在分配、发布、驱逐和解除引用时检查。
3. **F3：引擎、成本和回撤。** `engine.py` 根据开关组织 `PREFILL/DECODE/MIXED`；`perf_model.py` 实现本页 `P_cost/D_cost`；`coordinator.py` 严格按本页事件全序处理。两种模式均逐步分配 KV；Decode 无空间时先淘汰、后回撤，回撤重算到 `P+g-1`，输出及首 token 时间不倒退，单请求无法进展则终止。此阶段的完成条件是全路径无临时容量门槛或分叉缓存机制。
4. **F4：结果、外部交付与验收。** `runner.py`/`metrics.py` 按本页状态和指标公式结算完成/失败/跳过；`server` 的 schema 和 OpenAPI 与响应一致；`web` 表单默认互斥，可关闭开关，显示状态并仅绘完成请求。先执行本页验收用例与同 seed JSON 一致性，再分别报告 100/1k/10k 规模下两种模式的内核墙钟、步数、峰值驻留与前端响应时间。

## 完成、结果与对外接口

- 一步的时长为 `duration≥0`，步终时间为 `start+duration`；首次 Prefill 的 `prefill_start=start`、`prefill_end=first_token_time=start+duration`；回撤重算虽为 Prefill 工作量，但不重置这些首次时间字段；Decode 完成时 `finish=start+duration`，其中包含排队和重算时间；输出为 1 的请求在 Prefill 步终即完成。引擎只在完成事件真正处理时释放 KV，完成、客户端回调、发送、调度严格按上一节顺序执行，避免零 think-time 子轮抢占父请求尚未释放的页。
- 结果的 `meta.num_requests` 表示计划总数，增加 `num_completed/num_aborted/num_skipped/num_engine_steps`，三类数量之和等于计划总数。`requests` 只放已完成请求的区间记录，按 ID 排序；`aborted_requests` 按 ID 给出 `id, arrival, failed_at, reason, generated_tokens, retry_count`；`skipped_requests` 给出 `id, parent_request_id, reason=parent_not_completed`，未发送者没有到达时间。客户端在父请求 abort 后递归标记其未发送后代 skipped，绝不发送它们。配置/负载无效返回 HTTP 422；仿真期间的请求级 abort 以成功响应中的状态列表表示，不让一次请求失败中断其他请求。生成类型由 schema 统一导出，前端只给 `requests` 绘甘特图，同时展示失败/跳过数量与原因。
- 逐请求 `ttft=first_token_time-arrival`、`tpot=(finish-first_token_time)/(O-1)`（仅成功且 `O>1`）；TTFT 均值、p50、p99 只对成功请求计算，分位数在排序后的相邻元素间按 `(N-1)*q` 线性插值；TPOT 均值先逐请求计算再对 `O>1` 的成功请求取算术平均，没有此类请求时为 `None`。`makespan=latest_success_finish-earliest_sent_arrival`（有成功请求时，单位 ms）；吞吐 `completed_count/(makespan/1000)`（请求/秒）。makespan 为 0 时吞吐为 `None`；无成功请求时 TTFT 均值及分位数、TPOT、吞吐均为 `None`，makespan 为 0。逻辑输入命中与 Prefill token 守恒只对成功请求**首次准入**计算：`P=first_hit_tokens+first_prefill_tokens`；重放成本另记 `replayed_prefill_tokens`（包括失败请求已执行的重放），不能混入首次命中率分母。物理容量指标取最终缓存的 `free/resident/active/evictable` 页数及峰值，`free+resident=C/B`；任何未来生成估算均不进入物理容量指标。

## 验收用例（同时作为实现判据）

- `B=64` 时输入长 63/64/65 且对应旧页已缓存：按 `floor((P-1)/B)*B` 得到命中 0/0/64；手动改 `B=1` 后同场景命中长度按最后输入 token 边界变化。`max_total_tokens=130, B=64` 的有效容量为 128；少于一页的配置被拒绝。
- 新候选的 `M=200` 与已经生成 `g=40` 的运行请求 `M=200`：初始比例为 0.7 时，二者未来估算分别为 200 和 112 token；只 Decode 一步、发生回撤、引擎空闲分别触发衰减、重估和重置。构造同一 `A` 下即时预算可通过、总预算不能通过的案例，验证不发生实际未来分配；被 pin 的可淘汰页不得再算进 `A`。
- 默认互斥时 Prefill 与 Decode 不同一步；关闭互斥时输入到达和既有运行请求可产生 `MIXED`，只推进步首运行请求，成本取两类阶段的最大值。计算预算恰好耗尽、本批首候选超限、暂时无法准入但可 Decode、全部请求完成或全部失败均应有确定进展，不重排 FCFS 队头。
- 两个同批首次计算的相同输入在发布前分别占私有页，发布后索引只留一个 `chunk_id` 缓存项并归还重复页；共享该项的两个请求各有引用，任何一个先完成都不能使该页被提前淘汰。请求只持有继承前缀 ID 与 `K/F`，由 `(request_id, 页序号)` 重建新增输入/输出完整页 ID；回撤后仍能恢复继承身份，`free_pages+resident_pages=C/B` 在去重、解除引用和驱逐后均成立。输出长 1 时仅存输入 KV；输出长 2 时最终已建立 KV 长度为 `P+1`。回撤在输出长 3 且 `g=2` 时只重算到 `P+1`，不重新生成前两个输出，随后 Decode 推进到完成；失败父轮后代一律 skipped。
- 固定 seed/配置下两次运行的输出 JSON 和步骤序列完全一致。1k、10k 请求基准分别报告两种模式的墙钟、步数及峰值驻留；常规准入、容量读数不得全量复制/排序驻留页，只在实际分配不足时访问可淘汰队列。

## 后续计划（不阻塞首期）

- **Chunked prefill**：允许长 prompt 跨步、active chunk 优先、中途缓存已计算 KV，并新增 chunk 阶段/调度边界用例。
- **真实共享与 radix tree**：引入真实 token 或等价内容身份、命名空间/salt、跨独立请求共享、树节点分裂与缓存驱逐；不能把当前按请求 ID 编码的块 ID 当成真实公共前缀。
- **高级执行和容量**：CPU/GPU overlap、mixed chunk、speculative、SWA/Mamba、PD 拆分、HiCache/L2/L3 和解析硬件耗时不属于 F001；本期不可将这些模式作为可选但未实现的配置暴露。
