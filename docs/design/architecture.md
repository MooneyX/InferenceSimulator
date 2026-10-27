# 总体架构

## 模块划分

前后端分离的 monorepo，三个模块：

```
core/     仿真内核：纯 Python 包，零 IO
server/   服务层：FastAPI
web/      前端：React + TypeScript + Vite + ECharts
```

## 内核内部模块

`core` 按职责单向分解，模块间只允许向下依赖：

```
schemas      Pydantic 配置/结果与不可变内部消息（无依赖）
prefix_hash  完整块 ID 直接编码与独立随机流种子派生（无内部模块依赖）
workload     build_synthetic_closed_loop_plan 规划合成闭环长度、拓扑和理想复用；sample_length 采样（依赖 schemas、prefix_hash）
client       Client：_prepare_workload 统一负载准备，on_wakeup 统一请求构造与投放（依赖 schemas、workload、prefix_hash）
perf_model   唯一性能模型：不可变工作量与 StepCost、逐阶段 Roofline 和 TP/EP 通信纯函数（依赖 schemas）
kv_cache     CachePool：逐池页索引、容量、稳定 LRU；TieredKVCache：请求状态、连续匹配与跨池操作（依赖 schemas）
scheduler    F001 FCFS 调度：按 KV 页状态和未来估算选择 batch（依赖 kv_cache、schemas）
engine       单个实例的已到达请求、GPU 步与异步页传输、独占 KV 生命周期与完成结果（依赖 scheduler、kv_cache、perf_model）
events       客户端发送、GPU 步完成、L3 查询与页传输完成、请求完成事件（仅依赖标准库）
router       默认逐请求最低 token 负载优先，或新会话轮询、按 session_id 绑定的缓存亲和；接收不可变负载元组（依赖 schemas）
coordinator  唯一 DES 时钟和事件队列；按实例 ID 将请求及完成交给对应 engine（依赖 client、engine、events、router）
metrics      指标汇总：区间与 cache 快照 → Metrics（依赖 schemas、kv_cache）
runtime      RuntimeConfig、AnalyticalSpec、具名 TransferLinks 的不可变值对象
capacity     prepare_runtime：从 SimConfig 一次性准备容量、计时规格与链路（依赖 runtime、schemas）
runner       顶层仿真入口 simulate(config, trace_plan=None) -> SimResult（依赖 runtime、client、coordinator、engine、metrics）
```

`runner.simulate` 是统一的完整仿真入口，包顶层导出 `SimConfig`、`TracePlan` 与 `simulate`：

- `config: SimConfig`：唯一外部配置由 `num_instances`、`workload`、`scheduler`、`inference`、可选 `shared_l3` 和默认 `least_load` 的 `routing_policy` 组成。`inference` 从模型/硬件/存储推导容量，所有运行使用同一性能模型；不接受 `inference.timing` 或旧计时类型。
- `trace_plan: TracePlan | None`（仅关键字参数）：仅注入已解析的计划资源；开环/闭环、会话速率、种子和可选目标 QPS 都属于 `config.workload` 公共层；`workload.trace` 只保存可选 `plan_id`。Trace 缺少计划资源时拒绝运行，合成负载不接受 Trace 资源；服务层根据临时 `plan_id` 获取计划，内核不执行 IO。
- 始终返回单个 `SimResult`，新运行包含 `resolution` 和非空 `analysis`（即使没有 GPU 步也返回零值汇总）。结果字段 `analysis` 的类型仍允许为空；这不表示旧配置满足当前 schema，也不转换旧配置。`meta.config` 完整回显输入配置，不改写成运行时容量配置，也不修改调用方输入。
- 一场仿真只调用一次 `capacity.prepare_runtime`，再用 `Engine(runtime)` 将同一只读 `RuntimeConfig` 注入各实例。`RuntimeConfig` 包含内部 `engine`、`perf_model`、具名 `links`、非空 `analytical` 与 `resolution`；`AnalyticalSpec` 只共享模型/硬件与 TP/EP 规格，不持有动态状态。`TransferLinks.l2_to_l1` / `l3_to_l2` 预计算每页传输时间和固定延迟，保持原浮点运算顺序、成本公式和支持限制。
- `EngineConfig` / `PerfModelConfig` 是解析后使用的内部组件规格；`PerfModelConfig` 仅保存 KV 字节数、链路速率和延迟，不再保存 GPU 四系数。外部不能直接填写 KV token 容量。运行墙钟不包含配置解析；结果回显包含模型默认字段的原规范化 `SimConfig`。

四种负载组合均使用单个 `Client`，只读 `workload_mode` 与 `arrival_mode` 区分请求来源及投放模式，专用状态按需初始化；统一入口 `simulate` 使用 `Coordinator → start_step → 内部 _execute_step(selection 必填) → finish_step → complete` 生命周期。GPU 步完成与异步页传输分别通过事件推进，再统一汇总指标。

规则：

- 禁止跨层反向依赖（如 `scheduler` 不得引用 `engine`）。
- 模块间数据交换遵循下文「数据交换边界」。
- 每个模块可独立单测；只有 `coordinator` 感知并持有全局事件队列，`client` 与 `engine` 通过不可变事件和值对象参与调度。

### 数据交换边界

分两层边界：

- **对外边界（`core` ↔ `server`）**：可序列化的配置、计划和结果通过 `schemas` 中的 Pydantic 类型交换，`schemas` 是 HTTP 数据契约的单一事实源。服务层将请求中的 `SimConfig` 原样传给统一入口，仅根据 `workload.trace.plan_id` 另行注入不可变 `TracePlan` 资源；不存在第二套外部运行配置。
- **内核内部模块间**：只交换不可变值对象（`frozen` dataclass 或 `schemas` 类型）。内部计划与客户端/引擎消息在 `schemas` 中定义，其余值对象由提供方模块定义，只允许依赖图中的下游模块引用。
- **禁止跨模块共享可变状态**。唯一例外是依赖图中明确列出的协作者注入，须写明所有权（见下文）。

内部类型归属：

| 类型 | 定义于 | 使用方 | 说明 |
|---|---|---|---|
| `TurnPlan` / `WorkloadPlan` | `schemas`（由 `workload` 构造） | `client` | 不可变 session 拓扑、输入/输出长度与理想继承块数；不预生成块 ID |
| `ClientWakeup` | `schemas`（由 `client` 构造） | `coordinator` | 未来客户端发送事件，只含 session/turn 身份和发送时刻 |
| `InferenceRequest` | `schemas`（由 `client` 构造） | `coordinator`、`engine`、`scheduler`、`kv_cache`、`router` | 发送时构造：输入长度、生成上限、真实输出目标 `output_len` 和继承完整页 ID；不携带新页 ID 列表。当前调度不能以真实输出目标作为依据 |
| `StepWorkload` | `perf_model`（由 `engine` 构造） | `perf_model` | 不可变的 Prefill/Decode GPU 工作量、阶段、实际申请页数及 `restore_only` 请求 ID；不携带恢复页来源或页大小，传输另由事件计时 |
| `StepCost` | `perf_model` | `engine` | 不可变的 GPU 步总时长、计算、显存访问及 TP/EP 通信分项 |
| `AdmissionBatch` | `scheduler` | `engine` | FCFS 准入、恢复、失败及唯一提升页（值对象）；预算只在调度内部计算 |
| `L1Snapshot` 与分层快照 | `kv_cache` | `engine`、`metrics` | 各层页容量、驻留及峰值的不可变汇总 |
| `Event` / `EventKind` | `events` | `coordinator` | 全局事件队列载荷与同刻全序（值对象） |
| `EventFactory` | `events` | `coordinator` | 事件序号分配器，由协调器实例化并独占持有 |
| `EngineRun` | `engine` | `coordinator`、`runner` | 协调器返回的请求区间、告警、步数、分层快照及迁移/restore 计数 |

协作者注入：

- `TieredKVCache`：由 `engine` 独占并注入 `FcfsAdmissionScheduler`，管理请求 `K/F`、私有页和候选引用；`probe` / `pin` / `cancel_pin` / `plan` 组织连续匹配与无副作用预算，`admit` / `extend` / `publish_batch` / `finish` / `withdraw` 管理请求生命周期。
- `CachePool`：逐池拥有页索引、容量、LRU 和保护状态。每实例独占 L1/L2；`runner` 每场创建一个 L3 并以 `Engine(runtime, shared_l3=..., instance_id=...)` 注入所有实例，编号在本场唯一。共享 L3 是明确的可变协作者例外，不放进不可变 `RuntimeConfig`，不跨运行复用。上行传输由 `reserve_transfer` / `complete_transfer` 保护来源并完成目标占位；L3 回载保留来源副本。共享池统计由 `runner` 只读取一次。

调度事件仅用于推进仿真，不保存逐步诊断日志。`EngineRun` 只携带正常汇总使用的请求记录、分层快照与整体计数；`compute_metrics` 必须接收完整计划、三级快照和实际发送统计。

`coordinator` 只负责时间推进和事件路由，不参与块 ID 构造、调度或缓存决策。具体发送契约与同刻顺序分别见 [workload.md](workload.md) 和 [events.md](events.md)。

## 依赖规则

- `core` 不依赖任何外部 IO（无文件、无网络、无真实时钟读取），输入为配置对象，内部生成 `WorkloadPlan` 并由协调器推进模拟时钟，输出为请求区间和汇总结果对象。
- `server` 依赖 `core`，负责 HTTP 边界；`core` 不感知 `server`。
- `web` 与 `server` 仅通过 HTTP API 交互，类型由 OpenAPI 生成（见 [api.md](api.md)）。

## 工程规范

- 包管理：`uv`（`core` + `server` 为同一 workspace）。
- 可复现性为硬约束：同一随机种子的两次运行结果必须逐字节一致。
- `core` 自带基准测试：固定种子、固定规模的计时回归用例，从 M1 起常驻。
