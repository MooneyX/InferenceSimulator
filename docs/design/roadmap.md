# 里程碑与边界

## 里程碑

| 里程碑 | 内容 | 验收线 | 状态 |
|---|---|---|---|
| M1 内核 | DES 引擎 + FCFS 调度 + KV 预算 + SyntheticSource + 基准测试 | 1k 请求全场仿真 < 1s；同种子两次运行结果逐字节一致 | 已完成 |
| M2 服务层 | `POST /api/simulate` + OpenAPI→TS 类型生成 | curl 一场仿真返回完整区间数据与 metrics | 已完成 |
| M3 前端 | 请求甘特图（1k 请求渲染 + 缩放平移 + tooltip） | 加载 M2 响应正确渲染，hover 显示 TTFT 与各阶段时间戳 | 规划中 |

里程碑按 M1 → M2 → M3 顺序实施，每个独立可演示。

## MVP 明确排除

- 请求抢占 / KV 换出
- 多 GPU
- 基于真实 profiling 的性能模型标定
- trace replay 的具体设计（仅预留接口）
- 实时流式仿真（WebSocket、暂停/单步）
- 任务持久化与任务队列
- 调度决策溯源 UI

## 预留扩展接缝

只就位接口，不实现：

| 接缝 | 支撑的未来能力 |
|---|---|
| `WorkloadSource` 协议 + config `workload.mode` 字段 | 真实 trace replay |
| 内核零 IO + 步长纯函数 | 高性能内核替换（如 Rust） |
| 内核产出完整事件流但不暴露 API | 决策溯源、实时播放式仿真 |
