# 旧实现与废弃配置清理计划

**状态：待实施。** 一次变更同步更新 `core/`、`server/` 所依赖的 API 契约、`web/`、生成类型、测试和现行设计文档；不保留拟删除符号或配置字段的兼容层。前后端作为同一交付单元验收。

## 删除范围与保留边界

| 删除 | 保留 |
|---|---|
| `KVCache`、`KVStats`、`FCFSScheduler`、`StepPlan` 及其旧 token 账本/强制超预算准入实现 | `simcore.kv_cache`、`simcore.scheduler` 模块；`HierarchicalKVCache`、`PrefixScheduler`、`CachedStepPlan`、容量/OOM 异常、L1 事务及其测试；现行 `kv_token_budget`、步长计算和事件记录 |
| `SyntheticSource`、`build_source`、`WorkloadSource` 及专属的单随机流到达生成实现 | `SyntheticWorkloadConfig`、`build_plan`、`sample_length`、`plan_from_arrivals`、`run_engine`、`Arrival`，以及现行 `trace` 模式的 501 行为 |
| `SessionConfig.allow_cross_session_shared_prefix` 字段、`build_plan` 的专门拒绝分支、前端默认值及生成契约中的该字段 | `_Frozen(extra="forbid")`、同 session 父子块继承、不同根块 ID 隔离、L1 连续前缀探测和命中指标；已移除的 `content_seed` 仍须被拒绝 |

## 实施顺序

1. **清理旧 KV 路径。** 在 `core/src/simcore/kv_cache.py` 删除 `KVStats`、`KVCache`、旧说明和 `__all__` 条目，清理仅供旧账本使用的导入；在 `core/src/simcore/scheduler.py` 删除 `StepPlan`、`FCFSScheduler`、旧说明和专属导入/导出。删除只验证这两套旧实现的 `core/tests/test_kv_cache.py`、`core/tests/test_scheduler.py`；不得改动 `test_hierarchical_kv_cache.py`、`test_prefix_scheduler.py` 的现行断言。
2. **清理旧合成源。** 在 `core/src/simcore/workload.py` 删除 `WorkloadSource`、`SyntheticSource`、`build_source`、专属导入/导出与文件说明；保留 `sample_length` 及 `build_plan` 共用的采样和时间单位。整理 `core/tests/test_workload.py`：去掉旧源、旧协议及旧工厂的测试，保留分布采样测试。将 `core/tests/test_engine.py` 的 `synthetic_run` 改为通过 `build_plan`、`ClientDriver`、`Coordinator`、现行 `Engine` 生成 `EngineRun`；保留 300 请求的完成、容量、步数、FCFS、trace 和复现性断言，并确保新测试配置可执行。手写 `Arrival` 经 `run_engine` 的用例继续保留。
3. **移除跨 session 配置字段。** 在 `core/src/simcore/schemas.py` 删除 `SessionConfig.allow_cross_session_shared_prefix`；在 `core/src/simcore/workload.py` 删除读取该字段的专门拒绝分支，不改 `build_plan` 的根请求继承上限或 `ClientDriver`/`prefix_hash` 的块 ID 规则。在 `web/src/config/simulation.ts` 删除默认请求中的 `false`；`SimulationForm` 不新增开关或客户端兼容转换，服务端 `SimConfig` 校验继续拒绝未知字段。同步更新 `web/src/config/simulation.test.ts`、`web/src/components/SimulationForm.test.tsx` 中的默认值和提交断言。
4. **同步 API 类型与拒绝测试。** 在 `web/` 执行 `npm run gen:api`，提交重新生成的 `web/openapi.json`、`web/src/api/schema.d.ts`，不手改生成物。将 `core/tests/test_prefix_plan.py` 的旧专门拒绝测试改为断言不同根块 ID 隔离；在 `core/tests/test_schemas.py` 验证字段省略可解析、序列化不含该键、显式 `true` 和 `false` 均因未知字段失败；在 `server/tests/test_routes.py` 验证两种显式值均返回 422、无字段请求成功且结果 `meta.config` 不含该键。保留 `content_seed` 的拒绝测试及同 session 实际命中测试。前端测试验证序列化的提交体不含该键。
5. **更新现行文档。** 修订 `docs/design/kernel.md`、`docs/design/kv-prefix-reuse.md` 的旧调度器引用，并将现行步计划准确写为 `CachedStepPlan`；修订 `docs/design/workload.md`、`docs/design/architecture.md` 的旧合成源描述，保留 `Arrival` 适配链说明；修订 `docs/design/api.md`、`docs/design/frontend.md`、`docs/design/chunk-hash-encoding.md`、`docs/design/workload.md` 中该配置字段和专门拒绝路径的描述，继续说明跨 session 共享不支持、同 session 复用仍可用。检查 `core/src/simcore/schemas.py` 中 `Arrival` 的文案，不再将其归属已删除的源。

## 联合验收

- 从同一版源码生成类型、启动服务与前端；默认表单可提交并展示指标和甘特图，浏览器实际 POST 请求体不含删除的字段，响应 `meta.config` 也不含该字段。**不能只更新后端而保留发送显式 `false` 的旧前端。**
- 同 seed、同有效配置（旧基线中仅去掉被删除的 `false` 字段）下，现行 `simulate` 的请求区间、指标、告警保持一致；多轮 session 的父子块继承及 L1 命中、不同根块 ID 隔离、容量不足/OOM 行为保持一致。`run_engine` 的手工 `Arrival` 适配测试仍通过。
- 显式提交已删除字段的 `true` 或 `false` 均返回未知字段 422，不静默忽略；省略字段的请求成功；`content_seed` 仍返回 422，`trace` 仍返回 501。
- 执行根目录 `uv run pytest`、`uv run ruff check core server`、`uv run ruff format --check core server`、`uv run python -m mypy`；在 `web/` 执行 `npm test`、`npm run typecheck`、`npm run build`、`npm run check:types`、`npm run test:e2e`。确认 OpenAPI 新鲜度测试通过，并检查项目自身源码、测试、生成物和现行设计文档不再引用被删除的符号/字段（允许本计划列出删除目标）。

## 交付约束

本次删除会使旧的模块级 Python 导入及仍显式携带 `allow_cross_session_shared_prefix: false` 的外部请求失效；不提供兼容别名或静默剥离。发布或使用时须同步替换前后端，并通知独立 API/Python 调用方移除这些调用。除上述删除范围外，不改现行仿真算法、缓存身份、接口字段或未来 F001/F002/KV3 的计划。
