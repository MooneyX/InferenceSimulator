# 项目 Agent 约定

本文件是 InferenceSimulator 的项目级上下文。开始处理本项目任务时先遵循这里的约定；具体功能与开发命令以 `README.md` 和 `docs/index.md` 为准。

## 协作与实现约定

- 不要主动修改代码。只有用户明确说了“开始实现”之后，才可以修改代码；此前可以讨论、分析或提出方案。用户明确要求修改文档时，可以修改文档。
- 每次修改代码或文档的请求完成时，必须提交**本次请求的改动**；请求进行时也可以视需要分阶段提交。提交前检查变更范围，不要将用户或其他任务的未完成改动混入提交；没有本次改动时不创建空提交。
- 每次编码改动提交完成后，更新本地的系统部署，使本地运行的系统使用最新提交的代码；若无法更新，应说明原因。
- 编码时避免过度添加不必要的断言和验证；不必验证在正常使用仿真系统的流程中不可能出现的情况。
- 新实现替换旧实现时，直接删除旧实现，不必为兼容而保留被替换的实现。
- 可以根据实现需要修改测试脚本；不要为了通过原有测试而保留旧实现或增加其他冗余内容。

## 编码后的增量验证

- 以**本次任务实际改动及其直接影响范围**选择验证，不把工作区中其他未完成改动视为本次修改；完成一组相关改动后统一验证一次，不要每编辑一个文件就重复跑测试。文档或规则的纯文本改动无需运行代码测试。
- 优先运行对应的测试文件或用例，不默认执行全仓库 `uv run pytest`、`npm test`、`npm run build`、`npm run test:e2e`、基准脚本或全部静态检查。测试失败时先定位问题，按受影响的依赖链逐步扩大范围，而非立即全量重跑。没有对应测试时说明验证缺口，不以无关测试替代。
- 修改 `core/` 时，从对应的 `core/tests/test_*.py` 或具体 `::用例` 开始；涉及引擎、调度、KV、前缀或结果汇总的跨模块行为时，补跑相关的内核串联测试；影响 API 响应时再补 `server/tests/test_routes.py`。例如：`uv run pytest core/tests/test_engine.py core/tests/test_runner.py`。
- 修改 `server/` 时，优先运行 `uv run pytest server/tests/test_routes.py` 或 `server/tests/test_app.py` 的相关用例；若修改请求/响应模型、OpenAPI 或序列化，再补 `server/tests/test_openapi_export.py`、相关内核测试及必要的前端类型检查。不要因为只改路由就跑浏览器全链路。
- 修改 `web/` 时，在 `web` 目录运行对应 Vitest 测试，如 `npm test -- src/components/SimulationForm.test.tsx`；影响跨组件流程时补 `src/App.test.tsx` 或 API 客户端测试。修改 TypeScript 类型/API 使用时运行一次 `npm run typecheck`；仅修改样式且无逻辑变化时，可不跑单元测试。涉及真实浏览器交互、Canvas 或代理联调时才选择对应 Playwright 用例（`npm run test:e2e -- --grep "用例名"`）。
- 更改 `core` 的 Pydantic 模型、服务 API 契约或 `web/openapi.json` / `web/src/api/schema.d.ts` 时，针对性检查 `core/tests/test_schemas.py`、`server/tests/test_openapi_export.py` 及受影响的路由测试；需要校验生成类型时在 `web` 运行 `npm run check:types`，影响 TS 使用时再运行 `npm run typecheck`。`check:types` 不能替代 Python 侧 OpenAPI 新鲜度测试。
- 只有用户明确要求全量验收，或跨层大范围改动无法界定影响范围时，才考虑全量测试/构建；性能基准与含 10,000 请求的浏览器场景仅在性能或端到端交付相关任务中运行。报告实际运行的检查、结果及未覆盖的范围，不把局部通过表述成全量通过。

## 并行仿真阶段边界

- 多卡开发按 `docs/plans/expert-parallelism.md` 的“模型与硬件参数配置 → TP → EP”顺序推进；开始编码下一阶段前先确认对应的模型放置、KV 容量和计时契约。当前单 GPU Dense/GQA 的自动解析与 Roofline 只是第一阶段，不把参考模型/设备目录或 `Qwen3-235B-A22B × 8 B300` 标为已支持的 TP/EP 部署。
- 新增容量模式时，必须在规划负载与创建 `Engine` 之前解析物理规格；内核继续以整页 token 容量和现有 `page_size` 为唯一生效口径。解析性能须标明未经校准，不将标称硬件数据视作实测吞吐。
- 已实现行为以 `docs/design/`、代码和对应测试为准，`docs/plans/` 只记录未交付部分；阶段落地后同步状态与对外契约，不提前展示尚未实现的并行组合。

## Node.js 环境（Windows）

- 前端使用 Node.js 22（见 `README.md`）。当前开发环境已确认有 IDE 托管的 Node.js `22.22.2`，即使 `node` / `npm` 不在系统 `PATH` 中，也不要据此判断未安装。
- 先检查 `%USERPROFILE%\.workbuddy\binaries\node\versions\` 中的可用版本；如有兼容版本，将其可执行文件所在目录临时加入**当前进程**的 `PATH` 后再运行 `node` / `npm`。不要修改系统级 `PATH`。
- 只有该目录也没有兼容版本时，才考虑安装新的 Node.js 运行时。

## 项目入口

- `core/`：Python 仿真内核；`server/`：FastAPI 服务；`web/`：React + TypeScript 前端。
- 项目说明与开发命令见 `README.md`；设计与计划的索引见 `docs/index.md`。
