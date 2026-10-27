import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { clearSimulationCache, getJob, getJobResult, getPresets, getScanParameters, getScanTasks,
  getScanPointResult, listJobs, simulate, uploadReplayPlan, SimulationApiError } from "./api/client";
import type { InferenceConfig, SimResult } from "./api/types";
import { createDefaultSimConfig } from "./config/simulation";
import App from "./App";

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return { ...actual, simulate: vi.fn(), uploadReplayPlan: vi.fn(), clearSimulationCache: vi.fn(),
    listJobs: vi.fn(), getJob: vi.fn(), getJobResult: vi.fn(), getScanParameters: vi.fn(),
    getScanTasks: vi.fn(), getScanPointResult: vi.fn(), getPresets: vi.fn() };
});

vi.mock("./components/RequestGantt", () => ({
  RequestGantt: ({ requests = [] }: { requests?: unknown[] }) => (
    <div data-testid="request-gantt">{requests.length} requests</div>
  ),
}));

const result: SimResult = {
  resolution: null,
  analysis: null,
  meta: {
    config: createDefaultSimConfig(),
    num_requests: 1,
    num_completed: 1,
    num_aborted: 0,
    num_skipped: 0,
    num_engine_steps: 8,
    sim_start: 0,
    sim_end: 3,
    wall_clock_ms: 42.5,
  },
  metrics: {
    ttft_mean: 1,
    ttft_p50: 1,
    ttft_p99: 1,
    tpot_mean: 0.25,
    throughput: 333.33,
    makespan: 3,
    num_sessions: 1,
    root_request_ratio: 1,
    constructed_ideal_prefix_hit_rate: 0,
    actual_prefix_hit_rate: 0,
    l1_prefix_hit_rate: 0,
    l2_prefix_hit_rate: 0,
    l3_prefix_hit_rate: 0,
    requests_with_hit_ratio: 0,
    requests_fully_cached_ratio: 0,
    total_input_tokens: 16,
    ideal_hit_tokens: 0,
    matched_cache_tokens: 0,
    actual_hit_tokens: 0,
    l1_hit_tokens: 0,
    l2_hit_tokens: 0,
    l3_hit_tokens: 0,
    restore_bytes: 0,
    restore_time_ms: 0,
    promotions: 0,
    demotions: 0,
    actual_prefill_tokens: 16,
    replayed_prefill_tokens: 0,
    saved_prefill_tokens: 0,
    l1_evictions: 0,
    l1_peak_resident_blocks: 24,
    l1_capacity_blocks: 65_536,
    l1_resident_blocks: 24,
    l1_active_blocks: 0,
    l1_evictable_blocks: 24,
    l1_free_blocks: 65_512,
    l1_reserved_future_blocks: 0,
    l1_internal_fragmentation_tokens: 0,
    l2_evictions: 0,
    l2_peak_resident_blocks: 0,
    l2_capacity_blocks: 0,
    l2_resident_blocks: 0,
    l2_active_blocks: 0,
    l2_evictable_blocks: 0,
    l2_free_blocks: 0,
    l3_evictions: 0,
    l3_peak_resident_blocks: 0,
    l3_capacity_blocks: 0,
    l3_resident_blocks: 0,
    l3_active_blocks: 0,
    l3_evictable_blocks: 0,
    l3_free_blocks: 0,
  },
  instances: [],
  requests: [
    {
      id: 0,
      instance_id: 0,
      arrival: 0,
      prefill_start: 0,
      prefill_end: 1,
      finish: 3,
      prompt_len: 16,
      output_len: 8,
      session_id: 0,
      ideal_reuse_tokens: 0,
      matched_cache_tokens: 0,
      actual_hit_tokens: 0,
      actual_prefill_tokens: 16,
      replayed_prefill_tokens: 0,
      retry_count: 0,
      logical_context_tokens: 24,
      l1_hit_tokens: 0,
      l2_hit_tokens: 0,
      l3_hit_tokens: 0,
    },
  ],
  aborted_requests: [],
  skipped_requests: [],
  warnings: ["仿真提示"],
};
result.instances.push({ instance_id: 0, num_completed: 1, num_aborted: 0, num_skipped: 0,
  num_engine_steps: 8, metrics: result.metrics });

function resolvedResult(overrides: Partial<InferenceConfig> = {}): SimResult {
  const config = createDefaultSimConfig();
  config.inference = { ...config.inference, ...overrides };
  return {
    ...result,
    meta: { ...result.meta, config },
    resolution: {
      max_total_tokens: 4096, l2_capacity_tokens: 8192, l3_capacity_tokens: 16384,
      runnable: true,
      capacity: {
        weights_resident_bytes: 16_400_000_000, kv_bytes_per_token: 147456,
        kv_bytes_per_page: 9_437_184, estimated_total_parameters: 8_200_000_000,
        estimated_active_parameters: 8_200_000_000, estimated_expert_parameters: 0,
        l1_budget_bytes: 200_000_000_000, l1_kv_budget_bytes: 183_600_000_000,
        l2_budget_bytes: 2_048_000_000_000, l3_budget_bytes: 40_000_000_000_000,
        l1_capacity_pages: 64, l2_capacity_pages: 128, l3_capacity_pages: 256,
        rank_reports: [],
      },
    },
  };
}

const simulateMock = vi.mocked(simulate);
const uploadReplayPlanMock = vi.mocked(uploadReplayPlan);
const clearCacheMock = vi.mocked(clearSimulationCache);
const jobMock = vi.mocked(getJob);
const resultMock = vi.mocked(getJobResult);

function resolveSimulation(value: SimResult): void {
  simulateMock.mockResolvedValue({ id: "job-1" });
  resultMock.mockResolvedValue({ result: value, cache: { status: "miss", lookupMs: 0 } });
}

beforeEach(() => {
  simulateMock.mockReset();
  uploadReplayPlanMock.mockReset();
  clearCacheMock.mockReset();
  jobMock.mockReset();
  resultMock.mockReset();
  vi.mocked(getScanTasks).mockReset();
  vi.mocked(getScanPointResult).mockReset();
  vi.mocked(listJobs).mockReset();
  vi.mocked(listJobs).mockResolvedValue([]);
  vi.mocked(getScanParameters).mockReset();
  vi.mocked(getScanParameters).mockResolvedValue([]);
  vi.mocked(getPresets).mockReset().mockResolvedValue({ models: [], hardware: [] });
  jobMock.mockResolvedValue({ id: "job-1", kind: "single", status: "completed", total: 1,
    completed: 1, failed: 0, error: null, cache: "miss", lookup_ms: 0, axes: null, version: 1,
    code_fingerprint: "engine-1", code_version_status: "current",
    point_archive_version: 0 });
  vi.restoreAllMocks();
});

describe("App", () => {
  it("shows a cache hit and forces a recalculation with the current form", async () => {
    resolveSimulation(result);
    resultMock.mockResolvedValueOnce({ result, cache: { status: "hit", lookupMs: 1.25 } });
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(await screen.findByLabelText("结果缓存状态")).toHaveTextContent("结果缓存命中");
    expect(screen.getByLabelText("结果缓存状态")).toHaveTextContent("缓存查找/读取 1.25 ms");
    expect(screen.getByLabelText("结果缓存状态")).toHaveTextContent("原始仿真计算 42.50 ms");
    expect(screen.getByText("实际运行耗时").closest(".metric-card")).toHaveTextContent("42.5 ms");
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    fireEvent.click(screen.getByRole("button", { name: "强制重新计算" }));
    expect(simulateMock).toHaveBeenLastCalledWith(expect.objectContaining({
      workload: expect.objectContaining({ synthetic: expect.objectContaining({ num_requests: 20 }) }),
    }), true);
    expect(await screen.findByLabelText("结果缓存状态")).toHaveTextContent("本次结果为重新计算");
  });

  it("confirms cache clearing and keeps the displayed result", async () => {
    resolveSimulation(result);
    clearCacheMock.mockResolvedValue(undefined);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    await screen.findByRole("heading", { name: "仿真指标" });
    fireEvent.click(screen.getByRole("button", { name: "清空结果缓存" }));
    expect(clearCacheMock).not.toHaveBeenCalled();
    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: "清空结果缓存" }));
    expect(await screen.findByText("结果缓存已清空，下一次运行将重新计算。")).toBeInTheDocument();
    expect(clearCacheMock).toHaveBeenCalledOnce();
    expect(screen.getByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
  });

  it("reports cache clearing errors", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    clearCacheMock.mockRejectedValue(new SimulationApiError("清空失败"));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "清空结果缓存" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("清空失败");
    expect(screen.getByRole("button", { name: "清空结果缓存" })).toBeEnabled();
  });

  it("renders the configuration form and initial result placeholder", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { level: 1, name: "推理服务仿真器" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "负载配置" })).toBeInTheDocument();
    expect(screen.queryByRole("radio", { name: /自动解析|手动配置/ })).not.toBeInTheDocument();
    expect(screen.getByLabelText("当前仿真范围")).toHaveTextContent(
      "多实例 · TP 可选Synthetic / Trace workloadFCFS 调度",
    );
    expect(screen.getByRole("heading", { name: "等待仿真结果" })).toBeInTheDocument();
  });

  it("shows live request progress for a single background task", async () => {
    vi.mocked(listJobs).mockResolvedValue([{ id: "active", kind: "single", status: "running",
      total: 1, completed: 0, failed: 0, error: null, cache: null, lookup_ms: null,
      axes: null, version: 2, code_fingerprint: "engine", code_version_status: "current", point_archive_version: 0,
      request_progress: { total: 10, completed: 5, aborted: 1, skipped: 1 } }]);
    render(<App />);
    const tasks = await screen.findByLabelText("后台任务");
    expect(within(tasks).getByText("已处理 7 / 10 请求")).toBeInTheDocument();
    expect(within(tasks).getByText("70%")).toBeInTheDocument();
    expect(within(tasks).getByRole("progressbar")).toHaveAttribute("value", "7");
    expect(within(tasks).getByRole("button", { name: "取消任务" })).toBeInTheDocument();
    expect(within(tasks).queryByRole("button", { name: "查看结果" })).not.toBeInTheDocument();
  });

  it("keeps results hidden after refresh even when completed tasks are listed", async () => {
    vi.mocked(listJobs).mockResolvedValue([{ id: "previous", kind: "single", status: "completed",
      total: 1, completed: 1, failed: 0, error: null, cache: "miss", lookup_ms: 0,
      axes: null, version: 1, code_fingerprint: "engine-old", code_version_status: "stale",
      point_archive_version: 0 }]);
    resolveSimulation(result);
    render(<App />);
    const tasks = await screen.findByLabelText("后台任务");
    expect(within(tasks).getByRole("button", { name: "查看结果" })).toBeInTheDocument();
    expect(within(tasks).getByText("旧代码版本")).toHaveAttribute("title", "engine-old");
    expect(within(tasks).getByRole("link", { name: "查看原始 JSON" }))
      .toHaveAttribute("href", "/api/jobs/previous/result");
    expect(within(tasks).getByRole("link", { name: "导出 JSON" }))
      .toHaveAttribute("href", "/api/jobs/previous/result");
    expect(screen.getByRole("heading", { name: "等待仿真结果" })).toBeInTheDocument();
    expect(resultMock).not.toHaveBeenCalled();

    const scroll = vi.fn();
    document.getElementById("results-heading")!.scrollIntoView = scroll;
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    fireEvent.click(within(tasks).getByRole("button", { name: "查看结果" }));
    expect(scroll).toHaveBeenCalledWith({ behavior: "smooth" });
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(resultMock).toHaveBeenCalledWith("previous");
    fireEvent.click(screen.getByRole("button", { name: "导出本次参数" }));
    expect(JSON.parse((screen.getByLabelText("本次参数 JSON") as HTMLInputElement).value))
      .toEqual(result.meta.config);
    expect(simulateMock).not.toHaveBeenCalled();
  });

  it("keeps legacy imported results downloadable when the code version is unknown", async () => {
    vi.mocked(listJobs).mockResolvedValue([{ id: "legacy", kind: "single", status: "completed",
      total: 1, completed: 1, failed: 0, error: null, cache: "hit", lookup_ms: 1,
      axes: null, version: 1, code_fingerprint: "", code_version_status: "unknown",
      point_archive_version: 0 }]);
    render(<App />);
    const tasks = await screen.findByLabelText("后台任务");
    expect(within(tasks).getByText("代码版本未知")).toBeInTheDocument();
    expect(within(tasks).getByRole("link", { name: "查看原始 JSON" })).toBeInTheDocument();
    expect(within(tasks).getByRole("link", { name: "导出 JSON" }))
      .toHaveAttribute("download", "simulation-legacy.json");
  });

  it("expands scan points and opens a completed point in the existing result panel", async () => {
    vi.mocked(listJobs).mockResolvedValue([{ id: "scan-a", kind: "scan", status: "completed",
      total: 2, completed: 1, failed: 1, error: "point failed", cache: null, lookup_ms: null,
      axes: [{ key: "num_instances", values: [1, 0] }], version: 3,
      code_fingerprint: "engine-old", code_version_status: "stale",
      point_archive_version: 1 }]);
    vi.mocked(getScanTasks).mockResolvedValue({ total: 2, items: [
      { index: 0, values: { num_instances: 1 }, status: "completed", error: null,
        result_available: true, request_progress: { total: 10, completed: 7, aborted: 1, skipped: 2 } },
      { index: 1, values: { num_instances: 0 }, status: "failed", error: "point failed",
        result_available: false },
    ] });
    const pointResult = { ...result, meta: { ...result.meta,
      config: { ...result.meta.config, num_instances: 5 } } };
    vi.mocked(getScanPointResult).mockResolvedValue(pointResult);
    render(<App />);
    const tasks = await screen.findByLabelText("后台任务");
    expect(within(tasks).getByText(/1 点失败/)).toBeInTheDocument();
    expect(within(tasks).getByText("已结束组合 2/2")).toBeInTheDocument();
    fireEvent.click(within(tasks).getByRole("button", { name: "展开组合" }));
    expect(await within(tasks).findByText("num_instances: 1")).toBeInTheDocument();
    expect(within(tasks).getByText("已处理 10 / 10 请求")).toBeInTheDocument();
    expect(within(tasks).getByRole("progressbar", { name: "请求处理进度" })).toHaveAttribute("value", "10");
    expect(within(tasks).getByText("point failed")).toBeInTheDocument();
    expect(within(tasks).getAllByRole("button", { name: "查看结果" })).toHaveLength(1);
    expect(within(tasks).getByRole("link", { name: "查看原始 JSON" }))
      .toHaveAttribute("href", "/api/jobs/scan-a/points/0/result");
    expect(within(tasks).queryByRole("button", { name: "导出本次参数" })).not.toBeInTheDocument();
    const scroll = vi.fn();
    document.getElementById("results-heading")!.scrollIntoView = scroll;
    fireEvent.click(within(tasks).getByRole("button", { name: "查看结果" }));
    expect(scroll).toHaveBeenCalledWith({ behavior: "smooth" });
    expect(await screen.findByText("扫描 scan-a · 组合 #1")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.queryByLabelText("结果缓存状态")).not.toBeInTheDocument();
    expect(getScanPointResult).toHaveBeenCalledWith("scan-a", 0);
    fireEvent.click(screen.getByRole("button", { name: "导出本次参数" }));
    expect(JSON.parse((screen.getByLabelText("本次参数 JSON") as HTMLInputElement).value))
      .toEqual(pointResult.meta.config);
    expect(simulateMock).not.toHaveBeenCalled();
  });

  it("paginates recorded scan points and does not offer a fabricated historical result", async () => {
    vi.mocked(listJobs).mockResolvedValue([{ id: "old-scan", kind: "scan", status: "interrupted",
      total: 30, completed: 21, failed: 0, error: "service restarted", cache: null, lookup_ms: null,
      axes: [{ key: "num_instances", values: [1, 2] }], version: 3,
      code_fingerprint: "engine-old", code_version_status: "stale",
      point_archive_version: 0 }]);
    vi.mocked(getScanTasks).mockImplementation(async (_id, offset) => ({ total: 21, items: [
      { index: offset, values: { num_instances: offset + 1 }, status: "completed",
        error: null, result_available: false },
    ] }));
    render(<App />);
    const tasks = await screen.findByLabelText("后台任务");
    fireEvent.click(within(tasks).getByRole("button", { name: "展开组合" }));
    expect(await within(tasks).findByText("num_instances: 1")).toBeInTheDocument();
    expect(within(tasks).getByText(/历史扫描仅显示已记录的组合/)).toBeInTheDocument();
    expect(within(tasks).queryByRole("button", { name: "查看结果" })).not.toBeInTheDocument();
    fireEvent.click(within(tasks).getByRole("button", { name: "下一页" }));
    expect(await within(tasks).findByText("num_instances: 21")).toBeInTheDocument();
    expect(getScanTasks).toHaveBeenCalledWith("old-scan", 20, 20);
  });

  it("routes the configured replica count and filters instance results", async () => {
    resolveSimulation({
      ...result,
      meta: { ...result.meta, config: { ...result.meta.config, num_instances: 2 },
        num_requests: 2, num_completed: 2 },
      instances: [result.instances[0], { instance_id: 1, num_completed: 1,
        num_aborted: 0, num_skipped: 0, num_engine_steps: 1,
        metrics: { ...result.metrics, ttft_mean: 2, actual_prefix_hit_rate: 0.25,
          l1_prefix_hit_rate: 0.25, l1_hit_tokens: 4, actual_hit_tokens: 4,
          ideal_hit_tokens: 4, matched_cache_tokens: 4, saved_prefill_tokens: 4,
          actual_prefill_tokens: 12 } }],
      requests: [result.requests![0], { ...result.requests![0], id: 1, instance_id: 1 }],
    });
    render(<App />);
    fireEvent.change(screen.getByLabelText("推理实例数量"),
      { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(simulateMock).toHaveBeenCalledWith(expect.objectContaining({ num_instances: 2 }));
    expect(await screen.findByTestId("request-gantt")).toHaveTextContent("2 requests");
    fireEvent.change(screen.getByLabelText("查看范围："), { target: { value: "1" } });
    expect(screen.getByTestId("request-gantt")).toHaveTextContent("1 requests");
    expect(screen.getByLabelText("实例结果筛选")).toHaveTextContent("实例 1");
    expect(screen.getByText("TTFT 均值").closest(".metric-card")).toHaveTextContent("2 ms");
    expect(screen.getByText("L1 前缀命中率").closest(".metric-card")).toHaveTextContent("25 %");
  });

  it("navigates to real configuration and result sections", () => {
    render(<App />);

    const navigation = screen.getByRole("navigation", { name: "建模模块" });
    const engineLink = within(navigation).getByRole("link", { name: "引擎配置" });
    expect(engineLink).toHaveAttribute("href", "#engine-heading");
    fireEvent.click(engineLink);
    expect(engineLink).toHaveAttribute("aria-current", "location");
    expect(screen.getByRole("heading", { name: "引擎配置" })).toHaveAttribute("id", "engine-heading");

    const resultsLink = within(navigation).getByRole("link", { name: "仿真结果" });
    fireEvent.click(resultsLink);
    expect(resultsLink).toHaveAttribute("aria-current", "location");
    expect(document.querySelectorAll("#results-heading")).toHaveLength(1);
  });

  it("runs a simulation and renders every result section", async () => {
    resolveSimulation(result);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(simulateMock).toHaveBeenCalledWith(createDefaultSimConfig());
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getByText("实际运行耗时").closest(".metric-card")).toHaveTextContent("42.5 ms");
    expect(screen.getByRole("heading", { name: "仿真告警" })).toBeInTheDocument();
    expect(await screen.findByTestId("request-gantt")).toHaveTextContent("1 requests");
    expect(screen.getByLabelText("请求结果状态")).toHaveTextContent("已完成 1 · 失败 0 · 跳过 0");
    expect(screen.getByLabelText("请求结果状态")).toHaveClass("simulation-info");
    expect(screen.getByLabelText("实例结果筛选")).toHaveClass("simulation-info", "instance-filter");
    expect(screen.queryByRole("heading", { name: "等待仿真结果" })).not.toBeInTheDocument();
  });

  it("shows resolved capacity and cost after running an automatic trace plan", async () => {
    const traceResult = resolvedResult();
    traceResult.meta.config.workload = { mode: "trace", arrival_mode: "open_loop",
      session_rate: 1, seed: 0, target_qps: 9, trace: { plan_id: "plan-1" } };
    uploadReplayPlanMock.mockResolvedValue({ id: "plan-1", sessions: 1, requests: 1 });
    resolveSimulation(traceResult);
    render(<App />);
    fireEvent.change(screen.getAllByLabelText("负载模式")[0], { target: { value: "trace" } });
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json")] },
    });
    await screen.findByText(/计划已校验/);
    await waitFor(() => expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled());
    fireEvent.click(screen.getByLabelText("按目标 QPS 等比例缩放到达间隔"));
    fireEvent.change(screen.getByLabelText("目标发送 QPS（请求/秒）"), {
      target: { value: "9" },
    });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(simulateMock).toHaveBeenCalledWith(expect.objectContaining({
      workload: { mode: "trace", arrival_mode: "open_loop", session_rate: 1,
        seed: 0, target_qps: 9, trace: { plan_id: "plan-1" } },
      inference: createDefaultSimConfig().inference,
    }));
    expect(await screen.findByLabelText("性能模型配置与成本")).toHaveTextContent("性能模型");
    expect(screen.getByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getAllByText("目标发送 QPS").at(-1)?.closest(".metric-card"))
      .toHaveTextContent("9 请求/秒");
  });

  it("routes the single run button to the simulation API", async () => {
    resolveSimulation(resolvedResult());
    render(<App />);
    expect(screen.queryByRole("radio", { name: /自动解析|手动配置/ })).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "运行仿真" })).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(simulateMock).toHaveBeenCalledWith(createDefaultSimConfig());
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getByText("实际运行耗时").closest(".metric-card")).toHaveTextContent("42.5 ms");
    const summary = screen.getByLabelText("性能模型配置与成本");
    expect(summary).toHaveTextContent("性能模型");
    expect(summary).toHaveTextContent("模型 Qwen/Qwen3-8B");
    expect(summary).toHaveTextContent("L1 4,096 token · L2 8,192 token · 全场共享 L3 16,384 token");
  });

  it("shows TP communication inside the existing resolved result card", async () => {
    const resolved = resolvedResult({ tp_size: 4,
      hardware: { ...createDefaultSimConfig().inference.hardware, gpu_count: 4 } });
    resolved.analysis = {
      steps: 2, flops: 1000, hbm_bytes: 2000, compute_ms: 0.1,
      memory_ms: 0.2, device_ms: 5, restore_ms: 0,
      communication_ms: 1.25, communication_bytes: 500_000_000, collectives: 144,
      tp_communication_ms: 1.25, tp_communication_bytes: 500_000_000,
      ep_dispatch_ms: 0, ep_combine_ms: 0,
      ep_dispatch_bytes: 0, ep_combine_bytes: 0, ep_exchanges: 0,
    };
    resolveSimulation(resolved);
    render(<App />);
    fireEvent.change(screen.getByLabelText("TP 数（必须等于 GPU 卡数）"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("本实例 GPU 卡数"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("TP 每次集合通信固定延迟（ms，TP>1 运行必填）"), {
      target: { value: "0.1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(simulateMock).toHaveBeenCalledWith(expect.objectContaining({
      inference: expect.objectContaining({ tp_size: 4 }),
    }));
    const summary = await screen.findByLabelText("性能模型配置与成本");
    expect(summary).toHaveTextContent("TP4");
    expect(summary).toHaveTextContent("总计 4 GPU");
    expect(summary).toHaveTextContent("TP 同步 1.25 ms");
    expect(summary).toHaveTextContent("144 次");
  });

  it("shows TP and EP timing separately without adding communication to GPU time twice", async () => {
    const initial = createDefaultSimConfig().inference;
    const resolved = resolvedResult({ tp_size: 4, ep_size: 4,
      model: { ...initial.model, name: "toy-moe", moe_layers: 2 },
      hardware: { ...initial.hardware, name: "B300", gpu_count: 4 } });
    resolved.resolution!.capacity = { ...resolved.resolution!.capacity,
      weights_resident_bytes: 1_000_000, kv_bytes_per_token: 128,
      rank_reports: [{ rank: 0, sharded_weights_bytes: 200_000,
        replicated_weights_bytes: 50_000, expert_weights_bytes: 100_000,
        kv_bytes_per_token: 32, l1_kv_budget_bytes: 300_000, l1_capacity_pages: 25 }] };
    resolved.analysis = {
      steps: 2, flops: 1000, hbm_bytes: 2000, compute_ms: 0.1,
      memory_ms: 0.2, device_ms: 5, restore_ms: 1,
      communication_ms: 2, communication_bytes: 4000, collectives: 4,
      tp_communication_ms: 1, tp_communication_bytes: 2000,
      ep_dispatch_ms: 0.4, ep_combine_ms: 0.6,
      ep_dispatch_bytes: 900, ep_combine_bytes: 1100, ep_exchanges: 8,
    };
    resolveSimulation(resolved);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    const summary = await screen.findByLabelText("性能模型配置与成本");
    expect(summary).toHaveTextContent("TP4 + EP4");
    expect(summary).toHaveTextContent("整组权重 0.00 GB");
    expect(screen.getByLabelText("逐卡容量账单")).toHaveTextContent("GPU 0");
    expect(screen.getByLabelText("逐卡容量账单")).toHaveTextContent("专家驻留 0.000 GB");
    expect(summary).toHaveTextContent("TP 同步 1.00 ms");
    expect(summary).toHaveTextContent("EP dispatch 0.40 ms / combine 0.60 ms");
    expect(summary).toHaveTextContent("GPU 计时 5.00 ms");
    expect(summary).toHaveTextContent("逻辑流量");
  });

  it("shows abort and skip reasons but charts completed requests only", async () => {
    resolveSimulation({
      ...result,
      meta: { ...result.meta, num_requests: 3, num_aborted: 1, num_skipped: 1 },
      aborted_requests: [{
        id: 1, instance_id: 0, arrival: 0, failed_at: 0,
        reason: "output_exceeds_engine_limit", generated_tokens: 65_536, retry_count: 0,
      }],
      skipped_requests: [{ id: 2, instance_id: 0, parent_request_id: 1, reason: "parent_not_completed" }],
    });
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(await screen.findByLabelText("请求结果状态")).toHaveTextContent("已完成 1 · 失败 1 · 跳过 1");
    expect(screen.getByLabelText("失败请求原因")).toHaveTextContent("output_exceeds_engine_limit");
    expect(screen.getByLabelText("请求结果状态")).toHaveTextContent("引擎输出上限：65,536 token");
    expect(screen.getByLabelText("跳过请求原因")).toHaveTextContent("parent_not_completed");
    expect(await screen.findByTestId("request-gantt")).toHaveTextContent("1 requests");
  });

  it("shows an empty chart state when every request failed", async () => {
    resolveSimulation({
      ...result,
      meta: { ...result.meta, num_completed: 0, num_aborted: 1 },
      metrics: {
        ...result.metrics,
        ttft_mean: null, ttft_p50: null, ttft_p99: null,
        tpot_mean: null, throughput: null, makespan: 0,
      },
      requests: [],
      aborted_requests: [{
        id: 0, instance_id: 0, arrival: 0, failed_at: 0,
        reason: "input_exceeds_kv_capacity", generated_tokens: 0, retry_count: 0,
      }],
    });
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(await screen.findByText("本次没有已完成请求，无法绘制生命周期图。")).toBeInTheDocument();
    expect(screen.queryByTestId("request-gantt")).not.toBeInTheDocument();
    expect(screen.getByLabelText("请求结果状态")).toHaveTextContent("已完成 0 · 失败 1 · 跳过 0");
  });

  it("keeps the form editable while submitting and retrieves a background result", async () => {
    let resolveRequest: (value: { id: string }) => void = () => undefined;
    simulateMock.mockReturnValue(new Promise((resolve) => { resolveRequest = resolve; }));
    resultMock.mockResolvedValue({ result, cache: { status: "miss", lookupMs: 0 } });
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(screen.getByRole("status")).toHaveTextContent("正在提交后台任务…");
    expect(screen.getByLabelText("请求数量")).toBeEnabled();
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });

    resolveRequest({ id: "job-1" });
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
  });

  it("renders API validation errors without displaying stale results", async () => {
    simulateMock.mockRejectedValue(
      new SimulationApiError("仿真配置校验失败", {
        validationErrors: [
          {
            loc: ["body", "scheduler", "max_running_requests"],
            msg: "Input should be greater than 0",
            type: "greater_than",
          },
        ],
      }),
    );
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("仿真配置校验失败");
    expect(alert).toHaveTextContent(
      "body → scheduler → max_running_requests：Input should be greater than 0",
    );
    expect(screen.queryByRole("heading", { name: "仿真指标" })).not.toBeInTheDocument();
  });
});
