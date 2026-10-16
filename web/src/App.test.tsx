import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { simulate, uploadReplayPlan, SimulationApiError } from "./api/client";
import type { PhysicalInferenceConfig, SimResult } from "./api/types";
import { createDefaultSimConfig } from "./config/simulation";
import App from "./App";

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return { ...actual, simulate: vi.fn(), uploadReplayPlan: vi.fn() };
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

function physicalResult(overrides: Partial<PhysicalInferenceConfig> = {}): SimResult {
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

beforeEach(() => {
  simulateMock.mockReset();
  uploadReplayPlanMock.mockReset();
});

describe("App", () => {
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

  it("routes the configured replica count and filters instance results", async () => {
    simulateMock.mockResolvedValue({
      ...result,
      meta: { ...result.meta, config: { ...result.meta.config, num_instances: 2 },
        num_requests: 2, num_completed: 2 },
      instances: [result.instances[0], { instance_id: 1, num_completed: 1,
        num_aborted: 0, num_skipped: 0, num_engine_steps: 1,
        metrics: { ...result.metrics, ttft_mean: 2 } }],
      requests: [result.requests![0], { ...result.requests![0], id: 1, instance_id: 1 }],
    });
    render(<App />);
    fireEvent.change(screen.getByLabelText("推理实例数量（L1/L2 独占，L3 共享）"),
      { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(simulateMock).toHaveBeenCalledWith(expect.objectContaining({ num_instances: 2 }));
    expect(await screen.findByTestId("request-gantt")).toHaveTextContent("2 requests");
    fireEvent.change(screen.getByLabelText("查看范围："), { target: { value: "1" } });
    expect(screen.getByTestId("request-gantt")).toHaveTextContent("1 requests");
    expect(screen.getByLabelText("实例结果筛选")).toHaveTextContent("实例 1");
    expect(screen.getByText("TTFT 均值").closest(".metric-card")).toHaveTextContent("2 ms");
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
    simulateMock.mockResolvedValue(result);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(simulateMock).toHaveBeenCalledWith(createDefaultSimConfig());
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getByText("实际运行耗时").closest(".metric-card")).toHaveTextContent("42.5 ms");
    expect(screen.getByRole("heading", { name: "仿真告警" })).toBeInTheDocument();
    expect(await screen.findByTestId("request-gantt")).toHaveTextContent("1 requests");
    expect(screen.getByLabelText("请求结果状态")).toHaveTextContent("已完成 1 · 失败 0 · 跳过 0");
    expect(screen.queryByRole("heading", { name: "等待仿真结果" })).not.toBeInTheDocument();
  });

  it("shows resolved capacity and cost after running an automatic trace plan", async () => {
    const traceResult = physicalResult();
    traceResult.meta.config.workload = { mode: "trace", arrival_mode: "open_loop",
      session_rate: 1, seed: 0, target_qps: 9, trace: { plan_id: "plan-1" } };
    uploadReplayPlanMock.mockResolvedValue({ id: "plan-1", sessions: 1, requests: 1 });
    simulateMock.mockResolvedValue(traceResult);
    render(<App />);
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
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
      inference: expect.objectContaining({ mode: "physical", timing: { mode: "analytical" } }),
    }));
    expect(await screen.findByLabelText("解析配置与成本")).toHaveTextContent("自动解析");
    expect(screen.getByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getByText("目标发送 QPS").closest(".metric-card")).toHaveTextContent("9 请求/秒");
  });

  it("routes the single physical run button to the simulation API", async () => {
    simulateMock.mockResolvedValue(physicalResult());
    render(<App />);
    expect(screen.queryByRole("radio", { name: /自动解析|手动配置/ })).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "运行仿真" })).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(simulateMock).toHaveBeenCalledWith(createDefaultSimConfig());
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getByText("实际运行耗时").closest(".metric-card")).toHaveTextContent("42.5 ms");
    const summary = screen.getByLabelText("解析配置与成本");
    expect(summary).toHaveTextContent("自动解析");
    expect(summary).toHaveTextContent("模型 Qwen/Qwen3-8B");
    expect(summary).toHaveTextContent("L1 4,096 token · L2 8,192 token · 全场共享 L3 16,384 token");
  });

  it("shows TP communication inside the existing resolved result card", async () => {
    const resolved = physicalResult({ tp_size: 4,
      hardware: { ...createDefaultSimConfig().inference.hardware, gpu_count: 4 } });
    resolved.analysis = {
      steps: 2, flops: 1000, hbm_bytes: 2000, compute_ms: 0.1,
      memory_ms: 0.2, device_ms: 5, restore_ms: 0,
      communication_ms: 1.25, communication_bytes: 500_000_000, collectives: 144,
      tp_communication_ms: 1.25, tp_communication_bytes: 500_000_000,
      ep_dispatch_ms: 0, ep_combine_ms: 0,
      ep_dispatch_bytes: 0, ep_combine_bytes: 0, ep_exchanges: 0,
    };
    simulateMock.mockResolvedValue(resolved);
    render(<App />);
    fireEvent.change(screen.getByLabelText("TP 数（必须等于 GPU 卡数）"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("本实例 GPU 卡数"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("TP 每次集合通信固定延迟（ms，TP>1 运行必填）"), {
      target: { value: "0.1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(simulateMock).toHaveBeenCalledWith(expect.objectContaining({
      inference: expect.objectContaining({ mode: "physical", tp_size: 4 }),
    }));
    const summary = await screen.findByLabelText("解析配置与成本");
    expect(summary).toHaveTextContent("TP4");
    expect(summary).toHaveTextContent("总计 4 GPU");
    expect(summary).toHaveTextContent("TP 同步 1.25 ms");
    expect(summary).toHaveTextContent("144 次");
  });

  it("shows TP and EP timing separately without adding communication to GPU time twice", async () => {
    const initial = createDefaultSimConfig().inference;
    const resolved = physicalResult({ tp_size: 4, ep_size: 4,
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
    simulateMock.mockResolvedValue(resolved);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    const summary = await screen.findByLabelText("解析配置与成本");
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
    simulateMock.mockResolvedValue({
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
    simulateMock.mockResolvedValue({
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

  it("disables duplicate submission while the request is pending", async () => {
    let resolveRequest: (value: SimResult) => void = () => undefined;
    simulateMock.mockReturnValue(
      new Promise((resolve) => {
        resolveRequest = resolve;
      }),
    );
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(screen.getByRole("button", { name: "正在运行…" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("正在运行仿真…");

    resolveRequest(result);
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
