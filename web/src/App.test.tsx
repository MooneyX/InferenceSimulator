import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { simulate, simulateAutomatic, SimulationApiError } from "./api/client";
import type { ResolvedRun, SimResult } from "./api/types";
import { createDefaultSimConfig } from "./config/simulation";
import App from "./App";

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return { ...actual, simulate: vi.fn(), simulateAutomatic: vi.fn() };
});

vi.mock("./components/RequestGantt", () => ({
  RequestGantt: ({ requests = [] }: { requests?: unknown[] }) => (
    <div data-testid="request-gantt">{requests.length} requests</div>
  ),
}));

const result: SimResult = {
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
  requests: [
    {
      id: 0,
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

const simulateMock = vi.mocked(simulate);
const simulateAutomaticMock = vi.mocked(simulateAutomatic);

beforeEach(() => {
  simulateMock.mockReset();
  simulateAutomaticMock.mockReset();
});

describe("App", () => {
  it("renders the configuration form and initial result placeholder", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { level: 1, name: "推理服务仿真器" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "负载配置" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /自动解析/ })).toBeChecked();
    expect(screen.getByLabelText("当前仿真范围")).toHaveTextContent(
      "单实例 · TP 可选Synthetic workloadFCFS 调度",
    );
    expect(screen.getByRole("heading", { name: "等待仿真结果" })).toBeInTheDocument();
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

    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(simulateMock).toHaveBeenCalledWith(createDefaultSimConfig());
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getByText("实际运行耗时").closest(".metric-card")).toHaveTextContent("42.5 ms");
    expect(screen.getByRole("heading", { name: "仿真告警" })).toBeInTheDocument();
    expect(await screen.findByTestId("request-gantt")).toHaveTextContent("1 requests");
    expect(screen.getByLabelText("请求结果状态")).toHaveTextContent("已完成 1 · 失败 0 · 跳过 0");
    expect(screen.queryByRole("heading", { name: "等待仿真结果" })).not.toBeInTheDocument();
  });

  it("routes the single run button to the resolved API in automatic mode", async () => {
    simulateAutomaticMock.mockResolvedValue({
      result,
      resolution: {
        timing: "analytical",
        tp_size: 1,
        model: { name: "Qwen/Qwen3-8B" },
        hardware: { name: "NVIDIA B300" },
        capacity: { weights_resident_bytes: 16_400_000_000, kv_bytes_per_token: 147456 },
        config: result.meta.config,
      },
      analysis: null,
    } as ResolvedRun);
    render(<App />);
    expect(screen.getByRole("radio", { name: /自动解析/ })).toBeChecked();
    expect(screen.getAllByRole("button", { name: "运行仿真" })).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(simulateAutomaticMock).toHaveBeenCalledWith(expect.objectContaining({
      timing: "analytical", tp_size: 1,
      config: expect.objectContaining({ workload: expect.objectContaining({ mode: "synthetic" }) }),
    }));
    expect(simulateMock).not.toHaveBeenCalled();
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getByText("实际运行耗时").closest(".metric-card")).toHaveTextContent("42.5 ms");
    expect(screen.getByLabelText("解析配置与成本")).toHaveTextContent("自动解析");
  });

  it("shows TP communication inside the existing resolved result card", async () => {
    simulateAutomaticMock.mockResolvedValue({
      result,
      resolution: {
        timing: "analytical", tp_size: 4,
        model: { name: "Qwen/Qwen3-8B" }, hardware: { name: "NVIDIA B300" },
        capacity: { weights_resident_bytes: 16_400_000_000, kv_bytes_per_token: 147456 },
        config: result.meta.config,
      },
      analysis: {
        steps: 2, flops: 1000, hbm_bytes: 2000, compute_ms: 0.1,
        memory_ms: 0.2, device_ms: 5, restore_ms: 0,
        communication_ms: 1.25, communication_bytes: 500_000_000, collectives: 144,
      },
    } as ResolvedRun);
    render(<App />);
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
    fireEvent.change(screen.getByLabelText("TP 数（必须等于 GPU 卡数）"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("本实例 GPU 卡数"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("TP 每次集合通信固定延迟（ms，TP>1 运行必填）"), {
      target: { value: "0.1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(simulateAutomaticMock).toHaveBeenCalledWith(expect.objectContaining({ tp_size: 4 }));
    const summary = await screen.findByLabelText("解析配置与成本");
    expect(summary).toHaveTextContent("TP4");
    expect(summary).toHaveTextContent("TP 同步 1.25 ms");
    expect(summary).toHaveTextContent("144 次");
  });

  it("shows TP and EP timing separately without adding communication to GPU time twice", async () => {
    simulateAutomaticMock.mockResolvedValue({
      result,
      resolution: {
        timing: "analytical", tp_size: 4, ep_size: 4,
        model: { name: "toy-moe", moe_layers: 2 }, hardware: { name: "B300" },
        capacity: { weights_resident_bytes: 1_000_000, kv_bytes_per_token: 128,
          rank_reports: [{ rank: 0, sharded_weights_bytes: 200_000,
            replicated_weights_bytes: 50_000, expert_weights_bytes: 100_000,
            kv_bytes_per_token: 32, l1_kv_budget_bytes: 300_000, l1_capacity_pages: 25 }] },
        config: result.meta.config,
      },
      analysis: {
        steps: 2, flops: 1000, hbm_bytes: 2000, compute_ms: 0.1,
        memory_ms: 0.2, device_ms: 5, restore_ms: 1,
        communication_ms: 2, communication_bytes: 4000, collectives: 4,
        tp_communication_ms: 1, tp_communication_bytes: 2000,
        ep_dispatch_ms: 0.4, ep_combine_ms: 0.6,
        ep_dispatch_bytes: 900, ep_combine_bytes: 1100, ep_exchanges: 8,
      },
    } as ResolvedRun);
    render(<App />);
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
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
        id: 1, arrival: 0, failed_at: 0,
        reason: "input_exceeds_kv_capacity", generated_tokens: 0, retry_count: 0,
      }],
      skipped_requests: [{ id: 2, parent_request_id: 1, reason: "parent_not_completed" }],
    });
    render(<App />);
    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(await screen.findByLabelText("请求结果状态")).toHaveTextContent("已完成 1 · 失败 1 · 跳过 1");
    expect(screen.getByLabelText("失败请求原因")).toHaveTextContent("input_exceeds_kv_capacity");
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
        id: 0, arrival: 0, failed_at: 0,
        reason: "input_exceeds_kv_capacity", generated_tokens: 0, retry_count: 0,
      }],
    });
    render(<App />);
    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
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

    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(screen.getByRole("button", { name: "正在运行…" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("正在运行仿真…");

    resolveRequest(result);
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
  });

  it("renders API validation errors without displaying stale results", async () => {
    simulateMock.mockRejectedValue(
      new SimulationApiError("仿真配置校验失败", {
        kind: "validation",
        status: 422,
        validationErrors: [
          {
            loc: ["body", "engine", "max_running_requests"],
            msg: "Input should be greater than 0",
            type: "greater_than",
          },
        ],
      }),
    );
    render(<App />);

    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("仿真配置校验失败");
    expect(alert).toHaveTextContent(
      "body → engine → max_running_requests：Input should be greater than 0",
    );
    expect(screen.queryByRole("heading", { name: "仿真指标" })).not.toBeInTheDocument();
  });
});
