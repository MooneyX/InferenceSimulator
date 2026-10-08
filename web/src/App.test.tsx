import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { simulate, SimulationApiError } from "./api/client";
import type { SimResult } from "./api/types";
import { createDefaultSimConfig } from "./config/simulation";
import App from "./App";

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return { ...actual, simulate: vi.fn() };
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
  },
  metrics: {
    ttft_mean: 1,
    ttft_p50: 1,
    ttft_p99: 1,
    tpot_mean: 0.25,
    throughput: 333.33,
    makespan: 3,
    target_ideal_prefix_hit_rate: 0,
    constructed_ideal_prefix_hit_rate: 0,
    ideal_hit_quantization_error: 0,
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

beforeEach(() => {
  simulateMock.mockReset();
});

describe("App", () => {
  it("renders the configuration form and initial result placeholder", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { level: 1, name: "推理服务仿真器" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "负载配置" })).toBeInTheDocument();
    expect(screen.getByLabelText("当前仿真范围")).toHaveTextContent(
      "单 GPUSynthetic workloadFCFS 调度",
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

    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(simulateMock).toHaveBeenCalledWith(createDefaultSimConfig());
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "仿真告警" })).toBeInTheDocument();
    expect(await screen.findByTestId("request-gantt")).toHaveTextContent("1 requests");
    expect(screen.getByLabelText("请求结果状态")).toHaveTextContent("已完成 1 · 失败 0 · 跳过 0");
    expect(screen.queryByRole("heading", { name: "等待仿真结果" })).not.toBeInTheDocument();
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

    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("仿真配置校验失败");
    expect(alert).toHaveTextContent(
      "body → engine → max_running_requests：Input should be greater than 0",
    );
    expect(screen.queryByRole("heading", { name: "仿真指标" })).not.toBeInTheDocument();
  });
});
