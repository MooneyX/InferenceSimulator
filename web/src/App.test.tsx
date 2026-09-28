import { fireEvent, render, screen } from "@testing-library/react";
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
    },
  ],
  warnings: ["请求 0 超过 KV token 预算，已强制准入"],
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

  it("runs a simulation and renders every result section", async () => {
    simulateMock.mockResolvedValue(result);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(simulateMock).toHaveBeenCalledWith(createDefaultSimConfig());
    expect(await screen.findByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "仿真告警" })).toBeInTheDocument();
    expect(screen.getByTestId("request-gantt")).toHaveTextContent("1 requests");
    expect(screen.queryByRole("heading", { name: "等待仿真结果" })).not.toBeInTheDocument();
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
            loc: ["body", "engine", "max_batch_size"],
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
      "body → engine → max_batch_size：Input should be greater than 0",
    );
    expect(screen.queryByRole("heading", { name: "仿真指标" })).not.toBeInTheDocument();
  });
});
