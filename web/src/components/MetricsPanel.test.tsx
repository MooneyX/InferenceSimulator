import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { SimResult } from "../api/types";
import { MetricsPanel } from "./MetricsPanel";

type Metrics = SimResult["metrics"];

const metrics: Metrics = {
  ttft_mean: 12.345,
  ttft_p50: 10,
  ttft_p99: 98.765,
  tpot_mean: 2.5,
  throughput: 1234.567,
  makespan: 800.25,
};

function metricValue(label: string): HTMLElement {
  const term = screen.getByText(label);
  const card = term.closest<HTMLElement>(".metric-card");

  if (!card) {
    throw new Error(`找不到指标卡片：${label}`);
  }

  return within(card).getByRole("definition");
}

describe("MetricsPanel", () => {
  it("renders every service metric with deterministic units", () => {
    render(<MetricsPanel metrics={metrics} />);

    expect(screen.getByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(metricValue("TTFT 均值")).toHaveTextContent("12.35 ms");
    expect(metricValue("TTFT P50")).toHaveTextContent("10 ms");
    expect(metricValue("TTFT P99")).toHaveTextContent("98.77 ms");
    expect(metricValue("TPOT 均值")).toHaveTextContent("2.5 ms");
    expect(metricValue("端到端吞吐")).toHaveTextContent("1,234.57 请求/秒");
    expect(metricValue("Makespan")).toHaveTextContent("800.25 ms");
  });

  it("shows undefined instead of coercing nullable metrics to zero", () => {
    render(
      <MetricsPanel
        metrics={{
          ...metrics,
          tpot_mean: null,
          throughput: undefined,
        }}
      />,
    );

    expect(metricValue("TPOT 均值")).toHaveTextContent("未定义");
    expect(metricValue("端到端吞吐")).toHaveTextContent("未定义");
  });

  it("displays values supplied by the service without recomputing them", () => {
    render(
      <MetricsPanel
        metrics={{
          ttft_mean: 901,
          ttft_p50: 902,
          ttft_p99: 903,
          tpot_mean: 904,
          throughput: 905,
          makespan: 906,
        }}
      />,
    );

    expect(metricValue("TTFT 均值")).toHaveTextContent("901 ms");
    expect(metricValue("端到端吞吐")).toHaveTextContent("905 请求/秒");
  });
});
