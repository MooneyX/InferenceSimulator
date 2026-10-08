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
  target_ideal_prefix_hit_rate: 0.3,
  constructed_ideal_prefix_hit_rate: 0.28,
  ideal_hit_quantization_error: 0.02,
  actual_prefix_hit_rate: 0.17,
  requests_with_hit_ratio: 0.6,
  requests_fully_cached_ratio: 0.25,
  total_input_tokens: 1000,
  ideal_hit_tokens: 280,
  matched_cache_tokens: 170,
  actual_hit_tokens: 170,
  l1_hit_tokens: 170,
  l2_hit_tokens: 0,
  l3_hit_tokens: 0,
  restore_bytes: 0,
  restore_time_ms: 0,
  promotions: 0,
  demotions: 0,
  actual_prefill_tokens: 830,
  replayed_prefill_tokens: 4,
  saved_prefill_tokens: 170,
  l1_capacity_blocks: 256,
  l1_evictions: 4,
  l1_peak_resident_blocks: 242,
  l1_resident_blocks: 100,
  l1_active_blocks: 0,
  l1_evictable_blocks: 100,
  l1_free_blocks: 156,
  l1_reserved_future_blocks: 0,
  l1_internal_fragmentation_tokens: 3,
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
    expect(metricValue("目标理想命中率")).toHaveTextContent("30 %");
    expect(metricValue("构造理想命中率")).toHaveTextContent("28 %");
    expect(metricValue("实际前缀命中率")).toHaveTextContent("17 %");
    expect(metricValue("完整块全命中请求比例")).toHaveTextContent("25 %");
    expect(metricValue("节省 Prefill")).toHaveTextContent("170 token");
    expect(metricValue("重算 Prefill")).toHaveTextContent("4 token");
    expect(metricValue("L1 容量")).toHaveTextContent("256 页");
    expect(metricValue("L1 驱逐")).toHaveTextContent("4 次");
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
          ...metrics,
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
