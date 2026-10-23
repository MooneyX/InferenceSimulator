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
  arrival_qps: 1200,
  num_sessions: 3,
  root_request_ratio: 0.25,
  mean_prompt_tokens: 128,
  mean_output_tokens: 32,
  child_ideal_prefix_hit_rate: 0.4,
  child_actual_prefix_hit_rate: 0.2,
  constructed_ideal_prefix_hit_rate: 0.28,
  actual_prefix_hit_rate: 0.17,
  l1_prefix_hit_rate: 0.17,
  l2_prefix_hit_rate: 0,
  l3_prefix_hit_rate: 0,
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
  l2_evictions: 3,
  l2_peak_resident_blocks: 18,
  l2_capacity_blocks: 32,
  l2_resident_blocks: 12,
  l2_active_blocks: 0,
  l2_evictable_blocks: 12,
  l2_free_blocks: 20,
  l3_evictions: 1,
  l3_peak_resident_blocks: 8,
  l3_capacity_blocks: 16,
  l3_resident_blocks: 4,
  l3_active_blocks: 0,
  l3_evictable_blocks: 4,
  l3_free_blocks: 12,
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
    render(<MetricsPanel metrics={metrics} wallClockMs={42.5} targetQps={64} />);

    expect(screen.getByRole("heading", { name: "仿真指标" })).toBeInTheDocument();
    expect(metricValue("目标发送 QPS")).toHaveTextContent("64 请求/秒");
    expect(metricValue("TTFT 均值")).toHaveTextContent("12.35 ms");
    expect(metricValue("TTFT P50")).toHaveTextContent("10 ms");
    expect(metricValue("TTFT P99")).toHaveTextContent("98.77 ms");
    expect(metricValue("TPOT 均值")).toHaveTextContent("2.5 ms");
    expect(metricValue("实际请求 QPS")).toHaveTextContent("1,200 请求/秒");
    expect(metricValue("完成 QPS")).toHaveTextContent("1,234.57 请求/秒");
    expect(metricValue("新建会话数")).toHaveTextContent("3 次");
    expect(metricValue("根请求占比")).toHaveTextContent("25 %");
    expect(metricValue("子轮理想命中率")).toHaveTextContent("40 %");
    expect(metricValue("Makespan")).toHaveTextContent("800.25 ms");
    expect(metricValue("实际运行耗时")).toHaveTextContent("42.5 ms");
    expect(metricValue("构造理想命中率")).toHaveTextContent("28 %");
    expect(metricValue("实际前缀命中率")).toHaveTextContent("17 %");
    expect(metricValue("L1 前缀命中率")).toHaveTextContent("17 %");
    expect(metricValue("L2 前缀命中率")).toHaveTextContent("0 %");
    expect(metricValue("L3 前缀命中率")).toHaveTextContent("0 %");
    expect(metricValue("完整块全命中请求比例")).toHaveTextContent("25 %");
    expect(metricValue("节省 Prefill")).toHaveTextContent("170 token");
    expect(metricValue("重算 Prefill")).toHaveTextContent("4 token");
    expect(metricValue("L1 容量")).toHaveTextContent("256 页");
    expect(metricValue("L1 驱逐")).toHaveTextContent("4 次");
    expect(metricValue("L2 容量")).toHaveTextContent("32 页");
    expect(metricValue("L2 驻留")).toHaveTextContent("12 页");
    expect(metricValue("L2 空闲")).toHaveTextContent("20 页");
    expect(metricValue("L2 峰值驻留")).toHaveTextContent("18 页");
    expect(metricValue("L2 驱逐")).toHaveTextContent("3 次");
    expect(metricValue("L3 容量")).toHaveTextContent("16 页");
    expect(metricValue("L3 驻留")).toHaveTextContent("4 页");
    expect(metricValue("L3 空闲")).toHaveTextContent("12 页");
    expect(metricValue("L3 峰值驻留")).toHaveTextContent("8 页");
    expect(metricValue("L3 驱逐")).toHaveTextContent("1 次");
  });

  it("hides shared pool state for an instance but retains its L3 hits", () => {
    render(<MetricsPanel wallClockMs={42.5} metrics={{ ...metrics,
      l3_capacity_blocks: null, l3_resident_blocks: null, l3_active_blocks: null,
      l3_evictable_blocks: null, l3_free_blocks: null, l3_peak_resident_blocks: null,
      l3_evictions: null, l3_hit_tokens: 20, l3_prefix_hit_rate: 0.02,
    }} />);
    expect(screen.queryByText("L3 容量")).not.toBeInTheDocument();
    expect(screen.queryByText("L3 峰值驻留")).not.toBeInTheDocument();
    expect(metricValue("L3 命中")).toHaveTextContent("20 token");
    expect(metricValue("L3 前缀命中率")).toHaveTextContent("2 %");
  });

  it("hides shared L2 state for an instance but retains hits", () => {
    render(<MetricsPanel wallClockMs={42.5} sharedL2 metrics={{ ...metrics,
      l2_capacity_blocks: null, l2_resident_blocks: null, l2_active_blocks: null,
      l2_evictable_blocks: null, l2_free_blocks: null, l2_peak_resident_blocks: null,
      l2_evictions: null, l2_hit_tokens: 20,
    }} />);
    expect(screen.queryByText("L2 容量")).not.toBeInTheDocument();
    expect(screen.queryByText("L2 峰值驻留")).not.toBeInTheDocument();
    expect(metricValue("L2 命中")).toHaveTextContent("20 token");
  });

  it("uses the actual shared peak even in aggregate mode", () => {
    render(<MetricsPanel metrics={metrics} wallClockMs={42.5} peakAggregated sharedL2 />);
    expect(screen.getByText("L2 峰值驻留").closest(".metric-card"))
      .toHaveTextContent("全场共享 L2 的实际驻留峰值");
    expect(screen.getByText("L1 峰值驻留").closest(".metric-card"))
      .toHaveTextContent("各实例峰值之和");
    expect(screen.getByText("L3 峰值驻留").closest(".metric-card"))
      .toHaveTextContent("全场共享 L3 的实际驻留峰值");
  });

  it("shows observed request QPS without an input request-QPS target in closed_loop mode", () => {
    render(<MetricsPanel metrics={metrics} wallClockMs={42.5} targetSessionRate={1.28} />);
    expect(metricValue("新会话投放频率")).toHaveTextContent("1.28 会话/秒");
    expect(metricValue("实际请求 QPS")).toHaveTextContent("1,200 请求/秒");
    expect(screen.queryByText("目标发送 QPS")).not.toBeInTheDocument();
  });

  it("separates aggregate cache hits from tier attribution and restore costs", () => {
    render(<MetricsPanel wallClockMs={42.5} metrics={{
      ...metrics,
      actual_hit_tokens: 170,
      l1_hit_tokens: 50,
      l2_hit_tokens: 100,
      l3_hit_tokens: 20,
      l1_prefix_hit_rate: 0.05,
      l2_prefix_hit_rate: 0.1,
      l3_prefix_hit_rate: 0.02,
      promotions: 3,
      demotions: 5,
      restore_bytes: 8192,
      restore_time_ms: 15.5,
    }} />);

    expect(metricValue("总实际命中")).toHaveTextContent("170 token");
    expect(metricValue("L1 命中")).toHaveTextContent("50 token");
    expect(metricValue("L2 命中")).toHaveTextContent("100 token");
    expect(metricValue("L3 命中")).toHaveTextContent("20 token");
    expect(metricValue("L1 前缀命中率")).toHaveTextContent("5 %");
    expect(metricValue("L2 前缀命中率")).toHaveTextContent("10 %");
    expect(metricValue("L3 前缀命中率")).toHaveTextContent("2 %");
    expect(metricValue("提升到 L1")).toHaveTextContent("3 次");
    expect(metricValue("逐级降级")).toHaveTextContent("5 次");
    expect(metricValue("Restore 传输量")).toHaveTextContent("8,192 字节");
    expect(metricValue("Restore 耗时")).toHaveTextContent("15.5 ms");
  });

  it("shows undefined instead of coercing nullable metrics to zero", () => {
    render(
      <MetricsPanel
        wallClockMs={42.5}
        metrics={{
          ...metrics,
          tpot_mean: null,
          throughput: undefined,
        }}
      />,
    );

    expect(metricValue("TPOT 均值")).toHaveTextContent("未定义");
    expect(metricValue("完成 QPS")).toHaveTextContent("未定义");
  });

  it("displays values supplied by the service without recomputing them", () => {
    render(
      <MetricsPanel
        wallClockMs={42.5}
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
    expect(metricValue("完成 QPS")).toHaveTextContent("905 请求/秒");
  });
});
