import { describe, expect, it } from "vitest";

import type { JobSummary } from "../api/types";
import type { ScanData } from "./chart";
import { buildChart } from "./chart";
import { renderReport } from "./report";

const config = { inference: { tp_size: 1 } } as ScanData["points"][number]["config"];
const job: JobSummary = { id: "scan-a", kind: "scan", status: "completed", total: 4,
  completed: 4, failed: 0, error: null, cache: null, lookup_ms: null, version: 1,
  axes: { x: { key: "num_instances", values: [1, 2] },
    color: { key: "scheduler.max_running_requests", values: [1, 2] },
    shape: { key: "scheduler.pd_mutually_exclusive", values: [true] },
    line: { key: "inference.parallelism", values: [1] } } };
const point = (x: number, color: number, signature: string) => ({
  status: "completed" as const, signature, config,
  values: { num_instances: x, "scheduler.max_running_requests": color,
    "scheduler.pd_mutually_exclusive": true, "inference.parallelism": 1 },
  metrics: { throughput: x * color },
});
const data: ScanData[] = [{ job, points: [point(1, 1, "first"), point(2, 1, "second"),
  point(1, 2, "third"), { ...point(2, 2, "fourth"), status: "failed", metrics: null }] },
{ job: { ...job, id: "scan-b" }, points: [point(1, 1, "first")] }];

describe("scan projections", () => {
  it("deduplicates identical points across scans and leaves failed points empty", () => {
    const chart = buildChart(data, { x: "num_instances", color: "scheduler.max_running_requests",
      shape: "scheduler.pd_mutually_exclusive", line: "inference.parallelism" }, "throughput");
    expect(chart.deduplicated).toBe(1);
    expect(chart.xValues).toEqual(["1", "2"]);
    expect(chart.series).toHaveLength(2);
    expect(chart.series.some((series) => series.data[1] === null)).toBe(true);
    expect(new Set(chart.series.map((series) => series.itemStyle.color)).size).toBe(2);
  });

  it("embeds all scripts and data inside an offline HTML document", () => {
    const html = renderReport(data, []);
    expect(html).toContain('id="role-line"');
    expect(html).toContain('id="scan-data"');
    expect(html).toContain("scan-data.csv");
    expect(html).not.toMatch(/<script[^>]+src=/i);
    expect(html).toContain("window.echarts.init");
  });
});
