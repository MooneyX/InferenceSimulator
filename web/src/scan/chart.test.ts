import { fireEvent, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { JobSummary } from "../api/types";
import type { ScanData } from "./chart";
import { buildChart } from "./chart";
import { renderReport } from "./report";
import reportRuntime from "./reportRuntime.js?raw";

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
    expect(html).toContain('id="values-line"');
    expect(html).toContain('id="background-values"');
    expect(html).toContain('id="scan-data"');
    expect(html).toContain("scan-data.csv");
    expect(html).not.toMatch(/<script[^>]+src=/i);
    expect(html).toContain("window.echarts.init");
    expect(html).toContain("--bg:#0f1117");
    expect(html).toContain("backgroundColor: '#242837'");
  });

  function openReport(scanData: ScanData[]) {
    const html = new DOMParser().parseFromString(renderReport(scanData, []), "text/html");
    document.body.innerHTML = html.body.innerHTML;
    Object.defineProperty(window, "echarts", { configurable: true, value: {
      init: () => ({ setOption: vi.fn(), resize: vi.fn() }),
    } });
    new Function(reportRuntime)();
    const group = (id: string) => within(document.getElementById(id)!);
    return { group, rows: () => document.querySelectorAll("#data-body tr") };
  }

  it("only offers observed background groups and compatible values across batches", () => {
    const otherPoint = (x: number, color: number, signature: string) => ({
      ...point(x, color, signature), values: { ...point(x, color, signature).values,
        "scheduler.pd_mutually_exclusive": false, "inference.parallelism": 2 },
    });
    const second: ScanData = { job: { ...job, id: "scan-b" }, points: [
      otherPoint(1, 1, "b-first"), otherPoint(2, 2, "b-second"),
      { ...otherPoint(3, 3, "b-failed"), status: "failed", metrics: null },
    ] };
    const first: ScanData = { job, points: [point(1, 1, "a-first"),
      { ...point(2, 2, "a-failed"), status: "failed", metrics: null }] };
    const { group, rows } = openReport([first, second]);
    expect(group("role-x").getByRole("option", { name: "scheduler.max_running_requests" }))
      .toBeDisabled();
    fireEvent.change(document.getElementById("role-shape")!, { target: { value: "" } });
    fireEvent.change(document.getElementById("role-line")!, { target: { value: "" } });
    expect(group("background-values").getAllByRole("radio")).toHaveLength(2);
    expect(rows()).toHaveLength(1);
    expect(group("values-x").getByRole("checkbox", { name: "2" })).toBeDisabled();
    expect(group("values-color").getByRole("checkbox", { name: "2" })).toBeDisabled();
    expect(group("values-x").getByRole("checkbox", { name: "3" })).toBeDisabled();
    fireEvent.click(group("background-values").getByRole("radio", { name: /false.*2/ }));
    expect(rows()).toHaveLength(2);
    expect(document.getElementById("selection-notice")).toHaveTextContent("2 组勾选值组合无数据");
    fireEvent.click(group("values-x").getByRole("checkbox", { name: "1" }));
    expect(rows()).toHaveLength(1);
    expect(group("values-color").getByRole("checkbox", { name: "1" })).toBeDisabled();
    fireEvent.click(document.getElementById("reset-values")!);
    expect(rows()).toHaveLength(2);
    expect(group("values-color").getByRole("checkbox", { name: "1" })).toBeEnabled();
    fireEvent.click(group("background-values").getByRole("radio", { name: /true.*1/ }));
    expect(rows()).toHaveLength(1);
    expect(group("values-x").getByRole("checkbox", { name: "2" })).toBeDisabled();
    expect(group("values-x").getByRole("checkbox", { name: "1" })).toBeChecked();
    fireEvent.click(group("batches").getByRole("checkbox", { name: /scan-a/ }));
    expect(group("background-values").getByRole("radio", { name: /true.*1/ })).toBeDisabled();
    expect(group("background-values").getByRole("radio", { name: /false.*2/ })).toBeChecked();
    expect(rows()).toHaveLength(1);
  });

  it("limits selected visual levels to distinguishable styles", () => {
    const scans: ScanData[] = [{ job: { ...job, axes: { x: job.axes!.x!,
      color: { key: "scheduler.max_running_requests", values: [1, 2, 3, 4, 5, 6, 7, 8] } } },
      points: Array.from({ length: 8 }, (_, index) => point(1, index + 1, `color-${index}`)) }];
    const { group } = openReport(scans);
    const colors = group("values-color").getAllByRole("checkbox");
    expect(colors.filter(input => (input as HTMLInputElement).checked)).toHaveLength(7);
    expect(group("values-color").getByRole("checkbox", { name: "8" })).toBeDisabled();
  });
});
