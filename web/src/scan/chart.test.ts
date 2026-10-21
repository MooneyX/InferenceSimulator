import { fireEvent, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { JobSummary } from "../api/types";
import type { ScanData } from "./chart";
import { buildChart } from "./chart";
import { renderReport } from "./report";
import { resolveSelection } from "./selection";
import { formatScanTooltip } from "./tooltip.js";

const config = { inference: { tp_size: 1 } } as ScanData["points"][number]["config"];
const job: JobSummary = { id: "scan-a", kind: "scan", status: "completed", total: 4,
  completed: 4, failed: 0, error: null, cache: null, lookup_ms: null, version: 1,
  code_fingerprint: "engine-1", code_version_status: "current",
  axes: [{ key: "num_instances", values: [1, 2] },
    { key: "scheduler.max_running_requests", values: [1, 2] },
    { key: "scheduler.pd_mutually_exclusive", values: [true] },
    { key: "inference.parallelism", values: [1] }] };
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
    expect(chart.series.every((series) => series.smooth && !series.connectNulls)).toBe(true);
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
    expect(html).toContain("smooth: true, connectNulls: false");
    expect(html).toContain('<h3>背景参数</h3>');
    expect(html).not.toContain('背景参数（每项选择一个值）');
  });

  it("renders actual node and line styles in the axis tooltip without interpreting labels as HTML", () => {
    const html = formatScanTooltip([
      { axisValueLabel: "<轴>", seriesIndex: 0, value: 12 },
      { seriesIndex: 1, value: null },
    ], [
      { name: '<img src=x onerror=alert(1)>', symbol: "triangle",
        lineStyle: { color: "#ef4444", type: "dashed" } },
      { name: "second", symbol: "emptyCircle",
        lineStyle: { color: "#10b981", type: [10, 4, 2, 4] } },
    ]);
    expect(html).toContain('&lt;轴&gt;');
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');
    expect(html).not.toContain('<img');
    expect(html).toContain('stroke="#ef4444"');
    expect(html).toContain('stroke-dasharray="8,5"');
    expect(html).toContain('d="M28 5 L34 16 H22 Z"');
    expect(html).toContain('stroke-dasharray="10,4,2,4"');
    expect(html).toContain('fill="none" stroke="#10b981"');
    expect(html).toContain('second: —');
    expect(html).toContain('width:max-content');
    expect(html).toContain('white-space:normal;overflow-wrap:anywhere');
  });

  it("applies the report's value and background constraints to the live chart", () => {
    const other = (x: number, color: number, signature: string) => ({
      ...point(x, color, signature), values: { ...point(x, color, signature).values,
        "scheduler.pd_mutually_exclusive": false, "inference.parallelism": 2 },
    });
    const scans: ScanData[] = [{ job, points: [point(1, 1, "live-a"),
      { ...point(2, 2, "live-failed"), status: "failed", metrics: null }] },
    { job: { ...job, id: "scan-b" }, points: [other(1, 1, "live-b1"),
      other(2, 2, "live-b2")] }];
    const roles = { x: "num_instances", color: "scheduler.max_running_requests", shape: "", line: "" };
    const initial = resolveSelection(scans, roles, "throughput");
    expect(initial.background).toEqual({ "scheduler.pd_mutually_exclusive": "true",
      "inference.parallelism": "1" });
    expect(initial.visible.flatMap((batch) => batch.points)).toHaveLength(1);
    expect(initial.roleChoices.x.find((choice) => choice.value === 2)?.disabled).toBe(true);
    const empty = resolveSelection(scans, roles, "throughput", {}, {}, true);
    expect(empty.complete).toBe(false);
    expect(empty.visible.flatMap((batch) => batch.points)).toHaveLength(0);
    const partial = resolveSelection(scans, roles, "throughput",
      { "scheduler.pd_mutually_exclusive": "false" }, {}, true);
    expect(partial.backgroundChoices["inference.parallelism"].find((choice) => choice.value === 1)
      ?.disabled).toBe(true);
    const next = resolveSelection(scans, roles, "throughput",
      { "scheduler.pd_mutually_exclusive": "false", "inference.parallelism": "2" });
    expect(next.visible.flatMap((batch) => batch.points)).toHaveLength(2);
    expect(next.notice).toContain("2 组勾选值组合无数据");
    const narrowed = resolveSelection(scans, roles, "throughput", next.background,
      { ...next.selected, num_instances: ['2'] });
    expect(narrowed.visible.flatMap((batch) => batch.points)).toHaveLength(1);
    expect(narrowed.roleChoices.color.find((choice) => choice.value === 1)?.disabled).toBe(true);
  });

  function openReport(scanData: ScanData[]) {
    const html = new DOMParser().parseFromString(renderReport(scanData, []), "text/html");
    document.body.innerHTML = html.body.innerHTML;
    const setOption = vi.fn();
    Object.defineProperty(window, "echarts", { configurable: true, value: {
      init: () => ({ setOption, resize: vi.fn() }),
    } });
    new Function([...document.querySelectorAll("script")].at(-1)!.textContent!)();
    const group = (id: string) => within(document.getElementById(id)!);
    const background = (key: string) => within([...document.querySelectorAll<HTMLElement>(".background-group")]
      .find(item => item.querySelector("legend")?.textContent === key)!);
    return { group, background, setOption, rows: () => document.querySelectorAll("#data-body tr") };
  }

  it("uses the shared shape and dash sample for the offline chart's hover legend", () => {
    const other = { ...point(1, 2, "styled"), values: {
      ...point(1, 2, "styled").values,
      "scheduler.pd_mutually_exclusive": false, "inference.parallelism": 2,
    } };
    const { setOption } = openReport([{ job, points: [point(1, 1, "plain"), other] }]);
    const option = setOption.mock.lastCall?.[0] as { tooltip: { formatter: (params: unknown) => string;
      extraCssText: string };
      series: { name: string; symbol: string; lineStyle: { color: string; type: string | number[] } }[] };
    expect(option.tooltip.extraCssText).toContain('white-space:normal');
    expect(option.series).toHaveLength(2);
    const tooltip = option.tooltip.formatter([{ seriesIndex: 0, axisValueLabel: "1", value: 1 },
      { seriesIndex: 1, axisValueLabel: "1", value: 2 }]);
    expect(tooltip).toContain('stroke-dasharray="8,5"');
    expect(tooltip).toContain('<rect x="24" y="7"');
    expect(tooltip).toContain('stroke="#f59e0b"');
  });

  it("selects one value per background parameter and disables incompatible values", () => {
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
    const { group, background, rows } = openReport([first, second]);
    const shape = () => background("scheduler.pd_mutually_exclusive");
    const parallelism = () => background("inference.parallelism");
    expect(group("role-x").getByRole("option", { name: "scheduler.max_running_requests" }))
      .toBeDisabled();
    fireEvent.change(document.getElementById("role-shape")!, { target: { value: "" } });
    fireEvent.change(document.getElementById("role-line")!, { target: { value: "" } });
    expect(group("background-values").getAllByRole("radio")).toHaveLength(4);
    expect(shape().getByRole("radio", { name: "true" })).toBeChecked();
    expect(parallelism().getByRole("radio", { name: "1" })).toBeChecked();
    expect(shape().getByRole("radio", { name: "false" })).toBeDisabled();
    expect(parallelism().getByRole("radio", { name: "2" })).toBeDisabled();
    expect(rows()).toHaveLength(1);
    expect(group("values-x").getByRole("checkbox", { name: "2" })).toBeDisabled();
    expect(group("values-color").getByRole("checkbox", { name: "2" })).toBeDisabled();
    expect(group("values-x").getByRole("checkbox", { name: "3" })).toBeDisabled();
    fireEvent.click(document.getElementById("reset-background")!);
    expect(rows()).toHaveLength(0);
    expect(document.getElementById("csv")).toBeDisabled();
    expect(shape().getByRole("radio", { name: "false" })).toBeEnabled();
    fireEvent.click(shape().getByRole("radio", { name: "false" }));
    expect(rows()).toHaveLength(0);
    expect(parallelism().getByRole("radio", { name: "1" })).toBeDisabled();
    fireEvent.click(parallelism().getByRole("radio", { name: "2" }));
    expect(rows()).toHaveLength(2);
    expect(document.getElementById("selection-notice")).toHaveTextContent("2 组勾选值组合无数据");
    fireEvent.click(group("values-x").getByRole("checkbox", { name: "1" }));
    expect(rows()).toHaveLength(1);
    expect(group("values-color").getByRole("checkbox", { name: "1" })).toBeDisabled();
    fireEvent.click(document.getElementById("reset-values")!);
    expect(rows()).toHaveLength(2);
    fireEvent.click(document.getElementById("reset-background")!);
    fireEvent.click(parallelism().getByRole("radio", { name: "1" }));
    expect(shape().getByRole("radio", { name: "false" })).toBeDisabled();
    fireEvent.click(shape().getByRole("radio", { name: "true" }));
    expect(rows()).toHaveLength(1);
    fireEvent.click(group("batches").getByRole("checkbox", { name: /scan-a/ }));
    expect(shape().getByRole("radio", { name: "true" })).toBeDisabled();
    expect(shape().getByRole("radio", { name: "false" })).toBeChecked();
    expect(rows()).toHaveLength(1);
  });

  it("restricts background values by the currently selected axis levels", () => {
    const axes = [job.axes![0], job.axes![2]];
    const first: ScanData = { job: { ...job, axes }, points: [point(1, 1, "axis-first")] };
    const second: ScanData = { job: { ...job, id: "scan-b", axes }, points: [{
      ...point(2, 1, "axis-second"), values: { ...point(2, 1, "axis-second").values,
        "scheduler.pd_mutually_exclusive": false },
    }] };
    const { group, background, rows } = openReport([first, second]);
    const shape = () => background("scheduler.pd_mutually_exclusive");
    fireEvent.change(document.getElementById("role-color")!, { target: { value: "" } });
    expect(shape().getByRole("radio", { name: "false" })).toBeDisabled();
    fireEvent.click(document.getElementById("reset-background")!);
    expect(shape().getByRole("radio", { name: "false" })).toBeEnabled();
    fireEvent.click(shape().getByRole("radio", { name: "false" }));
    expect(rows()).toHaveLength(1);
    expect(group("values-x").getByRole("checkbox", { name: "2" })).toBeChecked();
    expect(shape().getByRole("radio", { name: "true" })).toBeDisabled();
  });

  it("defaults to four roles and requires a single value for a fifth parameter offline", () => {
    const key = "scheduler.max_prefill_tokens";
    const scan: ScanData = { job: { ...job, axes: [...job.axes!, { key, values: [1, 2] }] },
      points: [1, 2].map((level) => ({ ...point(1, 1, `fifth-${level}`),
        values: { ...point(1, 1, `fifth-${level}`).values, [key]: level } })) };
    const { background, rows } = openReport([scan]);
    expect((document.getElementById("role-x") as HTMLSelectElement).value).toBe("num_instances");
    expect((document.getElementById("role-color") as HTMLSelectElement).value)
      .toBe("scheduler.max_running_requests");
    expect((document.getElementById("role-shape") as HTMLSelectElement).value)
      .toBe("scheduler.pd_mutually_exclusive");
    expect((document.getElementById("role-line") as HTMLSelectElement).value)
      .toBe("inference.parallelism");
    expect(background(key).getByRole("radio", { name: "1" })).toBeChecked();
    expect(rows()).toHaveLength(1);
    fireEvent.click(document.getElementById("reset-background")!);
    fireEvent.click(background(key).getByRole("radio", { name: "2" }));
    expect(rows()).toHaveLength(1);
  });

  it("limits selected visual levels to distinguishable styles", () => {
    const scans: ScanData[] = [{ job: { ...job, axes: [job.axes![0],
      { key: "scheduler.max_running_requests", values: [1, 2, 3, 4, 5, 6, 7, 8] }] },
      points: Array.from({ length: 8 }, (_, index) => point(1, index + 1, `color-${index}`)) }];
    const { group } = openReport(scans);
    const colors = group("values-color").getAllByRole("checkbox");
    expect(colors.filter(input => (input as HTMLInputElement).checked)).toHaveLength(7);
    expect(group("values-color").getByRole("checkbox", { name: "8" })).toBeDisabled();
  });
});
