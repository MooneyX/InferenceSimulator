import { fireEvent, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { JobSummary } from "../api/types";
import type { ScanData } from "./chart";
import { availableKeys, buildChart, metricChannel, metricCharts, metricUnit } from "./chart";
import { renderReport } from "./report";
import { resolveSelection } from "./selection";
import { formatScanTooltip } from "./tooltip.js";

const config = { inference: { tp_size: 1 } } as ScanData["points"][number]["config"];
const job: JobSummary = { id: "scan-a", kind: "scan", status: "completed", total: 4,
  completed: 4, failed: 0, error: null, cache: null, lookup_ms: null, version: 1,
  code_fingerprint: "engine-1", code_version_status: "current",
  point_archive_version: 1,
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
const differentBases: ScanData[] = [10, 20].map((limit, index) => ({
  job: { ...job, id: `scan-${index}`, axes: [job.axes![0]] },
  points: [1, 2].map((x) => ({ ...point(x, 1, `base-${index}-${x}`),
    values: { num_instances: x },
    config: { num_instances: x, inference: { tp_size: 1 },
      scheduler: { max_prefill_tokens: limit }, workload: { seed: index + 7 } } as ScanData["points"][number]["config"],
  })),
}));

describe("scan projections", () => {
  it("includes changed background config fields, not constant axis or config fields", () => {
    const keys = availableKeys(differentBases);
    expect(keys).toEqual(["num_instances", "scheduler.max_prefill_tokens", "workload.seed"]);
    const roles = { x: "num_instances", color: "", shape: "", line: "" };
    const first = resolveSelection(differentBases, roles, "throughput", {}, {}, false, keys);
    expect(first.hidden).toEqual(keys.slice(1));
    expect(first.visible.map((batch) => batch.points.length)).toEqual([2, 0]);
    expect(buildChart(first.visible, roles, "throughput", keys).entries).toHaveLength(2);
    const next = resolveSelection(differentBases, roles, "throughput",
      { "scheduler.max_prefill_tokens": "20", "workload.seed": "8" }, {}, false, keys);
    expect(next.visible.map((batch) => batch.points.length)).toEqual([0, 2]);
    const asRoles = resolveSelection(differentBases, { x: "num_instances",
      color: "scheduler.max_prefill_tokens", shape: "workload.seed", line: "" },
    "throughput", {}, {}, false, keys);
    expect(asRoles.visible.flatMap((batch) => batch.points)).toHaveLength(4);
  });
  it("distinguishes array-valued background configuration by value in chart roles", () => {
    const scans: ScanData[] = differentBases.map(({ job: scan, points }, index) => ({ job: scan,
      points: points.map((entry) => ({ ...entry, config: {
        ...entry.config, inference: { model: { attention_groups: [index + 1] } },
      } as unknown as ScanData["points"][number]["config"] })),
    }));
    const key = "inference.model.attention_groups";
    expect(availableKeys(scans)).toContain(key);
    const chart = buildChart(scans, { x: "num_instances", color: key, shape: "", line: "" }, "throughput");
    expect(chart.series).toHaveLength(2);
    expect(new Set(chart.series.map((series) => series.itemStyle.color)).size).toBe(2);
    expect(chart.series[0].tooltipValues).toEqual(["[1]"]);
  });

  it("keeps a single-value scan viewable when no parameter varies", () => {
    const one: ScanData[] = [{ job: { ...job, axes: [job.axes![0]] }, points: [point(1, 1, "only")] }];
    expect(availableKeys(one)).toEqual(["num_instances"]);
    expect(resolveSelection(one, { x: "num_instances", color: "", shape: "", line: "" },
      "throughput").visible[0].points).toHaveLength(1);
  });

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

  it("uses available metrics from deduplicated scans without averaging", () => {
    const scans: ScanData[] = [
      { job, points: [{ ...point(1, 1, "shared"), metrics: { throughput: 2 } }] },
      { job: { ...job, id: "scan-b" }, points: [{ ...point(1, 1, "shared"),
        metrics: { ttft_mean: 12 } }] },
    ];
    const roles = { x: "num_instances", color: "", shape: "", line: "" };
    const chart = buildChart(scans, roles, ["throughput", "ttft_mean"]);
    expect(chart.deduplicated).toBe(1);
    expect(chart.series.map((series) => series.data)).toEqual([[2], [12]]);
    const { group, rows, setOption } = openReport(scans);
    fireEvent.click(group("metric-options").getByRole("checkbox", { name: /ttft_mean/ }));
    expect(rows()).toHaveLength(2);
    expect(rows()[1]).toHaveTextContent("12");
    expect((setOption.mock.lastCall?.[0] as { series: { data: number[] }[] }).series
      .map((series) => series.data)).toEqual([[2], [12]]);
  });

  it("uses only four ordered role values for live chart tooltips", () => {
    const background = "scheduler.max_prefill_tokens";
    const roles = { x: "num_instances", color: "scheduler.max_running_requests",
      shape: "scheduler.pd_mutually_exclusive", line: "inference.parallelism" };
    const scan: ScanData = { job: { ...job, axes: [...job.axes!, { key: background, values: [99] }] },
      points: [{ ...point(1, 2, "live-tooltip"), values: { ...point(1, 2, "live-tooltip").values,
        [background]: 99 } }] };
    const curve = buildChart([scan], roles, "throughput", [...scan.job.axes!.map((axis) => axis.key)]).series[0];
    expect(curve.name).toContain(background);
    expect(curve.tooltipValues).toEqual(["2", "true", "1"]);
    const html = formatScanTooltip({ axisValueLabel: "1", seriesIndex: 0, value: 2 }, [curve]);
    expect(html).toContain("2 · true · 1: 2");
    expect(html).not.toContain(background);
    expect(html).not.toContain("scheduler.pd_mutually_exclusive");
    expect(html).not.toContain("扫描 scan-a");
  });

  it("shares axes by unit, uses an unused style and keeps missing metrics as gaps", () => {
    const roles = { x: "num_instances", color: "scheduler.max_running_requests", shape: "", line: "" };
    const scans: ScanData[] = [{ job, points: [
      { ...point(1, 1, "a"), metrics: { throughput: 2, arrival_qps: 3, ttft_mean: 12,
        actual_prefix_hit_rate: 0.25 } },
      { ...point(2, 1, "b"), metrics: { throughput: 4, arrival_qps: 6,
        actual_prefix_hit_rate: 0.5 } },
    ] }];
    expect(metricUnit("actual_prefix_hit_rate")).toBe("%");
    expect(metricUnit("l3_capacity_blocks")).toBe("页");
    expect(metricUnit("restore_bytes")).toBe("字节");
    expect(metricCharts(["throughput", "arrival_qps", "ttft_mean"], roles))
      .toEqual([["throughput", "arrival_qps", "ttft_mean"]]);
    const metrics = ["throughput", "arrival_qps", "ttft_mean"];
    const model = buildChart(scans, roles, metrics);
    expect(model.series.map((series) => series.yAxisIndex)).toEqual([0, 0, 1]);
    expect(model.series.map((series) => series.lineStyle.type)).toEqual(["solid", "dashed", "dotted"]);
    expect(model.series[2].data).toEqual([12, null]);
    expect(model.series[1].name).toContain("arrival_qps");
    expect(model.series[1].tooltipValues[0]).toBe("arrival_qps (请求/秒)");
    expect(buildChart(scans, roles, "actual_prefix_hit_rate").series[0].data).toEqual([25, 50]);
    expect(resolveSelection(scans, roles, ["ttft_mean", "throughput"], {}, {}, false,
      [roles.x, roles.color]).visible[0].points).toHaveLength(2);
  });

  it("splits by unit or per metric without an available or sufficient style channel", () => {
    const roles = { x: "num_instances", color: "scheduler.max_running_requests", shape: "", line: "" };
    expect(metricCharts(["throughput", "ttft_mean", "restore_bytes"], roles))
      .toEqual([["throughput"], ["ttft_mean"], ["restore_bytes"]]);
    expect(metricChannel(roles, 5)).toBe("shape");
    expect(metricCharts(["throughput", "arrival_qps"],
      { ...roles, shape: "scheduler.pd_mutually_exclusive", line: "inference.parallelism" }))
      .toEqual([["throughput"], ["arrival_qps"]]);
    const stable = buildChart([{ job, points: [{ ...point(1, 1, "stable"),
      metrics: { throughput: 3, arrival_qps: 4 } }] }], roles,
    ["throughput", "arrival_qps"], undefined, { throughput: 1, arrival_qps: 3 });
    expect(stable.series.map((series) => series.lineStyle.type)).toEqual(["dashed", [10, 4, 2, 4]]);
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
      { name: '扫描 <img src=x onerror=alert(1)> · hidden=99',
        tooltipValues: ["<颜色>", "true", "2"], symbol: "triangle",
        lineStyle: { color: "#ef4444", type: "dashed" } },
      { name: "second · fixed=99", tooltipValues: ["false", "3"], symbol: "emptyCircle",
        lineStyle: { color: "#10b981", type: [10, 4, 2, 4] } },
    ]);
    expect(html).toContain('&lt;轴&gt;');
    expect(html).toContain('&lt;颜色&gt; · true · 2: 12');
    expect(html).not.toContain('扫描');
    expect(html).not.toContain('hidden=99');
    expect(html).not.toContain('&lt;img');
    expect(html).toContain('stroke="#ef4444"');
    expect(html).toContain('stroke-dasharray="8,5"');
    expect(html).toContain('d="M28 5 L34 16 H22 Z"');
    expect(html).toContain('stroke-dasharray="10,4,2,4"');
    expect(html).toContain('fill="none" stroke="#10b981"');
    expect(html).toContain('false · 3: —');
    expect(html).not.toContain('fixed=99');
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

  it("uses GB for GPU memory from point configs in live charts and offline reports", () => {
    const key = "inference.hardware.memory_bytes_per_gpu";
    const scans: ScanData[] = [{ job: { ...job, axes: [{ key, values: [128, 288] }] },
      points: [128, 288].map((memory) => ({ ...point(memory, 1, `memory-${memory}`), values: {},
        config: { inference: { hardware: { memory_bytes_per_gpu: memory * 1e9 } } } as ScanData["points"][number]["config"],
      })),
    }];
    expect(buildChart(scans, { x: key, color: "", shape: "", line: "" }, "throughput").xValues)
      .toEqual(["128", "288"]);
    const { group, rows } = openReport(scans);
    expect(group("values-x").getByRole("checkbox", { name: "128" })).toBeChecked();
    expect(group("values-x").getByRole("checkbox", { name: "288" })).toBeChecked();
    expect(rows()).toHaveLength(2);
    expect(within(rows()[0] as HTMLElement).getAllByRole("cell")[1]).toHaveTextContent(/^128$/);
    expect(within(rows()[1] as HTMLElement).getAllByRole("cell")[1]).toHaveTextContent(/^288$/);
  });

  it("filters identical scan axes by distinct base configs in the offline table and CSV selection", () => {
    const { group, background, rows } = openReport(differentBases);
    expect(group("role-color").getByRole("option", { name: "scheduler.max_prefill_tokens" }))
      .toBeInTheDocument();
    expect(group("role-shape").getByRole("option", { name: "workload.seed" }))
      .toBeInTheDocument();
    expect(rows()).toHaveLength(4);
    fireEvent.change(document.getElementById("role-color")!, { target: { value: "" } });
    fireEvent.change(document.getElementById("role-shape")!, { target: { value: "" } });
    expect(rows()).toHaveLength(2);
    expect(rows()[0]).toHaveTextContent("scan-0");
    fireEvent.click(document.getElementById("reset-background")!);
    expect(rows()).toHaveLength(0);
    fireEvent.click(background("scheduler.max_prefill_tokens").getByRole("radio", { name: "20" }));
    fireEvent.click(background("workload.seed").getByRole("radio", { name: "8" }));
    expect(rows()).toHaveLength(2);
    expect(rows()[0]).toHaveTextContent("scan-1");
    expect(document.getElementById("csv")).toBeEnabled();
  });

  it("uses the shared shape and dash sample for the offline chart's hover legend", () => {
    const other = { ...point(2, 2, "styled"), values: {
      ...point(2, 2, "styled").values,
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
      points: [1, 2].map((level) => ({ ...point(level, level, `fifth-${level}`),
        values: { ...point(level, level, `fifth-${level}`).values,
          "scheduler.pd_mutually_exclusive": level === 1,
          "inference.parallelism": level, [key]: level } })) };
    const { background, rows, setOption } = openReport([scan]);
    expect((document.getElementById("role-x") as HTMLSelectElement).value).toBe("num_instances");
    expect((document.getElementById("role-color") as HTMLSelectElement).value)
      .toBe("scheduler.max_running_requests");
    expect((document.getElementById("role-shape") as HTMLSelectElement).value)
      .toBe("scheduler.pd_mutually_exclusive");
    expect((document.getElementById("role-line") as HTMLSelectElement).value)
      .toBe("inference.parallelism");
    expect(background(key).getByRole("radio", { name: "1" })).toBeChecked();
    expect(rows()).toHaveLength(1);
    const option = setOption.mock.lastCall?.[0] as { tooltip: { formatter: (params: unknown) => string };
      series: { name: string; tooltipValues: string[] }[] };
    expect(option.series[0].name).toContain(key);
    expect(option.series[0].tooltipValues).toEqual(["1", "true", "1"]);
    const tooltip = option.tooltip.formatter({ axisValueLabel: "1", seriesIndex: 0, value: 1 });
    expect(tooltip).toContain("1 · true · 1: 1");
    expect(tooltip).not.toContain(key);
    expect(tooltip).not.toContain("num_instances");
    expect(tooltip).not.toContain("扫描 scan-a");
    fireEvent.click(document.getElementById("reset-background")!);
    fireEvent.click(background(key).getByRole("radio", { name: "2" }));
    expect(rows()).toHaveLength(1);
  });

  it("renders offline dual axes and then splits three units while keeping every selected metric in the table", () => {
    const scans: ScanData[] = [{ job, points: [{ ...point(1, 1, "multi"), metrics: {
      throughput: 2, ttft_mean: 10, restore_bytes: 2048,
    } }] }];
    const { group, setOption, rows } = openReport(scans);
    const metrics = () => group("metric-options");
    expect(metrics().getByRole("checkbox", { name: /throughput/ })).toBeDisabled();
    fireEvent.click(metrics().getByRole("checkbox", { name: /ttft_mean/ }));
    let option = setOption.mock.lastCall?.[0] as { yAxis: { name: string }[];
      series: { name: string; yAxisIndex: number; lineStyle: { type: string } }[] };
    expect(document.querySelectorAll(".chart")).toHaveLength(1);
    expect(option.yAxis.map((axis) => axis.name)).toEqual(["请求/秒", "ms"]);
    expect(option.series.map((series) => series.yAxisIndex)).toEqual([0, 1]);
    expect(option.series.map((series) => series.lineStyle.type)).toEqual(["solid", "dashed"]);
    expect(rows()).toHaveLength(2);
    fireEvent.click(metrics().getByRole("checkbox", { name: /restore_bytes/ }));
    expect(document.querySelectorAll(".chart")).toHaveLength(3);
    expect(rows()).toHaveLength(3);
    expect(rows()[2]).toHaveTextContent("restore_bytes (字节)");
    expect(document.getElementById("metric-summary")).toHaveTextContent("已选 3 项");
    fireEvent.click(metrics().getByRole("checkbox", { name: /ttft_mean/ }));
    expect(document.querySelectorAll(".chart")).toHaveLength(1);
    option = setOption.mock.lastCall?.[0] as typeof option;
    expect(option.yAxis.map((axis) => axis.name)).toEqual(["请求/秒", "字节"]);
  });

  it("splits offline metrics when all styles are assigned to parameters", () => {
    const scans: ScanData[] = [{ job, points: [
      { ...point(1, 1, "all-a"), metrics: { throughput: 1, arrival_qps: 2 } },
      { ...point(2, 2, "all-b"), values: { ...point(2, 2, "all-b").values,
        "scheduler.pd_mutually_exclusive": false, "inference.parallelism": 2 },
      metrics: { throughput: 3, arrival_qps: 4 } },
    ] }];
    const { group } = openReport(scans);
    fireEvent.click(group("metric-options").getByRole("checkbox", { name: /arrival_qps/ }));
    expect(document.querySelectorAll(".chart")).toHaveLength(2);
    expect(document.getElementById("selection-notice")).not.toHaveTextContent("释放通道");
  });

  it("limits selected visual levels to distinguishable styles", () => {
    const scans: ScanData[] = [{ job: { ...job, axes: [job.axes![0],
      { key: "scheduler.max_running_requests", values: [1, 2, 3, 4, 5, 6, 7, 8] }] },
      points: Array.from({ length: 8 }, (_, index) => point(index % 2 + 1, index + 1, `color-${index}`)) }];
    const { group } = openReport(scans);
    const colors = group("values-color").getAllByRole("checkbox");
    expect(colors.filter(input => (input as HTMLInputElement).checked)).toHaveLength(7);
    expect(group("values-color").getByRole("checkbox", { name: "8" })).toBeDisabled();
  });
});
