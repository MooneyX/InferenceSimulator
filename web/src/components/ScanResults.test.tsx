import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getScanParameters, getScanPoints } from "../api/client";
import type { JobSummary, ScanPoint } from "../api/types";
import { ScanResults } from "./ScanResults";

const { setOption } = vi.hoisted(() => ({ setOption: vi.fn() }));
vi.mock("../api/client", () => ({ getScanParameters: vi.fn(), getScanPoints: vi.fn() }));
vi.mock("echarts/core", () => ({ use: vi.fn(), init: () => ({ setOption,
  resize: vi.fn(), dispose: vi.fn() }) }));

const axes = [{ key: "num_instances", values: [1, 2] },
  { key: "scheduler.max_running_requests", values: [1, 2] },
  { key: "scheduler.pd_mutually_exclusive", values: [true, false] }];
const job = (id: string): JobSummary => ({ id, kind: "scan", status: "completed", total: 2,
  completed: 2, failed: 0, error: null, cache: null, lookup_ms: null, version: 1, axes,
  code_fingerprint: "engine-1", code_version_status: "current" });
const point = (x: number, color: number, shape: boolean, signature: string): ScanPoint => ({
  status: "completed", signature, values: { num_instances: x,
    "scheduler.max_running_requests": color, "scheduler.pd_mutually_exclusive": shape },
  metrics: { throughput: x * color }, config: null,
} as ScanPoint);

beforeEach(() => {
  setOption.mockClear();
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.mocked(getScanParameters).mockResolvedValue([
    { key: "num_instances", label: "实例数" },
    { key: "scheduler.max_running_requests", label: "同时运行请求" },
    { key: "scheduler.pd_mutually_exclusive", label: "分步执行" },
  ] as Awaited<ReturnType<typeof getScanParameters>>);
  vi.mocked(getScanPoints).mockImplementation(async (id) => id === "scan-a"
    ? [point(1, 1, true, "a-1"), point(2, 1, true, "a-2")]
    : [point(1, 2, true, "b-1"), point(2, 2, true, "b-2")]);
});

describe("ScanResults", () => {
  it("paginates scan checkboxes and applies global or page-only selection across pages", async () => {
    render(<ScanResults jobs={Array.from({ length: 12 }, (_, index) =>
      job(`scan-${String(index + 1).padStart(2, "0")}`))} />);
    const list = () => within(screen.getByRole("group", { name: "扫描结果列表" }));
    const checkboxes = () => list().getAllByRole("checkbox");
    const previous = screen.getByRole("button", { name: "上一页" });
    const next = screen.getByRole("button", { name: "下一页" });
    expect(checkboxes()).toHaveLength(10);
    expect(checkboxes().every((checkbox) => !(checkbox as HTMLInputElement).checked)).toBe(true);
    expect(screen.getByText("已选 0 / 12 项")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "全不选" })).toBeDisabled();
    expect(screen.queryByRole("img", { name: /参数扫描折线图/ })).not.toBeInTheDocument();
    expect(getScanPoints).not.toHaveBeenCalled();
    expect(previous).toBeDisabled();
    expect(screen.getByText("第 1 / 2 页")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "当页全选" }));
    expect(screen.getByText("已选 10 / 12 项")).toBeInTheDocument();
    fireEvent.click(next);
    expect(checkboxes()).toHaveLength(2);
    expect(checkboxes().every((checkbox) => !(checkbox as HTMLInputElement).checked)).toBe(true);
    expect(next).toBeDisabled();
    fireEvent.click(checkboxes()[0]);
    expect(screen.getByText("已选 11 / 12 项")).toBeInTheDocument();
    fireEvent.click(previous);
    expect(checkboxes().every((checkbox) => (checkbox as HTMLInputElement).checked)).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "当页全不选" }));
    expect(screen.getByText("已选 1 / 12 项")).toBeInTheDocument();
    fireEvent.click(next);
    expect(checkboxes()[0]).toBeChecked();
    expect(checkboxes()[1]).not.toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    expect(screen.getByText("已选 12 / 12 项")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "全不选" }));
    expect(screen.getByText("已选 0 / 12 项")).toBeInTheDocument();
    expect(checkboxes().every((checkbox) => !(checkbox as HTMLInputElement).checked)).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "当页全选" }));
    expect(screen.getByText("已选 2 / 12 项")).toBeInTheDocument();
    fireEvent.click(previous);
    expect(checkboxes().every((checkbox) => !(checkbox as HTMLInputElement).checked)).toBe(true);
  });

  it("clamps the current page when scans are removed", () => {
    const jobs = Array.from({ length: 12 }, (_, index) => job(`scan-${index + 1}`));
    const { rerender } = render(<ScanResults jobs={jobs} />);
    fireEvent.click(screen.getByRole("button", { name: "下一页" }));
    rerender(<ScanResults jobs={jobs.slice(0, 2)} />);
    expect(screen.getByText("第 1 / 1 页")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "下一页" })).toBeDisabled();
    expect(within(screen.getByRole("group", { name: "扫描结果列表" })).getAllByRole("checkbox"))
      .toHaveLength(2);
  });

  it("shows archived points and export for an interrupted older scan", async () => {
    render(<ScanResults jobs={[{ ...job("scan-a"), status: "interrupted", completed: 2,
      total: 4, code_version_status: "stale" }]} />);
    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    await waitFor(() => expect(getScanPoints).toHaveBeenCalledWith("scan-a"));
    expect(screen.getByText(/旧代码版本/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "导出离线可操作 HTML" })).toBeEnabled();
  });

  it("filters selected batches by a single background value and restores multi-value roles", async () => {
    render(<ScanResults jobs={[job("scan-a"), job("scan-b")]} />);
    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    await waitFor(() => expect(screen.getByText(/独立配置点 4/)).toBeInTheDocument());
    expect(screen.getByRole("combobox", { name: /颜色/ })).toHaveValue("scheduler.max_running_requests");
    expect(screen.getByRole("combobox", { name: /节点形状/ })).toHaveValue("");
    expect(screen.queryByRole("option", { name: "分步执行" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: /颜色/ }), { target: { value: "" } });
    expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument();
    const background = () => within(screen.getByRole("group", { name: "同时运行请求" }));
    expect(background().getByRole("radio", { name: "1" })).toBeChecked();
    expect(background().getByRole("radio", { name: "2" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "重新选择背景值" }));
    expect(screen.getByText("背景参数尚未选齐，图表暂不显示。")).toBeInTheDocument();
    fireEvent.click(background().getByRole("radio", { name: "2" }));
    expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument();
    const colorSelect = screen.getByRole("combobox", { name: /颜色/ });
    fireEvent.change(colorSelect, { target: { value: "scheduler.max_running_requests" } });
    expect(screen.getByText(/独立配置点 4/)).toBeInTheDocument();
    const colors = screen.getByLabelText("颜色参数值");
    expect(within(colors).getByRole("checkbox", { name: "1" })).toBeChecked();
    expect(within(colors).getByRole("checkbox", { name: "2" })).toBeChecked();
    fireEvent.click(within(colors).getByRole("checkbox", { name: "1" }));
    expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "分步执行" })).not.toBeInTheDocument();
  });

  it("keeps the fifth scan parameter as a single-choice background value", async () => {
    const key = "scheduler.max_prefill_tokens";
    vi.mocked(getScanPoints).mockResolvedValue([1, 2].map((value) => ({
      ...point(value, value, value === 1, `background-${value}`),
      values: { ...point(value, value, value === 1, `background-${value}`).values,
        "inference.parallelism": value, [key]: value },
    })));
    render(<ScanResults jobs={[{ ...job("scan-a"), axes: [...axes,
      { key: "inference.parallelism", values: [1] }, { key, values: [1, 2] }] }]} />);
    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    await waitFor(() => expect(screen.getByText(/独立配置点 1/)).toBeInTheDocument());
    expect(screen.getByRole("combobox", { name: /线样式/ })).toHaveValue("inference.parallelism");
    const group = within(screen.getByRole("group", { name: key }));
    expect(group.getByRole("radio", { name: "1" })).toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "重新选择背景值" }));
    fireEvent.click(group.getByRole("radio", { name: "2" }));
    expect(screen.getByText(/独立配置点 1/)).toBeInTheDocument();
  });

  it("shows varying base settings as selectable roles and filters scans when used as background", async () => {
    vi.mocked(getScanPoints).mockImplementation(async (id) => [1, 2].map((x) => ({
      ...point(x, 1, true, `${id}-${x}`), values: { num_instances: x },
      config: { num_instances: x, scheduler: { max_prefill_tokens: id === "scan-a" ? 10 : 20 },
        workload: { seed: id === "scan-a" ? 7 : 8 } } as ScanPoint["config"],
    })));
    vi.mocked(getScanParameters).mockResolvedValue([{ key: "scheduler.max_prefill_tokens",
      label: "单次 Prefill 额度" }] as Awaited<ReturnType<typeof getScanParameters>>);
    render(<ScanResults jobs={[job("scan-a"), job("scan-b")]} />);
    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    await waitFor(() => expect(screen.getByText(/独立配置点 4/)).toBeInTheDocument());
    const color = screen.getByRole("combobox", { name: /颜色/ });
    const shape = screen.getByRole("combobox", { name: /节点形状/ });
    expect(color).toHaveValue("scheduler.max_prefill_tokens");
    expect(shape).toHaveValue("workload.seed");
    expect(within(screen.getByLabelText("颜色参数值")).getByRole("checkbox", { name: "20" }))
      .toBeChecked();
    fireEvent.change(color, { target: { value: "" } });
    fireEvent.change(shape, { target: { value: "" } });
    expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "重新选择背景值" }));
    fireEvent.click(within(screen.getByRole("group", { name: "单次 Prefill 额度" }))
      .getByRole("radio", { name: "20" }));
    fireEvent.click(within(screen.getByRole("group", { name: "workload.seed" }))
      .getByRole("radio", { name: "8" }));
    expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument();
    expect(setOption.mock.lastCall?.[0].series[0].name).toContain("scan-b");
  });

  it("compares two units on dual axes and splits three units into separate charts", async () => {
    vi.mocked(getScanPoints).mockResolvedValue([1, 2].map((x) => ({
      ...point(x, x, x === 1, `multi-${x}`),
      metrics: { throughput: x, ttft_mean: x * 10, restore_bytes: x * 100 },
    })));
    render(<ScanResults jobs={[job("scan-a")]} />);
    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    await waitFor(() => expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument());
    const picker = within(screen.getByRole("group", { name: "纵轴指标" }));
    expect(picker.getByRole("checkbox", { name: /throughput/ })).toBeDisabled();
    fireEvent.click(picker.getByRole("checkbox", { name: /ttft_mean/ }));
    let option = setOption.mock.lastCall?.[0] as { yAxis: { name: string }[];
      series: { name: string; yAxisIndex: number; lineStyle: { type: string } }[] };
    expect(option.yAxis.map((axis) => axis.name)).toEqual(["请求/秒", "ms"]);
    expect(option.series.some((item) => item.lineStyle.type === "dashed")).toBe(true);
    expect(screen.getAllByRole("img", { name: /参数扫描折线图/ })).toHaveLength(1);
    fireEvent.click(picker.getByRole("checkbox", { name: /restore_bytes/ }));
    expect(screen.getAllByRole("img", { name: /参数扫描折线图/ })).toHaveLength(3);
    expect(screen.queryByText(/释放通道/)).not.toBeInTheDocument();
    fireEvent.click(picker.getByRole("checkbox", { name: /restore_bytes/ }));
    expect(screen.getAllByRole("img", { name: /参数扫描折线图/ })).toHaveLength(1);
    fireEvent.change(screen.getByRole("combobox", { name: /节点形状/ }),
      { target: { value: "" } });
    fireEvent.change(screen.getByRole("combobox", { name: /线样式/ }),
      { target: { value: "scheduler.pd_mutually_exclusive" } });
    option = setOption.mock.lastCall?.[0] as typeof option;
    expect(option.series.some((item) => item.name.includes("ttft_mean"))).toBe(true);
  });

  it("uses the series color and shape in the live hover legend", async () => {
    vi.mocked(getScanPoints).mockImplementation(async (id) => id === "scan-a"
      ? [point(1, 1, true, "plain")] : [point(1, 2, false, "styled")]);
    render(<ScanResults jobs={[job("scan-a"), job("scan-b")]} />);
    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    await waitFor(() => expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument());
    fireEvent.change(screen.getByRole("combobox", { name: /颜色/ }),
      { target: { value: "scheduler.max_running_requests" } });
    fireEvent.change(screen.getByRole("combobox", { name: /节点形状/ }),
      { target: { value: "scheduler.pd_mutually_exclusive" } });
    const option = setOption.mock.lastCall?.[0] as { tooltip: { formatter: (params: unknown) => string;
      extraCssText: string };
      series: { name: string; symbol: string; lineStyle: { color: string; type: string | number[] } }[] };
    expect(option.tooltip.extraCssText).toContain('white-space:normal');
    expect(option.series).toHaveLength(2);
    const tooltip = option.tooltip.formatter([{ seriesIndex: 0, axisValueLabel: "1", value: 1 },
      { seriesIndex: 1, axisValueLabel: "1", value: 2 }]);
    expect(tooltip).toContain('stroke="#6c63ff"');
    expect(tooltip).toContain('<circle cx="28"');
    expect(tooltip).toContain('<rect x="24" y="7"');
  });
});
