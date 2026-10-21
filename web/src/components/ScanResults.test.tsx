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
  it("shows archived points and export for an interrupted older scan", async () => {
    render(<ScanResults jobs={[{ ...job("scan-a"), status: "interrupted", completed: 2,
      total: 4, code_version_status: "stale" }]} />);
    await waitFor(() => expect(getScanPoints).toHaveBeenCalledWith("scan-a"));
    expect(screen.getByText(/旧代码版本/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "导出离线可操作 HTML" })).toBeEnabled();
  });

  it("filters selected batches by a single background value and restores multi-value roles", async () => {
    render(<ScanResults jobs={[job("scan-a"), job("scan-b")]} />);
    await waitFor(() => expect(screen.getByText(/独立配置点 4/)).toBeInTheDocument());
    expect(screen.getByRole("combobox", { name: /颜色/ })).toHaveValue("scheduler.max_running_requests");
    expect(screen.getByRole("combobox", { name: /节点形状/ })).toHaveValue("scheduler.pd_mutually_exclusive");
    fireEvent.change(screen.getByRole("combobox", { name: /颜色/ }), { target: { value: "" } });
    fireEvent.change(screen.getByRole("combobox", { name: /节点形状/ }), { target: { value: "" } });
    expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument();
    const background = () => within(screen.getByRole("group", { name: "同时运行请求" }));
    expect(background().getByRole("radio", { name: "1" })).toBeChecked();
    expect(background().getByRole("radio", { name: "2" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "重新选择背景值" }));
    expect(screen.getByText("背景参数尚未选齐，图表暂不显示。")).toBeInTheDocument();
    fireEvent.click(background().getByRole("radio", { name: "2" }));
    fireEvent.click(within(screen.getByRole("group", { name: "分步执行" }))
      .getByRole("radio", { name: "true" }));
    expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument();
    const colorSelect = screen.getByRole("combobox", { name: /颜色/ });
    fireEvent.change(colorSelect, { target: { value: "scheduler.max_running_requests" } });
    expect(screen.getByText(/独立配置点 4/)).toBeInTheDocument();
    const colors = screen.getByLabelText("颜色参数值");
    expect(within(colors).getByRole("checkbox", { name: "1" })).toBeChecked();
    expect(within(colors).getByRole("checkbox", { name: "2" })).toBeChecked();
    fireEvent.click(within(colors).getByRole("checkbox", { name: "1" }));
    expect(screen.getByText(/独立配置点 2/)).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "分步执行" })).toBeInTheDocument();
  });

  it("keeps the fifth scan parameter as a single-choice background value", async () => {
    const key = "scheduler.max_prefill_tokens";
    vi.mocked(getScanPoints).mockResolvedValue([1, 2].map((value) => ({
      ...point(1, 1, true, `background-${value}`),
      values: { ...point(1, 1, true, `background-${value}`).values,
        "inference.parallelism": 1, [key]: value },
    })));
    render(<ScanResults jobs={[{ ...job("scan-a"), axes: [...axes,
      { key: "inference.parallelism", values: [1] }, { key, values: [1, 2] }] }]} />);
    await waitFor(() => expect(screen.getByText(/独立配置点 1/)).toBeInTheDocument());
    expect(screen.getByRole("combobox", { name: /线样式/ })).toHaveValue("inference.parallelism");
    const group = within(screen.getByRole("group", { name: key }));
    expect(group.getByRole("radio", { name: "1" })).toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "重新选择背景值" }));
    fireEvent.click(group.getByRole("radio", { name: "2" }));
    expect(screen.getByText(/独立配置点 1/)).toBeInTheDocument();
  });

  it("uses the series color and shape in the live hover legend", async () => {
    vi.mocked(getScanPoints).mockImplementation(async (id) => id === "scan-a"
      ? [point(1, 1, true, "plain")] : [point(1, 2, false, "styled")]);
    render(<ScanResults jobs={[job("scan-a"), job("scan-b")]} />);
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
