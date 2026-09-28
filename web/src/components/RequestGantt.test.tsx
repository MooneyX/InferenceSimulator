import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RequestRecord } from "../api/types";
import { RequestGantt } from "./RequestGantt";

const chart = vi.hoisted(() => ({
  setOption: vi.fn(),
  resize: vi.fn(),
  dispose: vi.fn(),
}));
const init = vi.hoisted(() => vi.fn(() => chart));
const use = vi.hoisted(() => vi.fn());
const clipRectByRect = vi.hoisted(() => vi.fn((shape: unknown) => shape));

vi.mock("echarts/core", () => ({
  init,
  use,
  graphic: { clipRectByRect },
}));
vi.mock("echarts/charts", () => ({ CustomChart: {} }));
vi.mock("echarts/components", () => ({
  DataZoomComponent: {},
  GridComponent: {},
  TooltipComponent: {},
}));
vi.mock("echarts/renderers", () => ({ CanvasRenderer: {} }));

function request(id: number, offset = 0): RequestRecord {
  return {
    id,
    arrival: offset,
    prefill_start: offset + 2,
    prefill_end: offset + 5,
    finish: offset + 11,
    prompt_len: 16,
    output_len: 8,
  };
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("RequestGantt", () => {
  it("renders an empty state without initializing ECharts", () => {
    render(<RequestGantt requests={[]} />);

    expect(screen.getByText("运行仿真后将在此显示请求生命周期。")).toBeInTheDocument();
    expect(init).not.toHaveBeenCalled();
  });

  it("initializes a canvas chart with filtered zoom controls and three phases", () => {
    render(<RequestGantt requests={[request(1), request(2, 20)]} />);

    expect(init).toHaveBeenCalledWith(expect.any(HTMLDivElement), undefined, {
      renderer: "canvas",
    });
    expect(screen.getByRole("img")).toHaveAccessibleName(
      "2 个请求的排队、Prefill 和 Decode 生命周期甘特图",
    );
    expect(chart.setOption).toHaveBeenCalledTimes(1);

    const option = chart.setOption.mock.calls[0][0];
    expect(option.series[0]).toMatchObject({
      type: "custom",
      encode: { x: [1, 2], y: 0 },
    });
    expect(option.series[0].data).toHaveLength(6);
    expect(option.dataZoom).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ xAxisIndex: 0, filterMode: "weakFilter" }),
        expect.objectContaining({ yAxisIndex: 0, filterMode: "filter" }),
      ]),
    );
  });

  it("builds tooltip content from the selected request lifecycle", () => {
    render(<RequestGantt requests={[request(7, 3)]} />);
    const option = chart.setOption.mock.calls[0][0];
    const queueData = option.series[0].data[0];
    const tooltip = option.tooltip.formatter({ value: queueData });

    expect(tooltip).toContain("请求 #7");
    expect(tooltip).toContain("阶段：排队");
    expect(tooltip).toContain("TTFT：5 ms");
    expect(tooltip).toContain("Decode 时长：6 ms");
    expect(tooltip).toContain("到达：3 ms");
    expect(tooltip).toContain("Prefill 开始：5 ms");
    expect(tooltip).toContain("Prefill 结束：8 ms");
    expect(tooltip).toContain("完成：14 ms");
  });

  it("updates data without recreating the chart and disposes it on unmount", () => {
    const { rerender, unmount } = render(<RequestGantt requests={[request(1)]} />);

    rerender(<RequestGantt requests={[request(1), request(2, 20)]} />);
    expect(init).toHaveBeenCalledTimes(1);
    expect(chart.setOption).toHaveBeenCalledTimes(2);

    act(() => window.dispatchEvent(new Event("resize")));
    expect(chart.resize).toHaveBeenCalledTimes(1);

    unmount();
    expect(chart.dispose).toHaveBeenCalledTimes(1);
  });
});
