import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getScanParameters, uploadReplayPlan } from "../api/client";
import { createDefaultSimConfig } from "../config/simulation";
import { ScanPanel, parseLevels } from "./ScanPanel";

vi.mock("../api/client", () => ({ getScanParameters: vi.fn(), uploadReplayPlan: vi.fn() }));
const catalog = [
  { key: "workload.synthetic.num_requests", label: "请求数量", kind: "integer",
    modes: ["synthetic/open_loop", "synthetic/closed_loop"] },
  { key: "workload.session_rate", label: "会话投放频率", kind: "number",
    modes: ["synthetic/closed_loop", "trace/open_loop", "trace/closed_loop"] },
  { key: "workload.target_qps", label: "目标 QPS", kind: "number",
    modes: ["synthetic/open_loop", "trace/open_loop"] },
  { key: "workload.synthetic.session.num_sessions", label: "会话数量", kind: "integer",
    modes: ["synthetic/closed_loop"] },
  { key: "workload.synthetic.session.think_time_ms", label: "思考间隔", kind: "number",
    modes: ["synthetic/closed_loop"] },
  { key: "workload.synthetic.prompt_len", label: "平均输入长度", kind: "number",
    modes: ["synthetic/open_loop", "synthetic/closed_loop"] },
  { key: "workload.synthetic.turn_increment_len", label: "平均新增长度", kind: "number",
    modes: ["synthetic/open_loop", "synthetic/closed_loop"] },
  { key: "workload.synthetic.output_len", label: "平均输出长度", kind: "number",
    modes: ["synthetic/open_loop", "synthetic/closed_loop"] },
  { key: "inference.parallelism", label: "每实例 GPU / TP / EP 数", kind: "integer",
    modes: ["synthetic/open_loop", "synthetic/closed_loop", "trace/open_loop", "trace/closed_loop"] },
];

beforeEach(() => {
  vi.mocked(getScanParameters).mockReset().mockResolvedValue(catalog);
  vi.mocked(uploadReplayPlan).mockReset();
});

describe("ScanPanel", () => {
  it("matches parameter names and submits axis levels without changing the ordinary form", async () => {
    const onSubmit = vi.fn();
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={onSubmit} />);
    expect(screen.getByLabelText("负载模式")).toHaveValue("synthetic");
    await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
    fireEvent.focus(screen.getByLabelText("横轴参数（必选）"));
    fireEvent.change(screen.getByLabelText("横轴参数（必选）"), { target: { value: "请求" } });
    const options = screen.getByRole("listbox");
    fireEvent.click(within(options).getByRole("option", { name: /请求数量/ }));
    fireEvent.change(screen.getByLabelText("横轴档位"), { target: { value: "10,20" } });
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      axes: { x: { key: "workload.synthetic.num_requests", values: [10, 20] } },
    }));
  });

  it("offers request parameters as scan axes instead of separate inputs for each mode", async () => {
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={vi.fn()} />);
    await waitFor(() => expect(getScanParameters).toHaveBeenCalled());
    const search = screen.getByLabelText("横轴参数（必选）");
    fireEvent.focus(search);
    expect(screen.queryByLabelText("目标发送 QPS")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("会话投放频率（会话/秒）")).not.toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /目标 QPS/ })).toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /平均输入长度/ })).toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).queryByRole("option", { name: /会话数量/ })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("请求到达模式"), { target: { value: "closed_loop" } });
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /会话数量/ })).toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /会话投放频率/ })).toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).queryByRole("option", { name: /目标 QPS/ })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    expect(within(screen.getByRole("listbox")).queryByRole("option", { name: /会话数量/ })).not.toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /会话投放频率/ })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("请求到达模式"), { target: { value: "open_loop" } });
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /目标 QPS/ })).toBeInTheDocument();
  });

  it("uses a Trace open-loop target QPS axis without a separate scaling control", async () => {
    const onSubmit = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-scan", requests: 2, sessions: 1 });
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={onSubmit} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    expect(screen.queryByRole("checkbox", { name: "按目标 QPS 缩放 Trace" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("回放计划 JSON 文件"), {
      target: { files: [new File(["{}"], "replay-plan.json")] },
    });
    await screen.findByText(/replay-plan.json/);
    fireEvent.focus(screen.getByLabelText("横轴参数（必选）"));
    fireEvent.change(screen.getByLabelText("横轴参数（必选）"), { target: { value: "目标 QPS" } });
    fireEvent.click(screen.getByRole("option", { name: /目标 QPS/ }));
    fireEvent.change(screen.getByLabelText("横轴档位"), { target: { value: "8,16" } });
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      base: expect.objectContaining({ workload: expect.not.objectContaining({ target_qps: expect.anything() }) }),
      axes: { x: { key: "workload.target_qps", values: [8, 16] } },
    }));
  });

  it("requires an independently uploaded replay plan and removes invalid axes on mode switch", async () => {
    const onSubmit = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-scan", requests: 2, sessions: 1 });
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={onSubmit} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    expect(screen.getByLabelText("请求到达模式")).toHaveValue("open_loop");
    expect(screen.getByLabelText("回放计划 JSON 文件")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("回放计划 JSON 文件"), {
      target: { files: [new File(["{}"], "replay-plan.json")] },
    });
    await screen.findByText(/replay-plan.json/);
    fireEvent.focus(screen.getByLabelText("横轴参数（必选）"));
    fireEvent.change(screen.getByLabelText("横轴参数（必选）"), { target: { value: "会话" } });
    fireEvent.click(screen.getByRole("option", { name: /会话投放频率/ }));
    fireEvent.change(screen.getByLabelText("横轴档位"), { target: { value: "1,2" } });
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      base: expect.objectContaining({ workload: expect.objectContaining({
        mode: "trace", trace: { plan_id: "plan-scan" } }) }),
    }));
  });
});

it("accepts number, boolean and range levels", () => {
  expect(parseLevels("2:6:2", "integer")).toEqual([2, 4, 6]);
  expect(parseLevels("true,false", "boolean")).toEqual([true, false]);
  expect(() => parseLevels("1.5,2", "integer")).toThrow();
});
