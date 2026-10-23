import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { downloadTracePlan, uploadReplayPlan, uploadTrace, traceSessions, type TraceSessionSummary } from "../api/client";
import { TracePanel } from "./TracePanel";

vi.mock("../api/client", () => ({
  uploadReplayPlan: vi.fn(), uploadTrace: vi.fn(), traceStatus: vi.fn(),
  traceSessions: vi.fn(), downloadTracePlan: vi.fn(),
}));

afterEach(async () => {
  if (vi.isFakeTimers()) {
    await vi.runOnlyPendingTimersAsync();
    vi.useRealTimers();
  }
  vi.resetAllMocks();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function sessionSummary(id: string, overrides: Partial<TraceSessionSummary> = {}): TraceSessionSummary {
  return {
    id, requests: 2, subagent_groups: 0, input_tokens: 256,
    output_tokens: 10, start_s: 0, end_s: 1, open_loop_qps: 1, set_hit_rate: 0.5,
    prefix_hit_rate: 0.25, inferred_dependencies: 1, timed_roots: 1,
    ambiguous_dependencies: 0, estimated_plan_bytes: 1024, ...overrides,
  };
}

function uploadRawTrace(): void {
  fireEvent.change(screen.getByLabelText("原始轨迹 JSONL"), {
    target: { files: [new File(["{}\n"], "sample.jsonl")] },
  });
}

function expectStat(label: string, value: string): void {
  const stats = within(screen.getByRole("region", { name: "已选统计" }));
  expect(stats.getByText(label).nextElementSibling).toHaveTextContent(value);
}

function mockDownload(): void {
  vi.stubGlobal("URL", class extends URL {
    static createObjectURL = vi.fn(() => "blob:trace-plan");
    static revokeObjectURL = vi.fn();
  });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  vi.useFakeTimers();
}

describe("TracePanel", () => {
  it("restores imported replay settings but requires a newly uploaded plan", async () => {
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "fresh-plan", sessions: 1, requests: 2 });
    const onChange = vi.fn();
    render(<TracePanel disabled={false} onChange={onChange}
      initialReplay={{ arrivalMode: "open_loop", sessionRate: 2.5, seed: 17, targetQps: 12 }} />);
    expect(screen.getByLabelText("投放模式")).toHaveValue("open_loop");
    expect(screen.getByLabelText("会话/秒")).toHaveValue(2.5);
    expect(screen.getByLabelText("随机种子")).toHaveValue(17);
    expect(screen.getByLabelText("按目标 QPS 等比例缩放到达间隔")).toBeChecked();
    expect(screen.getByLabelText("目标发送 QPS（请求/秒）")).toHaveValue(12);
    expect(onChange).toHaveBeenLastCalledWith(null);
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json")] },
    });
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith({
      id: "fresh-plan", arrivalMode: "open_loop", sessionRate: 2.5, seed: 17, targetQps: 12,
    }));
  });

  it("restores imported closed_loop replay without an open-loop target", () => {
    render(<TracePanel disabled={false} onChange={vi.fn()}
      initialReplay={{ arrivalMode: "closed_loop", sessionRate: 3, seed: 11 }} />);
    expect(screen.getByLabelText("投放模式")).toHaveValue("closed_loop");
    expect(screen.getByLabelText("会话/秒")).toHaveValue(3);
    expect(screen.getByLabelText("随机种子")).toHaveValue(11);
    expect(screen.queryByLabelText("目标发送 QPS（请求/秒）")).not.toBeInTheDocument();
  });

  it("uploads an exported plan and reports its selected rate and seed", async () => {
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-1", sessions: 1, requests: 2 });
    const onChange = vi.fn();
    render(<TracePanel disabled={false} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json", { type: "application/json" })] },
    });
    await waitFor(() => expect(screen.getByText(/计划已校验/)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("投放模式"), { target: { value: "closed_loop" } });
    fireEvent.change(screen.getByLabelText("会话/秒"), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText("随机种子"), { target: { value: "11" } });
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith({
      id: "plan-1", arrivalMode: "closed_loop", sessionRate: 2, seed: 11,
    }));
    expect(screen.queryByRole("button", { name: "运行计划" })).not.toBeInTheDocument();
    vi.mocked(uploadReplayPlan).mockRejectedValueOnce(new Error("文件无效"));
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["invalid"], "bad.json", { type: "application/json" })] },
    });
    await waitFor(() => expect(screen.getByText("文件无效")).toBeInTheDocument());
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith(null));
  });

  it("only sends a valid target QPS when open-loop scaling is enabled", async () => {
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-1", sessions: 1, requests: 3 });
    const onChange = vi.fn();
    render(<TracePanel disabled={false} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json")] },
    });
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith({
      id: "plan-1", arrivalMode: "open_loop", sessionRate: 1, seed: 0,
    }));
    fireEvent.click(screen.getByLabelText("按目标 QPS 等比例缩放到达间隔"));
    fireEvent.change(screen.getByLabelText("目标发送 QPS（请求/秒）"), {
      target: { value: "8" },
    });
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith({
      id: "plan-1", arrivalMode: "open_loop", sessionRate: 1, seed: 0, targetQps: 8,
    }));
    fireEvent.change(screen.getByLabelText("目标发送 QPS（请求/秒）"), {
      target: { value: "0" },
    });
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith(null));
    fireEvent.change(screen.getByLabelText("投放模式"), { target: { value: "closed_loop" } });
    expect(screen.queryByLabelText("目标发送 QPS（请求/秒）")).not.toBeInTheDocument();
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith({
      id: "plan-1", arrivalMode: "closed_loop", sessionRate: 1, seed: 0,
    }));
  });

  it("summarizes selections across pages and subtracts deselected sessions", async () => {
    vi.mocked(uploadTrace).mockResolvedValue({ id: "trace-1", status: "ready", session_count: 51 });
    const sessions = Array.from({ length: 51 }, (_, index) => sessionSummary(`session-${index}`, {
      requests: index === 50 ? 5 : 2, input_tokens: index === 50 ? 1024 : 256,
      output_tokens: index === 50 ? 20 : 10, estimated_plan_bytes: index === 50 ? 2048 : 1024,
    }));
    vi.mocked(traceSessions).mockImplementation(async (_, offset) => ({
      total: 51, max_plan_bytes: 64 * 1024 ** 2, sessions: sessions.slice(offset, offset + 50),
    }));
    render(<TracePanel disabled={false} onChange={vi.fn()} />);
    uploadRawTrace();
    await screen.findByRole("checkbox", { name: "选择会话 session-0" });
    expectStat("会话总数", "0");
    expectStat("请求总数", "0");
    expectStat("计划大小（预计）", "0 B");
    expect(screen.getByRole("button", { name: "导出并载入选中会话计划" })).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "选择会话 session-0" }));
    expectStat("请求总数", "2");
    fireEvent.click(screen.getByRole("button", { name: "下一页" }));
    await screen.findByRole("checkbox", { name: "选择会话 session-50" });
    fireEvent.click(screen.getByRole("checkbox", { name: "选择会话 session-50" }));
    expectStat("会话总数", "2");
    expectStat("请求总数", "7");
    expectStat("输入 token 总数", (1280).toLocaleString());
    expectStat("输出 token 总数", "30");
    expectStat("计划大小（预计）", "3.00 KiB");
    expect(screen.getByText(/服务端上限：64.00 MiB/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "上一页" }));
    expect(await screen.findByRole("checkbox", { name: "选择会话 session-0" })).toBeChecked();
    fireEvent.click(screen.getByRole("checkbox", { name: "选择会话 session-0" }));
    expectStat("会话总数", "1");
    expectStat("请求总数", "5");
    expectStat("输入 token 总数", (1024).toLocaleString());
    expectStat("输出 token 总数", "20");
    expectStat("计划大小（预计）", "2.00 KiB");
    expect(downloadTracePlan).not.toHaveBeenCalled();
  });

  it("warns using the server limit without treating an estimate as a hard block", async () => {
    vi.mocked(uploadTrace).mockResolvedValue({ id: "trace-1", status: "ready", session_count: 1 });
    vi.mocked(traceSessions).mockResolvedValue({
      total: 1, max_plan_bytes: 1024, sessions: [sessionSummary("session-a", { estimated_plan_bytes: 2048 })],
    });
    render(<TracePanel disabled={false} onChange={vi.fn()} />);
    uploadRawTrace();
    fireEvent.click(await screen.findByRole("checkbox", { name: "选择会话 session-a" }));
    expectStat("计划大小（预计）", "2.00 KiB");
    expect(screen.getByText(/服务端上限：1.00 KiB/)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("预计计划大小超过服务端上限");
    expect(screen.getByRole("button", { name: "导出并载入选中会话计划" })).toBeEnabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "选择会话 session-a" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expectStat("计划大小（预计）", "0 B");
  });

  it("shows actual download bytes and resets them when selection or trace changes", async () => {
    vi.mocked(uploadTrace).mockResolvedValue({ id: "trace-1", status: "ready", session_count: 1 });
    vi.mocked(traceSessions).mockResolvedValue({
      total: 1, max_plan_bytes: 64 * 1024 ** 2, sessions: [sessionSummary("session-a")],
    });
    const blob = new Blob(['{"sessions":["会话"]}'], { type: "application/json" });
    vi.mocked(downloadTracePlan).mockResolvedValue(blob);
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-1", sessions: 1, requests: 2 });
    const onChange = vi.fn();
    render(<TracePanel disabled={false} onChange={onChange} />);
    uploadRawTrace();
    fireEvent.click(await screen.findByRole("checkbox", { name: "选择会话 session-a" }));
    mockDownload();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "导出并载入选中会话计划" }));
    });
    expect(screen.getByText(/已导出并载入计划/)).toBeInTheDocument();
    expectStat("计划大小（实际）", `${blob.size} B`);
    expect(downloadTracePlan).toHaveBeenCalledWith("trace-1", ["session-a"]);
    expect(uploadReplayPlan).toHaveBeenCalledWith(blob);
    expect(URL.createObjectURL).toHaveBeenCalledWith(blob);
    expect(onChange).toHaveBeenLastCalledWith({ id: "plan-1", arrivalMode: "open_loop", sessionRate: 1, seed: 0 });
    fireEvent.click(screen.getByRole("checkbox", { name: "选择会话 session-a" }));
    expect(screen.queryByText("计划大小（实际）")).not.toBeInTheDocument();
    expectStat("计划大小（预计）", "0 B");
    expect(onChange).toHaveBeenLastCalledWith(null);
    expect(screen.queryByText(/已导出并载入计划/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "选择会话 session-a" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "导出并载入选中会话计划" }));
    });
    expect(screen.getByText(/已导出并载入计划/)).toBeInTheDocument();
    vi.mocked(uploadTrace).mockResolvedValueOnce({ id: "trace-2", status: "ready", session_count: 1 });
    await act(async () => { uploadRawTrace(); });
    expect(traceSessions).toHaveBeenLastCalledWith("trace-2", 0);
    expectStat("会话总数", "0");
    expectStat("请求总数", "0");
    expectStat("输入 token 总数", "0");
    expectStat("输出 token 总数", "0");
    expectStat("计划大小（预计）", "0 B");
    expect(screen.queryByText("计划大小（实际）")).not.toBeInTheDocument();
  });

  it("keeps the estimate after export failure and actual bytes after reload failure", async () => {
    vi.mocked(uploadTrace).mockResolvedValue({ id: "trace-1", status: "ready", session_count: 1 });
    vi.mocked(traceSessions).mockResolvedValue({
      total: 1, max_plan_bytes: 64 * 1024 ** 2, sessions: [sessionSummary("session-a")],
    });
    vi.mocked(downloadTracePlan).mockRejectedValueOnce(new Error("选中的计划超过服务端允许的大小，请减少会话数量"));
    render(<TracePanel disabled={false} onChange={vi.fn()} />);
    uploadRawTrace();
    fireEvent.click(await screen.findByRole("checkbox", { name: "选择会话 session-a" }));
    fireEvent.click(screen.getByRole("button", { name: "导出并载入选中会话计划" }));
    await screen.findByText("选中的计划超过服务端允许的大小，请减少会话数量");
    expectStat("计划大小（预计）", "1.00 KiB");
    expect(screen.queryByText("计划大小（实际）")).not.toBeInTheDocument();
    const blob = new Blob(["{}"]);
    vi.mocked(downloadTracePlan).mockResolvedValueOnce(blob);
    vi.mocked(uploadReplayPlan).mockRejectedValueOnce(new Error("计划载入失败"));
    mockDownload();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "导出并载入选中会话计划" }));
    });
    expect(screen.getByText("计划载入失败")).toBeInTheDocument();
    expectStat("计划大小（实际）", "2 B");
  });
});
