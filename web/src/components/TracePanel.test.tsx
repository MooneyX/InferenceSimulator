import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { uploadReplayPlan, uploadTrace, traceSessions } from "../api/client";
import { TracePanel } from "./TracePanel";

vi.mock("../api/client", () => ({
  uploadReplayPlan: vi.fn(), uploadTrace: vi.fn(), traceStatus: vi.fn(),
  traceSessions: vi.fn(), downloadTracePlan: vi.fn(),
}));

afterEach(() => vi.clearAllMocks());

describe("TracePanel", () => {
  it("restores imported replay settings but requires a newly uploaded plan", async () => {
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "fresh-plan", sessions: 1, requests: 2 });
    const onChange = vi.fn();
    render(<TracePanel disabled={false} onChange={onChange}
      initialReplay={{ mode: "open_loop", rate: 2.5, seed: 17, targetQps: 12 }} />);
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
      id: "fresh-plan", mode: "open_loop", rate: 2.5, seed: 17, targetQps: 12,
    }));
  });

  it("restores imported closed_loop replay without an open-loop target", () => {
    render(<TracePanel disabled={false} onChange={vi.fn()}
      initialReplay={{ mode: "closed_loop", rate: 3, seed: 11 }} />);
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
      id: "plan-1", mode: "closed_loop", rate: 2, seed: 11,
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
      id: "plan-1", mode: "open_loop", rate: 1, seed: 0,
    }));
    fireEvent.click(screen.getByLabelText("按目标 QPS 等比例缩放到达间隔"));
    fireEvent.change(screen.getByLabelText("目标发送 QPS（请求/秒）"), {
      target: { value: "8" },
    });
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith({
      id: "plan-1", mode: "open_loop", rate: 1, seed: 0, targetQps: 8,
    }));
    fireEvent.change(screen.getByLabelText("目标发送 QPS（请求/秒）"), {
      target: { value: "0" },
    });
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith(null));
    fireEvent.change(screen.getByLabelText("投放模式"), { target: { value: "closed_loop" } });
    expect(screen.queryByLabelText("目标发送 QPS（请求/秒）")).not.toBeInTheDocument();
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith({
      id: "plan-1", mode: "closed_loop", rate: 1, seed: 0,
    }));
  });

  it("uploads raw trace and previews paginated sessions", async () => {
    vi.mocked(uploadTrace).mockResolvedValue({ id: "trace-1", status: "ready", session_count: 1, error: null });
    vi.mocked(traceSessions).mockResolvedValue({ total: 1, sessions: [{
      id: "session-a", requests: 2, subagent_groups: 0, input_tokens: 256,
      output_tokens: 10, start_s: 0, end_s: 1, open_loop_qps: 1, set_hit_rate: 0.5,
      prefix_hit_rate: 0.25, inferred_dependencies: 1, timed_roots: 1,
      ambiguous_dependencies: 0,
    }] });
    render(<TracePanel disabled={false} onChange={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("原始轨迹 JSONL"), {
      target: { files: [new File(["{}\n"], "sample.jsonl")] },
    });
    await waitFor(() => expect(screen.getByText(/session-a/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("checkbox", { name: /选择会话 session-a/ }));
    expect(screen.getByText(/已选 1 \/ 1/)).toBeInTheDocument();
  });
});
