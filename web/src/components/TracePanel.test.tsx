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
  it("uploads an exported plan and runs with the selected rate and seed", async () => {
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-1", sessions: 1, requests: 2 });
    const onRun = vi.fn();
    render(<TracePanel disabled={false} onRun={onRun} />);
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json", { type: "application/json" })] },
    });
    await waitFor(() => expect(screen.getByText(/计划已校验/)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("投放模式"), { target: { value: "session" } });
    fireEvent.change(screen.getByLabelText("会话/秒"), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText("随机种子"), { target: { value: "11" } });
    fireEvent.click(screen.getByRole("button", { name: "运行计划" }));
    expect(onRun).toHaveBeenCalledWith("plan-1", "session", 2, 11);
    vi.mocked(uploadReplayPlan).mockRejectedValueOnce(new Error("文件无效"));
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["invalid"], "bad.json", { type: "application/json" })] },
    });
    await waitFor(() => expect(screen.getByText("文件无效")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "运行计划" })).toBeDisabled();
  });

  it("uploads raw trace and previews paginated sessions", async () => {
    vi.mocked(uploadTrace).mockResolvedValue({ id: "trace-1", status: "ready", session_count: 1, error: null });
    vi.mocked(traceSessions).mockResolvedValue({ total: 1, sessions: [{
      id: "session-a", requests: 2, subagent_groups: 0, input_tokens: 256,
      output_tokens: 10, start_s: 0, end_s: 1, open_loop_qps: 1, set_hit_rate: 0.5,
      prefix_hit_rate: 0.25, inferred_dependencies: 1, timed_roots: 1,
      ambiguous_dependencies: 0,
    }] });
    render(<TracePanel disabled={false} onRun={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("原始轨迹 JSONL"), {
      target: { files: [new File(["{}\n"], "sample.jsonl")] },
    });
    await waitFor(() => expect(screen.getByText(/session-a/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("checkbox", { name: /选择会话 session-a/ }));
    expect(screen.getByText(/已选 1 \/ 1/)).toBeInTheDocument();
  });
});
