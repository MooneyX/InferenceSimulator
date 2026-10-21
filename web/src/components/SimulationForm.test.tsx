import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { getScanParameters, uploadReplayPlan } from "../api/client";
import { createDefaultSimConfig } from "../config/simulation";
import { SimulationForm } from "./SimulationForm";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return { ...actual, getScanParameters: vi.fn(), uploadReplayPlan: vi.fn() };
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.mocked(uploadReplayPlan).mockReset();
});

describe("SimulationForm", () => {
  it("validates force recalculation and disables it while running", () => {
    const onSubmit = vi.fn();
    const { rerender } = render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.click(screen.getByRole("button", { name: "强制重新计算" }));
    expect(onSubmit).toHaveBeenCalledWith(createDefaultSimConfig(), true);
    onSubmit.mockClear();
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "强制重新计算" }));
    expect(onSubmit).not.toHaveBeenCalled();
    rerender(<SimulationForm onSubmit={onSubmit} disabled />);
    expect(screen.getByRole("button", { name: "强制重新计算" })).toBeDisabled();
  });

  it("runs one trace plan with inference settings and restores synthetic workload", async () => {
    const onSubmit = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-1", sessions: 1, requests: 2 });
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    expect(screen.getByLabelText("负载模式")).toHaveValue("trace");
    expect(screen.queryByLabelText("请求数量")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
    expect(screen.getByText(/请先选择会话并导出计划/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json")] },
    });
    await waitFor(() => expect(screen.getByText(/计划已校验/)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("投放模式"), { target: { value: "closed_loop" } });
    fireEvent.change(screen.getByLabelText("会话/秒"), { target: { value: "2" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
      workload: { mode: "trace", arrival_mode: "closed_loop", session_rate: 2, seed: 0, trace: { plan_id: "plan-1" } },
      inference: expect.objectContaining({ timing: { mode: "analytical" } }),
    }));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "synthetic" } });
    expect(screen.getByLabelText("请求数量")).toHaveValue(1000);
    expect(screen.queryByLabelText("已导出的计划 JSON")).not.toBeInTheDocument();
  });

  it("passes optional open-loop target QPS through the single run button", async () => {
    const onReplay = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-1", sessions: 1, requests: 3 });
    render(<SimulationForm onSubmit={onReplay} />);
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json")] },
    });
    await waitFor(() => expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled());
    fireEvent.click(screen.getByLabelText("按目标 QPS 等比例缩放到达间隔"));
    fireEvent.change(screen.getByLabelText("目标发送 QPS（请求/秒）"), {
      target: { value: "7.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onReplay).toHaveBeenCalledWith(expect.objectContaining({ workload: { mode: "trace",
      arrival_mode: "open_loop", session_rate: 1, seed: 0, target_qps: 7.5, trace: { plan_id: "plan-1" } } }));
  });

  it("exports trace parameters without a plan and requires reupload after import", async () => {
    const onReplay = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-1", sessions: 1, requests: 2 });
    render(<SimulationForm onSubmit={onReplay} />);
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json")] },
    });
    await waitFor(() => expect(screen.getByText(/计划已校验/)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("投放模式"), { target: { value: "closed_loop" } });
    fireEvent.change(screen.getByLabelText("会话/秒"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));
    const exported = (screen.getByLabelText("参数 JSON") as HTMLInputElement).value;
    expect(JSON.parse(exported).workload).toEqual({ mode: "trace", arrival_mode: "closed_loop",
      session_rate: 3, seed: 0, trace: { plan_id: null } });
    expect(exported).not.toContain("plan-1");
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
    expect(onReplay).not.toHaveBeenCalled();
    expect(screen.getByLabelText("投放模式")).toHaveValue("closed_loop");
    expect(screen.getByLabelText("会话/秒")).toHaveValue(3);
    expect(screen.getByRole("status")).toHaveTextContent("请重新上传请求计划");
    expect(screen.getByText(/请先选择会话并导出计划/)).toBeInTheDocument();
  });

  it("imports a synthetic scan plan and updates the ordinary form before running the scan", async () => {
    vi.mocked(getScanParameters).mockResolvedValue([{ key: "workload.synthetic.num_requests", label: "请求数量",
      kind: "integer", modes: ["synthetic/open_loop"] }]);
    const onScanSubmit = vi.fn();
    render(<SimulationForm onSubmit={vi.fn()} onScanSubmit={onScanSubmit} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "导入扫描方案" })).toBeEnabled());
    const base = createDefaultSimConfig();
    base.workload.synthetic!.num_requests = 24;
    base.inference.page_size = 128;
    fireEvent.change(screen.getByLabelText("扫描方案 JSON"), { target: { value: JSON.stringify({ version: 1, base,
      axes: [{ key: "workload.synthetic.num_requests", text: "10:30:10" }] }) } });
    fireEvent.click(screen.getByRole("button", { name: "导入扫描方案" }));
    expect(screen.getByLabelText("请求数量")).toHaveValue(24);
    expect(screen.getByLabelText("每页 token 数")).toHaveValue(128);
    expect(screen.getByLabelText("扫描档位 1")).toHaveValue("10:30:10");
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onScanSubmit).toHaveBeenCalledWith(expect.objectContaining({
      base: expect.objectContaining({ inference: expect.objectContaining({ page_size: 128 }) }),
      axes: [{ key: "workload.synthetic.num_requests", values: [10, 20, 30] }],
    }));
  });

  it("imports Trace scan settings into both panels without reusing a plan ID", async () => {
    vi.mocked(getScanParameters).mockResolvedValue([{ key: "workload.target_qps", label: "目标 QPS",
      kind: "number", modes: ["trace/open_loop"] }]);
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "new-plan", sessions: 1, requests: 2 });
    const onScanSubmit = vi.fn();
    render(<SimulationForm onSubmit={vi.fn()} onScanSubmit={onScanSubmit} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "导入扫描方案" })).toBeEnabled());
    const base = { ...createDefaultSimConfig(), workload: { mode: "trace", arrival_mode: "open_loop",
      session_rate: 2, seed: 7, target_qps: 12, trace: { plan_id: "old-plan" } } };
    fireEvent.change(screen.getByLabelText("扫描方案 JSON"), { target: { value: JSON.stringify({ version: 1, base,
      axes: [{ key: "workload.target_qps", text: "8,16" }] }) } });
    fireEvent.click(screen.getByRole("button", { name: "导入扫描方案" }));
    expect(screen.getAllByLabelText("负载模式").map((element) => (element as HTMLSelectElement).value))
      .toEqual(["trace", "trace"]);
    expect(screen.getByLabelText("会话/秒")).toHaveValue(2);
    expect(screen.getByLabelText("目标发送 QPS（请求/秒）")).toHaveValue(12);
    expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("回放计划 JSON 文件"), {
      target: { files: [new File(["{}"], "new-plan.json")] },
    });
    await screen.findByText(/new-plan.json/);
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onScanSubmit.mock.lastCall?.[0]).toMatchObject({ base: { workload: {
      mode: "trace", session_rate: 2, seed: 7, target_qps: 12, trace: { plan_id: "new-plan" },
    } }, axes: [{ key: "workload.target_qps", values: [8, 16] }] });
  });

  it("starts with model and hardware settings", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={onSubmitAuto} />);

    expect(screen.queryByRole("radio", { name: /自动解析|手动配置/ })).not.toBeInTheDocument();
    expect(screen.getByLabelText("模型名称")).toHaveValue("Qwen/Qwen3-8B");
    expect(screen.getByLabelText("L2 可用容量（GB，仅 KV）")).toHaveValue(2048);
    expect(screen.getByLabelText("共享 L3 总容量（TB，仅 KV，0 为关闭）")).toHaveValue(40);
    expect(screen.getByLabelText("L2 → L1 带宽（GB/s）")).toHaveValue(256);
    expect(screen.getByLabelText("L3 → L2 带宽（GB/s）")).toHaveValue(50);
    expect(screen.getByLabelText("单算子固定开销（ms）")).toHaveValue(0.005);
    expect(screen.getByLabelText("L2 固定延迟（ms）")).toHaveValue(0.01);
    expect(screen.getByLabelText("L3 固定延迟（ms）")).toHaveValue(0.1);
    expect(screen.getByLabelText("L3 查找延迟（ms）")).toHaveValue(0.1);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      workload: expect.objectContaining({ arrival_mode: "open_loop", seed: 0,
        synthetic: expect.objectContaining({ turn_increment_len: { dist: "poisson", mean: 256 } }) }),
      inference: expect.objectContaining({ timing: { mode: "analytical" },
        storage: expect.objectContaining({ l2_available_gb: 2048,
          l2_to_l1_gb_s: 256, l3_to_l2_gb_s: 50 }) }),
      shared_l3: { available_gb: 40_000, lookup_latency_ms: 0.1 },
    }));
  });

  it.each([null, 0])("uses the lookup default only when re-enabling L3 without a value (%s)", (latency) => {
    const onSubmit = vi.fn();
    const config = createDefaultSimConfig();
    config.inference.hardware.kernel_overhead_ms = 0;
    config.inference.storage.l2_latency_ms = 0;
    config.inference.storage.l3_latency_ms = 0;
    config.shared_l3 = latency === null ? null : { available_gb: 0, lookup_latency_ms: latency };
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: JSON.stringify(config) } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByLabelText("单算子固定开销（ms）")).toHaveValue(0);
    expect(screen.getByLabelText("L2 固定延迟（ms）")).toHaveValue(0);
    expect(screen.getByLabelText("L3 固定延迟（ms）")).toHaveValue(0);
    fireEvent.change(screen.getByLabelText("共享 L3 总容量（TB，仅 KV，0 为关闭）"), { target: { value: "40" } });
    expect(screen.getByLabelText("L3 查找延迟（ms）")).toHaveValue(latency ?? 0.1);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).toHaveBeenCalledWith({ ...config,
      shared_l3: { available_gb: 40_000, lookup_latency_ms: latency ?? 0.1 } });
  });

  it("renders every synthetic configuration group with defaults", () => {
    render(<SimulationForm onSubmit={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "负载配置" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "引擎配置" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "性能模型" })).toBeInTheDocument();
    expect(screen.getByLabelText("负载模式")).toHaveValue("synthetic");
    expect(screen.getByLabelText("请求数量")).toHaveValue(1000);
    expect(screen.getByLabelText("目标发送 QPS（请求/秒，泊松到达）")).toHaveValue(64);
    expect(screen.getByLabelText("随机种子")).toHaveValue(0);
    const promptGroup = screen.getByRole("group", { name: "新会话根输入长度" });
    expect(within(promptGroup).getByLabelText("分布类型")).toHaveValue("poisson");
    expect(within(promptGroup).getByLabelText("平均长度（token）")).toHaveValue(2048);
    expect(within(screen.getByRole("group", { name: "子轮新增内容长度" }))
      .getByLabelText("平均长度（token）")).toHaveValue(256);
    expect(screen.queryByLabelText("允许跨会话共享公共前缀")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("每请求生成上限（token）")).not.toBeInTheDocument();
    expect(screen.getByLabelText("引擎输出上限（token）")).toHaveValue(65_536);
    expect(screen.getByText(/每请求生成上限由实际输出长度向上取二次幂自动确定/)).toBeInTheDocument();
    expect(screen.getByLabelText("最多同时运行请求")).toHaveValue(32);
    expect(screen.queryByLabelText("L1 KV 总容量（token）")).not.toBeInTheDocument();
    expect(screen.getByLabelText("L2 可用容量（GB，仅 KV）")).toHaveValue(2048);
    expect(screen.getByLabelText("共享 L3 总容量（TB，仅 KV，0 为关闭）")).toHaveValue(40);
    expect(screen.getByLabelText("L2 → L1 带宽（GB/s）")).toHaveValue(256);
    expect(screen.getByLabelText("单次 Prefill 额度（token）")).toHaveValue(16_384);
    expect(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）")).toHaveValue(256);
    expect(screen.getByLabelText("每页 token 数")).toHaveValue(64);
    expect(screen.getByLabelText(/Prefill \/ Decode 分步执行/)).toBeChecked();
    expect(screen.getByLabelText("计时模式")).toHaveValue("analytical");
  });

  it("updates the engine output limit without submitting a synthetic request limit", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("引擎输出上限（token）"), { target: { value: "512" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit.mock.lastCall![0].inference).toMatchObject({ max_output_tokens: 512 });
    expect(onSubmit.mock.lastCall![0].workload.synthetic).not.toHaveProperty("max_new_tokens");
    expect(screen.getByLabelText("引擎输出上限（token）")).toHaveValue(512);
  });

  it("allows clearing a closed-loop session count to use the request count default", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("请求投放方式"), { target: { value: "closed_loop" } });
    fireEvent.change(screen.getByLabelText("会话数量（根请求数）"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit.mock.lastCall![0].workload.synthetic.session.num_sessions).toBeNull();
    expect(screen.getByLabelText("会话数量（根请求数）")).not.toHaveAttribute("aria-invalid");
  });

  it("uses the selected closed_loop workload in automatic resolution too", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={onSubmitAuto} />);
    fireEvent.change(screen.getByLabelText("请求投放方式"), { target: { value: "closed_loop" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      workload: expect.objectContaining({ arrival_mode: "closed_loop", session_rate: 1.28,
        synthetic: expect.objectContaining({
          session: expect.objectContaining({ num_sessions: 20, think_time_ms: 10_000 }),
        }) }),
      inference: expect.objectContaining({ timing: { mode: "analytical" } }),
    }));
  });

  it("submits closed_loop mode using only session count, rate and think interval", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("请求投放方式"), { target: { value: "closed_loop" } });
    expect(screen.queryByLabelText("目标发送 QPS（请求/秒，泊松到达）")).not.toBeInTheDocument();
    const incrementGroup = screen.getByRole("group", { name: "子轮新增内容长度" });
    expect(within(incrementGroup).getByLabelText("平均长度（token）")).toHaveValue(256);
    fireEvent.change(within(incrementGroup).getByLabelText("平均长度（token）"),
      { target: { value: "48" } });
    fireEvent.change(screen.getByLabelText("会话数量（根请求数）"), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText("新会话投放频率（会话/秒，泊松到达）"),
      { target: { value: "3" } });
    fireEvent.change(screen.getByLabelText("轮次间平均思考时间（ms，指数分布）"),
      { target: { value: "200" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    const submitted = onSubmit.mock.calls[0][0].workload;
    expect(submitted).toMatchObject({ arrival_mode: "closed_loop", session_rate: 3,
      synthetic: { turn_increment_len: { dist: "poisson", mean: 48 },
        session: { num_sessions: 5, think_time_ms: 200, topology: "linear" } } });
    expect(submitted).not.toHaveProperty("target_qps");
    fireEvent.change(screen.getByLabelText("请求投放方式"), { target: { value: "open_loop" } });
    expect(screen.getByLabelText("目标发送 QPS（请求/秒，泊松到达）")).toBeInTheDocument();
    expect(within(screen.getByRole("group", { name: "子轮新增内容长度" }))
      .getByLabelText("平均长度（token）")).toHaveValue(48);
  });

  it("blocks invalid closed-loop fields before submission", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("请求投放方式"), { target: { value: "closed_loop" } });
    fireEvent.change(screen.getByLabelText("会话数量（根请求数）"), { target: { value: "1001" } });
    fireEvent.change(screen.getByLabelText("新会话投放频率（会话/秒，泊松到达）"),
      { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("轮次间平均思考时间（ms，指数分布）"),
      { target: { value: "-1" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).not.toHaveBeenCalled();
    for (const label of ["会话数量（根请求数）", "新会话投放频率（会话/秒，泊松到达）",
      "轮次间平均思考时间（ms，指数分布）"]) {
      expect(screen.getByLabelText(label)).toHaveAttribute("aria-invalid", "true");
    }
  });

  it("submits the complete default config", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    fireEvent.submit(screen.getByRole("button", { name: "运行仿真" }).closest("form")!);

    expect(onSubmit).toHaveBeenCalledWith(createDefaultSimConfig());
    expect(onSubmit.mock.calls[0][0].workload.synthetic).not.toHaveProperty("session");
  });

  it("updates scalar and distribution fields before submission", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText("请求数量"), {
      target: { value: "25" },
    });

    const promptGroup = screen.getByRole("group", { name: "新会话根输入长度" });
    fireEvent.change(within(promptGroup).getByLabelText("分布类型"), {
      target: { value: "fixed" },
    });
    fireEvent.change(within(promptGroup).getByLabelText("固定长度（token）"), {
      target: { value: "256" },
    });
    const incrementGroup = screen.getByRole("group", { name: "子轮新增内容长度" });
    fireEvent.change(within(incrementGroup).getByLabelText("平均长度（token）"), {
      target: { value: "64" },
    });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      workload: {
        arrival_mode: "open_loop",
        synthetic: {
          num_requests: 25,
          prompt_len: { dist: "fixed", value: 256 },
          turn_increment_len: { dist: "poisson", mean: 64 },
        },
      },
    });
  });

  it("submits edited open-loop distribution, page size and fused mode", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    const incrementGroup = screen.getByRole("group", { name: "子轮新增内容长度" });
    fireEvent.change(within(incrementGroup).getByLabelText("分布类型"), { target: { value: "fixed" } });
    fireEvent.change(within(incrementGroup).getByLabelText("固定长度（token）"), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText("每页 token 数"), {
      target: { value: "4" },
    });
    fireEvent.change(screen.getByLabelText("引擎输出上限（token）"), {
      target: { value: "12" },
    });
    fireEvent.click(screen.getByLabelText(/Prefill \/ Decode 分步执行/));
    expect(screen.getByLabelText("每页 token 数")).toHaveValue(4);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      workload: {
        synthetic: {
          num_requests: 20,
          turn_increment_len: { dist: "fixed", value: 20 },
        },
      },
      scheduler: { pd_mutually_exclusive: false },
      inference: { page_size: 4, max_output_tokens: 12 },
    });
  });

  it("requires a physical L2 link when shared L3 is enabled", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("L3 → L2 带宽（GB/s）"), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByLabelText("共享 L3 总容量（TB，仅 KV，0 为关闭）"))
      .toHaveAttribute("aria-invalid", "true");
    fireEvent.change(screen.getByLabelText("L3 → L2 带宽（GB/s）"), { target: { value: "75" } });
    fireEvent.change(screen.getByLabelText("L3 查找延迟（ms）"), { target: { value: "0.75" } });
    fireEvent.change(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）"), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      scheduler: expect.objectContaining({ l3_prefetch_min_tokens: 0 }),
      shared_l3: { available_gb: 40_000, lookup_latency_ms: 0.75 },
      inference: expect.objectContaining({ storage: expect.objectContaining({ l3_to_l2_gb_s: 75 }) }),
    }));
  });

  it.each([
    ["目标发送 QPS（请求/秒，泊松到达）", "0", "必须是大于 0 的有限数值"],
    ["共享 L3 总容量（TB，仅 KV，0 为关闭）", "-1", "必须是大于或等于 0 的有限数值"],
    ["L3 查找延迟（ms）", "-1", "必须是大于或等于 0 的有限数值"],
    ["L3 异步预取最小长度（token，0 为不设阈值）", "-1", "必须是大于或等于 0 的整数"],
    ["每页 token 数", "0", "必须是大于或等于 1 的整数"],
    ["引擎输出上限（token）", "0", "必须是大于或等于 1 的整数"],
  ])("blocks invalid %s with an accessible field error", (label, value, message) => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    const input = screen.getByLabelText(label);
    fireEvent.change(input, { target: { value } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(input).toHaveAttribute("aria-invalid", "true");
    const errorId = input.getAttribute("aria-describedby");
    expect(errorId).not.toBeNull();
    expect(document.getElementById(errorId!)).toHaveTextContent(message);
  });

  it("blocks invalid submissions and associates errors with their fields", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    const requestCount = screen.getByLabelText("请求数量");
    fireEvent.change(requestCount, { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    const error = screen.getByRole("alert");
    expect(onSubmit).not.toHaveBeenCalled();
    expect(requestCount).toHaveAttribute("aria-invalid", "true");
    expect(requestCount).toHaveAttribute("aria-describedby", error.id);
    expect(error).toHaveTextContent("必须是大于或等于 1 的整数");
  });

  it("exports edited parameters to the JSON box and copies them", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    render(<SimulationForm onSubmit={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "25" } });

    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));

    const json = (screen.getByLabelText("参数 JSON") as HTMLInputElement).value;
    expect(JSON.parse(json)).toMatchObject({ workload: { arrival_mode: "open_loop", synthetic: {
      num_requests: 25, turn_increment_len: { dist: "poisson", mean: 256 },
    } } });
    expect(JSON.parse(json)).not.toHaveProperty("metrics");
    expect(writeText).toHaveBeenCalledWith(json);
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("已生成并复制"));
  });

  it("keeps exported JSON available when clipboard copying is unavailable", () => {
    vi.stubGlobal("navigator", {});
    render(<SimulationForm onSubmit={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));
    expect((screen.getByLabelText("参数 JSON") as HTMLInputElement).value).toContain('"workload"');
    expect(screen.getByRole("status")).toHaveTextContent("可手动复制");
  });

  it("imports a valid JSON config into all form groups without running a simulation", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    const config = createDefaultSimConfig();
    config.workload.synthetic!.num_requests = 20;
    config.workload.synthetic!.turn_increment_len = { dist: "fixed", value: 35 };
    config.workload.synthetic!.prompt_len = { dist: "fixed", value: 256 };
    config.workload.synthetic!.output_len = { dist: "lognormal", mean: 50, sigma: 2 };
    config.inference.page_size = 128;
    config.inference.storage.l2_available_gb = 256;
    config.shared_l3!.available_gb = 128_000;
    config.scheduler.l3_prefetch_min_tokens = 512;
    config.inference.storage.l2_to_l1_gb_s = 1024;
    config.inference.storage.l3_to_l2_gb_s = 512;
    config.inference.storage.l2_latency_ms = 1;
    config.inference.storage.l3_latency_ms = 2;
    config.shared_l3!.lookup_latency_ms = 3;
    config.inference.timing = { mode: "empirical", prefill_c1: 0.1, prefill_c2: 0.000_01,
      decode_d1: 8, decode_d2: 0.002 };
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: JSON.stringify(config) } });

    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));

    expect(screen.getByRole("status")).toHaveTextContent("参数已导入");
    expect(screen.getByLabelText("请求数量")).toHaveValue(20);
    expect(within(screen.getByRole("group", { name: "子轮新增内容长度" }))
      .getByLabelText("固定长度（token）")).toHaveValue(35);
    expect(within(screen.getByRole("group", { name: "新会话根输入长度" }))
      .getByLabelText("固定长度（token）")).toHaveValue(256);
    expect(within(screen.getByRole("group", { name: "输出长度" })).getByLabelText("形状参数（sigma）"))
      .toHaveValue(2);
    expect(screen.getByLabelText("每页 token 数")).toHaveValue(128);
    expect(screen.getByLabelText("L2 可用容量（GB，仅 KV）")).toHaveValue(256);
    expect(screen.getByLabelText("共享 L3 总容量（TB，仅 KV，0 为关闭）")).toHaveValue(128);
    expect(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）")).toHaveValue(512);
    expect(screen.getByLabelText("L2 → L1 带宽（GB/s）")).toHaveValue(1024);
    expect(screen.getByLabelText("L3 → L2 带宽（GB/s）")).toHaveValue(512);
    expect(screen.getByLabelText("L2 固定延迟（ms）")).toHaveValue(1);
    expect(screen.getByLabelText("L3 固定延迟（ms）")).toHaveValue(2);
    expect(screen.getByLabelText("L3 查找延迟（ms）")).toHaveValue(3);
    expect(screen.getByLabelText("计时模式")).toHaveValue("empirical");
    expect(screen.getByLabelText("Prefill C1")).toHaveValue(0.1);
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).toHaveBeenCalledWith(config);
  });

  it("exports and imports the complete automatic configuration from the top transfer controls", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={onSubmitAuto} />);
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    fireEvent.change(screen.getByLabelText("隐藏维度"), { target: { value: "2048" } });
    fireEvent.change(screen.getByLabelText("单卡标称显存（GB）"), { target: { value: "300" } });
    fireEvent.change(screen.getByLabelText("共享 L3 总容量（TB，仅 KV，0 为关闭）"), { target: { value: "1.5" } });
    fireEvent.change(screen.getByLabelText("L3 查找延迟（ms）"), { target: { value: "2.5" } });
    fireEvent.change(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）"), { target: { value: "1024" } });
    expect(screen.getByLabelText("共享 L3 总容量（TB，仅 KV，0 为关闭）")).toHaveValue(1.5);
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));
    const exported = (screen.getByLabelText("参数 JSON") as HTMLInputElement).value;
    expect(JSON.parse(exported)).toMatchObject({
      workload: { synthetic: { num_requests: 20 } }, scheduler: { l3_prefetch_min_tokens: 1024 },
      inference: { timing: { mode: "empirical" }, model: { hidden_size: 2048 },
        hardware: { memory_bytes_per_gpu: 300_000_000_000 } },
      shared_l3: { available_gb: 1500, lookup_latency_ms: 2.5 },
    });
    fireEvent.change(screen.getByLabelText("隐藏维度"), { target: { value: "4096" } });
    fireEvent.change(screen.getByLabelText("单卡标称显存（GB）"), { target: { value: "288" } });
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "analytical" } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.queryByRole("radio", { name: /自动解析|手动配置/ })).not.toBeInTheDocument();
    expect(screen.getByLabelText("隐藏维度")).toHaveValue(2048);
    expect(screen.getByLabelText("单卡标称显存（GB）")).toHaveValue(300);
    expect(screen.getByLabelText("共享 L3 总容量（TB，仅 KV，0 为关闭）")).toHaveValue(1.5);
    expect(screen.getByLabelText("L3 查找延迟（ms）")).toHaveValue(2.5);
    expect(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）")).toHaveValue(1024);
    expect(screen.getByLabelText("计时模式")).toHaveValue("empirical");
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      scheduler: expect.objectContaining({ l3_prefetch_min_tokens: 1024 }),
      inference: expect.objectContaining({ timing: expect.objectContaining({ mode: "empirical" }),
        model: expect.objectContaining({ hidden_size: 2048 }),
        hardware: expect.objectContaining({ memory_bytes_per_gpu: 300_000_000_000 }),
        storage: expect.objectContaining({ l2_available_gb: 2048 }) }),
      shared_l3: { available_gb: 1500, lookup_latency_ms: 2.5 },
    }));
  });

  it("submits a single config draft with workload, scheduler and page-size edits", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "40" } });
    fireEvent.change(screen.getByLabelText("最多同时运行请求"), { target: { value: "8" } });
    fireEvent.change(screen.getByLabelText("每页 token 数"), { target: { value: "100" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      workload: expect.objectContaining({ synthetic: expect.objectContaining({ num_requests: 40 }) }),
      scheduler: expect.objectContaining({ max_running_requests: 8 }),
      inference: expect.objectContaining({ page_size: 100, timing: { mode: "analytical" } }),
    }));
    expect(onSubmit.mock.lastCall![0].inference).not.toHaveProperty("kv");
  });

  it("imports a config over the current draft", () => {
    render(<SimulationForm onSubmit={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "30" } });
    const imported = createDefaultSimConfig();
    imported.workload.synthetic!.num_requests = 12;
    imported.inference.storage.l2_available_gb = 512;
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: JSON.stringify(imported) } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByLabelText("请求数量")).toHaveValue(12);
    expect(screen.getByLabelText("L2 可用容量（GB，仅 KV）")).toHaveValue(512);
  });

  it("shows empirical coefficients only when automatic timing uses them", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={onSubmitAuto} />);
    expect(screen.queryByLabelText("Prefill C1")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    fireEvent.change(screen.getByLabelText("Prefill C1"), { target: { value: "0.2" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenLastCalledWith(expect.objectContaining({
      inference: expect.objectContaining({ timing: expect.objectContaining({ mode: "empirical", prefill_c1: 0.2 }) }),
    }));
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "analytical" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledTimes(2);
    expect(onSubmitAuto).toHaveBeenLastCalledWith(expect.objectContaining({
      inference: expect.objectContaining({ timing: { mode: "analytical" } }),
    }));
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    expect(screen.getByLabelText("Prefill C1")).toHaveValue(0.2);
    fireEvent.change(screen.getByLabelText("Prefill C1"), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "analytical" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenLastCalledWith(expect.objectContaining({
      inference: expect.objectContaining({ timing: { mode: "analytical" } }),
    }));
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    expect(screen.getByLabelText("Prefill C1")).toHaveValue(null);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledTimes(3);
  });

  it("round-trips EP topology and runs MoE analytical timing only", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={onSubmitAuto} />);
    fireEvent.change(screen.getByLabelText("MoE 层数"), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText("专家数"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("每 token 激活专家数"), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText("专家中间维度（MoE 必填）"), { target: { value: "512" } });
    fireEvent.change(screen.getByLabelText("TP 数（必须等于 GPU 卡数）"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("本实例 GPU 卡数"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("EP 数（MoE 须等于 TP）"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("EP 每次 dispatch/combine 固定开销（ms，EP>1 运行必填）"), {
      target: { value: "0.2" },
    });
    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));
    const exported = (screen.getByLabelText("参数 JSON") as HTMLInputElement).value;
    expect(JSON.parse(exported)).toMatchObject({ inference: {
      tp_size: 4, ep_size: 4,
      hardware: { ep_communication_latency_ms: 0.2, collective_latency_ms: 0.01,
        effective_interconnect_gb_s: 600 },
    } });
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled();
    fireEvent.change(screen.getByLabelText("TP 每次集合通信固定延迟（ms，TP>1 运行必填）"), {
      target: { value: "" },
    });
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("TP 每次集合通信固定延迟（ms，TP>1 运行必填）"), {
      target: { value: "0.1" },
    });
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      inference: expect.objectContaining({ tp_size: 4, ep_size: 4, timing: { mode: "analytical" } }),
    }));
    fireEvent.change(screen.getByLabelText("EP 数（MoE 须等于 TP）"), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByLabelText("EP 数（MoE 须等于 TP）")).toHaveValue(4);
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
  });

  it("round-trips TP topology and runs analytical multi-GPU only", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={onSubmitAuto} />);
    fireEvent.change(screen.getByLabelText("TP 数（必须等于 GPU 卡数）"), {
      target: { value: "4" },
    });
    fireEvent.change(screen.getByLabelText("本实例 GPU 卡数"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("单 rank 有效互联带宽（GB/s，TP>1 必填）"), {
      target: { value: "100" },
    });
    fireEvent.change(screen.getByLabelText("TP 每次集合通信固定延迟（ms，TP>1 运行必填）"), {
      target: { value: "0.1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));
    const exported = (screen.getByLabelText("参数 JSON") as HTMLInputElement).value;
    expect(JSON.parse(exported)).toMatchObject({ inference: {
      tp_size: 4, ep_size: 1, hardware: { gpu_count: 4, collective_latency_ms: 0.1 },
    } });
    fireEvent.change(screen.getByLabelText("TP 数（必须等于 GPU 卡数）"), {
      target: { value: "1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByLabelText("TP 数（必须等于 GPU 卡数）")).toHaveValue(4);
    expect(screen.getAllByRole("button", { name: "运行仿真" })).toHaveLength(1);
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      inference: expect.objectContaining({ tp_size: 4, timing: { mode: "analytical" },
        hardware: expect.objectContaining({ gpu_count: 4 }) }),
    }));
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "analytical" } });
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled();
  });

  it("accepts config JSON and rejects legacy wrappers without changing the form", () => {
    render(<SimulationForm onSubmit={vi.fn()} />);
    const config = createDefaultSimConfig();
    config.workload.synthetic!.num_requests = 12;
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: JSON.stringify(config) } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByLabelText("请求数量")).toHaveValue(12);
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: JSON.stringify({
      config, model: { name: "incomplete" }, hardware: {}, storage: {}, timing: "analytical",
    }) } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByRole("alert")).toHaveTextContent("参数导入失败");
    expect(screen.getByLabelText("请求数量")).toHaveValue(12);
    expect(screen.getByLabelText("模型名称")).toHaveValue("Qwen/Qwen3-8B");
    expect(screen.getByLabelText("词表大小（结构估算必填）")).toHaveValue(151936);
  });

  it.each(["inference", "shared_l3"] as const)("rejects obsolete %s.mode without changing the form", (section) => {
    render(<SimulationForm onSubmit={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    const config = createDefaultSimConfig();
    const legacy = { ...config, [section]: { ...config[section], mode: "physical" } };
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: JSON.stringify(legacy) } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByRole("alert")).toHaveTextContent(`不支持的参数：${section}.mode`);
    expect(screen.getByLabelText("请求数量")).toHaveValue(20);
  });

  it("preserves the current form when JSON is invalid or incomplete", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: '{"workload":{}}' } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByRole("alert")).toHaveTextContent("参数导入失败");
    expect(screen.getByLabelText("请求数量")).toHaveValue(20);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit.mock.calls[0][0].workload.synthetic.num_requests).toBe(20);
  });

  it("rejects exporting invalid drafts without replacing the JSON box", () => {
    render(<SimulationForm onSubmit={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: "previous" } });
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));
    expect(screen.getByRole("alert")).toHaveTextContent("参数导出失败");
    expect(screen.getByLabelText("参数 JSON")).toHaveValue("previous");
  });

  it("disables submission and parameter transfer while a simulation is running", () => {
    render(<SimulationForm onSubmit={vi.fn()} disabled />);

    expect(screen.getByRole("button", { name: "正在运行…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "导入参数" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "导出参数" })).toBeDisabled();
    expect(screen.getByLabelText("请求数量")).toBeDisabled();
    expect(screen.getByLabelText("每页 token 数")).toBeDisabled();
  });
});
