import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { uploadReplayPlan } from "../api/client";
import { createDefaultSimConfig } from "../config/simulation";
import { SimulationForm } from "./SimulationForm";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return { ...actual, uploadReplayPlan: vi.fn() };
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.mocked(uploadReplayPlan).mockReset();
});

describe("SimulationForm", () => {
  it("runs one trace plan with either automatic resolution or manual configuration", async () => {
    const onReplayAuto = vi.fn();
    const onReplay = vi.fn();
    const onSubmit = vi.fn();
    const onSubmitAuto = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-1", sessions: 1, requests: 2 });
    render(<SimulationForm onSubmit={onSubmit} onSubmitAuto={onSubmitAuto}
      onReplay={onReplay} onReplayAuto={onReplayAuto} />);
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    expect(screen.getByLabelText("负载模式")).toHaveValue("trace");
    expect(screen.queryByLabelText("请求数量")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
    expect(screen.getByText(/请先选择会话并导出计划/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json")] },
    });
    await waitFor(() => expect(screen.getByText(/计划已校验/)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("投放模式"), { target: { value: "session" } });
    fireEvent.change(screen.getByLabelText("会话/秒"), { target: { value: "2" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onReplayAuto).toHaveBeenCalledWith("plan-1", expect.objectContaining({
      config: expect.objectContaining({ workload: { mode: "trace" } }), timing: "analytical",
    }), "session", 2, 0);
    expect(onSubmitAuto).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    expect(screen.getByText(/计划已校验/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onReplay).toHaveBeenCalledWith("plan-1", expect.objectContaining({
      workload: { mode: "trace" },
    }), "session", 2, 0);
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "synthetic" } });
    expect(screen.getByLabelText("请求数量")).toHaveValue(1000);
    expect(screen.queryByLabelText("已导出的计划 JSON")).not.toBeInTheDocument();
  });

  it("exports trace parameters without a plan and requires reupload after import", async () => {
    const onReplay = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-1", sessions: 1, requests: 2 });
    render(<SimulationForm onSubmit={vi.fn()} onReplay={onReplay} />);
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    fireEvent.change(screen.getByLabelText("已导出的计划 JSON"), {
      target: { files: [new File(["{}"], "plan.json")] },
    });
    await waitFor(() => expect(screen.getByText(/计划已校验/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));
    const exported = (screen.getByLabelText("参数 JSON") as HTMLInputElement).value;
    expect(JSON.parse(exported).workload).toEqual({ mode: "trace" });
    expect(exported).not.toContain("plan-1");
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
    expect(onReplay).not.toHaveBeenCalled();
    expect(screen.getByText(/请先选择会话并导出计划/)).toBeInTheDocument();
  });

  it("selects automatic resolution first when available", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={vi.fn()} onSubmitAuto={onSubmitAuto} />);

    const modes = screen.getAllByRole("radio", { name: /自动解析|手动配置/ });
    expect(modes).toHaveLength(2);
    expect(modes[0]).toHaveAttribute("value", "automatic");
    expect(modes[0]).toBeChecked();
    expect(modes[1]).toHaveAttribute("value", "manual");
    expect(screen.getByRole("group", { name: "自动解析参数" })).toBeInTheDocument();
    expect(screen.getByLabelText("L2 可用容量（GB，仅 KV）")).toHaveValue(2048);
    expect(screen.getByLabelText("L3 可用容量（TB，仅 KV）")).toHaveValue(40);
    expect(screen.getByLabelText("L2 → L1 带宽（GB/s）")).toHaveValue(256);
    expect(screen.getByLabelText("L3 → L2 带宽（GB/s）")).toHaveValue(50);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      config: expect.objectContaining({ workload: expect.objectContaining({
        synthetic: expect.objectContaining({ arrival_mode: "open_loop",
          turn_increment_len: { dist: "poisson", mean: 256 } }),
      }) }),
      storage: expect.objectContaining({ l2_available_gb: 2048, l3_available_gb: 40_000,
        l2_to_l1_gb_s: 256, l3_to_l2_gb_s: 50 }),
    }));
  });

  it("renders every synthetic configuration group with defaults", () => {
    render(<SimulationForm onSubmit={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "负载配置" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "引擎配置" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "性能模型" })).toBeInTheDocument();
    expect(screen.getByLabelText("负载模式")).toHaveValue("synthetic");
    expect(screen.getByLabelText("请求数量")).toHaveValue(1000);
    expect(screen.getByLabelText("目标发送 QPS（请求/秒，泊松到达）")).toHaveValue(64);
    expect(screen.getByLabelText("随机种子")).toHaveValue(42);
    const promptGroup = screen.getByRole("group", { name: "新会话根输入长度" });
    expect(within(promptGroup).getByLabelText("分布类型")).toHaveValue("poisson");
    expect(within(promptGroup).getByLabelText("平均长度（token）")).toHaveValue(2048);
    expect(within(screen.getByRole("group", { name: "子轮新增内容长度" }))
      .getByLabelText("平均长度（token）")).toHaveValue(256);
    expect(screen.queryByLabelText("允许跨会话共享公共前缀")).not.toBeInTheDocument();
    expect(screen.getByLabelText("每请求生成上限（token）")).toHaveValue(1024);
    expect(screen.getByLabelText("最多同时运行请求")).toHaveValue(32);
    expect(screen.getByLabelText("L1 KV 总容量（token）")).toHaveValue(65_536);
    expect(screen.getByLabelText("L2 KV 容量（token，0 为关闭）")).toHaveValue(0);
    expect(screen.getByLabelText("L3 KV 容量（token，0 为关闭）")).toHaveValue(0);
    expect(screen.queryByLabelText("L2 → L1 带宽（字节/ms）")).not.toBeInTheDocument();
    expect(screen.getByLabelText("单次 Prefill 额度（token）")).toHaveValue(16_384);
    expect(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）")).toHaveValue(256);
    expect(screen.getByLabelText("每页 token 数")).toHaveValue(64);
    expect(screen.getByLabelText(/Prefill \/ Decode 分步执行/)).toBeChecked();
    expect(screen.getByText(/1024 页/)).toBeInTheDocument();
  });

  it("uses the selected session workload in automatic resolution too", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={vi.fn()} onSubmitAuto={onSubmitAuto} />);
    fireEvent.change(screen.getByLabelText("请求投放方式"), { target: { value: "session" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({ config: expect.objectContaining({
      workload: expect.objectContaining({ synthetic: expect.objectContaining({
        arrival_mode: "session", session_rate: 1.28,
        session: expect.objectContaining({ num_sessions: 20, think_time_ms: 10_000 }),
      }) }),
    }) }));
  });

  it("submits session mode using only session count, rate and think interval", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("请求投放方式"), { target: { value: "session" } });
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
    const submitted = onSubmit.mock.calls[0][0].workload.synthetic;
    expect(submitted).toMatchObject({ arrival_mode: "session", session_rate: 3,
      turn_increment_len: { dist: "poisson", mean: 48 },
      session: { num_sessions: 5, think_time_ms: 200, topology: "linear" } });
    expect(submitted).not.toHaveProperty("arrival_rate");
    fireEvent.change(screen.getByLabelText("请求投放方式"), { target: { value: "open_loop" } });
    expect(screen.getByLabelText("目标发送 QPS（请求/秒，泊松到达）")).toBeInTheDocument();
    expect(within(screen.getByRole("group", { name: "子轮新增内容长度" }))
      .getByLabelText("平均长度（token）")).toHaveValue(48);
  });

  it("blocks invalid closed-loop fields before submission", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("请求投放方式"), { target: { value: "session" } });
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

    expect(onSubmit).toHaveBeenCalledWith({
      num_instances: 1,
      workload: {
        mode: "synthetic",
        synthetic: {
          num_requests: 1000,
          arrival_rate: 64,
          arrival_mode: "open_loop",
          prompt_len: { dist: "poisson", mean: 2048 },
          turn_increment_len: { dist: "poisson", mean: 256 },
          output_len: { dist: "poisson", mean: 128 },
          max_new_tokens: 1024,
          seed: 42,
        },
      },
      engine: {
        max_running_requests: 32,
        max_total_tokens: 65_536,
        l2_capacity_tokens: 0,
        l3_capacity_tokens: 0,
        l3_prefetch_min_tokens: 256,
        max_prefill_tokens: 16_384,
        page_size: 64,
        pd_mutually_exclusive: true,
      },
      perf_model: {
        prefill_c1: 0.05,
        prefill_c2: 0.000_01,
        decode_d1: 8,
        decode_d2: 0.002,
        kv_bytes_per_token: null,
        l2_to_l1_bandwidth_bytes_per_ms: null,
        l3_to_l2_bandwidth_bytes_per_ms: null,
        l2_latency_ms: 0,
        l3_latency_ms: 0,
        l3_lookup_latency_ms: 0,
      },
    });
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
        synthetic: {
          num_requests: 25,
          arrival_mode: "open_loop",
          prompt_len: { dist: "fixed", value: 256 },
          turn_increment_len: { dist: "poisson", mean: 64 },
        },
      },
    });
  });

  it("submits edited open-loop distribution, whole-page capacity and fused mode", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    const incrementGroup = screen.getByRole("group", { name: "子轮新增内容长度" });
    fireEvent.change(within(incrementGroup).getByLabelText("分布类型"), { target: { value: "fixed" } });
    fireEvent.change(within(incrementGroup).getByLabelText("固定长度（token）"), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText("L1 KV 总容量（token）"), {
      target: { value: "128" },
    });
    fireEvent.change(screen.getByLabelText("每页 token 数"), {
      target: { value: "4" },
    });
    fireEvent.change(screen.getByLabelText("每请求生成上限（token）"), {
      target: { value: "12" },
    });
    fireEvent.click(screen.getByLabelText(/Prefill \/ Decode 分步执行/));
    expect(screen.getByText(/32 页/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      workload: {
        synthetic: {
          num_requests: 20,
          max_new_tokens: 12,
          turn_increment_len: { dist: "fixed", value: 20 },
        },
      },
      engine: { max_total_tokens: 128, page_size: 4, pd_mutually_exclusive: false },
    });
  });

  it("exposes L2/L3 links only when enabled and submits valid transfer settings", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("L2 KV 容量（token，0 为关闭）"), { target: { value: "128" } });
    expect(screen.getByLabelText("KV 每 token 字节数")).toBeInTheDocument();
    expect(screen.queryByLabelText("L3 → L2 带宽（字节/ms）")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("L3 KV 容量（token，0 为关闭）"), { target: { value: "64" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByLabelText("KV 每 token 字节数")).toHaveAttribute("aria-invalid", "true");
    fireEvent.change(screen.getByLabelText("KV 每 token 字节数"), { target: { value: "4096" } });
    fireEvent.change(screen.getByLabelText("L2 → L1 带宽（字节/ms）"), { target: { value: "1024" } });
    fireEvent.change(screen.getByLabelText("L3 → L2 带宽（字节/ms）"), { target: { value: "512" } });
    fireEvent.change(screen.getByLabelText("L3 链路固定延迟（ms）"), { target: { value: "1.5" } });
    fireEvent.change(screen.getByLabelText("L3 查找延迟（ms）"), { target: { value: "0.75" } });
    fireEvent.change(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）"), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      engine: expect.objectContaining({ l2_capacity_tokens: 128, l3_capacity_tokens: 64,
        l3_prefetch_min_tokens: 0 }),
      perf_model: expect.objectContaining({
        kv_bytes_per_token: 4096,
        l2_to_l1_bandwidth_bytes_per_ms: 1024,
        l3_to_l2_bandwidth_bytes_per_ms: 512,
        l3_latency_ms: 1.5,
        l3_lookup_latency_ms: 0.75,
      }),
    }));
  });

  it.each([
    ["目标发送 QPS（请求/秒，泊松到达）", "0", "必须是大于 0 的有限数值"],
    ["L1 KV 总容量（token）", "130", "必须是页大小的整数倍"],
    ["L2 KV 容量（token，0 为关闭）", "65", "容量必须是页大小的非负整数倍"],
    ["L3 KV 容量（token，0 为关闭）", "64", "启用 L3 前必须先启用 L2"],
    ["L3 异步预取最小长度（token，0 为不设阈值）", "-1", "必须是大于或等于 0 的整数"],
    ["每页 token 数", "0", "必须是大于或等于 1 的整数"],
    ["每请求生成上限（token）", "0", "必须是大于或等于 1 的整数"],
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
    expect(JSON.parse(json)).toMatchObject({ workload: { synthetic: {
      num_requests: 25, arrival_mode: "open_loop", turn_increment_len: { dist: "poisson", mean: 256 },
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
    config.engine.page_size = 128;
    config.engine.l2_capacity_tokens = 256;
    config.engine.l3_capacity_tokens = 128;
    config.engine.l3_prefetch_min_tokens = 512;
    config.perf_model.kv_bytes_per_token = 2048;
    config.perf_model.l2_to_l1_bandwidth_bytes_per_ms = 1024;
    config.perf_model.l3_to_l2_bandwidth_bytes_per_ms = 512;
    config.perf_model.l2_latency_ms = 1;
    config.perf_model.l3_latency_ms = 2;
    config.perf_model.l3_lookup_latency_ms = 3;
    config.perf_model.prefill_c1 = 0.1;
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
    expect(screen.getByLabelText("L2 KV 容量（token，0 为关闭）")).toHaveValue(256);
    expect(screen.getByLabelText("L3 KV 容量（token，0 为关闭）")).toHaveValue(128);
    expect(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）")).toHaveValue(512);
    expect(screen.getByLabelText("KV 每 token 字节数")).toHaveValue(2048);
    expect(screen.getByLabelText("L2 → L1 带宽（字节/ms）")).toHaveValue(1024);
    expect(screen.getByLabelText("L3 → L2 带宽（字节/ms）")).toHaveValue(512);
    expect(screen.getByLabelText("L2 链路固定延迟（ms）")).toHaveValue(1);
    expect(screen.getByLabelText("L3 链路固定延迟（ms）")).toHaveValue(2);
    expect(screen.getByLabelText("L3 查找延迟（ms）")).toHaveValue(3);
    expect(screen.getByLabelText("Prefill C1")).toHaveValue(0.1);
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).toHaveBeenCalledWith(config);
  });

  it("exports and imports the complete automatic configuration from the top transfer controls", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={vi.fn()} onSubmitAuto={onSubmitAuto} />);
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    fireEvent.change(screen.getByLabelText("隐藏维度"), { target: { value: "2048" } });
    fireEvent.change(screen.getByLabelText("单卡标称显存（GB）"), { target: { value: "300" } });
    fireEvent.change(screen.getByLabelText("L3 可用容量（TB，仅 KV）"), { target: { value: "1.5" } });
    fireEvent.change(screen.getByLabelText("L3 查找延迟（ms）"), { target: { value: "2.5" } });
    fireEvent.change(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）"), { target: { value: "1024" } });
    expect(screen.getByLabelText("L3 可用容量（TB，仅 KV）")).toHaveValue(1.5);
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));
    const exported = (screen.getByLabelText("参数 JSON") as HTMLInputElement).value;
    expect(JSON.parse(exported)).toMatchObject({
      timing: "empirical", model: { hidden_size: 2048 },
      hardware: { memory_bytes_per_gpu: 300_000_000_000 },
      storage: { l3_available_gb: 1500, l3_lookup_latency_ms: 2.5 },
      config: { workload: { synthetic: { num_requests: 20 } }, engine: { l3_prefetch_min_tokens: 1024 } },
    });
    fireEvent.change(screen.getByLabelText("隐藏维度"), { target: { value: "4096" } });
    fireEvent.change(screen.getByLabelText("单卡标称显存（GB）"), { target: { value: "288" } });
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "analytical" } });
    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByRole("radio", { name: /自动解析/ })).toBeChecked();
    expect(screen.getByLabelText("隐藏维度")).toHaveValue(2048);
    expect(screen.getByLabelText("单卡标称显存（GB）")).toHaveValue(300);
    expect(screen.getByLabelText("L3 可用容量（TB，仅 KV）")).toHaveValue(1.5);
    expect(screen.getByLabelText("L3 查找延迟（ms）")).toHaveValue(2.5);
    expect(screen.getByLabelText("L3 异步预取最小长度（token，0 为不设阈值）")).toHaveValue(1024);
    expect(screen.getByLabelText("计时模式")).toHaveValue("empirical");
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      timing: "empirical", model: expect.objectContaining({ hidden_size: 2048 }),
      hardware: expect.objectContaining({ memory_bytes_per_gpu: 300_000_000_000 }),
      storage: expect.objectContaining({ l3_available_gb: 1500, l3_lookup_latency_ms: 2.5 }),
      config: expect.objectContaining({ engine: expect.objectContaining({ l3_prefetch_min_tokens: 1024 }) }),
    }));
  });

  it("keeps a complete, independent set of parameters for each panel", () => {
    const onSubmit = vi.fn();
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} onSubmitAuto={onSubmitAuto} />);
    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    expect(screen.getByRole("group", { name: "手动配置参数" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "25" } });
    fireEvent.change(screen.getByLabelText("最多同时运行请求"), { target: { value: "12" } });
    fireEvent.change(screen.getByLabelText("L1 KV 总容量（token）"), { target: { value: "128" } });
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
    expect(screen.getByRole("group", { name: "自动解析参数" })).toBeInTheDocument();
    expect(screen.getByLabelText("请求数量")).toHaveValue(1000);
    expect(screen.getByLabelText("最多同时运行请求")).toHaveValue(32);
    expect(screen.queryByLabelText("L1 KV 总容量（token）")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Prefill C1")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "40" } });
    fireEvent.change(screen.getByLabelText("最多同时运行请求"), { target: { value: "8" } });
    fireEvent.change(screen.getByLabelText("每页 token 数"), { target: { value: "100" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      timing: "analytical", config: expect.objectContaining({
        workload: expect.objectContaining({ synthetic: expect.objectContaining({ num_requests: 40 }) }),
        engine: expect.objectContaining({ max_running_requests: 8, page_size: 100,
          max_total_tokens: 100, l2_capacity_tokens: 0, l3_capacity_tokens: 0 }),
      }),
    }));
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    expect(screen.getByLabelText("请求数量")).toHaveValue(25);
    expect(screen.getByLabelText("最多同时运行请求")).toHaveValue(12);
    expect(screen.getByLabelText("L1 KV 总容量（token）")).toHaveValue(128);
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      workload: expect.objectContaining({ synthetic: expect.objectContaining({ num_requests: 25 }) }),
      engine: expect.objectContaining({ max_running_requests: 12, max_total_tokens: 128 }),
    }));
  });

  it("imports a manual config from the automatic panel without losing the automatic draft", () => {
    render(<SimulationForm onSubmit={vi.fn()} onSubmitAuto={vi.fn()} />);
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "30" } });
    const manual = createDefaultSimConfig();
    manual.workload.synthetic!.num_requests = 12;
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: JSON.stringify(manual) } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByRole("radio", { name: /手动配置/ })).toBeChecked();
    expect(screen.getByLabelText("请求数量")).toHaveValue(12);
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
    expect(screen.getByLabelText("请求数量")).toHaveValue(30);
  });

  it("shows empirical coefficients only when automatic timing uses them", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={vi.fn()} onSubmitAuto={onSubmitAuto} />);
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
    expect(screen.queryByLabelText("Prefill C1")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    fireEvent.change(screen.getByLabelText("Prefill C1"), { target: { value: "0.2" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenLastCalledWith(expect.objectContaining({
      timing: "empirical", config: expect.objectContaining({
        perf_model: expect.objectContaining({ prefill_c1: 0.2 }),
      }),
    }));
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "analytical" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledTimes(2);
    expect(onSubmitAuto).toHaveBeenLastCalledWith(expect.objectContaining({
      timing: "analytical", config: expect.objectContaining({
        perf_model: expect.objectContaining({ prefill_c1: 0.2 }),
      }),
    }));
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    expect(screen.getByLabelText("Prefill C1")).toHaveValue(0.2);
    fireEvent.change(screen.getByLabelText("Prefill C1"), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "analytical" } });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenLastCalledWith(expect.objectContaining({
      timing: "analytical", config: expect.objectContaining({
        perf_model: expect.objectContaining({ prefill_c1: 0.05 }),
      }),
    }));
  });

  it("round-trips EP topology and runs MoE analytical timing only", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={vi.fn()} onSubmitAuto={onSubmitAuto} />);
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
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
    expect(JSON.parse(exported)).toMatchObject({
      tp_size: 4, ep_size: 4,
      hardware: { ep_communication_latency_ms: 0.2, collective_latency_ms: 0.01,
        effective_interconnect_gb_s: 600 },
    });
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
      tp_size: 4, ep_size: 4, timing: "analytical",
    }));
    fireEvent.change(screen.getByLabelText("EP 数（MoE 须等于 TP）"), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByLabelText("EP 数（MoE 须等于 TP）")).toHaveValue(4);
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
  });

  it("round-trips TP topology and runs analytical multi-GPU only", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={vi.fn()} onSubmitAuto={onSubmitAuto} />);
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
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
    expect(JSON.parse(exported)).toMatchObject({
      tp_size: 4, ep_size: 1, hardware: { gpu_count: 4, collective_latency_ms: 0.1 },
    });
    fireEvent.change(screen.getByLabelText("TP 数（必须等于 GPU 卡数）"), {
      target: { value: "1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByLabelText("TP 数（必须等于 GPU 卡数）")).toHaveValue(4);
    expect(screen.getAllByRole("button", { name: "运行仿真" })).toHaveLength(1);
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      tp_size: 4, timing: "analytical", hardware: expect.objectContaining({ gpu_count: 4 }),
    }));
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "empirical" } });
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeDisabled();
    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    expect(screen.getByRole("button", { name: "运行仿真" })).toBeEnabled();
  });

  it("accepts legacy config JSON and rejects incomplete physical specs without changing the form", () => {
    render(<SimulationForm onSubmit={vi.fn()} onSubmitAuto={vi.fn()} />);
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
    expect(screen.getByRole("radio", { name: /手动配置/ })).toBeChecked();
    expect(screen.queryByLabelText("总参数量（B，含全部专家）")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: /自动解析/ }));
    expect(screen.getByLabelText("词表大小（结构估算必填）")).toHaveValue(151936);
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
