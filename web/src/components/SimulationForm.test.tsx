import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createDefaultSimConfig } from "../config/simulation";
import { SimulationForm } from "./SimulationForm";

afterEach(() => vi.unstubAllGlobals());

describe("SimulationForm", () => {
  it("selects automatic resolution first when available", () => {
    const onSubmitAuto = vi.fn();
    render(<SimulationForm onSubmit={vi.fn()} onSubmitAuto={onSubmitAuto} />);

    const modes = screen.getAllByRole("radio", { name: /自动解析|手动配置/ });
    expect(modes).toHaveLength(2);
    expect(modes[0]).toHaveAttribute("value", "automatic");
    expect(modes[0]).toBeChecked();
    expect(modes[1]).toHaveAttribute("value", "manual");
    expect(screen.getByRole("group", { name: "自动解析参数" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      config: expect.objectContaining({ workload: expect.objectContaining({
        synthetic: expect.objectContaining({ session: expect.objectContaining({
          target_ideal_prefix_hit_rate: 0.9,
        }) }),
      }) }),
    }));
  });

  it("renders every synthetic configuration group with defaults", () => {
    render(<SimulationForm onSubmit={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "负载配置" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "引擎配置" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "性能模型" })).toBeInTheDocument();
    expect(screen.getByLabelText("负载模式")).toHaveValue("synthetic");
    expect(screen.getByLabelText("请求数量")).toHaveValue(1000);
    expect(screen.getByLabelText("总请求目标 QPS（请求/秒）")).toHaveValue(64);
    expect(screen.getByLabelText("随机种子")).toHaveValue(42);
    expect(screen.getByLabelText("会话数量（根请求数）")).toHaveValue(20);
    expect(screen.getByLabelText("轮次间平均思考时间（ms，指数分布）")).toHaveValue(10_000);
    expect(screen.getByLabelText("目标理想前缀命中率（%）")).toHaveValue(90);
    expect(screen.getByLabelText("目标理想前缀命中率（%）")).toHaveAttribute("max", "100");
    const promptGroup = screen.getByRole("group", { name: "Prompt 长度" });
    expect(within(promptGroup).getByLabelText("分布类型")).toHaveValue("poisson");
    expect(within(promptGroup).getByLabelText("平均长度（token）")).toHaveValue(32_768);
    expect(screen.queryByLabelText("允许跨会话共享公共前缀")).not.toBeInTheDocument();
    expect(screen.getByLabelText("每请求生成上限（token）")).toHaveValue(1024);
    expect(screen.getByLabelText("最多同时运行请求")).toHaveValue(32);
    expect(screen.getByLabelText("L1 KV 总容量（token）")).toHaveValue(65_536);
    expect(screen.getByLabelText("L2 KV 容量（token，0 为关闭）")).toHaveValue(0);
    expect(screen.getByLabelText("L3 KV 容量（token，0 为关闭）")).toHaveValue(0);
    expect(screen.queryByLabelText("L2 → L1 带宽（字节/ms）")).not.toBeInTheDocument();
    expect(screen.getByLabelText("单次 Prefill 额度（token）")).toHaveValue(16_384);
    expect(screen.getByLabelText("每页 token 数")).toHaveValue(64);
    expect(screen.getByLabelText(/Prefill \/ Decode 分步执行/)).toBeChecked();
    expect(screen.getByText(/1024 页/)).toBeInTheDocument();
  });

  it("submits the complete default config", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    fireEvent.submit(screen.getByRole("button", { name: "运行仿真" }).closest("form")!);

    expect(onSubmit).toHaveBeenCalledWith({
      workload: {
        mode: "synthetic",
        synthetic: {
          num_requests: 1000,
          arrival_rate: 64,
          prompt_len: { dist: "poisson", mean: 32_768 },
          output_len: { dist: "poisson", mean: 128 },
          max_new_tokens: 1024,
          seed: 42,
          session: {
            num_sessions: 20,
            topology: "linear",
            think_time_ms: 10_000,
            target_ideal_prefix_hit_rate: 0.9,
          },
        },
      },
      engine: {
        max_running_requests: 32,
        max_total_tokens: 65_536,
        l2_capacity_tokens: 0,
        l3_capacity_tokens: 0,
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
      },
    });
    const submittedSession = onSubmit.mock.calls[0][0].workload.synthetic.session;
    expect(JSON.parse(JSON.stringify(submittedSession))).not.toHaveProperty(
      "allow_cross_session_shared_prefix",
    );
  });

  it("updates scalar and distribution fields before submission", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText("请求数量"), {
      target: { value: "25" },
    });

    const promptGroup = screen.getByRole("group", { name: "Prompt 长度" });
    fireEvent.change(within(promptGroup).getByLabelText("分布类型"), {
      target: { value: "fixed" },
    });
    fireEvent.change(within(promptGroup).getByLabelText("固定长度（token）"), {
      target: { value: "256" },
    });
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      workload: {
        synthetic: {
          num_requests: 25,
          session: { num_sessions: 20 },
          prompt_len: { dist: "fixed", value: 256 },
        },
      },
    });
  });

  it("submits edited multi-turn sessions, whole-page capacity and fused mode", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText("会话数量（根请求数）"), {
      target: { value: "5" },
    });
    fireEvent.change(screen.getByLabelText("轮次间平均思考时间（ms，指数分布）"), {
      target: { value: "30" },
    });
    fireEvent.change(screen.getByLabelText("目标理想前缀命中率（%）"), {
      target: { value: "20" },
    });
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
          session: {
            num_sessions: 5,
            topology: "linear",
            think_time_ms: 30,
            target_ideal_prefix_hit_rate: 0.2,
          },
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
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      engine: expect.objectContaining({ l2_capacity_tokens: 128, l3_capacity_tokens: 64 }),
      perf_model: expect.objectContaining({
        kv_bytes_per_token: 4096,
        l2_to_l1_bandwidth_bytes_per_ms: 1024,
        l3_to_l2_bandwidth_bytes_per_ms: 512,
        l3_latency_ms: 1.5,
      }),
    }));
  });

  it.each([
    ["会话数量（根请求数）", "1001", "会话数量不能超过请求数量"],
    ["轮次间平均思考时间（ms，指数分布）", "-1", "必须是大于或等于 0 的有限数值"],
    ["目标理想前缀命中率（%）", "101", "必须是 0 到 100 之间的有限数值"],
    ["L1 KV 总容量（token）", "130", "必须是页大小的整数倍"],
    ["L2 KV 容量（token，0 为关闭）", "65", "容量必须是页大小的非负整数倍"],
    ["L3 KV 容量（token，0 为关闭）", "64", "启用 L3 前必须先启用 L2"],
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
      num_requests: 25, session: { num_sessions: 20, target_ideal_prefix_hit_rate: 0.9 },
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
    config.workload.synthetic!.session!.num_sessions = 5;
    config.workload.synthetic!.session!.target_ideal_prefix_hit_rate = 0.35;
    config.workload.synthetic!.prompt_len = { dist: "fixed", value: 256 };
    config.workload.synthetic!.output_len = { dist: "lognormal", mean: 50, sigma: 2 };
    config.engine.page_size = 128;
    config.engine.l2_capacity_tokens = 256;
    config.engine.l3_capacity_tokens = 128;
    config.perf_model.kv_bytes_per_token = 2048;
    config.perf_model.l2_to_l1_bandwidth_bytes_per_ms = 1024;
    config.perf_model.l3_to_l2_bandwidth_bytes_per_ms = 512;
    config.perf_model.l2_latency_ms = 1;
    config.perf_model.l3_latency_ms = 2;
    config.perf_model.prefill_c1 = 0.1;
    fireEvent.change(screen.getByLabelText("参数 JSON"), { target: { value: JSON.stringify(config) } });

    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));

    expect(screen.getByRole("status")).toHaveTextContent("参数已导入");
    expect(screen.getByLabelText("请求数量")).toHaveValue(20);
    expect(screen.getByLabelText("会话数量（根请求数）")).toHaveValue(5);
    expect(screen.getByLabelText("目标理想前缀命中率（%）")).toHaveValue(35);
    expect(within(screen.getByRole("group", { name: "Prompt 长度" })).getByLabelText("固定长度（token）"))
      .toHaveValue(256);
    expect(within(screen.getByRole("group", { name: "输出长度" })).getByLabelText("形状参数（sigma）"))
      .toHaveValue(2);
    expect(screen.getByLabelText("每页 token 数")).toHaveValue(128);
    expect(screen.getByLabelText("L2 KV 容量（token，0 为关闭）")).toHaveValue(256);
    expect(screen.getByLabelText("L3 KV 容量（token，0 为关闭）")).toHaveValue(128);
    expect(screen.getByLabelText("KV 每 token 字节数")).toHaveValue(2048);
    expect(screen.getByLabelText("L2 → L1 带宽（字节/ms）")).toHaveValue(1024);
    expect(screen.getByLabelText("L3 → L2 带宽（字节/ms）")).toHaveValue(512);
    expect(screen.getByLabelText("L2 链路固定延迟（ms）")).toHaveValue(1);
    expect(screen.getByLabelText("L3 链路固定延迟（ms）")).toHaveValue(2);
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
    fireEvent.change(screen.getByLabelText("总参数量（B，含全部专家）"), { target: { value: "10.5" } });
    fireEvent.change(screen.getByLabelText("单卡标称显存（GB）"), { target: { value: "300" } });
    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    fireEvent.click(screen.getByRole("button", { name: "导出参数" }));
    const exported = (screen.getByLabelText("参数 JSON") as HTMLInputElement).value;
    expect(JSON.parse(exported)).toMatchObject({
      timing: "empirical", model: { total_parameters: 10_500_000_000 },
      hardware: { memory_bytes_per_gpu: 300_000_000_000 },
      config: { workload: { synthetic: { num_requests: 20 } } },
    });
    fireEvent.change(screen.getByLabelText("总参数量（B，含全部专家）"), { target: { value: "8" } });
    fireEvent.change(screen.getByLabelText("单卡标称显存（GB）"), { target: { value: "288" } });
    fireEvent.change(screen.getByLabelText("计时模式"), { target: { value: "analytical" } });
    fireEvent.click(screen.getByRole("radio", { name: /手动配置/ }));
    fireEvent.click(screen.getByRole("button", { name: "导入参数" }));
    expect(screen.getByRole("radio", { name: /自动解析/ })).toBeChecked();
    expect(screen.getByLabelText("总参数量（B，含全部专家）")).toHaveValue(10.5);
    expect(screen.getByLabelText("单卡标称显存（GB）")).toHaveValue(300);
    expect(screen.getByLabelText("计时模式")).toHaveValue("empirical");
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));
    expect(onSubmitAuto).toHaveBeenCalledWith(expect.objectContaining({
      timing: "empirical", model: expect.objectContaining({ total_parameters: 10_500_000_000 }),
      hardware: expect.objectContaining({ memory_bytes_per_gpu: 300_000_000_000 }),
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
    manual.workload.synthetic!.session!.num_sessions = 12;
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
    fireEvent.change(screen.getByLabelText("专家中间维度（可选）"), { target: { value: "512" } });
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
      hardware: { ep_communication_latency_ms: 0.2, collective_latency_ms: null },
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
    config.workload.synthetic!.session!.num_sessions = 12;
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
    expect(screen.getByLabelText("总参数量（B，含全部专家）")).toHaveValue(8.2);
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
