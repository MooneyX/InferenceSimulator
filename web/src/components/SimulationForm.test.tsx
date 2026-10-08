import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SimulationForm } from "./SimulationForm";

describe("SimulationForm", () => {
  it("renders every synthetic configuration group with defaults", () => {
    render(<SimulationForm onSubmit={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "负载配置" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "引擎配置" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "性能模型" })).toBeInTheDocument();
    expect(screen.getByLabelText("负载模式")).toHaveValue("synthetic");
    expect(screen.getByLabelText("请求数量")).toHaveValue(1000);
    expect(screen.getByLabelText("总请求目标 QPS（请求/秒）")).toHaveValue(50);
    expect(screen.getByLabelText("会话数量（根请求数）")).toHaveValue(1000);
    expect(screen.getByLabelText("轮次间平均思考时间（ms，指数分布）")).toHaveValue(0);
    expect(screen.getByLabelText("目标理想前缀命中率（0–1）")).toHaveValue(0);
    expect(screen.queryByLabelText("允许跨会话共享公共前缀")).not.toBeInTheDocument();
    expect(screen.getByLabelText("每请求生成上限（token）")).toHaveValue(128);
    expect(screen.getByLabelText("最多同时运行请求")).toHaveValue(32);
    expect(screen.getByLabelText("L1 KV 总容量（token）")).toHaveValue(65_536);
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
          arrival_rate: 50,
          prompt_len: { dist: "lognormal", mean: 512, sigma: 1 },
          output_len: { dist: "poisson", mean: 128 },
          max_new_tokens: 128,
          seed: 0,
          session: {
            num_sessions: 1000,
            topology: "linear",
            think_time_ms: 0,
            target_ideal_prefix_hit_rate: 0,
          },
        },
      },
      engine: {
        max_running_requests: 32,
        max_total_tokens: 65_536,
        max_prefill_tokens: 16_384,
        page_size: 64,
        pd_mutually_exclusive: true,
      },
      perf_model: {
        prefill_c1: 0.05,
        prefill_c2: 0.000_01,
        decode_d1: 8,
        decode_d2: 0.002,
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
          session: { num_sessions: 25 },
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
    fireEvent.change(screen.getByLabelText("目标理想前缀命中率（0–1）"), {
      target: { value: "0.2" },
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

  it.each([
    ["会话数量（根请求数）", "1001", "会话数量不能超过请求数量"],
    ["轮次间平均思考时间（ms，指数分布）", "-1", "必须是大于或等于 0 的有限数值"],
    ["目标理想前缀命中率（0–1）", "1.1", "必须是 0 到 1 之间的有限数值"],
    ["L1 KV 总容量（token）", "130", "必须是页大小的整数倍"],
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

  it("disables submission while a simulation is running", () => {
    render(<SimulationForm onSubmit={vi.fn()} disabled />);

    expect(screen.getByRole("button", { name: "正在运行…" })).toBeDisabled();
  });
});
