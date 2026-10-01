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
    expect(screen.getByLabelText("轮次间思考时间（ms）")).toHaveValue(0);
    expect(screen.getByLabelText("目标理想前缀命中率（0–1）")).toHaveValue(0);
    expect(screen.queryByLabelText("允许跨会话共享公共前缀")).not.toBeInTheDocument();
    expect(screen.getByLabelText("最大批大小")).toHaveValue(32);
    expect(screen.getByLabelText("L1 KV token 预算")).toHaveValue(65_536);
    expect(screen.getByLabelText("Prefix cache 块大小（token）")).toHaveValue(1);
    expect(screen.getByText(/L1 容量由 KV token 预算换算： 65536 blocks/)).toBeInTheDocument();
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
        max_batch_size: 32,
        kv_token_budget: 65_536,
        prefix_cache: {
          block_size_tokens: 1,
          cache_salt: "",
          eviction_policy: "lru",
          hierarchy_policy: "exclusive_write_back",
          kv_bytes_per_token: 1,
          model_id: "default-model",
          tokenizer_id: "default-tokenizer",
          kv_dtype: "default-dtype",
          adapter_id: "default-adapter",
          multimodal_transform_id: "default-transform",
        },
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

  it("submits edited multi-turn sessions and derived block capacity", () => {
    const onSubmit = vi.fn();
    render(<SimulationForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText("请求数量"), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText("会话数量（根请求数）"), {
      target: { value: "5" },
    });
    fireEvent.change(screen.getByLabelText("轮次间思考时间（ms）"), {
      target: { value: "30" },
    });
    fireEvent.change(screen.getByLabelText("目标理想前缀命中率（0–1）"), {
      target: { value: "0.2" },
    });
    fireEvent.change(screen.getByLabelText("L1 KV token 预算"), {
      target: { value: "9" },
    });
    fireEvent.change(screen.getByLabelText("Prefix cache 块大小（token）"), {
      target: { value: "4" },
    });
    expect(screen.getByText(/L1 容量由 KV token 预算换算： 2 blocks/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "运行仿真" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      workload: {
        synthetic: {
          num_requests: 20,
          session: {
            num_sessions: 5,
            topology: "linear",
            think_time_ms: 30,
            target_ideal_prefix_hit_rate: 0.2,
          },
        },
      },
      engine: { kv_token_budget: 9, prefix_cache: { block_size_tokens: 4 } },
    });
  });

  it.each([
    ["会话数量（根请求数）", "1001", "会话数量不能超过请求数量"],
    ["轮次间思考时间（ms）", "-1", "必须是大于或等于 0 的有限数值"],
    ["目标理想前缀命中率（0–1）", "1.1", "必须是 0 到 1 之间的有限数值"],
    ["Prefix cache 块大小（token）", "65537", "块大小不能超过 KV token 预算"],
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
