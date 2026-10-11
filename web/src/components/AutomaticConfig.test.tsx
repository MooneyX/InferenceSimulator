import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getPresets, resolveAutomatic, SimulationApiError } from "../api/client";
import type { ResolvedConfig } from "../api/types";
import { createDefaultSimConfig } from "../config/simulation";
import { AutomaticConfig, initialHardware, initialModel, initialStorage } from "./AutomaticConfig";

vi.mock("../api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("../api/client")>(),
  getPresets: vi.fn(), resolveAutomatic: vi.fn(),
}));

const resolution = {
  config: {
    ...createDefaultSimConfig(),
    engine: { ...createDefaultSimConfig().engine, max_total_tokens: 1024 },
  },
  model: {}, hardware: {}, storage: {}, timing: "analytical",
  capacity: {
    kv_bytes_per_token: 147456, weights_resident_bytes: 16_381_470_720,
    estimated_total_parameters: 8_190_735_360, estimated_active_parameters: 8_190_735_360,
    estimated_expert_parameters: 0,
    l1_capacity_pages: 16, l2_capacity_pages: 0, l3_capacity_pages: 0,
  },
} as ResolvedConfig;

describe("AutomaticConfig", () => {
  beforeEach(() => {
    vi.mocked(getPresets).mockResolvedValue({ models: [], hardware: [], scenarios: [] });
    vi.mocked(resolveAutomatic).mockResolvedValue(resolution);
  });

  it("previews server-resolved capacity without a separate run button", async () => {
    render(<AutomaticConfig config={createDefaultSimConfig()} model={initialModel}
      hardware={initialHardware} storage={initialStorage} timing="analytical" tpSize={1} epSize={1} onEpSizeChange={vi.fn()}
      onTpSizeChange={vi.fn()} onModelChange={vi.fn()} onHardwareChange={vi.fn()} onStorageChange={vi.fn()}
      disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("1,024 token"));
    expect(screen.getByRole("status")).toHaveTextContent("结构估算总参数 8.191 B");
    expect(screen.getByRole("status")).toHaveTextContent("结构估算每 token 激活参数 8.191 B");
    expect(screen.getByRole("status")).toHaveTextContent("全部专家参数 0.000 B");
    expect(resolveAutomatic).toHaveBeenCalledWith(expect.objectContaining({
      timing: "analytical",
      model: expect.objectContaining({ name: "Qwen/Qwen3-8B", num_layers: 36 }),
      hardware: expect.objectContaining({ gpu_count: 1 }),
      config: expect.objectContaining({ engine: expect.objectContaining({ l2_capacity_tokens: 0 }) }),
    }));
    expect(screen.queryByRole("button", { name: "运行自动解析仿真" })).not.toBeInTheDocument();
  });

  it("selects HY4 BF16 with TP8/EP8 and a viable default L1 budget", async () => {
    const hy4 = { ...initialModel, name: "tencent/Hy4-preview", revision: "705d81ee51566a186d645b74c974d642ef2828fe",
      num_layers: 78, hidden_size: 6144, num_attention_heads: 64, num_kv_heads: 8,
      head_dim: 256, ffn_intermediate_size: 18432, vocab_size: 120832,
      moe_layers: 77, num_experts: 256, experts_per_token: 8,
      expert_intermediate_size: 2048, attention_layout: "hy4_preview" as const,
      attention_groups: [{ kind: "mla_dsa" as const, layers: 78, num_attention_heads: 64,
        q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256,
        index_heads: 32, index_dim: 128, topk: 2048, full_indexer_layers: 21 }] };
    vi.mocked(getPresets).mockResolvedValue({ models: [initialModel, hy4], hardware: [], scenarios: [] });
    const onModelChange = vi.fn();
    const onTpSizeChange = vi.fn();
    const onEpSizeChange = vi.fn();
    const onHardwareChange = vi.fn();
    const onStorageChange = vi.fn();
    render(<AutomaticConfig config={createDefaultSimConfig()} model={initialModel}
      hardware={initialHardware} storage={initialStorage} timing="analytical"
      tpSize={1} epSize={1} onTpSizeChange={onTpSizeChange} onEpSizeChange={onEpSizeChange}
      onModelChange={onModelChange} onHardwareChange={onHardwareChange}
      onStorageChange={onStorageChange} disabled={false} />);
    await waitFor(() => expect(screen.getByRole("option", { name: "tencent/Hy4-preview" })).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("模型选择"), { target: { value: "tencent/Hy4-preview" } });
    expect(onModelChange).toHaveBeenCalledWith(hy4);
    expect(onTpSizeChange).toHaveBeenCalledWith(8);
    expect(onEpSizeChange).toHaveBeenCalledWith(8);
    expect(onHardwareChange).toHaveBeenCalledWith(expect.objectContaining({ gpu_count: 8 }));
    expect(onStorageChange).toHaveBeenCalledWith(expect.objectContaining({ l1_available_gb: 260 }));
  });

  it("edits independent MLA and MLA+DSA attention groups without requiring layer indices", async () => {
    const model = { ...initialModel, name: "自定义模型", num_layers: 2,
      attention_layout: "grouped" as const, attention_groups: [
        { kind: "mla" as const, layers: 1, num_attention_heads: 32,
          q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256 },
        { kind: "mla_dsa" as const, layers: 1, num_attention_heads: 32,
          q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256,
          index_heads: 32, index_dim: 128, topk: 2048, full_indexer_layers: 1 },
      ] };
    const onModelChange = vi.fn();
    render(<AutomaticConfig config={createDefaultSimConfig()} model={model}
      hardware={initialHardware} storage={initialStorage} timing="analytical"
      tpSize={1} epSize={1} onTpSizeChange={vi.fn()} onEpSizeChange={vi.fn()}
      onModelChange={onModelChange} onHardwareChange={vi.fn()}
      onStorageChange={vi.fn()} disabled={false} />);
    expect(screen.getByText(/已配置 2 \/ 2 层/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("模型名称"), { target: { value: "another-mla-model" } });
    expect(onModelChange).toHaveBeenCalledWith(expect.objectContaining({ name: "another-mla-model" }));
    fireEvent.change(screen.getByLabelText("第 2 组选取 token 数"), { target: { value: "1024" } });
    expect(onModelChange).toHaveBeenCalledWith(expect.objectContaining({ attention_groups: [
      model.attention_groups[0], expect.objectContaining({ kind: "mla_dsa", topk: 1024 }),
    ] }));
    expect(screen.queryByLabelText("第 1 组选取 token 数")).not.toBeInTheDocument();
  });

  it("shows TP rank preview and the analytical timing scope", async () => {
    const preview = {
      ...resolution, tp_size: 4, runnable: true,
      capacity: { ...resolution.capacity, unassigned_parameters: 50,
        rank_reports: [{ rank: 0, sharded_weights_bytes: 1000,
          replicated_weights_bytes: 200, expert_weights_bytes: 0, kv_bytes_per_token: 36864,
          l1_kv_budget_bytes: 800, l1_capacity_pages: 2 }] },
    } as ResolvedConfig;
    vi.mocked(resolveAutomatic).mockResolvedValue(preview);
    render(<AutomaticConfig config={createDefaultSimConfig()} model={initialModel}
      hardware={{ ...initialHardware, gpu_count: 4 }} storage={initialStorage}
      timing="analytical" tpSize={4} epSize={1} onTpSizeChange={vi.fn()} onEpSizeChange={vi.fn()}
      onModelChange={vi.fn()} onHardwareChange={vi.fn()} onStorageChange={vi.fn()}
      disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("GPU 0"));
    expect(resolveAutomatic).toHaveBeenCalledWith(expect.objectContaining({ tp_size: 4 }));
    expect(screen.getByText(/每层两次等效同步/)).toBeInTheDocument();
  });

  it("shows MoE expert placement in a non-runnable EP capacity preview", async () => {
    const model = { ...initialModel, moe_layers: 36, num_experts: 8,
      experts_per_token: 2, expert_intermediate_size: 512 };
    vi.mocked(resolveAutomatic).mockResolvedValue({
      ...resolution, model, tp_size: 4, ep_size: 4, runnable: false,
      capacity: { ...resolution.capacity, rank_reports: [{
        rank: 0, sharded_weights_bytes: 1000, replicated_weights_bytes: 200,
        expert_weights_bytes: 2_000_000_000, kv_bytes_per_token: 36864,
        l1_kv_budget_bytes: 800, l1_capacity_pages: 2,
      }] },
    } as ResolvedConfig);
    render(<AutomaticConfig config={createDefaultSimConfig()} model={model}
      hardware={{ ...initialHardware, gpu_count: 4 }} storage={initialStorage}
      timing="analytical" tpSize={4} epSize={4} onTpSizeChange={vi.fn()} onEpSizeChange={vi.fn()}
      onModelChange={vi.fn()} onHardwareChange={vi.fn()} onStorageChange={vi.fn()} disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("全部驻留专家权重 2.000 GB"));
    expect(screen.getByRole("status")).toHaveTextContent("TP4 + EP4");
    expect(resolveAutomatic).toHaveBeenCalledWith(expect.objectContaining({ tp_size: 4, ep_size: 4 }));
    expect(screen.getByText(/未经校准的平均负载解析计时/)).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("填写互联带宽、TP 集合通信延迟与 EP 单次通信开销");
  });

  it("shows the exact field path for server validation errors during preview", async () => {
    vi.mocked(resolveAutomatic).mockRejectedValueOnce(new SimulationApiError("模型规格错误", {
      kind: "validation", validationErrors: [{
        loc: ["body", "model", "num_experts"], msg: "专家数必须为正", type: "value_error",
      }],
    }));
    render(<AutomaticConfig config={createDefaultSimConfig()} model={initialModel}
      hardware={initialHardware} storage={initialStorage} timing="analytical"
      tpSize={1} epSize={1} onTpSizeChange={vi.fn()} onEpSizeChange={vi.fn()}
      onModelChange={vi.fn()} onHardwareChange={vi.fn()} onStorageChange={vi.fn()} disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("alert"))
      .toHaveTextContent("body → model → num_experts：专家数必须为正"));
  });

  it("separates model and hardware cards and converts GB inputs to API units", async () => {
    vi.mocked(getPresets).mockResolvedValue({
      models: [initialModel, { ...initialModel, name: "Qwen/Qwen3-235B-A22B",
        moe_layers: 94 }],
      hardware: [{ name: "NVIDIA B300", gpu_count: 8, memory_bytes_per_gpu: 288_000_000_000,
        peak_fp16_tflops: 2250, peak_hbm_gb_s: 8000, peak_interconnect_gb_s: 1800 }],
      scenarios: [],
    });
    const onModelChange = vi.fn();
    const onHardwareChange = vi.fn();
    const onStorageChange = vi.fn();
    render(<AutomaticConfig config={createDefaultSimConfig()} model={initialModel}
      hardware={initialHardware} storage={initialStorage} timing="analytical" tpSize={1} epSize={1} onEpSizeChange={vi.fn()}
      onTpSizeChange={vi.fn()} onModelChange={onModelChange} onHardwareChange={onHardwareChange}
      onStorageChange={onStorageChange} disabled={false} />);
    await waitFor(() => expect(screen.getByRole("combobox", { name: "GPU 选择" })).toHaveValue("NVIDIA B300"));
    expect(screen.getByRole("heading", { name: "模型参数" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "硬件配置" })).toBeInTheDocument();
    expect(screen.queryByLabelText(/总参数量/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/实际驻留权重/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("TP 每次集合通信固定延迟（ms，TP>1 运行必填）")).toHaveValue(0.01);
    expect(screen.getByLabelText("EP 每次 dispatch/combine 固定开销（ms，EP>1 运行必填）")).toHaveValue(0.02);
    expect(screen.getByLabelText("单卡标称显存（GB）")).toHaveValue(288);
    expect(screen.getByLabelText("L3 查找延迟（ms）")).toHaveValue(0);
    fireEvent.change(screen.getByLabelText("L3 查找延迟（ms）"), { target: { value: "1.5" } });
    expect(onStorageChange).toHaveBeenCalledWith(expect.objectContaining({ l3_lookup_latency_ms: 1.5 }));
    expect(screen.getByText(/2,250 TFLOPS 稠密 BF16\/FP16 算力/)).toBeInTheDocument();
    expect(screen.getByText(/8,000 GB\/s HBM 带宽/)).toBeInTheDocument();
    expect(screen.getByText(/1,800 GB\/s 单卡 NVLink 双向标称带宽/)).toBeInTheDocument();
    expect(screen.getByLabelText("单 rank 有效互联带宽（GB/s，TP>1 必填）")).toHaveValue(600);
    fireEvent.change(screen.getByLabelText("GPU 选择"), { target: { value: "NVIDIA B300" } });
    expect(onHardwareChange).toHaveBeenLastCalledWith(expect.objectContaining({
      effective_interconnect_gb_s: 600,
    }));
    expect(screen.queryByLabelText("参考组合")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("模型名称")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "导入自动参数" })).not.toBeInTheDocument();
    expect(Array.from((screen.getByLabelText("模型选择") as HTMLSelectElement).options)
      .map((option) => option.textContent)).toEqual([
        "Qwen/Qwen3-8B", "Qwen/Qwen3-235B-A22B", "自定义",
      ]);
    fireEvent.change(screen.getByLabelText("模型选择"), { target: { value: "Qwen/Qwen3-235B-A22B" } });
    expect(onModelChange).toHaveBeenCalledWith(expect.objectContaining({ moe_layers: 94 }));
    fireEvent.change(screen.getByLabelText("隐藏维度"), { target: { value: "2048" } });
    expect(onModelChange).toHaveBeenLastCalledWith(expect.objectContaining({
      name: "自定义模型", hidden_size: 2048,
    }));
    fireEvent.change(screen.getByLabelText("单卡标称显存（GB）"), { target: { value: "300" } });
    expect(onHardwareChange).toHaveBeenCalledWith(expect.objectContaining({
      name: "自定义 GPU", memory_bytes_per_gpu: 300_000_000_000,
    }));
    fireEvent.change(screen.getByLabelText("额外运行时预留（GB）"), { target: { value: "2" } });
    expect(onStorageChange).toHaveBeenCalledWith(expect.objectContaining({ runtime_reserve_bytes: 2_000_000_000 }));
  });
});
