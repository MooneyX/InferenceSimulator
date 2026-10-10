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
    kv_bytes_per_token: 147456, weights_resident_bytes: 16400000000,
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
    expect(resolveAutomatic).toHaveBeenCalledWith(expect.objectContaining({
      timing: "analytical",
      model: expect.objectContaining({ name: "Qwen/Qwen3-8B", num_layers: 36 }),
      hardware: expect.objectContaining({ gpu_count: 1 }),
      config: expect.objectContaining({ engine: expect.objectContaining({ l2_capacity_tokens: 0 }) }),
    }));
    expect(screen.queryByRole("button", { name: "运行自动解析仿真" })).not.toBeInTheDocument();
  });

  it("shows TP rank preview and the analytical timing scope", async () => {
    const preview = {
      ...resolution, tp_size: 4, runnable: true,
      capacity: { ...resolution.capacity, unassigned_parameters: 50,
        rank_reports: [{ rank: 0, sharded_weights_bytes: 1000,
          replicated_weights_bytes: 200, kv_bytes_per_token: 36864,
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

  it("separates model and hardware cards and converts B/GB inputs to API units", async () => {
    vi.mocked(getPresets).mockResolvedValue({
      models: [initialModel, { ...initialModel, name: "Qwen/Qwen3-235B-A22B",
        total_parameters: 235_000_000_000, moe_layers: 94 }],
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
    expect(screen.getByLabelText("总参数量（B，含全部专家）")).toHaveValue(8.2);
    expect(screen.getByLabelText("单卡标称显存（GB）")).toHaveValue(288);
    expect(screen.getByText(/2,250 TFLOPS 稠密 BF16\/FP16 算力/)).toBeInTheDocument();
    expect(screen.getByText(/8,000 GB\/s HBM 带宽/)).toBeInTheDocument();
    expect(screen.getByText(/1,800 GB\/s 单卡 NVLink 双向标称带宽/)).toBeInTheDocument();
    expect(screen.getByLabelText("单 rank 有效互联带宽（GB/s，TP>1 必填）")).toHaveValue(1800);
    expect(screen.queryByLabelText("参考组合")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("模型名称")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "导入自动参数" })).not.toBeInTheDocument();
    expect(Array.from((screen.getByLabelText("模型选择") as HTMLSelectElement).options)
      .map((option) => option.textContent)).toEqual([
        "Qwen/Qwen3-8B", "Qwen/Qwen3-235B-A22B", "自定义",
      ]);
    fireEvent.change(screen.getByLabelText("模型选择"), { target: { value: "Qwen/Qwen3-235B-A22B" } });
    expect(onModelChange).toHaveBeenCalledWith(expect.objectContaining({ total_parameters: 235_000_000_000 }));
    fireEvent.change(screen.getByLabelText("总参数量（B，含全部专家）"), { target: { value: "10.5" } });
    expect(onModelChange).toHaveBeenLastCalledWith(expect.objectContaining({
      name: "自定义模型", total_parameters: 10_500_000_000,
    }));
    fireEvent.change(screen.getByLabelText("单卡标称显存（GB）"), { target: { value: "300" } });
    expect(onHardwareChange).toHaveBeenCalledWith(expect.objectContaining({
      name: "自定义 GPU", memory_bytes_per_gpu: 300_000_000_000,
    }));
    fireEvent.change(screen.getByLabelText("额外运行时预留（GB）"), { target: { value: "2" } });
    expect(onStorageChange).toHaveBeenCalledWith(expect.objectContaining({ runtime_reserve_bytes: 2_000_000_000 }));
  });
});
