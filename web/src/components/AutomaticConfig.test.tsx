import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getPresets, resolveConfig, SimulationApiError } from "../api/client";
import type { ResolutionReport } from "../api/types";
import { createDefaultSimConfig, initialHardware, initialModel, initialStorage,
  type PhysicalSimConfig } from "../config/simulation";
import { AutomaticConfig } from "./AutomaticConfig";

function physicalConfig(patch: Partial<PhysicalSimConfig["inference"]> = {}): PhysicalSimConfig {
  const config = createDefaultSimConfig();
  return { ...config, inference: { ...config.inference, ...patch } };
}

vi.mock("../api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("../api/client")>(),
  getPresets: vi.fn(), resolveConfig: vi.fn(),
}));

const resolution = {
  max_total_tokens: 1024, l2_capacity_tokens: 0, l3_capacity_tokens: 0, runnable: true,
  capacity: {
    kv_bytes_per_token: 147456, weights_resident_bytes: 16_381_470_720,
    estimated_total_parameters: 8_190_735_360, estimated_active_parameters: 8_190_735_360,
    estimated_expert_parameters: 0,
    l1_capacity_pages: 16, l2_capacity_pages: 0, l3_capacity_pages: 0,
  },
} as ResolutionReport;

const hy4 = { ...initialModel, name: "tencent/Hy4-preview", revision: "705d81ee51566a186d645b74c974d642ef2828fe",
  num_layers: 78, hidden_size: 6144, num_attention_heads: 64, num_kv_heads: 8,
  head_dim: 256, ffn_intermediate_size: 18432, vocab_size: 120832,
  moe_layers: 77, num_experts: 256, experts_per_token: 8,
  expert_intermediate_size: 2048, num_shared_experts: 1, shared_expert_intermediate_size: 2048,
  attention_layout: "grouped" as const,
  norms: { kind: "rms" as const, block_per_layer: 2, final_norm: true, dtype: null },
  router: { dtype: "fp32" as const, correction_bias: true, correction_dtype: "fp32" as const },
  ihc: { mult: 4, dtype: "fp32" as const },
  attention_groups: [{ kind: "mla_dsa" as const, layers: 78, num_attention_heads: 64,
    q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256,
    index_heads: 32, index_dim: 128, topk: 2048, full_indexer_layers: 21, output_gate: true,
    qk_norm: true, norm_dtype: null, index_norm: "layer" as const,
    index_norm_dtype: "fp32" as const, sink_dtype: "fp32" as const }] };

describe("AutomaticConfig", () => {
  beforeEach(() => {
    vi.mocked(getPresets).mockResolvedValue({ models: [], hardware: [] });
    vi.mocked(resolveConfig).mockResolvedValue(resolution);
  });

  it("previews server-resolved capacity without a separate run button", async () => {
    render(<AutomaticConfig config={createDefaultSimConfig()} onInferenceChange={vi.fn()} disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("1,024 token"));
    expect(screen.getByRole("status")).toHaveTextContent("结构估算总参数 8.191 B");
    expect(screen.getByRole("status")).toHaveTextContent("结构估算每 token 激活参数 8.191 B");
    expect(screen.getByRole("status")).toHaveTextContent("全部专家参数 0.000 B");
    expect(resolveConfig).toHaveBeenCalledWith(expect.objectContaining({
      scheduler: expect.objectContaining({ max_running_requests: 32 }),
      inference: expect.objectContaining({ mode: "physical", timing: { mode: "analytical" },
        model: expect.objectContaining({ name: "Qwen/Qwen3-8B", num_layers: 36 }),
        hardware: expect.objectContaining({ gpu_count: 1 }) }),
    }));
    expect(screen.queryByRole("button", { name: "运行自动解析仿真" })).not.toBeInTheDocument();
  });

  it("invalidates a capacity preview when the shared L3 budget changes", async () => {
    const config = createDefaultSimConfig();
    const { rerender } = render(<AutomaticConfig config={config} onInferenceChange={vi.fn()} disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("全场共享 L3"));
    const changed = { ...config, shared_l3: { mode: "physical" as const, available_gb: 1000, lookup_latency_ms: 1 } };
    rerender(<AutomaticConfig config={changed} onInferenceChange={vi.fn()} disabled={false} />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(resolveConfig).toHaveBeenLastCalledWith(changed));
    await waitFor(() => expect(screen.getByRole("status")).toBeInTheDocument());
  });

  it("matches complete Qwen defaults independently of API object key order", async () => {
    const qwen = JSON.parse(JSON.stringify(initialModel)) as typeof initialModel;
    qwen.attention_groups = qwen.attention_groups!.map((group) =>
      Object.fromEntries(Object.entries(group).reverse()) as typeof group);
    qwen.norms = Object.fromEntries(Object.entries(qwen.norms!).reverse()) as typeof qwen.norms;
    vi.mocked(getPresets).mockResolvedValue({ models: [qwen], hardware: [] });
    const onInferenceChange = vi.fn();
    render(<AutomaticConfig config={createDefaultSimConfig()} onInferenceChange={onInferenceChange} disabled={false} />);
    await waitFor(() => expect(screen.getByLabelText("模型选择")).toHaveValue("Qwen/Qwen3-8B"));
    expect(screen.getByLabelText("第 1 组注意力方式")).toHaveValue("gqa");
    expect(screen.getByLabelText("第 1 组层数")).toHaveValue(36);
    expect(screen.getByLabelText("第 1 组 Query 头数")).toHaveValue(32);
    expect(screen.getByLabelText("第 1 组KV 头数")).toHaveValue(8);
    expect(screen.getByLabelText("第 1 组每头维度")).toHaveValue(128);
    expect(screen.getByLabelText("第 1 组 Q/K 归一化")).toBeChecked();
    expect(screen.getByLabelText("第 1 组注意力输出门控")).not.toBeChecked();
    expect(onInferenceChange).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("第 1 组 Query 头数"), { target: { value: "64" } });
    expect(onInferenceChange).toHaveBeenCalledWith({
      ...createDefaultSimConfig().inference,
      model: { ...initialModel, name: "自定义模型",
        attention_groups: [{ ...initialModel.attention_groups![0], num_attention_heads: 64 }] },
    });
  });

  it("uses themed collapsible auxiliary sections and attention cards without disabling editable fields", () => {
    const { container } = render(<fieldset disabled><AutomaticConfig config={createDefaultSimConfig()} onInferenceChange={vi.fn()} disabled /></fieldset>);
    const details = screen.getByText("辅助权重容量配置", { exact: true }).closest("details")!;
    expect(details).toHaveClass("model-subsection", "auxiliary-config");
    expect(details).not.toHaveAttribute("open");
    fireEvent.click(screen.getByText("辅助权重容量配置", { exact: true }));
    expect(details).toHaveAttribute("open");
    expect(screen.getByLabelText("主干归一化类型")).toBeDisabled();
    expect(container.querySelector(".auxiliary-config-fields")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "第 1 组注意力" }).closest(".model-group-card")).toBeInTheDocument();
  });

  it("selects HY4 BF16 with TP8/EP8 and a viable default L1 budget", async () => {
    vi.mocked(getPresets).mockResolvedValue({ models: [initialModel, hy4], hardware: [] });
    const onInferenceChange = vi.fn();
    render(<AutomaticConfig config={createDefaultSimConfig()} onInferenceChange={onInferenceChange} disabled={false} />);
    await waitFor(() => expect(screen.getByRole("option", { name: "tencent/Hy4-preview" })).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("模型选择"), { target: { value: "tencent/Hy4-preview" } });
    expect(onInferenceChange).toHaveBeenCalledTimes(1);
    expect(onInferenceChange).toHaveBeenCalledWith({ ...createDefaultSimConfig().inference,
      model: hy4, tp_size: 8, ep_size: 8,
      hardware: { ...initialHardware, gpu_count: 8 }, storage: { ...initialStorage, l1_available_gb: 260 } });
  });

  it("directly edits HY4 structure, precision, auxiliary weights and groups while preserving the preview request", async () => {
    vi.mocked(getPresets).mockResolvedValue({ models: [hy4], hardware: [] });
    const onInferenceChange = vi.fn();
    const props = { config: physicalConfig({ model: hy4, tp_size: 8, ep_size: 8 }),
      onInferenceChange: onInferenceChange, disabled: false };
    const { rerender } = render(<AutomaticConfig {...props} />);
    const refresh = () => rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
    await waitFor(() => expect(screen.getByLabelText("模型选择")).toHaveValue(hy4.name));
    expect(screen.getByText(/HY4-Preview 为可编辑参考预设/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "复制为自定义模型" })).not.toBeInTheDocument();
    for (const label of ["模型层数", "隐藏维度", "权重精度", "KV 精度", "输入输出词嵌入共享",
      "共享专家数（0 为关闭）", "共享专家中间维度（启用时必填）",
      "主干归一化类型", "每层主干归一化数量", "最终主干归一化", "主干归一化精度",
      "路由权重精度", "路由选位修正向量", "路由选位修正向量精度", "启用 iHC", "iHC 信息通路数（mult）", "iHC 精度",
      "第 1 组 Q/K 归一化", "第 1 组 Q/K 归一化精度", "第 1 组索引归一化类型",
      "第 1 组索引归一化精度", "第 1 组启用 attention sink", "第 1 组 attention sink 精度",
      "第 1 组注意力方式", "第 1 组注意力输出门控", "第 1 组层数", "第 1 组 Query 头数",
      "第 1 组Q 压缩维度", "第 1 组KV 压缩维度", "第 1 组位置 Key 维度", "第 1 组非位置 Q/K 维度",
      "第 1 组V 维度", "第 1 组索引头数", "第 1 组索引 Key 维度（128 倍数）",
      "第 1 组选取 token 数", "第 1 组重算索引的层数"]) {
      expect(screen.getByLabelText(label)).toBeEnabled();
    }
    for (const label of ["Query 头数", "KV 头数", "每头维度"]) {
      expect(screen.getByLabelText(label)).toBeDisabled();
    }
    fireEvent.change(screen.getByLabelText("隐藏维度"), { target: { value: "8192" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: { ...hy4, name: "自定义模型", revision: null, hidden_size: 8192 } });
    refresh();
    expect(screen.getByLabelText("模型选择")).toHaveValue("custom");
    for (const [label, value] of [["权重精度", "fp16"], ["KV 精度", "fp16"],
      ["共享专家数（0 为关闭）", "2"], ["共享专家中间维度（启用时必填）", "1024"],
      ["主干归一化精度", "fp32"], ["路由权重精度", "bf16"], ["iHC 信息通路数（mult）", "2"],
      ["第 1 组索引归一化精度", "bf16"], ["第 1 组选取 token 数", "1024"],
      ["第 1 组索引 Key 维度（128 倍数）", "256"], ["第 1 组重算索引的层数", "20"]]) {
      fireEvent.change(screen.getByLabelText(label), { target: { value } });
      refresh();
    }
    fireEvent.click(screen.getByLabelText("输入输出词嵌入共享"));
    refresh();
    fireEvent.click(screen.getByLabelText("第 1 组注意力输出门控"));
    refresh();
    const edited = onInferenceChange.mock.lastCall![0].model;
    expect(edited).toEqual({ ...hy4, name: "自定义模型", revision: null, hidden_size: 8192,
      weight_dtype: "fp16", kv_dtype: "fp16", tie_word_embeddings: true,
      num_shared_experts: 2, shared_expert_intermediate_size: 1024,
      norms: { ...hy4.norms, dtype: "fp32" }, router: { ...hy4.router, dtype: "bf16" },
      ihc: { ...hy4.ihc, mult: 2 }, attention_groups: [{ ...hy4.attention_groups[0],
        index_norm_dtype: "bf16", topk: 1024, index_dim: 256, full_indexer_layers: 20, output_gate: false }],
    });
    fireEvent.click(screen.getByRole("button", { name: "添加注意力组" }));
    const added = onInferenceChange.mock.lastCall![0].model;
    expect(added.attention_groups).toEqual([
      { ...edited.attention_groups[0], layers: 77 }, expect.objectContaining({ kind: "gqa", layers: 1 }),
    ]);
    refresh();
    fireEvent.click(screen.getByRole("button", { name: "删除第 2 组" }));
    refresh();
    fireEvent.change(screen.getByLabelText("第 1 组层数"), { target: { value: "78" } });
    refresh();
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference, model: edited });
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(resolveConfig).toHaveBeenCalledWith({
      ...props.config, inference: { ...props.config.inference,
        model: edited, hardware: initialHardware, storage: initialStorage, tp_size: 8, ep_size: 8 },
    }));
  });

  it.each(["grouped", "hy4_preview"] as const)("customizes every model edit path directly from a %s HY4 preset", async (attention_layout) => {
    const model = { ...hy4, attention_layout };
    vi.mocked(getPresets).mockResolvedValue({ models: [model], hardware: [] });
    const onInferenceChange = vi.fn();
    const props = { config: physicalConfig({ model, tp_size: 8, ep_size: 8 }),
      onInferenceChange: onInferenceChange, disabled: false };
    const { rerender } = render(<AutomaticConfig {...props} />);
    await waitFor(() => expect(screen.getByLabelText("模型选择")).toHaveValue(hy4.name));
    for (const [label, value, patch] of [
      ["隐藏维度", "8192", { hidden_size: 8192 }],
      ["权重精度", "fp16", { weight_dtype: "fp16" }],
      ["KV 精度", "fp16", { kv_dtype: "fp16" }],
      ["共享专家数（0 为关闭）", "2", { num_shared_experts: 2 }],
      ["共享专家中间维度（启用时必填）", "1024", { shared_expert_intermediate_size: 1024 }],
      ["主干归一化精度", "fp32", { norms: { ...hy4.norms, dtype: "fp32" } }],
      ["路由权重精度", "bf16", { router: { ...hy4.router, dtype: "bf16" } }],
      ["iHC 信息通路数（mult）", "2", { ihc: { ...hy4.ihc, mult: 2 } }],
      ["第 1 组选取 token 数", "1024", { attention_groups: [{ ...hy4.attention_groups[0], topk: 1024 }] }],
      ["模型名称", "user-model", { name: "user-model" }],
      ["修订号（可选）", "user-revision", { revision: "user-revision" }],
    ] as const) {
      fireEvent.change(screen.getByLabelText(label), { target: { value } });
      expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
        model: { ...hy4, name: "自定义模型", revision: null, ...patch } });
      rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
      expect(screen.getByLabelText("模型选择")).toHaveValue("custom");
      rerender(<AutomaticConfig {...props} />);
    }
    fireEvent.click(screen.getByLabelText("输入输出词嵌入共享"));
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: { ...hy4, name: "自定义模型", revision: null, tie_word_embeddings: true } });
    rerender(<AutomaticConfig {...props} />);
    fireEvent.click(screen.getByLabelText("第 1 组注意力输出门控"));
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: { ...hy4, name: "自定义模型", revision: null,
        attention_groups: [{ ...hy4.attention_groups[0], output_gate: false }] } });
    rerender(<AutomaticConfig {...props} />);
    fireEvent.change(screen.getByLabelText("模型选择"), { target: { value: "custom" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: { ...hy4, name: "自定义模型", revision: null } });
  });

  it("does not apply HY4 parallelism or storage defaults to other grouped presets", async () => {
    const other = { ...hy4, name: "another-grouped-model" };
    vi.mocked(getPresets).mockResolvedValue({ models: [other], hardware: [] });
    const onInferenceChange = vi.fn();
    const config = physicalConfig({ tp_size: 2 });
    render(<AutomaticConfig config={config} onInferenceChange={onInferenceChange} disabled={false} />);
    await waitFor(() => expect(screen.getByRole("option", { name: other.name })).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("模型选择"), { target: { value: other.name } });
    expect(onInferenceChange).toHaveBeenCalledTimes(1);
    expect(onInferenceChange).toHaveBeenCalledWith({ ...config.inference, model: other, ep_size: 2 });
  });

  it("preserves a custom grouped model identity when adjusting its parameters", () => {
    const model = { ...hy4, name: "user-model", revision: "user-revision" };
    const onInferenceChange = vi.fn();
    const config = physicalConfig({ model, tp_size: 8, ep_size: 8 });
    render(<AutomaticConfig config={config} onInferenceChange={onInferenceChange} disabled={false} />);
    fireEvent.change(screen.getByLabelText("隐藏维度"), { target: { value: "8192" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...config.inference,
      model: { ...model, hidden_size: 8192 } });
  });

  it.each(["mla", "mla_dsa"] as const)("edits %s output gates and resets them when switching to GQA", (kind) => {
    const model = { ...initialModel, name: "自定义模型", attention_layout: "grouped" as const,
      attention_groups: [{ ...hy4.attention_groups[0], kind, layers: initialModel.num_layers,
        full_indexer_layers: kind === "mla_dsa" ? 1 : undefined,
        index_heads: kind === "mla_dsa" ? 32 : undefined,
        index_dim: kind === "mla_dsa" ? 128 : undefined,
        topk: kind === "mla_dsa" ? 2048 : undefined,
        index_norm_dtype: kind === "mla_dsa" ? "fp32" as const : null, output_gate: false }] };
    const onInferenceChange = vi.fn();
    const props = { config: physicalConfig({ model }), onInferenceChange: onInferenceChange, disabled: false };
    const { rerender } = render(<AutomaticConfig {...props} />);
    expect(screen.getByLabelText("第 1 组注意力输出门控")).not.toBeChecked();
    fireEvent.click(screen.getByLabelText("第 1 组注意力输出门控"));
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: { ...model, attention_groups: [{ ...model.attention_groups[0], output_gate: true }] } });
    const gated = onInferenceChange.mock.lastCall![0];
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: gated }} />);
    fireEvent.change(screen.getByLabelText("第 1 组注意力方式"), {
      target: { value: kind === "mla" ? "mla_dsa" : "mla" },
    });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: expect.objectContaining({ attention_groups: [expect.objectContaining({ output_gate: true })] }) });
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
    fireEvent.change(screen.getByLabelText("第 1 组注意力方式"), { target: { value: "gqa" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: expect.objectContaining({ attention_groups: [expect.objectContaining({ kind: "gqa", output_gate: false })] }) });
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
    expect(screen.getByLabelText("第 1 组注意力输出门控")).toBeDisabled();
    expect(screen.getByLabelText("第 1 组注意力输出门控")).not.toBeChecked();
  });

  it("creates attention groups with output gates disabled by default", () => {
    const onInferenceChange = vi.fn();
    const legacy = { ...initialModel, attention_layout: "full_gqa" as const, attention_groups: null };
    const props = { config: physicalConfig({ model: legacy }), onInferenceChange: onInferenceChange, disabled: false };
    const { rerender } = render(<AutomaticConfig {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "启用注意力组合配置" }));
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: expect.objectContaining({ attention_groups: [expect.objectContaining({ kind: "gqa", output_gate: false })] }) });
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
    fireEvent.change(screen.getByLabelText("第 1 组注意力方式"), { target: { value: "mla" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: expect.objectContaining({ attention_groups: [expect.objectContaining({ kind: "mla", output_gate: false })] }) });
  });

  it.each(["bf16", "fp16", "fp32"] as const)("edits every auxiliary precision as %s without dropping settings", (dtype) => {
    const model = { ...hy4, name: "自定义模型", revision: null, attention_layout: "grouped" as const };
    const onInferenceChange = vi.fn();
    const props = { config: physicalConfig({ model }), onInferenceChange: onInferenceChange, disabled: false };
    const { rerender } = render(<AutomaticConfig {...props} />);
    for (const [label, field] of [["主干归一化精度", "norms"], ["路由权重精度", "router"],
      ["路由选位修正向量精度", "router"], ["iHC 精度", "ihc"],
      ["第 1 组 Q/K 归一化精度", "attention_groups"], ["第 1 组索引归一化精度", "attention_groups"],
      ["第 1 组 attention sink 精度", "attention_groups"]] as const) {
      fireEvent.change(screen.getByLabelText(label), { target: { value: dtype } });
      const inference = onInferenceChange.mock.lastCall![0];
      const changed = inference.model;
      expect(inference).toEqual({ ...props.config.inference, model: changed });
      const actual = field === "attention_groups" ? changed.attention_groups[0] : changed[field];
      const key = label === "路由选位修正向量精度" ? "correction_dtype" :
        label === "第 1 组 Q/K 归一化精度" ? "norm_dtype" :
        label === "第 1 组索引归一化精度" ? "index_norm_dtype" :
        label === "第 1 组 attention sink 精度" ? "sink_dtype" : "dtype";
      expect(actual[key]).toBe(dtype);
      expect(changed.ihc.mult).toBe(4);
      expect(changed.norms.block_per_layer).toBe(2);
      expect(changed.attention_groups[0].output_gate).toBe(true);
      rerender(<AutomaticConfig {...props} config={{ ...props.config, inference }} />);
    }
    for (const label of ["主干归一化精度", "路由权重精度", "第 1 组 Q/K 归一化精度", "第 1 组索引归一化精度"]) {
      fireEvent.change(screen.getByLabelText(label), { target: { value: "" } });
      rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
      expect(screen.getByLabelText(label)).toHaveValue("");
    }
  });

  it("edits auxiliary counts, kinds and switches including GQA sinks", () => {
    const onInferenceChange = vi.fn();
    const props = { config: physicalConfig({ model: { ...hy4,
      name: "自定义模型", revision: null, attention_layout: "grouped" as const } }),
      onInferenceChange: onInferenceChange, disabled: false };
    const { rerender } = render(<AutomaticConfig {...props} />);
    const refresh = () => rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
    fireEvent.change(screen.getByLabelText("每层主干归一化数量"), { target: { value: "3" } });
    expect(onInferenceChange.mock.lastCall![0].model.norms.block_per_layer).toBe(3);
    refresh();
    fireEvent.click(screen.getByLabelText("最终主干归一化"));
    expect(onInferenceChange.mock.lastCall![0].model.norms.final_norm).toBe(false);
    refresh();
    fireEvent.change(screen.getByLabelText("主干归一化类型"), { target: { value: "none" } });
    expect(onInferenceChange.mock.lastCall![0].model.norms.kind).toBe("none");
    refresh();
    expect(screen.getByLabelText("每层主干归一化数量")).toBeDisabled();
    fireEvent.click(screen.getByLabelText("路由选位修正向量"));
    expect(onInferenceChange.mock.lastCall![0].model.router.correction_bias).toBe(false);
    refresh();
    expect(screen.getByLabelText("路由选位修正向量精度")).toBeDisabled();
    fireEvent.change(screen.getByLabelText("iHC 信息通路数（mult）"), { target: { value: "2" } });
    expect(onInferenceChange.mock.lastCall![0].model.ihc.mult).toBe(2);
    refresh();
    fireEvent.click(screen.getByLabelText("启用 iHC"));
    expect(onInferenceChange.mock.lastCall![0].model.ihc).toBeNull();
    refresh();
    expect(screen.getByLabelText("iHC 精度")).toBeDisabled();
    fireEvent.click(screen.getByLabelText("启用 iHC"));
    expect(onInferenceChange.mock.lastCall![0].model.ihc).toEqual({ mult: 4, dtype: "fp32" });
    refresh();
    fireEvent.click(screen.getByLabelText("第 1 组 Q/K 归一化"));
    expect(onInferenceChange.mock.lastCall![0].model.attention_groups[0].qk_norm).toBe(false);
    refresh();
    expect(screen.getByLabelText("第 1 组 Q/K 归一化精度")).toBeDisabled();
    fireEvent.change(screen.getByLabelText("第 1 组索引归一化类型"), { target: { value: "rms" } });
    expect(onInferenceChange.mock.lastCall![0].model.attention_groups[0].index_norm).toBe("rms");
    refresh();
    fireEvent.change(screen.getByLabelText("第 1 组注意力方式"), { target: { value: "gqa" } });
    expect(onInferenceChange.mock.lastCall![0].model.attention_groups[0]).toEqual(expect.objectContaining({
      kind: "gqa", qk_norm: false, index_norm: "layer", index_norm_dtype: null,
      sink_dtype: "fp32", output_gate: false,
    }));
    refresh();
    expect(screen.queryByLabelText("第 1 组索引归一化类型")).not.toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("第 1 组启用 attention sink"));
    expect(onInferenceChange.mock.lastCall![0].model.attention_groups[0].sink_dtype).toBeNull();
    refresh();
    expect(screen.getByLabelText("第 1 组 attention sink 精度")).toBeDisabled();
    fireEvent.click(screen.getByLabelText("第 1 组启用 attention sink"));
    expect(onInferenceChange.mock.lastCall![0].model.attention_groups[0].sink_dtype).toBe("fp32");
  });

  it("enables shared experts only for MoE and clears their dimension when disabled", () => {
    const onInferenceChange = vi.fn();
    const props = { config: createDefaultSimConfig(), onInferenceChange: onInferenceChange, disabled: false };
    const { rerender } = render(<AutomaticConfig {...props} />);
    expect(screen.getByLabelText("共享专家数（0 为关闭）")).toHaveValue(0);
    expect(screen.getByLabelText("共享专家数（0 为关闭）")).toBeDisabled();
    expect(screen.getByLabelText("共享专家中间维度（启用时必填）")).toHaveValue(null);
    expect(screen.getByLabelText("共享专家中间维度（启用时必填）")).toBeDisabled();
    expect(screen.getByLabelText("路由权重精度")).toBeDisabled();
    expect(screen.getByLabelText("路由选位修正向量")).toBeDisabled();
    expect(screen.getByLabelText("路由选位修正向量精度")).toBeDisabled();
    const moe = { ...initialModel, moe_layers: 1, num_experts: 8,
      experts_per_token: 2, expert_intermediate_size: 512, router: hy4.router };
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: { ...props.config.inference, model: moe } }} />);
    expect(screen.getByLabelText("路由权重精度")).toBeEnabled();
    expect(screen.getByLabelText("路由选位修正向量")).toBeEnabled();
    expect(screen.getByLabelText("路由选位修正向量精度")).toBeEnabled();
    expect(screen.getByLabelText("共享专家数（0 为关闭）")).toBeEnabled();
    fireEvent.change(screen.getByLabelText("共享专家数（0 为关闭）"), { target: { value: "2" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: { ...moe, num_shared_experts: 2, shared_expert_intermediate_size: null } });
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
    expect(screen.getByLabelText("共享专家中间维度（启用时必填）")).toBeEnabled();
    fireEvent.change(screen.getByLabelText("共享专家中间维度（启用时必填）"), { target: { value: "1024" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...props.config.inference,
      model: { ...moe, num_shared_experts: 2, shared_expert_intermediate_size: 1024 } });
    const shared = onInferenceChange.mock.lastCall![0];
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: shared }} />);
    fireEvent.change(screen.getByLabelText("共享专家中间维度（启用时必填）"), { target: { value: "" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...shared,
      model: { ...shared.model, shared_expert_intermediate_size: null } });
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
    fireEvent.change(screen.getByLabelText("共享专家数（0 为关闭）"), { target: { value: "0" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...shared,
      model: { ...shared.model, num_shared_experts: 0, shared_expert_intermediate_size: null } });
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
    expect(screen.getByLabelText("共享专家中间维度（启用时必填）")).toBeDisabled();
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: shared }} />);
    fireEvent.change(screen.getByLabelText("MoE 层数"), { target: { value: "0" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...shared,
      model: { ...shared.model, moe_layers: 0, num_shared_experts: 0, shared_expert_intermediate_size: null,
        router: { dtype: null, correction_bias: false, correction_dtype: "fp32" } } });
    rerender(<AutomaticConfig {...props} config={{ ...props.config, inference: onInferenceChange.mock.lastCall![0] }} />);
    expect(screen.getByLabelText("路由权重精度")).toBeDisabled();
    expect(screen.getByLabelText("路由权重精度")).toHaveValue("");
    expect(screen.getByLabelText("路由选位修正向量")).not.toBeChecked();
  });

  it("edits independent MLA and MLA+DSA attention groups without requiring layer indices", async () => {
    const model = { ...initialModel, name: "自定义模型", num_layers: 2,
      attention_layout: "grouped" as const, attention_groups: [
        { kind: "mla" as const, layers: 1, num_attention_heads: 32, output_gate: false,
          qk_norm: true, index_norm: "layer" as const,
          q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256 },
        { kind: "mla_dsa" as const, layers: 1, num_attention_heads: 32, output_gate: false,
          qk_norm: true, index_norm: "layer" as const,
          q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256,
          index_heads: 32, index_dim: 128, topk: 2048, full_indexer_layers: 1 },
      ] };
    const onInferenceChange = vi.fn();
    const config = physicalConfig({ model });
    const { rerender } = render(<AutomaticConfig config={config} onInferenceChange={onInferenceChange} disabled={false} />);
    expect(screen.getByText(/已配置 2 \/ 2 层/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("模型名称"), { target: { value: "another-mla-model" } });
    expect(onInferenceChange).toHaveBeenCalledWith({ ...config.inference,
      model: { ...model, name: "another-mla-model" } });
    rerender(<AutomaticConfig config={{ ...config, inference: onInferenceChange.mock.lastCall![0] }}
      onInferenceChange={onInferenceChange} disabled={false} />);
    fireEvent.change(screen.getByLabelText("第 2 组选取 token 数"), { target: { value: "1024" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...config.inference,
      model: { ...model, name: "another-mla-model", attention_groups: [
        model.attention_groups[0], { ...model.attention_groups[1], topk: 1024 },
      ] } });
    expect(screen.queryByLabelText("第 1 组选取 token 数")).not.toBeInTheDocument();
  });

  it("shows TP rank preview and the analytical timing scope", async () => {
    const preview = {
      ...resolution, runnable: true,
      capacity: { ...resolution.capacity, unassigned_parameters: 50,
        rank_reports: [{ rank: 0, sharded_weights_bytes: 1000,
          replicated_weights_bytes: 200, expert_weights_bytes: 0, kv_bytes_per_token: 36864,
          l1_kv_budget_bytes: 800, l1_capacity_pages: 2 }] },
    } as ResolutionReport;
    vi.mocked(resolveConfig).mockResolvedValue(preview);
    const config = physicalConfig({ hardware: { ...initialHardware, gpu_count: 4 }, tp_size: 4 });
    render(<AutomaticConfig config={config} onInferenceChange={vi.fn()} disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("GPU 0"));
    expect(resolveConfig).toHaveBeenCalledWith(expect.objectContaining({ inference: expect.objectContaining({ tp_size: 4 }) }));
    expect(screen.getByText(/每层两次等效同步/)).toBeInTheDocument();
  });

  it("shows MoE expert placement in a non-runnable EP capacity preview", async () => {
    const model = { ...initialModel, moe_layers: 36, num_experts: 8,
      experts_per_token: 2, expert_intermediate_size: 512 };
    vi.mocked(resolveConfig).mockResolvedValue({
      ...resolution, runnable: false,
      capacity: { ...resolution.capacity, rank_reports: [{
        rank: 0, sharded_weights_bytes: 1000, replicated_weights_bytes: 200,
        expert_weights_bytes: 2_000_000_000, kv_bytes_per_token: 36864,
        l1_kv_budget_bytes: 800, l1_capacity_pages: 2,
      }] },
    } as ResolutionReport);
    const config = physicalConfig({ model, hardware: { ...initialHardware, gpu_count: 4 }, tp_size: 4, ep_size: 4 });
    render(<AutomaticConfig config={config} onInferenceChange={vi.fn()} disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("全部驻留专家权重 2.000 GB"));
    expect(screen.getByRole("status")).toHaveTextContent("TP4 + EP4");
    expect(resolveConfig).toHaveBeenCalledWith(expect.objectContaining({ inference: expect.objectContaining({ tp_size: 4, ep_size: 4 }) }));
    expect(screen.getByText(/未经校准的平均负载解析计时/)).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("填写互联带宽、TP 集合通信延迟与 EP 单次通信开销");
  });

  it("shows the exact field path for server validation errors during preview", async () => {
    vi.mocked(resolveConfig).mockRejectedValueOnce(new SimulationApiError("模型规格错误", {
      validationErrors: [{
        loc: ["body", "inference", "physical", "model", "num_experts"], msg: "专家数必须为正", type: "value_error",
      }],
    }));
    render(<AutomaticConfig config={createDefaultSimConfig()} onInferenceChange={vi.fn()} disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("alert"))
      .toHaveTextContent("body → inference → physical → model → num_experts：专家数必须为正"));
  });

  it("separates model and hardware cards and converts GB inputs to API units", async () => {
    vi.mocked(getPresets).mockResolvedValue({
      models: [initialModel, { ...initialModel, name: "Qwen/Qwen3-235B-A22B",
        num_layers: 94, num_attention_heads: 64, num_kv_heads: 4,
        moe_layers: 94, num_experts: 128, experts_per_token: 8, expert_intermediate_size: 1536,
        attention_groups: [{ ...initialModel.attention_groups![0],
          layers: 94, num_attention_heads: 64, num_kv_heads: 4 }] }],
      hardware: [{ name: "NVIDIA B300", gpu_count: 8, memory_bytes_per_gpu: 288_000_000_000,
        peak_fp16_tflops: 2250, peak_hbm_gb_s: 8000, peak_interconnect_gb_s: 1800 }],
    });
    const onInferenceChange = vi.fn();
    let config = createDefaultSimConfig();
    const { rerender } = render(<AutomaticConfig config={config} onInferenceChange={onInferenceChange} disabled={false} />);
    const refresh = () => {
      config = { ...config, inference: onInferenceChange.mock.lastCall![0] };
      rerender(<AutomaticConfig config={config} onInferenceChange={onInferenceChange} disabled={false} />);
    };
    await waitFor(() => expect(screen.getByRole("combobox", { name: "GPU 选择" })).toHaveValue("NVIDIA B300"));
    expect(screen.getByRole("heading", { name: "模型参数" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "硬件配置" })).toBeInTheDocument();
    expect(screen.queryByLabelText(/总参数量/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/实际驻留权重/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("TP 每次集合通信固定延迟（ms，TP>1 运行必填）")).toHaveValue(0.01);
    expect(screen.getByLabelText("EP 每次 dispatch/combine 固定开销（ms，EP>1 运行必填）")).toHaveValue(0.02);
    expect(screen.getByLabelText("单卡标称显存（GB）")).toHaveValue(288);
    expect(screen.queryByLabelText("L3 查找延迟（ms）")).not.toBeInTheDocument();
    expect(screen.getByText(/2,250 TFLOPS 稠密 BF16\/FP16 算力/)).toBeInTheDocument();
    expect(screen.getByText(/8,000 GB\/s HBM 带宽/)).toBeInTheDocument();
    expect(screen.getByText(/1,800 GB\/s 单卡 NVLink 双向标称带宽/)).toBeInTheDocument();
    expect(screen.getByLabelText("单 rank 有效互联带宽（GB/s，TP>1 必填）")).toHaveValue(600);
    fireEvent.change(screen.getByLabelText("GPU 选择"), { target: { value: "NVIDIA B300" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...config.inference,
      hardware: { ...config.inference.hardware, name: "NVIDIA B300", memory_bytes_per_gpu: 288_000_000_000 } });
    refresh();
    expect(config.inference.hardware.effective_interconnect_gb_s).toBe(600);
    expect(screen.queryByLabelText("参考组合")).not.toBeInTheDocument();
    expect(screen.getByLabelText("模型名称")).toHaveValue("Qwen/Qwen3-8B");
    expect(screen.queryByRole("button", { name: "导入自动参数" })).not.toBeInTheDocument();
    expect(Array.from((screen.getByLabelText("模型选择") as HTMLSelectElement).options)
      .map((option) => option.textContent)).toEqual([
        "Qwen/Qwen3-8B", "Qwen/Qwen3-235B-A22B", "自定义",
      ]);
    fireEvent.change(screen.getByLabelText("模型选择"), { target: { value: "Qwen/Qwen3-235B-A22B" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...config.inference,
      model: expect.objectContaining({ name: "Qwen/Qwen3-235B-A22B", moe_layers: 94 }),
      ep_size: config.inference.tp_size });
    refresh();
    fireEvent.change(screen.getByLabelText("隐藏维度"), { target: { value: "2048" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...config.inference,
      model: { ...config.inference.model, name: "自定义模型", hidden_size: 2048 } });
    refresh();
    fireEvent.change(screen.getByLabelText("单卡标称显存（GB）"), { target: { value: "300" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...config.inference,
      hardware: { ...config.inference.hardware, name: "自定义 GPU", memory_bytes_per_gpu: 300_000_000_000 } });
    refresh();
    fireEvent.change(screen.getByLabelText("额外运行时预留（GB）"), { target: { value: "2" } });
    expect(onInferenceChange).toHaveBeenLastCalledWith({ ...config.inference,
      storage: { ...config.inference.storage, runtime_reserve_bytes: 2_000_000_000 } });
    refresh();
    expect(config.inference.model).toEqual(expect.objectContaining({ moe_layers: 94, hidden_size: 2048 }));
    expect(config.shared_l3?.lookup_latency_ms).toBe(0);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(resolveConfig).toHaveBeenCalledWith(config));
  });
});
