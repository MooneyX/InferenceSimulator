import { useEffect, useState } from "react";

import { getPresets, resolveAutomatic } from "../api/client";
import type { PresetCatalog, ResolvedConfig, ResolvedRequest, SimConfig } from "../api/types";

export type Model = ResolvedRequest["model"];
export type Hardware = ResolvedRequest["hardware"];
export type Storage = ResolvedRequest["storage"];

interface Props {
  config: SimConfig;
  model: Model;
  hardware: Hardware;
  storage: Storage;
  timing: ResolvedRequest["timing"];
  onModelChange: (value: Model) => void;
  onHardwareChange: (value: Hardware) => void;
  onStorageChange: (value: Storage) => void;
  disabled: boolean;
  onRun: (request: ResolvedRequest) => void;
}

export const initialModel: Model = {
  name: "Qwen/Qwen3-8B", revision: null, num_layers: 36, hidden_size: 4096,
  num_attention_heads: 32, num_kv_heads: 8, head_dim: 128,
  ffn_intermediate_size: 12288, total_parameters: 8_200_000_000,
  weight_dtype: "bf16", kv_dtype: "bf16", resident_weights_bytes: null,
  moe_layers: 0, expert_parameters_total: null, active_parameters: null,
  num_experts: 0, experts_per_token: 0, expert_intermediate_size: null,
  attention_layout: "full_gqa",
};
export const initialHardware: Hardware = {
  name: "NVIDIA B300", gpu_count: 1, memory_bytes_per_gpu: 288_000_000_000,
  effective_fp16_tflops: 100, effective_fp8_tflops: null,
  effective_hbm_gb_s: 1000, effective_interconnect_gb_s: null,
  kernel_overhead_ms: 0,
};
export const initialStorage: Storage = {
  l1_available_gb: 200, l2_available_gb: 0, l3_available_gb: 0,
  runtime_reserve_bytes: 0, l2_to_l1_gb_s: null, l3_to_l2_gb_s: null,
  l2_latency_ms: 0, l3_latency_ms: 0,
};

function Input({ label, value, onChange, optional = false }: {
  label: string; value: number | null | undefined;
  onChange: (value: number | null) => void; optional?: boolean;
}) {
  return <div className="form-field"><label>{label}
    <input type="number" step="any" value={value ?? ""}
      onChange={(event) => onChange(event.currentTarget.value === ""
        ? (optional ? null : NaN) : event.currentTarget.valueAsNumber)} />
  </label></div>;
}

export function AutomaticConfig({ config, model, hardware, storage, timing,
  onModelChange, onHardwareChange, onStorageChange, disabled, onRun }: Props) {
  const [catalog, setCatalog] = useState<PresetCatalog | null>(null);
  const [preview, setPreview] = useState<ResolvedConfig | null>(null);
  const [previewKey, setPreviewKey] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    void getPresets().then((data) => { if (active) setCatalog(data); }).catch(() => {
      if (active) setCatalog(null);
    });
    return () => { active = false; };
  }, []);

  const request: ResolvedRequest = { model, hardware, storage, timing, config };
  const key = JSON.stringify(request);
  const selectedModel = catalog?.models.find((entry) => entry.name === model.name &&
    Object.entries(entry).every(([field, value]) => model[field as keyof Model] === value));
  const selectedHardware = catalog?.hardware.find((entry) => entry.name === hardware.name &&
    entry.memory_bytes_per_gpu === hardware.memory_bytes_per_gpu);

  async function inspect() {
    setError("");
    try {
      const response = await resolveAutomatic(request);
      setPreview(response);
      setPreviewKey(key);
    } catch (cause) {
      setPreview(null);
      setError(cause instanceof Error ? cause.message : "无法解析配置");
    }
  }

  return <>
    <section className="form-section" aria-labelledby="auto-model-heading">
      <div className="section-heading"><div>
        <p className="section-kicker">Analytical · Stage 1</p>
        <h2 id="auto-model-heading">模型参数</h2>
      </div><p>选择预置模型，或切换到自定义并调整结构与精度</p></div>
      <div className="form-grid" style={{ marginBottom: 14 }}><div className="form-field"><label htmlFor="auto-model-preset">模型选择</label>
        <select id="auto-model-preset" value={selectedModel?.name ?? "custom"}
          onChange={(event) => {
            const selected = catalog?.models.find((entry) => entry.name === event.target.value);
            onModelChange(selected ?? { ...model, name: "自定义模型" });
          }}>{catalog?.models.map((entry) =>
            <option key={entry.name} value={entry.name}>{entry.name}</option>)}
          <option value="custom">自定义</option></select>
      </div></div>
    <div className="form-grid">{([
      ["num_layers", "模型层数"], ["hidden_size", "隐藏维度"],
      ["num_attention_heads", "Query 头数"], ["num_kv_heads", "KV 头数"],
      ["head_dim", "每头维度"], ["ffn_intermediate_size", "FFN 中间维度"],
      ["total_parameters", "总参数量（B，含全部专家）"], ["moe_layers", "MoE 层数"],
      ["expert_parameters_total", "全部专家参数量（B）"], ["active_parameters", "激活参数量（B，可选）"],
      ["num_experts", "专家数"], ["experts_per_token", "每 token 激活专家数"],
      ["expert_intermediate_size", "专家中间维度（可选）"],
      ["resident_weights_bytes", "实际驻留权重（GB，可选覆盖）"],
    ] as const).map(([field, label]) => {
      const inBillions = field === "total_parameters" || field === "expert_parameters_total" ||
        field === "active_parameters" || field === "resident_weights_bytes";
      return <Input key={field} label={label}
        value={inBillions && model[field] != null ? model[field] / 1e9 : model[field]}
        optional={field === "expert_parameters_total" || field === "active_parameters" ||
          field === "expert_intermediate_size" || field === "resident_weights_bytes"}
        onChange={(value) => onModelChange({ ...model, name: "自定义模型",
          [field]: inBillions && value !== null ? value * 1e9 : value })} />;
    })}
      <div className="form-field"><label>权重精度<select value={model.weight_dtype} onChange={(event) =>
        onModelChange({ ...model, name: "自定义模型", weight_dtype: event.target.value as Model["weight_dtype"] })}>
        <option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp8">FP8（待支持）</option>
      </select></label></div>
      <div className="form-field"><label>KV 精度<select value={model.kv_dtype} onChange={(event) =>
        onModelChange({ ...model, name: "自定义模型", kv_dtype: event.target.value as Model["kv_dtype"] })}>
        <option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp8">FP8（待支持）</option>
      </select></label></div>
    </div>
    </section>
    <section className="form-section" aria-labelledby="auto-hardware-heading">
      <div className="section-heading"><div>
        <p className="section-kicker">Hardware · Stage 1</p>
        <h2 id="auto-hardware-heading">硬件配置</h2>
      </div><p>标称峰值供参考；有效算力与带宽需按实际负载校准</p></div>
      <div className="form-grid" style={{ marginBottom: 14 }}><div className="form-field"><label htmlFor="auto-gpu-preset">GPU 选择</label>
        <select id="auto-gpu-preset" value={selectedHardware?.name ?? "custom"}
          onChange={(event) => {
            const selected = catalog?.hardware.find((entry) => entry.name === event.target.value);
            onHardwareChange(selected ? { ...hardware, name: selected.name,
              gpu_count: 1, memory_bytes_per_gpu: selected.memory_bytes_per_gpu } :
              { ...hardware, name: "自定义 GPU" });
          }}>{catalog?.hardware.map((entry) =>
            <option key={entry.name} value={entry.name}>{entry.name}</option>)}
          <option value="custom">自定义</option></select>
      </div></div>
    <div className="form-grid">{([
      ["gpu_count", "GPU 卡数（首阶段须为 1）"],
      ["memory_bytes_per_gpu", "单卡标称显存（GB）"],
      ["effective_fp16_tflops", "有效 FP16/BF16 算力（TFLOPS）"],
      ["effective_fp8_tflops", "有效 FP8 算力（TFLOPS，可选）"],
      ["effective_hbm_gb_s", "有效 HBM 带宽（GB/s）"],
      ["effective_interconnect_gb_s", "有效互联带宽（GB/s，可选）"],
      ["kernel_overhead_ms", "单算子固定开销（ms）"],
    ] as const).map(([field, label]) => <Input key={field} label={label}
      value={field === "memory_bytes_per_gpu" ? hardware[field] / 1e9 : hardware[field]}
      optional={field === "effective_fp8_tflops" || field === "effective_interconnect_gb_s"}
      onChange={(value) => onHardwareChange({ ...hardware,
        ...(field === "memory_bytes_per_gpu" ? { name: "自定义 GPU" } : {}),
        [field]: field === "memory_bytes_per_gpu" && value !== null ? value * 1e9 : value })} />)}
      {([
        ["l1_available_gb", "L1 可用容量（GB，含权重）"],
        ["l2_available_gb", "L2 可用容量（GB，仅 KV）"],
        ["l3_available_gb", "L3 可用容量（GB，仅 KV）"],
        ["runtime_reserve_bytes", "额外运行时预留（GB）"],
        ["l2_to_l1_gb_s", "L2 → L1 带宽（GB/s）"],
        ["l3_to_l2_gb_s", "L3 → L2 带宽（GB/s）"],
        ["l2_latency_ms", "L2 固定延迟（ms）"],
        ["l3_latency_ms", "L3 固定延迟（ms）"],
      ] as const).map(([field, label]) => <Input key={field} label={label}
        value={field === "runtime_reserve_bytes" ? storage[field] / 1e9 : storage[field]}
        optional={field === "l2_to_l1_gb_s" || field === "l3_to_l2_gb_s"}
        onChange={(value) => onStorageChange({ ...storage,
          [field]: field === "runtime_reserve_bytes" && value !== null ? value * 1e9 : value })} />)}
    </div>
    {selectedHardware && <p>单卡标称规格：{selectedHardware.memory_bytes_per_gpu / 1e9} GB 显存、
      {selectedHardware.peak_fp16_tflops.toLocaleString()} TFLOPS 稠密 BF16/FP16 算力、
      {selectedHardware.peak_hbm_gb_s.toLocaleString()} GB/s HBM 带宽。
      <a href="https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html" target="_blank" rel="noreferrer">显存来源</a>、
      <a href="https://inferencex.semianalysis.com/zh/chips/b300" target="_blank" rel="noreferrer">性能来源</a>；
      上述有效值为独立的仿真假设。</p>}
    <div className="form-actions">
      <button type="button" className="transfer-button" disabled={disabled} onClick={() => void inspect()}>预览解析容量</button>
      <button type="button" className="run-button" disabled={disabled} onClick={() => onRun(request)}>
        {disabled ? "正在运行自动解析…" : "运行自动解析仿真"}
      </button>
    </div>
    {error && <p role="alert" className="field-error">{error}</p>}
    {preview && previewKey === key && <div role="status" className="simulation-status">
      <strong>服务端解析：</strong>KV {preview.capacity.kv_bytes_per_token.toLocaleString()} bytes/token；
      权重 { (preview.capacity.weights_resident_bytes / 1e9).toFixed(2) } GB；
      L1 {preview.capacity.l1_capacity_pages.toLocaleString()} 页 / {preview.config.engine.max_total_tokens.toLocaleString()} token；
      L2 {preview.capacity.l2_capacity_pages.toLocaleString()} 页；
      L3 {preview.capacity.l3_capacity_pages.toLocaleString()} 页。
      {timing === "analytical" && " 时延为未经校准的 Roofline 估计。"}
    </div>}
  </section>
  </>;
}
