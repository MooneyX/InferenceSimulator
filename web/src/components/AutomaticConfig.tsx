import { useEffect, useState } from "react";

import { getPresets, resolveAutomatic } from "../api/client";
import type { PresetCatalog, ResolvedConfig, ResolvedRequest, SimConfig } from "../api/types";
import { parseSimConfigJson } from "../config/parameterTransfer";

interface Props {
  config: SimConfig;
  disabled: boolean;
  onRun: (request: ResolvedRequest) => void;
  onImportConfig?: (config: SimConfig) => void;
}

type Model = ResolvedRequest["model"];
type Hardware = ResolvedRequest["hardware"];
type Storage = ResolvedRequest["storage"];

const initialModel: Model = {
  name: "Qwen/Qwen3-8B", revision: null, num_layers: 36, hidden_size: 4096,
  num_attention_heads: 32, num_kv_heads: 8, head_dim: 128,
  ffn_intermediate_size: 12288, total_parameters: 8_200_000_000,
  weight_dtype: "bf16", kv_dtype: "bf16", resident_weights_bytes: null,
  moe_layers: 0, expert_parameters_total: null, active_parameters: null,
  num_experts: 0, experts_per_token: 0, expert_intermediate_size: null,
  attention_layout: "full_gqa",
};
const initialHardware: Hardware = {
  name: "NVIDIA B300（单卡）", gpu_count: 1, memory_bytes_per_gpu: 288_000_000_000,
  effective_fp16_tflops: 100, effective_fp8_tflops: null,
  effective_hbm_gb_s: 1000, effective_interconnect_gb_s: null,
  kernel_overhead_ms: 0,
};
const initialStorage: Storage = {
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

export function AutomaticConfig({ config, disabled, onRun, onImportConfig }: Props) {
  const [model, setModel] = useState<Model>(initialModel);
  const [hardware, setHardware] = useState<Hardware>(initialHardware);
  const [storage, setStorage] = useState<Storage>(initialStorage);
  const [timing, setTiming] = useState<ResolvedRequest["timing"]>("analytical");
  const [catalog, setCatalog] = useState<PresetCatalog | null>(null);
  const [preview, setPreview] = useState<ResolvedConfig | null>(null);
  const [previewKey, setPreviewKey] = useState("");
  const [error, setError] = useState("");
  const [json, setJson] = useState("");

  useEffect(() => {
    let active = true;
    void getPresets().then((data) => { if (active) setCatalog(data); }).catch(() => {
      if (active) setCatalog(null);
    });
    return () => { active = false; };
  }, []);

  const request: ResolvedRequest = { model, hardware, storage, timing, config };
  const key = JSON.stringify(request);

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

  function importJson() {
    try {
      const value: unknown = JSON.parse(json);
      if (typeof value !== "object" || value === null || !("model" in value) ||
          !("hardware" in value) || !("storage" in value) || !("config" in value)) {
        throw new Error("须包含 config、model、hardware 和 storage");
      }
      const imported = value as ResolvedRequest;
      for (const part of [imported.model, imported.hardware, imported.storage]) {
        if (!part || typeof part !== "object" || Array.isArray(part)) {
          throw new Error("模型、设备和存储规格须为 JSON 对象");
        }
      }
      const importedConfig = parseSimConfigJson(JSON.stringify(imported.config));
      setModel(imported.model);
      setHardware(imported.hardware);
      setStorage(imported.storage);
      setTiming(imported.timing ?? "analytical");
      onImportConfig?.(importedConfig);
      setPreview(null);
      setError("正在校验导入的物理规格…");
      void resolveAutomatic(imported).then((response) => {
        setPreview(response);
        setPreviewKey(JSON.stringify(imported));
        setError("");
      }).catch((cause: unknown) => {
        setError(cause instanceof Error ? cause.message : "导入的物理规格无法解析");
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "参数 JSON 不正确");
    }
  }

  return <section className="form-section" aria-labelledby="automatic-heading">
    <div className="section-heading"><div>
      <p className="section-kicker">Analytical · Stage 1</p>
      <h2 id="automatic-heading">模型与硬件自动解析（单 GPU）</h2>
    </div><p>现有手填模式不变；有效算力与带宽是用户假设，非厂商实测</p></div>
    <div className="form-grid">
      <div className="form-field"><label htmlFor="auto-scenario">参考组合</label>
        <select id="auto-scenario" defaultValue="custom" onChange={(event) => {
          const scenario = catalog?.scenarios.find((entry) => entry.id === event.target.value);
          const selected = catalog?.models.find((entry) => entry.name === scenario?.model_name);
          const device = catalog?.hardware.find((entry) => entry.name === scenario?.hardware_name);
          if (selected && scenario && device) {
            setModel(selected);
            setHardware({ ...hardware, name: device.name,
              gpu_count: scenario.gpu_count, memory_bytes_per_gpu: device.memory_bytes_per_gpu });
          }
        }}><option value="custom">自定义</option>{catalog?.scenarios.map((entry) =>
          <option key={entry.id} value={entry.id}>{entry.model_name} × {entry.hardware_name} × {entry.gpu_count}
            {entry.stage === "requires_tp_ep" ? "（待 TP/EP）" : "（单卡示例）"}</option>)}</select>
      </div>
      <div className="form-field"><label htmlFor="auto-model-preset">参考模型</label>
        <select id="auto-model-preset" value={catalog?.models.find((entry) => entry.name === model.name)?.name ?? "custom"}
          onChange={(event) => {
            const selected = catalog?.models.find((entry) => entry.name === event.target.value);
            if (selected) setModel(selected);
          }}><option value="custom">自定义</option>{catalog?.models.map((entry) =>
            <option key={entry.name} value={entry.name}>{entry.name}{entry.moe_layers ? "（TP/EP 阶段）" : ""}</option>)}</select>
      </div>
      <div className="form-field"><label>模型名称<input value={model.name}
        onChange={(event) => setModel({ ...model, name: event.target.value })} /></label></div>
      <div className="form-field"><label>GPU 名称<input value={hardware.name}
        onChange={(event) => setHardware({ ...hardware, name: event.target.value })} /></label></div>
      <div className="form-field"><label htmlFor="auto-timing">计时模式</label>
        <select id="auto-timing" value={timing} onChange={(event) =>
          setTiming(event.target.value as ResolvedRequest["timing"])}>
          <option value="analytical">逐步 Roofline 估计</option>
          <option value="empirical">原四系数经验模型</option>
        </select></div>
    </div>
    <h3>模型结构与精度</h3>
    <div className="form-grid">{([
      ["num_layers", "模型层数"], ["hidden_size", "隐藏维度"],
      ["num_attention_heads", "Query 头数"], ["num_kv_heads", "KV 头数"],
      ["head_dim", "每头维度"], ["ffn_intermediate_size", "FFN 中间维度"],
      ["total_parameters", "总参数量（含全部专家）"], ["moe_layers", "MoE 层数"],
      ["expert_parameters_total", "全部专家参数量"], ["active_parameters", "激活参数量（可选）"],
      ["num_experts", "专家数"], ["experts_per_token", "每 token 激活专家数"],
      ["expert_intermediate_size", "专家中间维度（可选）"],
      ["resident_weights_bytes", "实际驻留权重（bytes，可选覆盖）"],
    ] as const).map(([field, label]) => <Input key={field} label={label} value={model[field]}
      optional={field === "expert_parameters_total" || field === "active_parameters" ||
        field === "expert_intermediate_size" || field === "resident_weights_bytes"}
      onChange={(value) => setModel({ ...model, [field]: value })} />)}
      <div className="form-field"><label>权重精度<select value={model.weight_dtype} onChange={(event) =>
        setModel({ ...model, weight_dtype: event.target.value as Model["weight_dtype"] })}>
        <option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp8">FP8（待支持）</option>
      </select></label></div>
      <div className="form-field"><label>KV 精度<select value={model.kv_dtype} onChange={(event) =>
        setModel({ ...model, kv_dtype: event.target.value as Model["kv_dtype"] })}>
        <option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp8">FP8（待支持）</option>
      </select></label></div>
    </div>
    <h3>设备性能与三级容量</h3>
    <div className="form-grid">{([
      ["gpu_count", "GPU 卡数（首阶段须为 1）"],
      ["memory_bytes_per_gpu", "单卡标称显存（bytes）"],
      ["effective_fp16_tflops", "有效 FP16/BF16 算力（TFLOPS）"],
      ["effective_fp8_tflops", "有效 FP8 算力（TFLOPS，可选）"],
      ["effective_hbm_gb_s", "有效 HBM 带宽（GB/s）"],
      ["effective_interconnect_gb_s", "有效互联带宽（GB/s，可选）"],
      ["kernel_overhead_ms", "单算子固定开销（ms）"],
    ] as const).map(([field, label]) => <Input key={field} label={label} value={hardware[field]}
      optional={field === "effective_fp8_tflops" || field === "effective_interconnect_gb_s"}
      onChange={(value) => setHardware({ ...hardware, [field]: value })} />)}
      {([
        ["l1_available_gb", "L1 可用容量（GB，含权重）"],
        ["l2_available_gb", "L2 可用容量（GB，仅 KV）"],
        ["l3_available_gb", "L3 可用容量（GB，仅 KV）"],
        ["runtime_reserve_bytes", "额外运行时预留（bytes）"],
        ["l2_to_l1_gb_s", "L2 → L1 带宽（GB/s）"],
        ["l3_to_l2_gb_s", "L3 → L2 带宽（GB/s）"],
        ["l2_latency_ms", "L2 固定延迟（ms）"],
        ["l3_latency_ms", "L3 固定延迟（ms）"],
      ] as const).map(([field, label]) => <Input key={field} label={label} value={storage[field]}
        optional={field === "l2_to_l1_gb_s" || field === "l3_to_l2_gb_s"}
        onChange={(value) => setStorage({ ...storage, [field]: value })} />)}
    </div>
    <p>参考硬件：{catalog?.hardware.map((entry) =>
      `${entry.name} × ${entry.gpu_count}（单卡 ${entry.memory_bytes_per_gpu / 1e9} GB）`).join("；") ?? "加载中"}。
      参考规格未提供有效算力和带宽；表单初始值仅用于情景估算。</p>
    <div className="parameter-transfer">
      <button type="button" className="transfer-button" onClick={() => setJson(JSON.stringify(request, null, 2))}>导出自动参数</button>
      <button type="button" className="transfer-button" onClick={importJson}>导入自动参数</button>
      <label className="transfer-label" htmlFor="auto-json">自动参数 JSON</label>
      <textarea id="auto-json" value={json} onChange={(event) => setJson(event.target.value)} rows={3} />
    </div>
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
  </section>;
}
