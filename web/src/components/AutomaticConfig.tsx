import { useEffect, useState } from "react";

import { getPresets, resolveAutomatic, SimulationApiError } from "../api/client";
import type { PresetCatalog, ResolvedConfig, ResolvedRequest, SimConfig } from "../api/types";

export type Model = ResolvedRequest["model"];
type AttentionGroup = NonNullable<Model["attention_groups"]>[number];
export type Hardware = ResolvedRequest["hardware"];
export type Storage = ResolvedRequest["storage"];

interface Props {
  config: SimConfig;
  model: Model;
  hardware: Hardware;
  storage: Storage;
  timing: ResolvedRequest["timing"];
  tpSize: number;
  epSize: number;
  onTpSizeChange: (value: number) => void;
  onEpSizeChange: (value: number) => void;
  onModelChange: (value: Model) => void;
  onHardwareChange: (value: Hardware) => void;
  onStorageChange: (value: Storage) => void;
  disabled: boolean;
}

export const initialModel: Model = {
  name: "Qwen/Qwen3-8B", revision: null, num_layers: 36, hidden_size: 4096,
  num_attention_heads: 32, num_kv_heads: 8, head_dim: 128,
  ffn_intermediate_size: 12288, vocab_size: 151_936, tie_word_embeddings: false,
  weight_dtype: "bf16", kv_dtype: "bf16", moe_layers: 0,
  num_experts: 0, experts_per_token: 0, expert_intermediate_size: null,
  num_shared_experts: 0, shared_expert_intermediate_size: null,
  attention_layout: "full_gqa", attention_groups: null,
};
export const initialHardware: Hardware = {
  name: "NVIDIA B300", gpu_count: 1, memory_bytes_per_gpu: 288_000_000_000,
  effective_fp16_tflops: 100, effective_fp8_tflops: null,
  effective_hbm_gb_s: 1000, effective_interconnect_gb_s: 600,
  collective_latency_ms: 0.01, ep_communication_latency_ms: 0.02,
  kernel_overhead_ms: 0,
};
export const initialStorage: Storage = {
  l1_available_gb: 200, l2_available_gb: 2048, l3_available_gb: 40_000,
  runtime_reserve_bytes: 0, l2_to_l1_gb_s: 256, l3_to_l2_gb_s: 50,
  l2_latency_ms: 0, l3_latency_ms: 0, l3_lookup_latency_ms: 0,
};

function Input({ label, value, onChange, optional = false, disabled = false }: {
  label: string; value: number | null | undefined;
  onChange: (value: number | null) => void; optional?: boolean; disabled?: boolean;
}) {
  return <div className="form-field"><label>{label}
    <input type="number" step="any" disabled={disabled} value={value ?? ""}
      onChange={(event) => onChange(event.currentTarget.value === ""
        ? (optional ? null : NaN) : event.currentTarget.valueAsNumber)} />
  </label></div>;
}

export function AutomaticConfig({ config, model, hardware, storage, timing, tpSize, epSize,
  onTpSizeChange, onEpSizeChange, onModelChange, onHardwareChange, onStorageChange, disabled }: Props) {
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

  const request: ResolvedRequest = { model, hardware, storage, timing, tp_size: tpSize, ep_size: epSize, config };
  const key = JSON.stringify(request);
  const selectedModel = catalog?.models.find((entry) => entry.name === model.name &&
    Object.entries(entry).every(([field, value]) =>
      JSON.stringify(model[field as keyof Model]) === JSON.stringify(value)));
  const selectedHardware = catalog?.hardware.find((entry) => entry.name === hardware.name &&
    entry.memory_bytes_per_gpu === hardware.memory_bytes_per_gpu);

  function updateGroups(groups: AttentionGroup[] | null) {
    onModelChange({ ...model,
      name: selectedModel || model.attention_layout === "hy4_preview" ? "自定义模型" : model.name,
      revision: selectedModel || model.attention_layout === "hy4_preview" ? null : model.revision,
      attention_layout: groups ? "grouped" : "full_gqa", attention_groups: groups });
  }

  function newGroup(kind: AttentionGroup["kind"], layers: number, outputGate = false): AttentionGroup {
    const common = { kind, layers, num_attention_heads: model.num_attention_heads,
      output_gate: kind === "gqa" ? false : outputGate };
    if (kind === "gqa") return { ...common, num_kv_heads: model.num_kv_heads, head_dim: model.head_dim };
    const mla = { ...common, q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256 };
    return kind === "mla" ? mla : { ...mla, index_heads: 32, index_dim: 128,
      topk: 2048, full_indexer_layers: 1 };
  }

  async function inspect() {
    setError("");
    try {
      const response = await resolveAutomatic(request);
      setPreview(response);
      setPreviewKey(key);
    } catch (cause) {
      setPreview(null);
      const fields = cause instanceof SimulationApiError
        ? cause.validationErrors.map(({ loc, msg }) => `${loc.join(" → ")}：${msg}`)
        : [];
      setError(fields.length ? fields.join("；") : cause instanceof Error ? cause.message : "无法解析配置");
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
            onModelChange(selected ?? { ...model, name: "自定义模型", revision: null,
              attention_layout: model.attention_groups ? "grouped" : "full_gqa" });
            if (selected?.attention_layout === "hy4_preview") {
              onTpSizeChange(8);
              onEpSizeChange(8);
              onHardwareChange({ ...hardware, gpu_count: 8 });
              if (storage.l1_available_gb === 200 && hardware.memory_bytes_per_gpu >= 260e9) {
                onStorageChange({ ...storage, l1_available_gb: 260 });
              }
            } else if (selected) onEpSizeChange(selected.moe_layers ? tpSize : 1);
          }}>{catalog?.models.map((entry) =>
            <option key={entry.name} value={entry.name}>{entry.name}</option>)}
          <option value="custom">自定义</option></select>
      </div></div>
    {model.attention_layout === "hy4_preview" && <p>HY4-Preview 仅支持 BF16 主干：含输出门控与共享专家的 MLA + DSA 容量和时延粗估，不含 MTP；保留包含 iHC、attention sink 等 FP32 辅助参数的专用容量账本。复制为自定义模型后，输出门控、共享专家与性能组件保留，但不包含 iHC、attention sink 等辅助容量。结构字段固定，调整 TP/EP、硬件和负载可进行情景试算。逐卡权重按 SGLang 模块放置估算，不等于实测显存。</p>}
    <div className="form-grid">{([
      ["num_layers", "模型层数"], ["hidden_size", "隐藏维度"],
      ["num_attention_heads", "Query 头数"], ["num_kv_heads", "KV 头数"],
      ["head_dim", "每头维度"], ["ffn_intermediate_size", "FFN 中间维度"],
      ["vocab_size", "词表大小（结构估算必填）"], ["moe_layers", "MoE 层数"],
      ["num_experts", "专家数"], ["experts_per_token", "每 token 激活专家数"],
      ["expert_intermediate_size", "专家中间维度（MoE 必填）"],
    ] as const).map(([field, label]) => <Input key={field} label={label}
      value={model[field]} optional={field === "expert_intermediate_size"}
      disabled={model.attention_layout === "hy4_preview" ||
        (model.attention_groups != null && (["num_attention_heads", "num_kv_heads", "head_dim"] as string[]).includes(field))}
      onChange={(value) => onModelChange({ ...model, name: "自定义模型", [field]: value,
        ...(field === "moe_layers" && value === 0
          ? { num_shared_experts: 0, shared_expert_intermediate_size: null } : {}) })} />)}
      <Input label="共享专家数（0 为关闭）" value={model.num_shared_experts ?? 0}
        disabled={model.attention_layout === "hy4_preview" || !model.moe_layers}
        onChange={(value) => onModelChange({ ...model, name: "自定义模型",
          num_shared_experts: value ?? NaN,
          ...(value === 0 ? { shared_expert_intermediate_size: null } : {}) })} />
      <Input label="共享专家中间维度（启用时必填）" value={model.shared_expert_intermediate_size}
        optional disabled={model.attention_layout === "hy4_preview" || !model.moe_layers || !model.num_shared_experts}
        onChange={(value) => onModelChange({ ...model, name: "自定义模型",
          shared_expert_intermediate_size: value })} />
      <div className="form-field"><label>权重精度<select disabled={model.attention_layout === "hy4_preview"} value={model.weight_dtype} onChange={(event) =>
        onModelChange({ ...model, name: "自定义模型", weight_dtype: event.target.value as Model["weight_dtype"] })}>
        <option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp8">FP8（待支持）</option>
      </select></label></div>
      <div className="form-field"><label>KV 精度<select disabled={model.attention_layout === "hy4_preview"} value={model.kv_dtype} onChange={(event) =>
        onModelChange({ ...model, name: "自定义模型", kv_dtype: event.target.value as Model["kv_dtype"] })}>
        <option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp8">FP8（待支持）</option>
      </select></label></div>
      <div className="form-field"><label>输入输出词嵌入共享
        <input type="checkbox" disabled={model.attention_layout === "hy4_preview"} checked={model.tie_word_embeddings} onChange={(event) =>
          onModelChange({ ...model, name: "自定义模型", tie_word_embeddings: event.target.checked })} />
      </label></div>
    </div>
    <div className="section-heading"><div><h3>注意力组合</h3></div>
      <p>每组填写层数；同一组的 MLA 与 DSA 作用在同一批层。FFN/MoE 层数独立计算；启用组合后上方头数仅作为旧模型基线。</p></div>
    {model.attention_layout === "hy4_preview" && <>
      <div className="form-grid">{model.attention_groups?.map((group, index) =>
        <div key={index} className="form-field"><label>第 {index + 1} 组注意力输出门控
          <input type="checkbox" checked={group.output_gate ?? false} disabled />
        </label></div>)}</div>
      <button type="button" className="transfer-button"
        onClick={() => updateGroups(model.attention_groups ?? null)}>
        复制为自定义模型
      </button>
    </>}
    {model.attention_layout !== "hy4_preview" && <>
      {model.attention_groups != null && <div className="form-grid"><div className="form-field">
        <label>模型名称<input type="text" value={model.name} onChange={(event) =>
          onModelChange({ ...model, name: event.target.value })} /></label>
      </div><div className="form-field"><label>修订号（可选）<input type="text"
        value={model.revision ?? ""} onChange={(event) =>
          onModelChange({ ...model, revision: event.target.value || null })} /></label></div></div>}
      {model.attention_groups == null ? <button type="button" className="transfer-button"
        onClick={() => updateGroups([newGroup("gqa", model.num_layers)])}>启用注意力组合配置</button> : <>
        <p>已配置 {model.attention_groups.reduce((total, group) => total + group.layers, 0)} / {model.num_layers} 层。
          不按具体层号排列；不支持顺序敏感的跨层依赖。</p>
        {model.attention_groups.map((group, index) => <div key={index} className="form-section">
          <div className="form-grid">
            <div className="form-field"><label>第 {index + 1} 组注意力方式
              <select value={group.kind} onChange={(event) => updateGroups(model.attention_groups!.map((item, i) =>
                i === index ? newGroup(event.target.value as AttentionGroup["kind"], item.layers,
                  item.output_gate ?? false) : item))}>
                <option value="gqa">全注意力 GQA/MQA</option>
                <option value="mla">MLA</option>
                <option value="mla_dsa">MLA + DSA</option>
              </select></label></div>
            <div className="form-field"><label>第 {index + 1} 组注意力输出门控
              <input type="checkbox" checked={group.output_gate ?? false} disabled={group.kind === "gqa"}
                onChange={(event) => updateGroups(model.attention_groups!.map((item, i) =>
                  i === index ? { ...item, output_gate: event.target.checked } : item))} />
            </label></div>
            <Input label={`第 ${index + 1} 组层数`} value={group.layers} onChange={(value) =>
              updateGroups(model.attention_groups!.map((item, i) => i === index ? { ...item, layers: value ?? NaN } : item))} />
            <Input label={`第 ${index + 1} 组 Query 头数`} value={group.num_attention_heads} onChange={(value) =>
              updateGroups(model.attention_groups!.map((item, i) => i === index ? { ...item, num_attention_heads: value ?? NaN } : item))} />
            {(group.kind === "gqa" ? ([
              ["num_kv_heads", "KV 头数"], ["head_dim", "每头维度"],
            ] as const) : ([
              ["q_rank", "Q 压缩维度"], ["kv_rank", "KV 压缩维度"],
              ["rope_dim", "位置 Key 维度"], ["nope_dim", "非位置 Q/K 维度"],
              ["value_dim", "V 维度"],
              ...(group.kind === "mla_dsa" ? ([
                ["index_heads", "索引头数"], ["index_dim", "索引 Key 维度（128 倍数）"],
                ["topk", "选取 token 数"], ["full_indexer_layers", "重算索引的层数"],
              ] as const) : []),
            ] as const)).map(([field, label]) => <Input key={field} label={`第 ${index + 1} 组${label}`}
              value={group[field]} onChange={(value) => updateGroups(model.attention_groups!.map((item, i) =>
                i === index ? { ...item, [field]: value ?? NaN } : item))} />)}
          </div>
          <button type="button" className="transfer-button" onClick={() => updateGroups(
            model.attention_groups!.length === 1 ? null : model.attention_groups!.filter((_, i) => i !== index)
          )}>删除第 {index + 1} 组</button>
        </div>)}
        <button type="button" className="transfer-button" onClick={() => {
          const groups = model.attention_groups!;
          const last = groups[groups.length - 1];
          updateGroups([...groups.slice(0, -1), { ...last, layers: Math.max(1, last.layers - 1) }, newGroup("gqa", 1)]);
        }}>添加注意力组</button>
      </>}
      <p>自定义 MLA/DSA 按 BF16/FP16 压缩 KV、FP8 索引 Key（每 128 元素 4 字节 scale）和同构 TP 复制口径粗估；可选输出门控及共享专家按填写参数计入容量与时延，iHC 与 attention sink 参数尚未计入容量，其他运行时空间不自动推断。</p>
    </>}
    </section>
    <section className="form-section" aria-labelledby="auto-hardware-heading">
      <div className="section-heading"><div>
        <p className="section-kicker">Hardware · Stage 1</p>
        <h2 id="auto-hardware-heading">硬件配置</h2>
      </div><p>以下硬件与存储均为单实例规格；总 GPU 数 = 实例数量 × 每实例 GPU 数</p></div>
      <div className="form-grid" style={{ marginBottom: 14 }}><div className="form-field"><label htmlFor="auto-gpu-preset">GPU 选择</label>
        <select id="auto-gpu-preset" value={selectedHardware?.name ?? "custom"}
          onChange={(event) => {
            const selected = catalog?.hardware.find((entry) => entry.name === event.target.value);
            onHardwareChange(selected ? { ...hardware, name: selected.name,
              memory_bytes_per_gpu: selected.memory_bytes_per_gpu } :
              { ...hardware, name: "自定义 GPU" });
          }}>{catalog?.hardware.map((entry) =>
            <option key={entry.name} value={entry.name}>{entry.name}</option>)}
          <option value="custom">自定义</option></select>
      </div></div>
    <div className="form-grid">
      <Input label="TP 数（必须等于 GPU 卡数）" value={tpSize} onChange={(value) => {
        onTpSizeChange(value ?? NaN);
        if (model.moe_layers) onEpSizeChange(value ?? NaN);
      }} />
      <Input label="EP 数（MoE 须等于 TP）" value={epSize} onChange={(value) =>
        onEpSizeChange(value ?? NaN)} />
      {([
      ["gpu_count", "本实例 GPU 卡数"],
      ["memory_bytes_per_gpu", "单卡标称显存（GB）"],
      ["effective_fp16_tflops", "有效 FP16/BF16 算力（TFLOPS）"],
      ["effective_fp8_tflops", "有效 FP8 算力（TFLOPS，可选）"],
      ["effective_hbm_gb_s", "有效 HBM 带宽（GB/s）"],
      ["effective_interconnect_gb_s", "单 rank 有效互联带宽（GB/s，TP>1 必填）"],
      ["collective_latency_ms", "TP 每次集合通信固定延迟（ms，TP>1 运行必填）"],
      ["ep_communication_latency_ms", "EP 每次 dispatch/combine 固定开销（ms，EP>1 运行必填）"],
      ["kernel_overhead_ms", "单算子固定开销（ms）"],
    ] as const).map(([field, label]) => <Input key={field} label={label}
      value={field === "memory_bytes_per_gpu" ? hardware[field] / 1e9 : hardware[field]}
      optional={field === "effective_fp8_tflops" || field === "effective_interconnect_gb_s" ||
        field === "collective_latency_ms" || field === "ep_communication_latency_ms"}
      onChange={(value) => onHardwareChange({ ...hardware,
        ...(field === "memory_bytes_per_gpu" ? { name: "自定义 GPU" } : {}),
        [field]: field === "memory_bytes_per_gpu" && value !== null ? value * 1e9 : value })} />)}
      {([
        ["l1_available_gb", "L1 可用容量（GB，含权重）"],
        ["l2_available_gb", "L2 可用容量（GB，仅 KV）"],
        ["l3_available_gb", "L3 可用容量（TB，仅 KV）"],
        ["runtime_reserve_bytes", "额外运行时预留（GB）"],
        ["l2_to_l1_gb_s", "L2 → L1 带宽（GB/s）"],
        ["l3_to_l2_gb_s", "L3 → L2 带宽（GB/s）"],
        ["l2_latency_ms", "L2 固定延迟（ms）"],
        ["l3_latency_ms", "L3 固定延迟（ms）"],
        ["l3_lookup_latency_ms", "L3 查找延迟（ms）"],
      ] as const).map(([field, label]) => <Input key={field} label={label}
        value={field === "runtime_reserve_bytes" ? storage[field] / 1e9 :
          field === "l3_available_gb" ? storage[field] / 1000 : storage[field]}
        optional={field === "l2_to_l1_gb_s" || field === "l3_to_l2_gb_s"}
        onChange={(value) => onStorageChange({ ...storage,
          [field]: field === "runtime_reserve_bytes" && value !== null ? value * 1e9 :
            field === "l3_available_gb" && value !== null ? value * 1000 : value })} />)}
    </div>
    {selectedHardware && <p>单卡标称规格：{selectedHardware.memory_bytes_per_gpu / 1e9} GB 显存、
      {selectedHardware.peak_fp16_tflops.toLocaleString()} TFLOPS 稠密 BF16/FP16 算力、
      {selectedHardware.peak_hbm_gb_s.toLocaleString()} GB/s HBM 带宽、
      {selectedHardware.peak_interconnect_gb_s.toLocaleString()} GB/s 单卡 NVLink 双向标称带宽。
      <a href="https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html" target="_blank" rel="noreferrer">显存来源</a>、
      <a href="https://inferencex.semianalysis.com/zh/chips/b300" target="_blank" rel="noreferrer">性能来源</a>、
      <a href="https://www.nvidia.com/en-us/data-center/hgx/" target="_blank" rel="noreferrer">互联来源</a>；
      有效互联 600 GB/s、TP 固定延迟 0.01 ms/次、EP 固定开销 0.02 ms/次均为未经校准的试算假设，可在上方修改；1800 GB/s 是双向标称值，不等于单 rank 有效带宽。</p>}
    <div className="form-actions">
      <button type="button" className="transfer-button" disabled={disabled} onClick={() => void inspect()}>预览解析容量</button>
    </div>
    {model.moe_layers ? <p>{timing === "analytical"
      ? "MoE/EP 采用未经校准的平均负载解析计时：含 Prefill 用 normal，纯 Decode 用 low_latency；可先预览容量。"
      : "MoE/EP 不支持四系数计时，请切换到解析计时。"}</p> :
      tpSize > 1 && <p>{timing === "analytical"
        ? "TP 解析计时已接入：每层两次等效同步，未按实测校准。"
        : "TP>1 不支持四系数计时，请切换到解析计时。"}</p>}
    {error && <p role="alert" className="field-error">{error}</p>}
    {preview && previewKey === key && <div role="status" className="simulation-status">
      <strong>服务端单实例解析：</strong>共 {config.num_instances} 个实例，总 GPU {config.num_instances * hardware.gpu_count} 张；KV {preview.capacity.kv_bytes_per_token.toLocaleString()} bytes/token；
      {preview.tp_size > 1 ? "整组结构估算驻留权重" : "结构估算驻留权重"} { (preview.capacity.weights_resident_bytes / 1e9).toFixed(2) } GB；
      结构估算总参数 {(preview.capacity.estimated_total_parameters / 1e9).toFixed(3)} B；
      结构估算每 token 激活参数 {(preview.capacity.estimated_active_parameters / 1e9).toFixed(3)} B；
      全部专家参数 {(preview.capacity.estimated_expert_parameters / 1e9).toFixed(3)} B；
      L1 {preview.capacity.l1_capacity_pages.toLocaleString()} 页 / {preview.config.engine.max_total_tokens.toLocaleString()} token；
      L2 {preview.capacity.l2_capacity_pages.toLocaleString()} 页；
      L3 {preview.capacity.l3_capacity_pages.toLocaleString()} 页。
      {(preview.capacity.rank_reports ?? []).length > 0 && <>
        <p>TP{preview.tp_size}{preview.model.moe_layers ? ` + EP${preview.ep_size}` : ""} 逐卡账单：每个逻辑 token 的整组 KV 物理成本为
          {preview.capacity.kv_bytes_per_token.toLocaleString()} bytes；权重按分片与逐卡复制的结构分别计入。</p>
        {(preview.capacity.rank_reports ?? []).map((rank) => <p key={rank.rank}>
          GPU {rank.rank}：分片权重 {(rank.sharded_weights_bytes / 1e9).toFixed(3)} GB；
          逐卡复制权重 {(rank.replicated_weights_bytes / 1e9).toFixed(3)} GB；
          {preview.model.moe_layers ? <>全部驻留专家权重 {((rank.expert_weights_bytes ?? 0) / 1e9).toFixed(3)} GB；</> : null}
          KV {rank.kv_bytes_per_token.toLocaleString()} bytes/token；
          L1 {rank.l1_capacity_pages.toLocaleString()} 页。
        </p>)}
      </>}
      {!preview.runnable && timing === "analytical" && preview.tp_size > 1 &&
        (preview.model.moe_layers
          ? " 若要运行 TP/EP>1，请填写互联带宽、TP 集合通信延迟与 EP 单次通信开销。"
          : " 多卡容量可预览；填写互联带宽和 TP 集合通信延迟后才能运行。")}
      {preview.runnable && timing === "analytical" && " 时延为未经校准的平均负载/Roofline 估计。"}
    </div>}
  </section>
  </>;
}
