import { useEffect, useState } from "react";

import { getPresets, resolveConfig, SimulationApiError } from "../api/client";
import type { InferenceConfig, PresetCatalog, ResolutionReport } from "../api/types";
import type { FormSimConfig } from "../config/simulation";
export { initialModel, initialHardware, initialStorage } from "../config/simulation";

export type Model = InferenceConfig["model"];
type AttentionGroup = NonNullable<Model["attention_groups"]>[number];
export type Hardware = InferenceConfig["hardware"];
export type Storage = InferenceConfig["storage"];
type AuxiliaryDtype = "bf16" | "fp16" | "fp32";
const defaultNorms = { kind: "rms" as const, block_per_layer: 2, final_norm: true, dtype: null };
const defaultRouter = { dtype: null, correction_bias: false, correction_dtype: "fp32" as const };

interface Props {
  config: FormSimConfig;
  onInferenceChange: (inference: FormSimConfig["inference"]) => void;
  disabled: boolean;
}

function sameConfig(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((value, index) => sameConfig(value, right[index]));
  }
  if (left !== null && right !== null && typeof left === "object" && typeof right === "object") {
    const a = left as Record<string, unknown>;
    const b = right as Record<string, unknown>;
    return Object.keys(a).length === Object.keys(b).length &&
      Object.keys(a).every((key) => Object.hasOwn(b, key) && sameConfig(a[key], b[key]));
  }
  return false;
}

function DtypeInput({ label, value, onChange, disabled = false, nullable = true }: {
  label: string; value: AuxiliaryDtype | null | undefined;
  onChange: (value: AuxiliaryDtype | null) => void; disabled?: boolean; nullable?: boolean;
}) {
  return <div className="form-field"><label>{label}
    <select value={value ?? ""} disabled={disabled}
      onChange={(event) => onChange((event.currentTarget.value || null) as AuxiliaryDtype | null)}>
      {nullable && <option value="">跟随主权重</option>}
      <option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp32">FP32</option>
    </select>
  </label></div>;
}

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

export function ModelHardwareConfig({ config, onInferenceChange, disabled }: Props) {
  const { model, hardware, storage, tp_size: tpSize, ep_size: epSize } = config.inference;
  const timing = config.inference.timing.mode;
  function updateInference(patch: Partial<FormSimConfig["inference"]>) {
    onInferenceChange({ ...config.inference, ...patch });
  }
  const onModelChange = (model: Model) => updateInference({ model });
  const onHardwareChange = (hardware: Hardware) => updateInference({ hardware });
  const onStorageChange = (storage: Storage) => updateInference({ storage });
  const onEpSizeChange = (ep_size: number) => updateInference({ ep_size });
  const [catalog, setCatalog] = useState<PresetCatalog | null>(null);
  const [preview, setPreview] = useState<ResolutionReport | null>(null);
  const [previewKey, setPreviewKey] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    void getPresets().then((data) => { if (active) setCatalog(data); }).catch(() => {
      if (active) setCatalog(null);
    });
    return () => { active = false; };
  }, []);

  const key = JSON.stringify(config);
  const selectedModel = catalog?.models.find((entry) => entry.name === model.name &&
    Object.entries(entry).every(([field, value]) => sameConfig(model[field as keyof Model], value)));
  const selectedHardware = catalog?.hardware.find((entry) => entry.name === hardware.name &&
    entry.memory_bytes_per_gpu === hardware.memory_bytes_per_gpu);

  function updateGroups(groups: AttentionGroup[] | null) {
    updateModel({ attention_layout: groups ? "grouped" : "full_gqa", attention_groups: groups });
  }

  const norms = { ...defaultNorms, ...model.norms };
  const router = { ...defaultRouter, ...model.router };

  function updateModel(patch: Partial<Model>) {
    const custom = selectedModel || model.attention_layout === "hy4_preview";
    onModelChange({ ...model, name: custom ? "自定义模型" : model.name,
      revision: custom ? null : model.revision,
      attention_layout: model.attention_layout === "hy4_preview" ? "grouped" : model.attention_layout,
      ...patch });
  }

  function updateGroup(index: number, patch: Partial<AttentionGroup>) {
    updateGroups(model.attention_groups!.map((group, i) => i === index ? { ...group, ...patch } : group));
  }

  function newGroup(kind: AttentionGroup["kind"], layers: number, previous?: AttentionGroup): AttentionGroup {
    const common = { kind, layers, num_attention_heads: model.num_attention_heads,
      output_gate: kind === "gqa" ? false : previous?.output_gate ?? false,
      qk_norm: previous?.qk_norm ?? true, norm_dtype: previous?.norm_dtype ?? null,
      index_norm: kind === "mla_dsa" ? previous?.index_norm ?? "layer" : "layer",
      index_norm_dtype: kind === "mla_dsa" ? previous?.index_norm_dtype ?? null : null,
      sink_dtype: previous?.sink_dtype ?? null } as AttentionGroup;
    if (kind === "gqa") return { ...common, num_kv_heads: model.num_kv_heads, head_dim: model.head_dim };
    const mla = { ...common, q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256 };
    return kind === "mla" ? mla : { ...mla, index_heads: 32, index_dim: 128,
      topk: 2048, full_indexer_layers: 1 };
  }

  async function inspect() {
    setError("");
    try {
      const response = await resolveConfig(config);
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
    <section className="form-section" aria-labelledby="model-heading">
      <div className="section-heading"><div>
        <p className="section-kicker">Analytical · Stage 1</p>
        <h2 id="model-heading">模型参数</h2>
      </div><p>选择预置模型，或切换到自定义并调整结构与精度</p></div>
      <div className="form-grid" style={{ marginBottom: 14 }}><div className="form-field"><label htmlFor="model-preset">模型选择</label>
        <select id="model-preset" value={selectedModel?.name ?? "custom"}
          onChange={(event) => {
            const selected = catalog?.models.find((entry) => entry.name === event.target.value);
            updateInference({ model: selected ?? { ...model, name: "自定义模型", revision: null,
              attention_layout: model.attention_groups ? "grouped" : "full_gqa" },
              ...(selected?.name === "tencent/Hy4-preview" ? {
                tp_size: 8, ep_size: 8, hardware: { ...hardware, gpu_count: 8 },
              } : selected ? { ep_size: selected.moe_layers ? tpSize : 1 } : {}),
            });
          }}>{catalog?.models.map((entry) =>
            <option key={entry.name} value={entry.name}>{entry.name}</option>)}
          <option value="custom">自定义</option></select>
      </div></div>
    {(selectedModel?.name === "tencent/Hy4-preview" || model.attention_layout === "hy4_preview") && <p>HY4-Preview 为可编辑参考预设：含输出门控与共享专家的 MLA + DSA 容量和时延粗估，不含 MTP。可直接调整模型结构、精度、辅助权重与注意力组合；修改后转为自定义模型并保留其他参数。归一化、路由、iHC 与 attention sink 使用统一通用权重账本。逐卡权重按 SGLang 模块放置估算，不等于实测显存。</p>}
    <div className="form-grid">{([
      ["num_layers", "模型层数"], ["hidden_size", "隐藏维度"],
      ["num_attention_heads", "Query 头数"], ["num_kv_heads", "KV 头数"],
      ["head_dim", "每头维度"], ["ffn_intermediate_size", "FFN 中间维度"],
      ["vocab_size", "词表大小（结构估算必填）"], ["moe_layers", "MoE 层数"],
      ["num_experts", "专家数"], ["experts_per_token", "每 token 激活专家数"],
      ["expert_intermediate_size", "专家中间维度（MoE 必填）"],
    ] as const).map(([field, label]) => <Input key={field} label={label}
      value={model[field]} optional={field === "expert_intermediate_size"}
      disabled={model.attention_groups != null && (["num_attention_heads", "num_kv_heads", "head_dim"] as string[]).includes(field)}
      onChange={(value) => updateModel({ [field]: value,
        ...(field === "moe_layers" && value === 0
          ? { num_shared_experts: 0, shared_expert_intermediate_size: null, router: defaultRouter } : {}) })} />)}
      <Input label="共享专家数（0 为关闭）" value={model.num_shared_experts ?? 0}
        disabled={!model.moe_layers}
        onChange={(value) => updateModel({ num_shared_experts: value ?? NaN,
          ...(value === 0 ? { shared_expert_intermediate_size: null } : {}) })} />
      <Input label="共享专家中间维度（启用时必填）" value={model.shared_expert_intermediate_size}
        optional disabled={!model.moe_layers || !model.num_shared_experts}
        onChange={(value) => updateModel({ shared_expert_intermediate_size: value })} />
      <div className="form-field"><label>权重精度<select value={model.weight_dtype} onChange={(event) =>
        updateModel({ weight_dtype: event.target.value as Model["weight_dtype"] })}>
        <option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp8">FP8（待支持）</option>
      </select></label></div>
      <div className="form-field"><label>KV 精度<select value={model.kv_dtype} onChange={(event) =>
        updateModel({ kv_dtype: event.target.value as Model["kv_dtype"] })}>
        <option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp8">FP8（待支持）</option>
      </select></label></div>
      <div className="form-field"><label>输入输出词嵌入共享
        <input type="checkbox" checked={model.tie_word_embeddings} onChange={(event) =>
          updateModel({ tie_word_embeddings: event.target.checked })} />
      </label></div>
    </div>
    <details className="model-subsection auxiliary-config">
      <summary>辅助权重容量配置</summary>
      <div className="auxiliary-config-body">
      <p>辅助参数只计入容量，不增加性能耗时。精度留空跟随主权重；none 归一化不含参数。路由选位修正向量不是 Linear bias；iHC 固定为每层两处和最终 head。</p>
      <fieldset className="auxiliary-config-fields">
        <div className="form-grid">
          <div className="form-field"><label>主干归一化类型
            <select value={norms.kind} onChange={(event) => updateModel({
              norms: { ...norms, kind: event.currentTarget.value as typeof norms.kind },
            })}><option value="none">关闭（无参数）</option><option value="rms">RMSNorm</option><option value="layer">LayerNorm</option></select>
          </label></div>
          <Input label="每层主干归一化数量" value={norms.block_per_layer} disabled={norms.kind === "none"}
            onChange={(value) => updateModel({ norms: { ...norms, block_per_layer: value ?? NaN } })} />
          <div className="form-field"><label>最终主干归一化
            <input type="checkbox" checked={norms.final_norm} disabled={norms.kind === "none"}
              onChange={(event) => updateModel({ norms: { ...norms, final_norm: event.currentTarget.checked } })} />
          </label></div>
          <DtypeInput label="主干归一化精度" value={norms.dtype} disabled={norms.kind === "none"}
            onChange={(dtype) => updateModel({ norms: { ...norms, dtype } })} />
          <DtypeInput label="路由权重精度" value={router.dtype} disabled={!model.moe_layers}
            onChange={(dtype) => updateModel({ router: { ...router, dtype } })} />
          <div className="form-field"><label>路由选位修正向量
            <input type="checkbox" checked={router.correction_bias} disabled={!model.moe_layers}
              onChange={(event) => updateModel({ router: { ...router, correction_bias: event.currentTarget.checked } })} />
          </label></div>
          <DtypeInput label="路由选位修正向量精度" value={router.correction_dtype}
            nullable={false} disabled={!model.moe_layers || !router.correction_bias}
            onChange={(dtype) => updateModel({ router: { ...router, correction_dtype: dtype! } })} />
          <div className="form-field"><label>启用 iHC
            <input type="checkbox" checked={model.ihc != null}
              onChange={(event) => updateModel({ ihc: event.currentTarget.checked ? { mult: 4, dtype: "fp32" } : null })} />
          </label></div>
          <Input label="iHC 信息通路数（mult）" value={model.ihc?.mult} disabled={model.ihc == null}
            onChange={(value) => updateModel({ ihc: { mult: value ?? NaN, dtype: model.ihc?.dtype ?? "fp32" } })} />
          <DtypeInput label="iHC 精度" value={model.ihc?.dtype ?? "fp32"} nullable={false} disabled={model.ihc == null}
            onChange={(dtype) => updateModel({ ihc: { mult: model.ihc!.mult, dtype: dtype! } })} />
        </div>
        {model.attention_groups?.map((group, index) => <div key={index} className="model-group-card">
          <h4>第 {index + 1} 组辅助权重</h4>
          <div className="form-grid">
          <div className="form-field"><label>第 {index + 1} 组 Q/K 归一化
            <input type="checkbox" checked={group.qk_norm ?? true}
              onChange={(event) => updateGroup(index, { qk_norm: event.currentTarget.checked })} />
          </label></div>
          <DtypeInput label={`第 ${index + 1} 组 Q/K 归一化精度`} value={group.norm_dtype}
            disabled={group.qk_norm === false} onChange={(norm_dtype) => updateGroup(index, { norm_dtype })} />
          {group.kind === "mla_dsa" && <>
            <div className="form-field"><label>第 {index + 1} 组索引归一化类型
              <select value={group.index_norm ?? "layer"} onChange={(event) => updateGroup(index, {
                index_norm: event.currentTarget.value as AttentionGroup["index_norm"],
              })}><option value="none">关闭（无参数）</option><option value="rms">RMSNorm</option><option value="layer">LayerNorm</option></select>
            </label></div>
            <DtypeInput label={`第 ${index + 1} 组索引归一化精度`} value={group.index_norm_dtype}
              disabled={group.index_norm === "none"} onChange={(index_norm_dtype) => updateGroup(index, { index_norm_dtype })} />
          </>}
          <div className="form-field"><label>第 {index + 1} 组启用 attention sink
            <input type="checkbox" checked={group.sink_dtype != null}
              onChange={(event) => updateGroup(index, { sink_dtype: event.currentTarget.checked ? "fp32" : null })} />
          </label></div>
          <DtypeInput label={`第 ${index + 1} 组 attention sink 精度`} value={group.sink_dtype ?? "fp32"}
            nullable={false} disabled={group.sink_dtype == null} onChange={(sink_dtype) => updateGroup(index, { sink_dtype })} />
          </div>
        </div>)}
      </fieldset>
      </div>
    </details>
    <div className="model-subsection">
    <div className="section-heading"><div><h3>注意力组合</h3></div>
      <p>每组填写层数；同一组的 MLA 与 DSA 作用在同一批层。FFN/MoE 层数独立计算；组合内头数决定容量与注意力计算，上方头数仅保留在当前性能基线。</p></div>
      {model.attention_groups != null && <div className="form-grid"><div className="form-field">
        <label>模型名称<input type="text" value={model.name} onChange={(event) =>
          updateModel({ name: event.target.value })} /></label>
      </div><div className="form-field"><label>修订号（可选）<input type="text"
        value={model.revision ?? ""} onChange={(event) =>
          updateModel({ revision: event.target.value || null })} /></label></div></div>}
      {model.attention_groups == null ? <button type="button" className="transfer-button"
        onClick={() => updateGroups([newGroup("gqa", model.num_layers)])}>启用注意力组合配置</button> : <>
        <p>已配置 {model.attention_groups.reduce((total, group) => total + group.layers, 0)} / {model.num_layers} 层。
          不按具体层号排列；不支持顺序敏感的跨层依赖。</p>
        {model.attention_groups.map((group, index) => <div key={index} className="model-group-card">
          <h4>第 {index + 1} 组注意力</h4>
          <div className="form-grid">
            <div className="form-field"><label>第 {index + 1} 组注意力方式
              <select value={group.kind} onChange={(event) => updateGroups(model.attention_groups!.map((item, i) =>
                i === index ? newGroup(event.target.value as AttentionGroup["kind"], item.layers, item) : item))}>
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
          <button type="button" className="secondary-button secondary-button-small model-remove-button" onClick={() => updateGroups(
            model.attention_groups!.length === 1 ? null : model.attention_groups!.filter((_, i) => i !== index)
          )}>删除第 {index + 1} 组</button>
        </div>)}
        <button type="button" className="transfer-button" onClick={() => {
          const groups = model.attention_groups!;
          const last = groups[groups.length - 1];
          updateGroups([...groups.slice(0, -1), { ...last, layers: Math.max(1, last.layers - 1) }, newGroup("gqa", 1)]);
        }}>添加注意力组</button>
      </>}
      <p>自定义 MLA/DSA 按 BF16/FP16 压缩 KV、FP8 索引 Key（每 128 元素 4 字节 scale）和同构 TP 复制口径粗估；可选输出门控及共享专家按填写参数计入容量与时延。归一化、路由修正向量、iHC 与 attention sink 使用统一通用容量公式，只计容量、不增加性能耗时；其他运行时空间不自动推断。</p>
    </div>
    </section>
    <section className="form-section" aria-labelledby="hardware-heading">
      <div className="section-heading"><div>
        <p className="section-kicker">Hardware · Stage 1</p>
        <h2 id="hardware-heading">硬件配置</h2>
      </div><p>以下硬件与存储均为单实例规格；总 GPU 数 = 实例数量 × 每实例 GPU 数</p></div>
      <div className="form-grid" style={{ marginBottom: 14 }}><div className="form-field"><label htmlFor="gpu-preset">GPU 选择</label>
        <select id="gpu-preset" value={selectedHardware?.name ?? "custom"}
          onChange={(event) => {
            const selected = catalog?.hardware.find((entry) => entry.name === event.target.value);
            onHardwareChange(selected ? { ...hardware, name: selected.name,
              memory_bytes_per_gpu: selected.memory_bytes_per_gpu,
              effective_fp16_tflops: selected.peak_fp16_tflops,
              effective_fp8_tflops: selected.peak_fp8_tflops } :
              { ...hardware, name: "自定义 GPU" });
          }}>{catalog?.hardware.map((entry) =>
            <option key={entry.name} value={entry.name}>{entry.name}</option>)}
          <option value="custom">自定义</option></select>
      </div></div>
    <div className="form-grid">
      <Input label="TP 数（必须等于 GPU 卡数）" value={tpSize} onChange={(value) => {
        updateInference({ tp_size: value ?? NaN, ...(model.moe_layers ? { ep_size: value ?? NaN } : {}) });
      }} />
      <Input label="EP 数（MoE 须等于 TP）" value={epSize} onChange={(value) =>
        onEpSizeChange(value ?? NaN)} />
      {([
      ["gpu_count", "本实例 GPU 卡数"],
      ["memory_bytes_per_gpu", "单卡显存容量（GB）"],
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
        ["l2_available_gb", "L2 可用容量（GB，仅 KV）"],
        ["l2_to_l1_gb_s", "L2 → L1 带宽（GB/s）"],
        ["l3_to_l2_gb_s", "L3 → L2 带宽（GB/s）"],
        ["l2_latency_ms", "L2 固定延迟（ms）"],
        ["l3_latency_ms", "L3 固定延迟（ms）"],
      ] as const).map(([field, label]) => <Input key={field} label={label}
        value={storage[field]}
        optional={field === "l2_to_l1_gb_s" || field === "l3_to_l2_gb_s"}
        onChange={(value) => onStorageChange({ ...storage, [field]: value })} />)}
    </div>
    <p>单卡显存容量默认采用所选 GPU 的显存规格，可手动调整；扣除每卡模型权重后全部用于 L1 KV，不额外扣除运行时空间。</p>
    <p>单算子开销默认 0.005 ms/建模计算阶段，仅用于解析计时；L2（本地 DRAM）→ L1 固定延迟默认 0.01 ms/页，L3（低延迟共享存储）→ L2 默认 0.1 ms/页，另计数据传输耗时。均为未经实测校准的试算假设，可按部署调整，设为 0 可对照理想基线。</p>
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
      {tpSize > 1 ? "整组结构估算驻留权重" : "结构估算驻留权重"} { (preview.capacity.weights_resident_bytes / 1e9).toFixed(2) } GB；
      结构估算总参数 {(preview.capacity.estimated_total_parameters / 1e9).toFixed(3)} B；
      结构估算每 token 激活参数 {(preview.capacity.estimated_active_parameters / 1e9).toFixed(3)} B；
      全部专家参数 {(preview.capacity.estimated_expert_parameters / 1e9).toFixed(3)} B；
      L1 {preview.capacity.l1_capacity_pages.toLocaleString()} 页 / {preview.max_total_tokens.toLocaleString()} token；
      L2 {preview.capacity.l2_capacity_pages.toLocaleString()} 页 / {preview.l2_capacity_tokens.toLocaleString()} token；
      全场共享 L3 {preview.capacity.l3_capacity_pages.toLocaleString()} 页 / {preview.l3_capacity_tokens.toLocaleString()} token。
      {preview.reason && <p>{preview.reason}</p>}
      {(preview.capacity.rank_reports ?? []).length > 0 && <>
        <p>TP{tpSize}{model.moe_layers ? ` + EP${epSize}` : ""} 逐卡账单：每个逻辑 token 的整组 KV 物理成本为
          {preview.capacity.kv_bytes_per_token.toLocaleString()} bytes；权重按分片与逐卡复制的结构分别计入。</p>
        {(preview.capacity.rank_reports ?? []).map((rank) => <p key={rank.rank}>
          GPU {rank.rank}：分片权重 {(rank.sharded_weights_bytes / 1e9).toFixed(3)} GB；
          逐卡复制权重 {(rank.replicated_weights_bytes / 1e9).toFixed(3)} GB；
          {model.moe_layers ? <>全部驻留专家权重 {((rank.expert_weights_bytes ?? 0) / 1e9).toFixed(3)} GB；</> : null}
          KV {rank.kv_bytes_per_token.toLocaleString()} bytes/token；
          L1 {rank.l1_capacity_pages.toLocaleString()} 页。
        </p>)}
      </>}
      {!preview.runnable && timing === "analytical" && tpSize > 1 &&
        (model.moe_layers
          ? " 若要运行 TP/EP>1，请填写互联带宽、TP 集合通信延迟与 EP 单次通信开销。"
          : " 多卡容量可预览；填写互联带宽和 TP 集合通信延迟后才能运行。")}
      {preview.runnable && timing === "analytical" && " 时延为未经校准的平均负载/Roofline 估计。"}
    </div>}
  </section>
  </>;
}
