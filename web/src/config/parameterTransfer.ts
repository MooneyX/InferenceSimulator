import type { ResolvedRequest, SimConfig } from "../api/types";
import { validateSimConfig } from "./simulation";

function objectWithKeys(value: unknown, path: string, keys: readonly string[]): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path || "参数配置"} 必须为 JSON 对象`);
  }

  const record = value as Record<string, unknown>;
  for (const key of keys) {
    if (!Object.hasOwn(record, key)) {
      throw new Error(`缺少参数：${path ? `${path}.` : ""}${key}`);
    }
  }
  for (const key of Object.keys(record)) {
    if (!keys.includes(key)) {
      throw new Error(`不支持的参数：${path ? `${path}.` : ""}${key}`);
    }
  }
  return record;
}

function numericFields(record: Record<string, unknown>, path: string, keys: readonly string[]): void {
  for (const key of keys) {
    if (typeof record[key] !== "number") {
      throw new Error(`${path}.${key} 必须是数值`);
    }
  }
}

function lengthDistribution(value: unknown, path: string): void {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} 必须为 JSON 对象`);
  }
  if (!Object.hasOwn(value, "dist")) {
    throw new Error(`缺少参数：${path}.dist`);
  }
  const dist = (value as Record<string, unknown>).dist;
  if (dist !== "fixed" && dist !== "poisson" && dist !== "lognormal") {
    throw new Error(`${path}.dist 不支持该分布类型`);
  }
  const fields = dist === "fixed" ? ["value"] : dist === "poisson" ? ["mean"] : ["mean", "sigma"];
  const distribution = objectWithKeys(value, path, ["dist", ...fields]);
  numericFields(distribution, path, fields);
}

export function parseSimConfigJson(json: string): SimConfig {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    throw new Error("JSON 格式不正确");
  }

  const root = objectWithKeys(parsed, "", ["num_instances", "workload", "engine", "perf_model"]);
  numericFields(root, "参数配置", ["num_instances"]);
  const workloadMode = (root.workload as Record<string, unknown> | null)?.mode;
  if (workloadMode === "trace") {
    objectWithKeys(root.workload, "workload", ["mode"]);
  } else if (workloadMode === "synthetic") {
    const workload = objectWithKeys(root.workload, "workload", ["mode", "synthetic"]);
    const arrivalMode = (workload.synthetic as Record<string, unknown> | null)?.arrival_mode;
    if (arrivalMode !== "open_loop" && arrivalMode !== "session") {
      throw new Error("workload.synthetic.arrival_mode 仅支持 open_loop 或 session");
    }
    const fields = arrivalMode === "open_loop"
      ? ["arrival_rate"] : ["session_rate", "session"];
    const synthetic = objectWithKeys(workload.synthetic, "workload.synthetic", [
      "num_requests", "arrival_mode", "prompt_len", "turn_increment_len", "output_len",
      "max_new_tokens", "seed", ...fields,
    ]);
    numericFields(synthetic, "workload.synthetic", [
      "num_requests", "max_new_tokens", "seed", arrivalMode === "open_loop" ? "arrival_rate" : "session_rate",
    ]);
    lengthDistribution(synthetic.prompt_len, "workload.synthetic.prompt_len");
    lengthDistribution(synthetic.turn_increment_len, "workload.synthetic.turn_increment_len");
    lengthDistribution(synthetic.output_len, "workload.synthetic.output_len");
    if (arrivalMode === "session") {
      const session = objectWithKeys(synthetic.session, "workload.synthetic.session", [
        "num_sessions", "topology", "think_time_ms",
      ]);
      if (session.topology !== "linear") throw new Error("workload.synthetic.session.topology 仅支持 linear");
      numericFields(session, "workload.synthetic.session", [
        "num_sessions", "think_time_ms",
      ]);
    }
  } else {
    throw new Error("workload.mode 仅支持 synthetic 或 trace 模式");
  }

  const engine = objectWithKeys(root.engine, "engine", [
    "max_running_requests", "max_total_tokens", "l2_capacity_tokens", "l3_capacity_tokens",
    "max_prefill_tokens", "l3_prefetch_min_tokens", "page_size", "pd_mutually_exclusive",
  ]);
  numericFields(engine, "engine", [
    "max_running_requests", "max_total_tokens", "l2_capacity_tokens", "l3_capacity_tokens",
    "max_prefill_tokens", "l3_prefetch_min_tokens", "page_size",
  ]);
  if (typeof engine.pd_mutually_exclusive !== "boolean") {
    throw new Error("engine.pd_mutually_exclusive 必须是布尔值");
  }

  const perfModel = objectWithKeys(root.perf_model, "perf_model", [
    "prefill_c1", "prefill_c2", "decode_d1", "decode_d2", "kv_bytes_per_token",
    "l2_to_l1_bandwidth_bytes_per_ms", "l3_to_l2_bandwidth_bytes_per_ms",
    "l2_latency_ms", "l3_latency_ms", "l3_lookup_latency_ms",
  ]);
  numericFields(perfModel, "perf_model", [
    "prefill_c1", "prefill_c2", "decode_d1", "decode_d2", "l2_latency_ms", "l3_latency_ms",
    "l3_lookup_latency_ms",
  ]);
  for (const key of [
    "kv_bytes_per_token", "l2_to_l1_bandwidth_bytes_per_ms", "l3_to_l2_bandwidth_bytes_per_ms",
  ]) {
    if (perfModel[key] !== null && typeof perfModel[key] !== "number") {
      throw new Error(`perf_model.${key} 必须是数值或 null`);
    }
  }

  const config = parsed as SimConfig;
  const error = validateSimConfig(config)[0];
  if (error) {
    throw new Error(`${error.field}：${error.message}`);
  }
  return config;
}

function physicalObject(value: unknown, path: string, required: string[], optional: string[]): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} 必须为 JSON 对象`);
  }
  const record = value as Record<string, unknown>;
  for (const key of required) {
    if (!Object.hasOwn(record, key) || record[key] === undefined) {
      throw new Error(`缺少参数：${path}.${key}`);
    }
  }
  for (const key of Object.keys(record)) {
    if (![...required, ...optional].includes(key)) throw new Error(`不支持的参数：${path}.${key}`);
  }
  return record;
}

function physicalNumber(record: Record<string, unknown>, path: string, key: string,
  minimum: number, integer = false, nullable = false): void {
  const value = record[key];
  if (value === undefined || (nullable && value === null)) return;
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum ||
      (integer && !Number.isInteger(value))) {
    throw new Error(`${path}.${key} 必须是${integer ? "整数且" : ""}不小于 ${minimum} 的有限数值${nullable ? "或 null" : ""}`);
  }
}

function physicalBoolean(record: Record<string, unknown>, path: string, key: string): void {
  if (record[key] !== undefined && typeof record[key] !== "boolean") {
    throw new Error(`${path}.${key} 必须是布尔值`);
  }
}

function physicalDtype(record: Record<string, unknown>, path: string, key: string, nullable = true): void {
  const value = record[key];
  if (value === undefined || (nullable && value === null)) return;
  if (!["bf16", "fp16", "fp32"].includes(value as string)) {
    throw new Error(`${path}.${key} 仅支持 BF16/FP16/FP32${nullable ? " 或 null" : ""}`);
  }
}

function validateResolvedRequest(value: unknown): ResolvedRequest {
  const root = physicalObject(value, "自动解析参数", ["config", "model", "hardware", "storage"],
    ["timing", "tp_size", "ep_size"]);
  const config = parseSimConfigJson(JSON.stringify(root.config));
  const model = physicalObject(root.model, "model", ["name", "num_layers", "hidden_size",
    "num_attention_heads", "num_kv_heads", "head_dim", "ffn_intermediate_size", "vocab_size"],
    ["revision", "tie_word_embeddings", "weight_dtype", "kv_dtype", "moe_layers",
      "num_experts", "experts_per_token", "expert_intermediate_size", "num_shared_experts",
      "shared_expert_intermediate_size", "attention_layout", "attention_groups", "norms", "router", "ihc"]);
  const hardware = physicalObject(root.hardware, "hardware", ["name", "memory_bytes_per_gpu",
    "effective_fp16_tflops", "effective_hbm_gb_s"], ["gpu_count", "effective_fp8_tflops",
    "effective_interconnect_gb_s", "collective_latency_ms", "ep_communication_latency_ms",
    "kernel_overhead_ms"]);
  const storage = physicalObject(root.storage, "storage", ["l1_available_gb"],
    ["l2_available_gb", "l3_available_gb", "runtime_reserve_bytes", "l2_to_l1_gb_s",
      "l3_to_l2_gb_s", "l2_latency_ms", "l3_latency_ms", "l3_lookup_latency_ms"]);
  for (const [part, record] of [["model", model], ["hardware", hardware]] as const) {
    if (typeof record.name !== "string" || !record.name.trim()) throw new Error(`${part}.name 不能为空`);
  }
  if (model.revision !== undefined && model.revision !== null && typeof model.revision !== "string") {
    throw new Error("model.revision 必须是字符串或 null");
  }
  if (model.tie_word_embeddings !== undefined && typeof model.tie_word_embeddings !== "boolean") {
    throw new Error("model.tie_word_embeddings 必须是布尔值");
  }
  for (const [key, allowed] of [["weight_dtype", ["bf16", "fp16", "fp8"]],
    ["kv_dtype", ["bf16", "fp16", "fp8"]], ["attention_layout", ["full_gqa", "grouped", "hy4_preview", "other"]]] as const) {
    if (model[key] !== undefined && !allowed.includes(model[key] as never)) {
      throw new Error(`model.${key} 不支持该配置`);
    }
  }
  if (model.norms !== undefined) {
    const norms = physicalObject(model.norms, "model.norms", [], ["kind", "block_per_layer", "final_norm", "dtype"]);
    if (norms.kind !== undefined && !["none", "rms", "layer"].includes(norms.kind as string)) {
      throw new Error("model.norms.kind 仅支持 none/rms/layer");
    }
    physicalNumber(norms, "model.norms", "block_per_layer", 0, true);
    physicalBoolean(norms, "model.norms", "final_norm");
    physicalDtype(norms, "model.norms", "dtype");
  }
  if (model.router !== undefined) {
    const router = physicalObject(model.router, "model.router", [], ["dtype", "correction_bias", "correction_dtype"]);
    physicalDtype(router, "model.router", "dtype");
    physicalBoolean(router, "model.router", "correction_bias");
    physicalDtype(router, "model.router", "correction_dtype", false);
    if (Number(model.moe_layers ?? 0) === 0 && (router.dtype != null ||
        (router.correction_bias ?? false) !== false || (router.correction_dtype ?? "fp32") !== "fp32")) {
      throw new Error("Router 辅助配置需要 MoE 层");
    }
  }
  if (model.ihc !== undefined && model.ihc !== null) {
    const ihc = physicalObject(model.ihc, "model.ihc", ["mult"], ["dtype"]);
    physicalNumber(ihc, "model.ihc", "mult", 1, true);
    physicalDtype(ihc, "model.ihc", "dtype", false);
  }
  for (const key of ["num_layers", "hidden_size", "num_attention_heads", "num_kv_heads",
    "head_dim", "ffn_intermediate_size", "vocab_size"] as const) physicalNumber(model, "model", key, 1, true);
  for (const key of ["moe_layers", "num_experts", "experts_per_token", "num_shared_experts"] as const) {
    physicalNumber(model, "model", key, 0, true);
  }
  physicalNumber(model, "model", "expert_intermediate_size", 1, true, true);
  physicalNumber(model, "model", "shared_expert_intermediate_size", 1, true, true);
  if (Number(model.num_shared_experts ?? 0) > 0) {
    if (!Number(model.moe_layers ?? 0) || model.shared_expert_intermediate_size == null) {
      throw new Error("启用共享专家需要 MoE 层及 model.shared_expert_intermediate_size");
    }
  } else if (model.shared_expert_intermediate_size != null) {
    throw new Error("关闭共享专家时 model.shared_expert_intermediate_size 必须为 null");
  }
  if (Number(model.num_attention_heads) % Number(model.num_kv_heads) ||
      Number(model.moe_layers ?? 0) > Number(model.num_layers)) {
    throw new Error("model 注意力头数或 MoE 层数不符合模型结构");
  }
  if (model.weight_dtype === "fp8" || model.kv_dtype === "fp8" || model.attention_layout === "other") {
    throw new Error("当前解析仅支持 BF16/FP16 全注意力或 HY4-Preview BF16");
  }
  if ((model.attention_layout === "grouped" && model.attention_groups == null) ||
      (model.attention_layout !== "grouped" && model.attention_layout !== "hy4_preview" &&
        model.attention_groups != null)) {
    throw new Error("grouped 注意力布局必须提供注意力组，其他布局不能混用");
  }
  if (model.attention_groups !== undefined && model.attention_groups !== null) {
    if (!Array.isArray(model.attention_groups) || model.attention_groups.length === 0) {
      throw new Error("model.attention_groups 必须包含至少一个配置组");
    }
    const fields = ["num_kv_heads", "head_dim", "q_rank", "kv_rank", "rope_dim", "nope_dim",
      "value_dim", "index_heads", "index_dim", "topk", "full_indexer_layers"];
    let layers = 0;
    for (const [index, raw] of model.attention_groups.entries()) {
      const path = `model.attention_groups[${index}]`;
      const group = physicalObject(raw, path, ["kind", "layers", "num_attention_heads"],
        [...fields, "output_gate", "qk_norm", "norm_dtype", "index_norm", "index_norm_dtype", "sink_dtype"]);
      physicalBoolean(group, path, "qk_norm");
      for (const field of ["norm_dtype", "index_norm_dtype", "sink_dtype"]) physicalDtype(group, path, field);
      if (group.index_norm !== undefined && !["none", "rms", "layer"].includes(group.index_norm as string)) {
        throw new Error(`${path}.index_norm 仅支持 none/rms/layer`);
      }
      if (group.kind !== "mla_dsa" && ((group.index_norm ?? "layer") !== "layer" || group.index_norm_dtype != null)) {
        throw new Error(`${path}.index_norm/index_norm_dtype 仅支持 DSA 自定义配置`);
      }
      if (!["gqa", "mla", "mla_dsa"].includes(group.kind as string)) throw new Error(`${path}.kind 不支持`);
      if (group.output_gate !== undefined && typeof group.output_gate !== "boolean") {
        throw new Error(`${path}.output_gate 必须是布尔值`);
      }
      if (group.kind === "gqa" && group.output_gate === true) {
        throw new Error(`${path}.output_gate 仅支持 MLA/DSA`);
      }
      for (const field of ["layers", "num_attention_heads"])
        physicalNumber(group, path, field, 1, true);
      for (const field of fields) physicalNumber(group, path, field, 1, true, true);
      const present = (field: string) => group[field] != null;
      const mla = ["q_rank", "kv_rank", "rope_dim", "nope_dim", "value_dim"];
      const dsa = ["index_heads", "index_dim", "topk", "full_indexer_layers"];
      if (group.kind === "gqa") {
        if (!present("num_kv_heads") || !present("head_dim") ||
            [...mla, ...dsa].some(present) || Number(group.num_attention_heads) % Number(group.num_kv_heads)) {
          throw new Error(`${path} 的 GQA 头数或字段无效`);
        }
      } else if (present("num_kv_heads") || present("head_dim") || mla.some((field) => !present(field)) ||
        (group.kind === "mla_dsa" ? dsa.some((field) => !present(field)) ||
          Number(group.full_indexer_layers) > Number(group.layers) || Number(group.index_dim) % 128 !== 0 :
          dsa.some(present))) {
        throw new Error(`${path} 的 MLA/DSA 参数无效`);
      }
      layers += Number(group.layers);
    }
    if (layers !== Number(model.num_layers)) throw new Error("注意力组层数之和必须等于模型总层数");
  }
  if (model.attention_layout === "hy4_preview") {
    const expected = {
      name: "tencent/Hy4-preview", revision: "705d81ee51566a186d645b74c974d642ef2828fe",
      num_layers: 78, hidden_size: 6144,
      num_attention_heads: 64, num_kv_heads: 8, head_dim: 256,
      ffn_intermediate_size: 18432, vocab_size: 120832,
      moe_layers: 77, num_experts: 256, experts_per_token: 8,
      expert_intermediate_size: 2048, num_shared_experts: 1, shared_expert_intermediate_size: 2048,
      tie_word_embeddings: false, weight_dtype: "bf16", kv_dtype: "bf16",
    };
    if (Object.entries(expected).some(([field, expectedValue]) =>
      (model[field] ?? (field === "tie_word_embeddings" ? false : undefined)) !== expectedValue)) {
      throw new Error("HY4-Preview 仅支持已核对的 BF16 主干结构；不含 MTP");
    }
    const expectedAuxiliary = {
      norms: { kind: "rms", block_per_layer: 2, final_norm: true, dtype: null },
      router: { dtype: "fp32", correction_bias: true, correction_dtype: "fp32" },
      ihc: { mult: 4, dtype: "fp32" },
    };
    const auxiliaryDefaults = {
      norms: { kind: "rms", block_per_layer: 2, final_norm: true, dtype: null },
      router: { dtype: null, correction_bias: false, correction_dtype: "fp32" },
      ihc: { dtype: "fp32" },
    };
    for (const key of ["norms", "router", "ihc"] as const) {
      const actual = model[key] === null || (key === "ihc" && model[key] === undefined)
        ? null : { ...auxiliaryDefaults[key], ...model[key] as Record<string, unknown> | undefined };
      if (actual === null || Object.entries(expectedAuxiliary[key]).some(([field, value]) =>
        (actual as Record<string, unknown>)[field] !== value)) {
        throw new Error(`HY4-Preview 的 model.${key} 须与固定修订结构一致`);
      }
    }
    if (model.attention_groups == null) {
      throw new Error("HY4-Preview 必须提供完整固定注意力组");
    }
    const group = model.attention_groups[0] as Record<string, unknown>;
    const expectedGroup = { kind: "mla_dsa", layers: 78, num_attention_heads: 64,
      q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256,
      index_heads: 32, index_dim: 128, topk: 2048, full_indexer_layers: 21, output_gate: true };
    const expectedGroupAuxiliary = { qk_norm: true, norm_dtype: null, index_norm: "layer",
      index_norm_dtype: "fp32", sink_dtype: "fp32" };
    const effectiveGroupAuxiliary = { qk_norm: true, norm_dtype: null, index_norm: "layer",
      index_norm_dtype: null, sink_dtype: null,
      ...Object.fromEntries(Object.entries(group).filter(([, value]) => value !== undefined)) };
    if (model.attention_groups.length !== 1 ||
        Object.entries(expectedGroup).some(([field, value]) => group[field] !== value) ||
        Object.entries(expectedGroupAuxiliary).some(([field, value]) =>
          (effectiveGroupAuxiliary as Record<string, unknown>)[field] !== value)) {
      throw new Error("HY4-Preview 的注意力组须与固定修订结构一致");
    }
  }
  for (const key of ["gpu_count", "memory_bytes_per_gpu"] as const) {
    physicalNumber(hardware, "hardware", key, 1, true);
  }
  for (const key of ["effective_fp16_tflops", "effective_hbm_gb_s"] as const) {
    physicalNumber(hardware, "hardware", key, Number.MIN_VALUE);
  }
  for (const key of ["effective_fp8_tflops", "effective_interconnect_gb_s"] as const) {
    physicalNumber(hardware, "hardware", key, Number.MIN_VALUE, false, true);
  }
  for (const key of ["collective_latency_ms", "ep_communication_latency_ms"] as const) {
    physicalNumber(hardware, "hardware", key, 0, false, true);
  }
  physicalNumber(hardware, "hardware", "kernel_overhead_ms", 0);
  physicalNumber(storage, "storage", "l1_available_gb", Number.MIN_VALUE);
  for (const key of ["l2_available_gb", "l3_available_gb", "l2_latency_ms", "l3_latency_ms",
    "l3_lookup_latency_ms"] as const) {
    physicalNumber(storage, "storage", key, 0);
  }
  physicalNumber(storage, "storage", "runtime_reserve_bytes", 0, true);
  for (const key of ["l2_to_l1_gb_s", "l3_to_l2_gb_s"] as const) {
    physicalNumber(storage, "storage", key, Number.MIN_VALUE, false, true);
  }
  if (root.timing !== undefined && root.timing !== "analytical" && root.timing !== "empirical") {
    throw new Error("timing 不支持该计时模式");
  }
  for (const field of ["tp_size", "ep_size"] as const) {
    if (root[field] !== undefined && (!Number.isInteger(root[field]) || (root[field] as number) < 1)) {
      throw new Error(`${field} 必须是正整数`);
    }
  }
  const tpSize = Number(root.tp_size ?? 1);
  const epSize = Number(root.ep_size ?? 1);
  if (Number(hardware.gpu_count ?? 1) !== tpSize) throw new Error("TP 数必须等于 GPU 卡数");
  if (Number(model.moe_layers ?? 0) > 0) {
    if (epSize !== tpSize) throw new Error("MoE 的 EP 数必须等于 TP 数");
    if (!model.vocab_size || !model.expert_intermediate_size ||
        !model.num_experts || !model.experts_per_token ||
        Number(model.experts_per_token) > Number(model.num_experts)) {
      throw new Error("MoE 需要词表、专家数、激活专家数和专家中间维度");
    }
  } else if (epSize !== 1 || Number(model.num_experts ?? 0) > 0 ||
             Number(model.experts_per_token ?? 0) > 0 || model.expert_intermediate_size != null) {
    throw new Error("Dense 模型不能设置 EP 或 MoE 专家参数");
  }
  return { ...root, config, timing: root.timing ?? "analytical",
    tp_size: tpSize, ep_size: epSize } as ResolvedRequest;
}

export function parseResolvedRequestJson(json: string): ResolvedRequest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    throw new Error("JSON 格式不正确");
  }
  return validateResolvedRequest(parsed);
}

export function serializeResolvedRequest(request: ResolvedRequest): string {
  serializeSimConfig(request.config);
  validateResolvedRequest(request);
  return JSON.stringify(request);
}

export function serializeSimConfig(config: SimConfig): string {
  const error = validateSimConfig(config)[0];
  if (error) {
    throw new Error(`${error.field}：${error.message}`);
  }
  const json = JSON.stringify(config);
  parseSimConfigJson(json);
  return json;
}
