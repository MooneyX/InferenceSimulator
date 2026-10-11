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

function validateResolvedRequest(value: unknown): ResolvedRequest {
  const root = physicalObject(value, "自动解析参数", ["config", "model", "hardware", "storage"],
    ["timing", "tp_size", "ep_size"]);
  const config = parseSimConfigJson(JSON.stringify(root.config));
  const model = physicalObject(root.model, "model", ["name", "num_layers", "hidden_size",
    "num_attention_heads", "num_kv_heads", "head_dim", "ffn_intermediate_size", "vocab_size"],
    ["revision", "tie_word_embeddings", "weight_dtype", "kv_dtype", "moe_layers",
      "num_experts", "experts_per_token", "expert_intermediate_size", "attention_layout"]);
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
    ["kv_dtype", ["bf16", "fp16", "fp8"]], ["attention_layout", ["full_gqa", "hy4_preview", "other"]]] as const) {
    if (model[key] !== undefined && !allowed.includes(model[key] as never)) {
      throw new Error(`model.${key} 不支持该配置`);
    }
  }
  for (const key of ["num_layers", "hidden_size", "num_attention_heads", "num_kv_heads",
    "head_dim", "ffn_intermediate_size", "vocab_size"] as const) physicalNumber(model, "model", key, 1, true);
  for (const key of ["moe_layers", "num_experts", "experts_per_token"] as const) {
    physicalNumber(model, "model", key, 0, true);
  }
  physicalNumber(model, "model", "expert_intermediate_size", 1, true, true);
  if (Number(model.num_attention_heads) % Number(model.num_kv_heads) ||
      Number(model.moe_layers ?? 0) > Number(model.num_layers)) {
    throw new Error("model 注意力头数或 MoE 层数不符合模型结构");
  }
  if (model.weight_dtype === "fp8" || model.kv_dtype === "fp8" || model.attention_layout === "other") {
    throw new Error("当前解析仅支持 BF16/FP16 全注意力或 HY4-Preview BF16");
  }
  if (model.attention_layout === "hy4_preview") {
    const expected = {
      name: "tencent/Hy4-preview", revision: "705d81ee51566a186d645b74c974d642ef2828fe",
      num_layers: 78, hidden_size: 6144,
      num_attention_heads: 64, num_kv_heads: 8, head_dim: 256,
      ffn_intermediate_size: 18432, vocab_size: 120832,
      moe_layers: 77, num_experts: 256, experts_per_token: 8,
      expert_intermediate_size: 2048, tie_word_embeddings: false,
      weight_dtype: "bf16", kv_dtype: "bf16",
    };
    if (Object.entries(expected).some(([field, expectedValue]) =>
      (model[field] ?? (field === "tie_word_embeddings" ? false : undefined)) !== expectedValue)) {
      throw new Error("HY4-Preview 仅支持已核对的 BF16 主干结构；不含 MTP");
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
