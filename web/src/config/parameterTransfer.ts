import type { InferenceConfig, SimConfig, TimingConfig } from "../api/types";
import { initialHardware, initialSharedL3, initialStorage, validateSimConfig } from "./simulation";

function objectWithKeys(value: unknown, path: string, keys: readonly string[], defaults: Record<string, unknown> = {}): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path || "参数配置"} 必须为 JSON 对象`);
  }

  const record = value as Record<string, unknown>;
  for (const key of keys) {
    if (!Object.hasOwn(record, key) && !Object.hasOwn(defaults, key)) {
      throw new Error(`缺少参数：${path ? `${path}.` : ""}${key}`);
    }
  }
  for (const key of Object.keys(record)) {
    if (!keys.includes(key)) {
      throw new Error(`不支持的参数：${path ? `${path}.` : ""}${key}`);
    }
  }
  return { ...defaults, ...record };
}

function numericFields(record: Record<string, unknown>, path: string, keys: readonly string[]): void {
  for (const key of keys) {
    if (typeof record[key] !== "number" || !Number.isFinite(record[key])) {
      throw new Error(`${path}.${key} 必须是数值且为有限数值`);
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

  const root = objectWithKeys(parsed, "", ["num_instances", "routing_policy", "workload", "scheduler", "inference", "shared_l3"],
    { shared_l3: null, routing_policy: "least_load" });
  if (root.shared_l3 != null) {
    root.shared_l3 = objectWithKeys(root.shared_l3, "shared_l3", ["available_gb", "lookup_latency_ms"], { available_gb: 0, lookup_latency_ms: initialSharedL3.lookup_latency_ms });
    numericFields(root.shared_l3 as Record<string, unknown>, "shared_l3", ["available_gb", "lookup_latency_ms"]);
  }
  numericFields(root, "参数配置", ["num_instances"]);
  const workload: Record<string, unknown> = { mode: "synthetic", arrival_mode: "closed_loop", seed: 0,
    ...configObject(root.workload, "workload", [], [
      "mode", "arrival_mode", "seed", "session_rate", "target_qps", "synthetic", "trace",
    ]) };
  if (workload.arrival_mode !== "open_loop" && workload.arrival_mode !== "closed_loop") {
    throw new Error("workload.arrival_mode 仅支持 open_loop 或 closed_loop");
  }
  numericFields(workload, "workload", ["seed"]);
  for (const field of ["session_rate", "target_qps"]) {
    if (workload[field] != null) numericFields(workload, "workload", [field]);
  }
  if (workload.mode === "trace") {
    if (workload.synthetic != null) throw new Error("Trace 负载不能同时配置 workload.synthetic");
    const trace = objectWithKeys(workload.trace ?? {}, "workload.trace", ["plan_id"], { plan_id: null });
    if (trace.plan_id != null && (typeof trace.plan_id !== "string" || trace.plan_id.length === 0)) {
      throw new Error("workload.trace.plan_id 必须是非空字符串或 null");
    }
    const { synthetic: _synthetic, ...traceWorkload } = workload;
    root.workload = { ...traceWorkload, trace };
  } else if (workload.mode === "synthetic") {
    if (workload.trace != null) throw new Error("合成负载不能同时配置 workload.trace");
    const synthetic = objectWithKeys(workload.synthetic, "workload.synthetic", [
      "num_requests", "prompt_len", "turn_increment_len", "output_len", "session",
    ], { session: {} });
    numericFields(synthetic, "workload.synthetic", ["num_requests"]);
    lengthDistribution(synthetic.prompt_len, "workload.synthetic.prompt_len");
    lengthDistribution(synthetic.turn_increment_len, "workload.synthetic.turn_increment_len");
    lengthDistribution(synthetic.output_len, "workload.synthetic.output_len");
    const session = objectWithKeys(synthetic.session, "workload.synthetic.session", ["num_sessions", "topology", "think_time_ms"], {
      num_sessions: null, topology: "linear", think_time_ms: 0,
    });
    if (session.topology !== "linear") throw new Error("workload.synthetic.session.topology 仅支持 linear");
    numericFields(session, "workload.synthetic.session", ["think_time_ms"]);
    if (session.num_sessions != null) numericFields(session, "workload.synthetic.session", ["num_sessions"]);
    const { trace: _trace, ...syntheticWorkload } = workload;
    root.workload = { ...syntheticWorkload, synthetic: { ...synthetic,
      session: workload.arrival_mode === "open_loop" && !Object.hasOwn(workload.synthetic as object, "session")
        ? undefined : session } };
  } else throw new Error("workload.mode 仅支持 synthetic 或 trace 模式");

  const scheduler = objectWithKeys(root.scheduler, "scheduler", ["max_running_requests", "max_prefill_tokens", "l3_prefetch_min_tokens", "pd_mutually_exclusive"], {
    max_prefill_tokens: 16_384, l3_prefetch_min_tokens: 256, pd_mutually_exclusive: true,
  });
  numericFields(scheduler, "scheduler", ["max_running_requests", "max_prefill_tokens", "l3_prefetch_min_tokens"]);
  if (typeof scheduler.pd_mutually_exclusive !== "boolean") throw new Error("scheduler.pd_mutually_exclusive 必须是布尔值");
  root.scheduler = scheduler;
  root.inference = validateInference(root.inference);

  const config = root as SimConfig;
  const error = validateSimConfig(config)[0];
  if (error) {
    throw new Error(`${error.field}：${error.message}`);
  }
  return config;
}

function configObject(value: unknown, path: string, required: string[], optional: string[]): Record<string, unknown> {
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

function configNumber(record: Record<string, unknown>, path: string, key: string,
  minimum: number, integer = false, nullable = false): void {
  const value = record[key];
  if (value === undefined || (nullable && value === null)) return;
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum ||
      (integer && !Number.isInteger(value))) {
    throw new Error(`${path}.${key} 必须是${integer ? "整数且" : ""}不小于 ${minimum} 的有限数值${nullable ? "或 null" : ""}`);
  }
}

function configBoolean(record: Record<string, unknown>, path: string, key: string): void {
  if (record[key] !== undefined && typeof record[key] !== "boolean") {
    throw new Error(`${path}.${key} 必须是布尔值`);
  }
}

function configDtype(record: Record<string, unknown>, path: string, key: string, nullable = true): void {
  const value = record[key];
  if (value === undefined || (nullable && value === null)) return;
  if (!["bf16", "fp16", "fp32"].includes(value as string)) {
    throw new Error(`${path}.${key} 仅支持 BF16/FP16/FP32${nullable ? " 或 null" : ""}`);
  }
}

function validateTiming(value: unknown): TimingConfig {
  const mode = (value as Record<string, unknown> | null)?.mode;
  if (mode === "analytical") {
    return objectWithKeys(value, "inference.timing", ["mode"]) as TimingConfig;
  } else if (mode === "empirical") {
    const timing = objectWithKeys(value, "inference.timing", ["mode", "prefill_c1", "prefill_c2", "decode_d1", "decode_d2"], { mode: "empirical", prefill_c2: 0 });
    numericFields(timing, "inference.timing", ["prefill_c1", "prefill_c2", "decode_d1", "decode_d2"]);
    return timing as TimingConfig;
  } else throw new Error("inference.timing.mode 不支持该计时模式");
}

function validateInference(value: unknown): InferenceConfig {
  const root = objectWithKeys(value, "inference", ["page_size", "max_output_tokens", "model", "hardware", "storage", "timing", "tp_size", "ep_size"], {
    page_size: 64, max_output_tokens: 65_536, timing: { mode: "analytical" }, tp_size: 1, ep_size: 1,
  });
  numericFields(root, "inference", ["page_size", "max_output_tokens"]);
  root.timing = validateTiming(root.timing);
  const model = { tie_word_embeddings: false, weight_dtype: "bf16", kv_dtype: "bf16", moe_layers: 0,
    num_experts: 0, experts_per_token: 0, num_shared_experts: 0, attention_layout: "full_gqa",
    ...configObject(root.model, "model", ["name", "num_layers", "hidden_size",
    "num_attention_heads", "num_kv_heads", "head_dim", "ffn_intermediate_size", "vocab_size"],
    ["revision", "tie_word_embeddings", "weight_dtype", "kv_dtype", "moe_layers",
      "num_experts", "experts_per_token", "expert_intermediate_size", "num_shared_experts",
      "shared_expert_intermediate_size", "attention_layout", "attention_groups", "norms", "router", "ihc"]) } as Record<string, unknown>;
  const hardware = { gpu_count: 1, kernel_overhead_ms: initialHardware.kernel_overhead_ms,
    ...configObject(root.hardware, "hardware", ["name", "memory_bytes_per_gpu",
    "effective_fp16_tflops", "effective_hbm_gb_s"], ["gpu_count", "effective_fp8_tflops",
    "effective_interconnect_gb_s", "collective_latency_ms", "ep_communication_latency_ms",
    "kernel_overhead_ms"]) } as Record<string, unknown>;
  const storage = { l2_available_gb: 0, runtime_reserve_bytes: 0, l2_latency_ms: initialStorage.l2_latency_ms,
    l3_latency_ms: initialStorage.l3_latency_ms, ...configObject(root.storage, "storage", ["l1_available_gb"],
    ["l2_available_gb", "runtime_reserve_bytes", "l2_to_l1_gb_s",
      "l3_to_l2_gb_s", "l2_latency_ms", "l3_latency_ms"]) } as Record<string, unknown>;
  root.model = model;
  root.hardware = hardware;
  root.storage = storage;
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
    const norms = configObject(model.norms, "model.norms", [], ["kind", "block_per_layer", "final_norm", "dtype"]);
    if (norms.kind !== undefined && !["none", "rms", "layer"].includes(norms.kind as string)) {
      throw new Error("model.norms.kind 仅支持 none/rms/layer");
    }
    configNumber(norms, "model.norms", "block_per_layer", 0, true);
    configBoolean(norms, "model.norms", "final_norm");
    configDtype(norms, "model.norms", "dtype");
  }
  if (model.router !== undefined) {
    const router = configObject(model.router, "model.router", [], ["dtype", "correction_bias", "correction_dtype"]);
    configDtype(router, "model.router", "dtype");
    configBoolean(router, "model.router", "correction_bias");
    configDtype(router, "model.router", "correction_dtype", false);
    if (Number(model.moe_layers ?? 0) === 0 && (router.dtype != null ||
        (router.correction_bias ?? false) !== false || (router.correction_dtype ?? "fp32") !== "fp32")) {
      throw new Error("Router 辅助配置需要 MoE 层");
    }
  }
  if (model.ihc !== undefined && model.ihc !== null) {
    const ihc = configObject(model.ihc, "model.ihc", ["mult"], ["dtype"]);
    configNumber(ihc, "model.ihc", "mult", 1, true);
    configDtype(ihc, "model.ihc", "dtype", false);
  }
  for (const key of ["num_layers", "hidden_size", "num_attention_heads", "num_kv_heads",
    "head_dim", "ffn_intermediate_size", "vocab_size"] as const) configNumber(model, "model", key, 1, true);
  for (const key of ["moe_layers", "num_experts", "experts_per_token", "num_shared_experts"] as const) {
    configNumber(model, "model", key, 0, true);
  }
  configNumber(model, "model", "expert_intermediate_size", 1, true, true);
  configNumber(model, "model", "shared_expert_intermediate_size", 1, true, true);
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
    throw new Error("当前解析仅支持 BF16/FP16 全注意力或注意力组合");
  }
  if (((model.attention_layout === "grouped" || model.attention_layout === "hy4_preview") &&
        model.attention_groups == null) ||
      (model.attention_layout !== "grouped" && model.attention_layout !== "hy4_preview" &&
        model.attention_groups != null)) {
    throw new Error("grouped/hy4_preview 注意力布局必须提供注意力组，其他布局不能混用");
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
      const group = configObject(raw, path, ["kind", "layers", "num_attention_heads"],
        [...fields, "output_gate", "qk_norm", "norm_dtype", "index_norm", "index_norm_dtype", "sink_dtype"]);
      configBoolean(group, path, "qk_norm");
      for (const field of ["norm_dtype", "index_norm_dtype", "sink_dtype"]) configDtype(group, path, field);
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
        configNumber(group, path, field, 1, true);
      for (const field of fields) configNumber(group, path, field, 1, true, true);
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
  for (const key of ["gpu_count", "memory_bytes_per_gpu"] as const) {
    configNumber(hardware, "hardware", key, 1, true);
  }
  for (const key of ["effective_fp16_tflops", "effective_hbm_gb_s"] as const) {
    configNumber(hardware, "hardware", key, Number.MIN_VALUE);
  }
  for (const key of ["effective_fp8_tflops", "effective_interconnect_gb_s"] as const) {
    configNumber(hardware, "hardware", key, Number.MIN_VALUE, false, true);
  }
  for (const key of ["collective_latency_ms", "ep_communication_latency_ms"] as const) {
    configNumber(hardware, "hardware", key, 0, false, true);
  }
  configNumber(hardware, "hardware", "kernel_overhead_ms", 0);
  configNumber(storage, "storage", "l1_available_gb", Number.MIN_VALUE);
  for (const key of ["l2_available_gb", "l2_latency_ms", "l3_latency_ms"] as const) {
    configNumber(storage, "storage", key, 0);
  }
  configNumber(storage, "storage", "runtime_reserve_bytes", 0, true);
  for (const key of ["l2_to_l1_gb_s", "l3_to_l2_gb_s"] as const) {
    configNumber(storage, "storage", key, Number.MIN_VALUE, false, true);
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
  return root as InferenceConfig;
}

export function serializeSimConfig(config: SimConfig): string {
  const error = validateSimConfig(config)[0];
  if (error) {
    throw new Error(`${error.field}：${error.message}`);
  }
  validateInference(config.inference);
  const reusable = config.workload.mode === "trace" && config.workload.trace
    ? { ...config, workload: { ...config.workload, trace: { ...config.workload.trace, plan_id: null } } }
    : config;
  const json = JSON.stringify(reusable);
  parseSimConfigJson(json);
  return json;
}

export function serializeScanPlan(base: SimConfig, axes: { key: string; text: string }[]): string {
  return JSON.stringify({ version: 1, base: JSON.parse(serializeSimConfig(base)), axes });
}
