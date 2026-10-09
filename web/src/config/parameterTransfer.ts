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

  const root = objectWithKeys(parsed, "", ["workload", "engine", "perf_model"]);
  const workload = objectWithKeys(root.workload, "workload", ["mode", "synthetic"]);
  if (workload.mode !== "synthetic") {
    throw new Error("workload.mode 当前仅支持 synthetic 模式");
  }
  const synthetic = objectWithKeys(workload.synthetic, "workload.synthetic", [
    "num_requests", "arrival_rate", "prompt_len", "output_len", "max_new_tokens", "seed", "session",
  ]);
  numericFields(synthetic, "workload.synthetic", [
    "num_requests", "arrival_rate", "max_new_tokens", "seed",
  ]);
  lengthDistribution(synthetic.prompt_len, "workload.synthetic.prompt_len");
  lengthDistribution(synthetic.output_len, "workload.synthetic.output_len");

  const session = objectWithKeys(synthetic.session, "workload.synthetic.session", [
    "num_sessions", "topology", "think_time_ms", "target_ideal_prefix_hit_rate",
  ]);
  if (session.topology !== "linear") {
    throw new Error("workload.synthetic.session.topology 当前仅支持 linear");
  }
  numericFields(session, "workload.synthetic.session", [
    "num_sessions", "think_time_ms", "target_ideal_prefix_hit_rate",
  ]);

  const engine = objectWithKeys(root.engine, "engine", [
    "max_running_requests", "max_total_tokens", "l2_capacity_tokens", "l3_capacity_tokens",
    "max_prefill_tokens", "page_size", "pd_mutually_exclusive",
  ]);
  numericFields(engine, "engine", [
    "max_running_requests", "max_total_tokens", "l2_capacity_tokens", "l3_capacity_tokens",
    "max_prefill_tokens", "page_size",
  ]);
  if (typeof engine.pd_mutually_exclusive !== "boolean") {
    throw new Error("engine.pd_mutually_exclusive 必须是布尔值");
  }

  const perfModel = objectWithKeys(root.perf_model, "perf_model", [
    "prefill_c1", "prefill_c2", "decode_d1", "decode_d2", "kv_bytes_per_token",
    "l2_to_l1_bandwidth_bytes_per_ms", "l3_to_l2_bandwidth_bytes_per_ms",
    "l2_latency_ms", "l3_latency_ms",
  ]);
  numericFields(perfModel, "perf_model", [
    "prefill_c1", "prefill_c2", "decode_d1", "decode_d2", "l2_latency_ms", "l3_latency_ms",
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

export function parseResolvedRequestJson(json: string): ResolvedRequest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    throw new Error("JSON 格式不正确");
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("自动解析参数必须为 JSON 对象");
  }
  const root = parsed as Record<string, unknown>;
  const config = parseSimConfigJson(JSON.stringify(root.config));
  for (const part of ["model", "hardware", "storage"] as const) {
    if (root[part] === null || typeof root[part] !== "object" || Array.isArray(root[part])) {
      throw new Error(`${part} 必须为 JSON 对象`);
    }
  }
  const model = root.model as Record<string, unknown>;
  const hardware = root.hardware as Record<string, unknown>;
  const storage = root.storage as Record<string, unknown>;
  if (typeof model.name !== "string" || !model.name ||
      typeof hardware.name !== "string" || !hardware.name) {
    throw new Error("模型和 GPU 名称不能为空");
  }
  numericFields(model, "model", ["num_layers", "hidden_size", "num_attention_heads",
    "num_kv_heads", "head_dim", "ffn_intermediate_size", "total_parameters"]);
  numericFields(hardware, "hardware", ["gpu_count", "memory_bytes_per_gpu",
    "effective_fp16_tflops", "effective_hbm_gb_s"]);
  numericFields(storage, "storage", ["l1_available_gb", "l2_available_gb", "l3_available_gb"]);
  if (root.timing !== undefined && root.timing !== "analytical" && root.timing !== "empirical") {
    throw new Error("timing 不支持该计时模式");
  }
  if (root.tp_size !== undefined &&
      (!Number.isInteger(root.tp_size) || (root.tp_size as number) < 1)) {
    throw new Error("tp_size 必须是正整数");
  }
  return { ...root, config, timing: root.timing ?? "analytical",
    tp_size: root.tp_size ?? 1 } as ResolvedRequest;
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
