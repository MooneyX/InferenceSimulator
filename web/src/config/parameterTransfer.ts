import type { SimConfig } from "../api/types";
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
    "max_running_requests", "max_total_tokens", "max_prefill_tokens", "page_size", "pd_mutually_exclusive",
  ]);
  numericFields(engine, "engine", [
    "max_running_requests", "max_total_tokens", "max_prefill_tokens", "page_size",
  ]);
  if (typeof engine.pd_mutually_exclusive !== "boolean") {
    throw new Error("engine.pd_mutually_exclusive 必须是布尔值");
  }

  const perfModel = objectWithKeys(root.perf_model, "perf_model", [
    "prefill_c1", "prefill_c2", "decode_d1", "decode_d2",
  ]);
  numericFields(perfModel, "perf_model", ["prefill_c1", "prefill_c2", "decode_d1", "decode_d2"]);

  const config = parsed as SimConfig;
  const error = validateSimConfig(config)[0];
  if (error) {
    throw new Error(`${error.field}：${error.message}`);
  }
  return config;
}

export function serializeSimConfig(config: SimConfig): string {
  const json = JSON.stringify(config);
  parseSimConfigJson(json);
  return json;
}
