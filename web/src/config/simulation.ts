import type { LengthDistribution, SimConfig } from "../api/types";

export type ConfigField =
  | "workload.mode"
  | "workload.synthetic"
  | "workload.synthetic.num_requests"
  | "workload.synthetic.arrival_rate"
  | "workload.synthetic.session_rate"
  | "workload.synthetic.session.num_sessions"
  | "workload.synthetic.session.think_time_ms"
  | "workload.synthetic.session.target_ideal_prefix_hit_rate"
  | "workload.synthetic.prompt_len.value"
  | "workload.synthetic.prompt_len.mean"
  | "workload.synthetic.prompt_len.sigma"
  | "workload.synthetic.turn_increment_len.value"
  | "workload.synthetic.turn_increment_len.mean"
  | "workload.synthetic.turn_increment_len.sigma"
  | "workload.synthetic.output_len.value"
  | "workload.synthetic.output_len.mean"
  | "workload.synthetic.output_len.sigma"
  | "workload.synthetic.seed"
  | "workload.synthetic.max_new_tokens"
  | "engine.max_running_requests"
  | "engine.max_total_tokens"
  | "engine.l2_capacity_tokens"
  | "engine.l3_capacity_tokens"
  | "engine.max_prefill_tokens"
  | "engine.page_size"
  | "engine.pd_mutually_exclusive"
  | "perf_model.kv_bytes_per_token"
  | "perf_model.l2_to_l1_bandwidth_bytes_per_ms"
  | "perf_model.l3_to_l2_bandwidth_bytes_per_ms"
  | "perf_model.l2_latency_ms"
  | "perf_model.l3_latency_ms"
  | "perf_model.prefill_c1"
  | "perf_model.prefill_c2"
  | "perf_model.decode_d1"
  | "perf_model.decode_d2";

export interface ConfigValidationError {
  field: ConfigField;
  message: string;
}

export function createDefaultSimConfig(): SimConfig {
  return {
    workload: {
      mode: "synthetic",
      synthetic: {
        num_requests: 1000,
        arrival_rate: 64,
        arrival_mode: "open_loop",
        prompt_len: {
          dist: "poisson",
          mean: 2048,
        },
        turn_increment_len: {
          dist: "poisson",
          mean: 256,
        },
        output_len: {
          dist: "poisson",
          mean: 128,
        },
        seed: 42,
        max_new_tokens: 1024,
      },
    },
    engine: {
      max_running_requests: 32,
      max_total_tokens: 65_536,
      l2_capacity_tokens: 0,
      l3_capacity_tokens: 0,
      max_prefill_tokens: 16_384,
      page_size: 64,
      pd_mutually_exclusive: true,
    },
    perf_model: {
      prefill_c1: 0.05,
      prefill_c2: 0.000_01,
      decode_d1: 8,
      decode_d2: 0.002,
      kv_bytes_per_token: null,
      l2_to_l1_bandwidth_bytes_per_ms: null,
      l3_to_l2_bandwidth_bytes_per_ms: null,
      l2_latency_ms: 0,
      l3_latency_ms: 0,
    },
  };
}

function validateFiniteNumber(
  errors: ConfigValidationError[],
  field: ConfigField,
  value: number,
  minimum: number,
  exclusiveMinimum = false,
): void {
  const belowMinimum = exclusiveMinimum ? value <= minimum : value < minimum;

  if (!Number.isFinite(value) || belowMinimum) {
    errors.push({
      field,
      message: exclusiveMinimum
        ? `必须是大于 ${minimum} 的有限数值`
        : `必须是大于或等于 ${minimum} 的有限数值`,
    });
  }
}

function validatePositiveInteger(
  errors: ConfigValidationError[],
  field: ConfigField,
  value: number,
): void {
  if (!Number.isInteger(value) || value < 1) {
    errors.push({ field, message: "必须是大于或等于 1 的整数" });
  }
}

function validateLengthDistribution(
  errors: ConfigValidationError[],
  field: "workload.synthetic.prompt_len" | "workload.synthetic.turn_increment_len" | "workload.synthetic.output_len",
  distribution: LengthDistribution,
): void {
  switch (distribution.dist) {
    case "fixed":
      validatePositiveInteger(errors, `${field}.value`, distribution.value);
      break;
    case "poisson":
      validateFiniteNumber(errors, `${field}.mean`, distribution.mean, 0, true);
      break;
    case "lognormal":
      validateFiniteNumber(errors, `${field}.mean`, distribution.mean, 0, true);
      validateFiniteNumber(errors, `${field}.sigma`, distribution.sigma, 0, true);
      break;
  }
}

export function validateSimConfig(config: SimConfig): ConfigValidationError[] {
  const errors: ConfigValidationError[] = [];
  const { workload, engine, perf_model: perfModel } = config;

  if (workload.mode !== "synthetic") {
    errors.push({ field: "workload.mode", message: "当前仅支持 synthetic 模式" });
  }

  const synthetic = workload.synthetic;
  if (workload.mode === "synthetic" && (synthetic === undefined || synthetic === null)) {
    errors.push({
      field: "workload.synthetic",
      message: "synthetic 模式必须提供负载配置",
    });
  } else if (synthetic !== undefined && synthetic !== null) {
    validatePositiveInteger(
      errors,
      "workload.synthetic.num_requests",
      synthetic.num_requests,
    );
    if (synthetic.arrival_mode === "open_loop") {
      validateFiniteNumber(errors, "workload.synthetic.arrival_rate", synthetic.arrival_rate ?? NaN, 0, true);
      if (!synthetic.turn_increment_len) {
        errors.push({ field: "workload.synthetic", message: "开环模式必须设置每轮新增长度" });
      } else {
        validateLengthDistribution(errors, "workload.synthetic.turn_increment_len", synthetic.turn_increment_len);
      }
    } else {
      const session = synthetic.session;
      validateFiniteNumber(errors, "workload.synthetic.session_rate", synthetic.session_rate ?? NaN, 0, true);
      validatePositiveInteger(errors, "workload.synthetic.session.num_sessions", session?.num_sessions ?? NaN);
      if (session?.num_sessions != null && session.num_sessions > synthetic.num_requests) {
        errors.push({ field: "workload.synthetic.session.num_sessions", message: "会话数量不能超过请求数量" });
      }
      validateFiniteNumber(errors, "workload.synthetic.session.think_time_ms", session?.think_time_ms ?? NaN, 0);
      const target = session?.target_ideal_prefix_hit_rate;
      if (target == null || !Number.isFinite(target) || target < 0 || target > 1) {
        errors.push({ field: "workload.synthetic.session.target_ideal_prefix_hit_rate",
          message: "理想命中率须在 0 到 1 之间" });
      }
    }
    validateLengthDistribution(
      errors,
      "workload.synthetic.prompt_len",
      synthetic.prompt_len,
    );
    validateLengthDistribution(
      errors,
      "workload.synthetic.output_len",
      synthetic.output_len,
    );
    validatePositiveInteger(errors, "workload.synthetic.max_new_tokens", synthetic.max_new_tokens);

    if (!Number.isInteger(synthetic.seed) || synthetic.seed < 0) {
      errors.push({
        field: "workload.synthetic.seed",
        message: "必须是大于或等于 0 的整数",
      });
    }
  }

  validatePositiveInteger(errors, "engine.max_running_requests", engine.max_running_requests);
  validatePositiveInteger(errors, "engine.max_total_tokens", engine.max_total_tokens);
  validatePositiveInteger(errors, "engine.max_prefill_tokens", engine.max_prefill_tokens);
  validatePositiveInteger(errors, "engine.page_size", engine.page_size);
  if (typeof engine.pd_mutually_exclusive !== "boolean") {
    errors.push({ field: "engine.pd_mutually_exclusive", message: "必须选择 P/D 执行模式" });
  }
  if (Number.isInteger(engine.max_total_tokens) && engine.max_total_tokens > 0 &&
      Number.isInteger(engine.page_size) && engine.page_size > 0 &&
      (engine.max_total_tokens < engine.page_size ||
       engine.max_total_tokens % engine.page_size !== 0)) {
    errors.push({
      field: "engine.max_total_tokens",
      message: "KV token 容量须至少为一页，且必须是页大小的整数倍",
    });
  }
  for (const tier of ["l2", "l3"] as const) {
    const field = `engine.${tier}_capacity_tokens` as const;
    const capacity = engine[`${tier}_capacity_tokens`];
    if (!Number.isInteger(capacity) || capacity < 0 ||
        (Number.isInteger(engine.page_size) && engine.page_size > 0 &&
         capacity % engine.page_size !== 0)) {
      errors.push({ field, message: "容量必须是页大小的非负整数倍；0 表示关闭该层" });
    }
  }
  const l2Enabled = engine.l2_capacity_tokens > 0;
  const l3Enabled = engine.l3_capacity_tokens > 0;
  if (l3Enabled && !l2Enabled) {
    errors.push({ field: "engine.l3_capacity_tokens", message: "启用 L3 前必须先启用 L2" });
  }

  validateFiniteNumber(errors, "perf_model.prefill_c1", perfModel.prefill_c1, 0);
  validateFiniteNumber(errors, "perf_model.prefill_c2", perfModel.prefill_c2, 0);
  validateFiniteNumber(errors, "perf_model.decode_d1", perfModel.decode_d1, 0);
  validateFiniteNumber(errors, "perf_model.decode_d2", perfModel.decode_d2, 0);

  if (perfModel.kv_bytes_per_token == null) {
    if (l2Enabled) {
      errors.push({ field: "perf_model.kv_bytes_per_token", message: "启用 L2 时必须填写正整数" });
    }
  } else {
    validatePositiveInteger(errors, "perf_model.kv_bytes_per_token", perfModel.kv_bytes_per_token);
  }
  for (const [field, value, required] of [
    ["perf_model.l2_to_l1_bandwidth_bytes_per_ms", perfModel.l2_to_l1_bandwidth_bytes_per_ms, l2Enabled],
    ["perf_model.l3_to_l2_bandwidth_bytes_per_ms", perfModel.l3_to_l2_bandwidth_bytes_per_ms, l3Enabled],
  ] as const) {
    if (value == null) {
      if (required) errors.push({ field, message: "启用该层时必须填写大于 0 的带宽" });
    } else {
      validateFiniteNumber(errors, field, value, 0, true);
    }
  }
  validateFiniteNumber(errors, "perf_model.l2_latency_ms", perfModel.l2_latency_ms, 0);
  validateFiniteNumber(errors, "perf_model.l3_latency_ms", perfModel.l3_latency_ms, 0);

  return errors;
}
