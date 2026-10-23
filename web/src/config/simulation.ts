import type { EmpiricalTimingConfig, InferenceConfig, LengthDistribution, SimConfig } from "../api/types";

export type FormSimConfig = SimConfig & {
  inference: InferenceConfig & { timing: NonNullable<InferenceConfig["timing"]> };
};

export type ClosedLoopDerivedField = "requests" | "sessions" | "turns";

export type ConfigField =
  | "turns_per_session"
  | "num_instances"
  | "routing_policy"
  | "workload.mode"
  | "workload.synthetic"
  | "workload.synthetic.num_requests"
  | "workload.arrival_mode"
  | "workload.target_qps"
  | "workload.session_rate"
  | "workload.seed"
  | "workload.synthetic.session.num_sessions"
  | "workload.synthetic.session.think_time_ms"
  | "workload.synthetic.prompt_len.value"
  | "workload.synthetic.prompt_len.mean"
  | "workload.synthetic.prompt_len.sigma"
  | "workload.synthetic.turn_increment_len.value"
  | "workload.synthetic.turn_increment_len.mean"
  | "workload.synthetic.turn_increment_len.sigma"
  | "workload.synthetic.output_len.value"
  | "workload.synthetic.output_len.mean"
  | "workload.synthetic.output_len.sigma"
  | "workload.trace"
  | "inference.max_output_tokens"
  | "scheduler.max_running_requests"
  | "scheduler.max_prefill_tokens"
  | "scheduler.l3_prefetch_min_tokens"
  | "scheduler.pd_mutually_exclusive"
  | "inference.page_size"
  | "shared_l3.available_gb"
  | "shared_l3.lookup_latency_ms"
  | "inference.timing.prefill_c1"
  | "inference.timing.prefill_c2"
  | "inference.timing.decode_d1"
  | "inference.timing.decode_d2";

export interface ConfigValidationError {
  field: ConfigField;
  message: string;
}

function createDefaultCommonConfig(): Pick<SimConfig, "num_instances" | "routing_policy" | "workload" | "scheduler" | "shared_l2_enabled"> {
  return {
    num_instances: 1,
    shared_l2_enabled: true,
    routing_policy: "least_load",
    workload: {
      mode: "synthetic",
      arrival_mode: "open_loop",
      target_qps: 64,
      seed: 0,
      synthetic: {
        num_requests: 1000,
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
      },
    },
    scheduler: {
      max_running_requests: 32,
      l3_prefetch_min_tokens: 256,
      max_prefill_tokens: 16_384,
      pd_mutually_exclusive: true,
    },
  };
}

export function createDefaultEmpiricalTiming(): EmpiricalTimingConfig {
  return { mode: "empirical", prefill_c1: 0.05, prefill_c2: 0.000_01, decode_d1: 8, decode_d2: 0.002 };
}

export function createDefaultSimConfig(): FormSimConfig {
  return {
    ...createDefaultCommonConfig(),
    shared_l3: { ...initialSharedL3 },
    inference: { page_size: 64, max_output_tokens: 65_536,
      model: structuredClone(initialModel), hardware: { ...initialHardware }, storage: { ...initialStorage },
      tp_size: 1, ep_size: 1, timing: { mode: "analytical" } },
  };
}

export const initialModel: InferenceConfig["model"] = {
  name: "Qwen/Qwen3-8B", revision: null, num_layers: 36, hidden_size: 4096,
  num_attention_heads: 32, num_kv_heads: 8, head_dim: 128,
  ffn_intermediate_size: 12288, vocab_size: 151_936, tie_word_embeddings: false,
  weight_dtype: "bf16", kv_dtype: "bf16", moe_layers: 0,
  num_experts: 0, experts_per_token: 0, expert_intermediate_size: null,
  num_shared_experts: 0, shared_expert_intermediate_size: null,
  attention_layout: "grouped", attention_groups: [{
    kind: "gqa", layers: 36, num_attention_heads: 32, num_kv_heads: 8, head_dim: 128,
    q_rank: null, kv_rank: null, rope_dim: null, nope_dim: null, value_dim: null,
    index_heads: null, index_dim: null, topk: null, full_indexer_layers: null,
    output_gate: false, qk_norm: true, norm_dtype: null,
    index_norm: "layer", index_norm_dtype: null, sink_dtype: null,
  }],
  norms: { kind: "rms", block_per_layer: 2, final_norm: true, dtype: null },
  router: { dtype: null, correction_bias: false, correction_dtype: "fp32" }, ihc: null,
};
export const initialHardware: InferenceConfig["hardware"] = {
  name: "NVIDIA B300", gpu_count: 1, memory_bytes_per_gpu: 288_000_000_000,
  effective_fp16_tflops: 100, effective_fp8_tflops: null,
  effective_hbm_gb_s: 1000, effective_interconnect_gb_s: 600,
  collective_latency_ms: 0.01, ep_communication_latency_ms: 0.02, kernel_overhead_ms: 0.005,
};
export const initialStorage: InferenceConfig["storage"] = {
  l1_available_gb: 200, l2_available_gb: 2048,
  runtime_reserve_bytes: 0, l2_to_l1_gb_s: 256, l3_to_l2_gb_s: 50,
  l2_latency_ms: 0.01, l3_latency_ms: 0.1,
};
export const initialSharedL3: NonNullable<SimConfig["shared_l3"]> = {
  available_gb: 100_000, lookup_latency_ms: 0.1,
};

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

export function resolveClosedLoopCounts(
  numRequests: number,
  numSessions: number,
  turnsPerSession: number,
  derivedField: ClosedLoopDerivedField,
) {
  const errors: ConfigValidationError[] = [];
  if (derivedField !== "requests") {
    validatePositiveInteger(errors, "workload.synthetic.num_requests", numRequests);
  }
  if (derivedField !== "sessions") {
    validatePositiveInteger(errors, "workload.synthetic.session.num_sessions", numSessions);
  }
  if (derivedField !== "turns") {
    validatePositiveInteger(errors, "turns_per_session", turnsPerSession);
  }

  if (derivedField === "requests") {
    numRequests = errors.length ? NaN : numSessions * turnsPerSession;
    if (!errors.length) validatePositiveInteger(errors, "workload.synthetic.num_requests", numRequests);
  } else if (derivedField === "sessions") {
    if (!errors.length && numRequests % turnsPerSession !== 0) {
      errors.push({ field: "workload.synthetic.num_requests", message: "请求数量必须是每会话轮次数的整数倍" });
    }
    numSessions = errors.length ? NaN : numRequests / turnsPerSession;
  } else if (!errors.length && numSessions > numRequests) {
    errors.push({ field: "workload.synthetic.session.num_sessions", message: "会话数量不能超过请求数量" });
  }

  const baseTurns = errors.length ? NaN : Math.floor(numRequests / numSessions);
  const turnsDisplay = errors.length ? "" : numRequests % numSessions === 0
    ? String(baseTurns) : `(${baseTurns} / ${baseTurns + 1})`;
  return { numRequests, numSessions, turnsPerSession: derivedField === "turns" ? baseTurns : turnsPerSession,
    turnsDisplay, errors };
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
  const { workload, scheduler, inference } = config;
  validatePositiveInteger(errors, "num_instances", config.num_instances);
  if (config.routing_policy !== undefined && config.routing_policy !== "cache_affinity" && config.routing_policy !== "least_load") {
    errors.push({ field: "routing_policy", message: "routing_policy 仅支持 cache_affinity 或 least_load" });
  }

  const synthetic = workload.synthetic;
  if (workload.mode === "synthetic" && (synthetic === undefined || synthetic === null)) {
    errors.push({
      field: "workload.synthetic",
      message: "synthetic 模式必须提供负载配置",
    });
  } else if (workload.mode === "synthetic" && synthetic !== undefined && synthetic !== null) {
    validatePositiveInteger(
      errors,
      "workload.synthetic.num_requests",
      synthetic.num_requests,
    );
    const session = synthetic.session;
    if ((workload.arrival_mode ?? "closed_loop") === "open_loop") {
      validateFiniteNumber(errors, "workload.target_qps", workload.target_qps ?? NaN, 0, true);
      if (workload.session_rate != null) errors.push({ field: "workload.session_rate", message: "合成开环不能设置会话投放频率" });
      if (session?.num_sessions != null) errors.push({ field: "workload.synthetic.session.num_sessions", message: "开环负载不能设置会话数量" });
      if ((session?.think_time_ms ?? 0) !== 0) errors.push({ field: "workload.synthetic.session.think_time_ms", message: "开环负载的思考时间必须为 0" });
    } else {
      validateFiniteNumber(errors, "workload.session_rate", workload.session_rate ?? NaN, 0, true);
      if (workload.target_qps != null) errors.push({ field: "workload.target_qps", message: "闭环负载不能设置目标 QPS" });
      if (session?.num_sessions != null) {
        validatePositiveInteger(errors, "workload.synthetic.session.num_sessions", session.num_sessions);
        if (session.num_sessions > synthetic.num_requests) {
          errors.push({ field: "workload.synthetic.session.num_sessions", message: "会话数量不能超过请求数量" });
        }
      }
      validateFiniteNumber(errors, "workload.synthetic.session.think_time_ms", session?.think_time_ms ?? 0, 0);
    }
    if (!synthetic.turn_increment_len) {
      errors.push({ field: "workload.synthetic", message: "必须设置子轮新增内容长度" });
    } else {
      validateLengthDistribution(errors, "workload.synthetic.turn_increment_len", synthetic.turn_increment_len);
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
  }

  if (workload.mode === "trace") {
    validateFiniteNumber(errors, "workload.session_rate", workload.session_rate ?? NaN, 0, true);
    if (workload.target_qps != null) {
      if ((workload.arrival_mode ?? "closed_loop") === "closed_loop") errors.push({ field: "workload.target_qps", message: "目标 QPS 仅适用于开环 trace 回放" });
      else validateFiniteNumber(errors, "workload.target_qps", workload.target_qps, 0, true);
    }
  }
  const seed = workload.seed ?? 0;
  if (!Number.isInteger(seed) || seed < 0) errors.push({ field: "workload.seed", message: "必须是大于或等于 0 的整数" });
  validatePositiveInteger(errors, "scheduler.max_running_requests", scheduler.max_running_requests);
  validatePositiveInteger(errors, "scheduler.max_prefill_tokens", scheduler.max_prefill_tokens);
  if (!Number.isInteger(scheduler.l3_prefetch_min_tokens) || scheduler.l3_prefetch_min_tokens < 0) {
    errors.push({ field: "scheduler.l3_prefetch_min_tokens", message: "必须是大于或等于 0 的整数" });
  }
  validatePositiveInteger(errors, "inference.page_size", inference.page_size);
  validatePositiveInteger(errors, "inference.max_output_tokens", inference.max_output_tokens ?? 65_536);
  if (typeof scheduler.pd_mutually_exclusive !== "boolean") {
    errors.push({ field: "scheduler.pd_mutually_exclusive", message: "必须选择 P/D 执行模式" });
  }
  const timing = inference.timing ?? { mode: "analytical" as const };
  if (timing.mode === "empirical") {
    for (const field of ["prefill_c1", "prefill_c2", "decode_d1", "decode_d2"] as const) {
      validateFiniteNumber(errors, `inference.timing.${field}`, timing[field], 0);
    }
  }
  const shared = config.shared_l3;
  if (shared != null) {
    validateFiniteNumber(errors, "shared_l3.lookup_latency_ms", shared.lookup_latency_ms ?? 0, 0);
    validateFiniteNumber(errors, "shared_l3.available_gb", shared.available_gb ?? 0, 0);
    if ((shared.available_gb ?? 0) > 0 &&
        (!(inference.storage.l2_available_gb! > 0) || !(inference.storage.l3_to_l2_gb_s! > 0))) {
      errors.push({ field: "shared_l3.available_gb", message: "启用共享 L3 须有 L2 和正的 L3 → L2 带宽" });
    }
  }
  return errors;
}
