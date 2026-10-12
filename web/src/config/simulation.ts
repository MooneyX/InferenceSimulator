import type { LengthDistribution, ManualInferenceConfig, PhysicalInferenceConfig, SimConfig } from "../api/types";

export type ManualSimConfig = SimConfig & { inference: ManualInferenceConfig };
export type PhysicalSimConfig = SimConfig & { inference: PhysicalInferenceConfig & {
  timing: NonNullable<PhysicalInferenceConfig["timing"]>;
} };

export type ConfigField =
  | "num_instances"
  | "workload.mode"
  | "workload.synthetic"
  | "workload.synthetic.num_requests"
  | "workload.synthetic.arrival_rate"
  | "workload.synthetic.session_rate"
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
  | "workload.synthetic.seed"
  | "workload.synthetic.max_new_tokens"
  | "workload.trace"
  | "workload.trace.session_rate"
  | "workload.trace.seed"
  | "workload.trace.target_qps"
  | "scheduler.max_running_requests"
  | "scheduler.max_prefill_tokens"
  | "scheduler.l3_prefetch_min_tokens"
  | "scheduler.pd_mutually_exclusive"
  | "inference.page_size"
  | "inference.kv.max_total_tokens"
  | "inference.kv.l2_capacity_tokens"
  | "inference.kv.l3_capacity_tokens"
  | "inference.kv.kv_bytes_per_token"
  | "inference.kv.l2_to_l1_bandwidth_bytes_per_ms"
  | "inference.kv.l3_to_l2_bandwidth_bytes_per_ms"
  | "inference.kv.l2_latency_ms"
  | "inference.kv.l3_latency_ms"
  | "inference.kv.l3_lookup_latency_ms"
  | "inference.timing.prefill_c1"
  | "inference.timing.prefill_c2"
  | "inference.timing.decode_d1"
  | "inference.timing.decode_d2";

export interface ConfigValidationError {
  field: ConfigField;
  message: string;
}

export function createDefaultSimConfig(): ManualSimConfig {
  return {
    num_instances: 1,
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
    scheduler: {
      max_running_requests: 32,
      l3_prefetch_min_tokens: 256,
      max_prefill_tokens: 16_384,
      pd_mutually_exclusive: true,
    },
    inference: {
      mode: "manual",
      page_size: 64,
      kv: {
        max_total_tokens: 65_536, l2_capacity_tokens: 0, l3_capacity_tokens: 0,
        kv_bytes_per_token: null,
        l2_to_l1_bandwidth_bytes_per_ms: null,
        l3_to_l2_bandwidth_bytes_per_ms: null,
        l2_latency_ms: 0, l3_latency_ms: 0, l3_lookup_latency_ms: 0,
      },
      timing: createDefaultEmpiricalTiming(),
    },
  };
}

export function createDefaultEmpiricalTiming(): ManualInferenceConfig["timing"] {
  return { mode: "empirical", prefill_c1: 0.05, prefill_c2: 0.000_01, decode_d1: 8, decode_d2: 0.002 };
}

export function createDefaultPhysicalSimConfig(): PhysicalSimConfig {
  const manual = createDefaultSimConfig();
  return { num_instances: manual.num_instances, workload: manual.workload, scheduler: manual.scheduler,
    inference: { mode: "physical", page_size: manual.inference.page_size,
      model: structuredClone(initialModel), hardware: { ...initialHardware }, storage: { ...initialStorage },
      tp_size: 1, ep_size: 1, timing: { mode: "analytical" } } };
}

export const initialModel: PhysicalInferenceConfig["model"] = {
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
export const initialHardware: PhysicalInferenceConfig["hardware"] = {
  name: "NVIDIA B300", gpu_count: 1, memory_bytes_per_gpu: 288_000_000_000,
  effective_fp16_tflops: 100, effective_fp8_tflops: null,
  effective_hbm_gb_s: 1000, effective_interconnect_gb_s: 600,
  collective_latency_ms: 0.01, ep_communication_latency_ms: 0.02, kernel_overhead_ms: 0,
};
export const initialStorage: PhysicalInferenceConfig["storage"] = {
  l1_available_gb: 200, l2_available_gb: 2048, l3_available_gb: 40_000,
  runtime_reserve_bytes: 0, l2_to_l1_gb_s: 256, l3_to_l2_gb_s: 50,
  l2_latency_ms: 0, l3_latency_ms: 0, l3_lookup_latency_ms: 0,
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
    if (synthetic.arrival_mode === "open_loop") {
      validateFiniteNumber(errors, "workload.synthetic.arrival_rate", synthetic.arrival_rate ?? NaN, 0, true);
    } else {
      const session = synthetic.session;
      validateFiniteNumber(errors, "workload.synthetic.session_rate", synthetic.session_rate ?? NaN, 0, true);
      validatePositiveInteger(errors, "workload.synthetic.session.num_sessions", session?.num_sessions ?? NaN);
      if (session?.num_sessions != null && session.num_sessions > synthetic.num_requests) {
        errors.push({ field: "workload.synthetic.session.num_sessions", message: "会话数量不能超过请求数量" });
      }
      validateFiniteNumber(errors, "workload.synthetic.session.think_time_ms", session?.think_time_ms ?? NaN, 0);
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
    validatePositiveInteger(errors, "workload.synthetic.max_new_tokens", synthetic.max_new_tokens);

    if (!Number.isInteger(synthetic.seed) || synthetic.seed < 0) {
      errors.push({
        field: "workload.synthetic.seed",
        message: "必须是大于或等于 0 的整数",
      });
    }
  }

  if (workload.mode === "trace") {
    const trace = workload.trace;
    if (!trace) errors.push({ field: "workload.trace", message: "trace 模式必须提供回放配置" });
    else {
      validateFiniteNumber(errors, "workload.trace.session_rate", trace.session_rate, 0, true);
      if (!Number.isInteger(trace.seed) || trace.seed < 0) errors.push({ field: "workload.trace.seed", message: "必须是大于或等于 0 的整数" });
      if (trace.target_qps != null) validateFiniteNumber(errors, "workload.trace.target_qps", trace.target_qps, 0, true);
    }
  }
  validatePositiveInteger(errors, "scheduler.max_running_requests", scheduler.max_running_requests);
  validatePositiveInteger(errors, "scheduler.max_prefill_tokens", scheduler.max_prefill_tokens);
  if (!Number.isInteger(scheduler.l3_prefetch_min_tokens) || scheduler.l3_prefetch_min_tokens < 0) {
    errors.push({ field: "scheduler.l3_prefetch_min_tokens", message: "必须是大于或等于 0 的整数" });
  }
  validatePositiveInteger(errors, "inference.page_size", inference.page_size);
  if (typeof scheduler.pd_mutually_exclusive !== "boolean") {
    errors.push({ field: "scheduler.pd_mutually_exclusive", message: "必须选择 P/D 执行模式" });
  }
  const timing = inference.timing ?? { mode: "analytical" as const };
  if (timing.mode === "empirical") {
    for (const field of ["prefill_c1", "prefill_c2", "decode_d1", "decode_d2"] as const) {
      validateFiniteNumber(errors, `inference.timing.${field}`, timing[field], 0);
    }
  }
  if (inference.mode === "physical") return errors;
  const { kv, page_size: pageSize } = inference;
  validatePositiveInteger(errors, "inference.kv.max_total_tokens", kv.max_total_tokens);
  if (Number.isInteger(kv.max_total_tokens) && kv.max_total_tokens > 0 &&
      Number.isInteger(pageSize) && pageSize > 0 &&
      (kv.max_total_tokens < pageSize || kv.max_total_tokens % pageSize !== 0)) {
    errors.push({ field: "inference.kv.max_total_tokens", message: "KV token 容量须至少为一页，且必须是页大小的整数倍" });
  }
  for (const tier of ["l2", "l3"] as const) {
    const capacity = kv[`${tier}_capacity_tokens`];
    if (!Number.isInteger(capacity) || capacity < 0 ||
        (Number.isInteger(pageSize) && pageSize > 0 && capacity % pageSize !== 0)) {
      errors.push({ field: `inference.kv.${tier}_capacity_tokens`, message: "容量必须是页大小的非负整数倍；0 表示关闭该层" });
    }
  }
  const l2Enabled = kv.l2_capacity_tokens > 0;
  const l3Enabled = kv.l3_capacity_tokens > 0;
  if (l3Enabled && !l2Enabled) errors.push({ field: "inference.kv.l3_capacity_tokens", message: "启用 L3 前必须先启用 L2" });
  if (kv.kv_bytes_per_token == null) {
    if (l2Enabled) errors.push({ field: "inference.kv.kv_bytes_per_token", message: "启用 L2 时必须填写正整数" });
  } else validatePositiveInteger(errors, "inference.kv.kv_bytes_per_token", kv.kv_bytes_per_token);
  for (const [field, value, required] of [
    ["inference.kv.l2_to_l1_bandwidth_bytes_per_ms", kv.l2_to_l1_bandwidth_bytes_per_ms, l2Enabled],
    ["inference.kv.l3_to_l2_bandwidth_bytes_per_ms", kv.l3_to_l2_bandwidth_bytes_per_ms, l3Enabled],
  ] as const) {
    if (value == null) {
      if (required) errors.push({ field, message: "启用该层时必须填写大于 0 的带宽" });
    } else validateFiniteNumber(errors, field, value, 0, true);
  }
  for (const field of ["l2_latency_ms", "l3_latency_ms", "l3_lookup_latency_ms"] as const) {
    validateFiniteNumber(errors, `inference.kv.${field}`, kv[field], 0);
  }

  return errors;
}
