import type { LengthDistribution, ManualInferenceConfig, PhysicalInferenceConfig, SimConfig } from "../api/types";

export type ManualSimConfig = SimConfig & {
  shared_l3: Extract<NonNullable<SimConfig["shared_l3"]>, { mode: "manual" }> | null;
  inference: ManualInferenceConfig;
};
export type PhysicalSimConfig = SimConfig & {
  shared_l3: Extract<NonNullable<SimConfig["shared_l3"]>, { mode: "physical" }> | null;
  inference: PhysicalInferenceConfig & { timing: NonNullable<PhysicalInferenceConfig["timing"]> };
};

export type ConfigField =
  | "num_instances"
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
  | "inference.kv.max_total_tokens"
  | "inference.kv.l2_capacity_tokens"
  | "shared_l3.capacity_tokens"
  | "shared_l3.available_gb"
  | "shared_l3.lookup_latency_ms"
  | "inference.kv.kv_bytes_per_token"
  | "inference.kv.l2_to_l1_bandwidth_bytes_per_ms"
  | "inference.kv.l3_to_l2_bandwidth_bytes_per_ms"
  | "inference.kv.l2_latency_ms"
  | "inference.kv.l3_latency_ms"

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
    shared_l3: { mode: "manual", capacity_tokens: 0, lookup_latency_ms: 0 },
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
    inference: {
      mode: "manual",
      page_size: 64,
      max_output_tokens: 65_536,
      kv: {
        max_total_tokens: 65_536, l2_capacity_tokens: 0,
        kv_bytes_per_token: null,
        l2_to_l1_bandwidth_bytes_per_ms: null,
        l3_to_l2_bandwidth_bytes_per_ms: null,
        l2_latency_ms: 0, l3_latency_ms: 0,
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
    shared_l3: { mode: "physical", available_gb: 40_000, lookup_latency_ms: 0 },
    inference: { mode: "physical", page_size: manual.inference.page_size,
      max_output_tokens: manual.inference.max_output_tokens,
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
  l1_available_gb: 200, l2_available_gb: 2048,
  runtime_reserve_bytes: 0, l2_to_l1_gb_s: 256, l3_to_l2_gb_s: 50,
  l2_latency_ms: 0, l3_latency_ms: 0,
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
    if (shared.mode !== inference.mode) errors.push({ field: "shared_l3.capacity_tokens", message: "共享 L3 配置方式须与推理配置一致" });
    validateFiniteNumber(errors, "shared_l3.lookup_latency_ms", shared.lookup_latency_ms ?? 0, 0);
    if (shared.mode === "physical") {
      validateFiniteNumber(errors, "shared_l3.available_gb", shared.available_gb ?? 0, 0);
      if (inference.mode === "physical" && (shared.available_gb ?? 0) > 0 &&
          (!(inference.storage.l2_available_gb! > 0) || !(inference.storage.l3_to_l2_gb_s! > 0))) {
        errors.push({ field: "shared_l3.available_gb", message: "启用共享 L3 须有 L2 和正的 L3 → L2 带宽" });
      }
    } else {
      const capacity = shared.capacity_tokens ?? 0;
      if (!Number.isInteger(capacity) || capacity < 0 ||
          (inference.page_size > 0 && capacity % inference.page_size !== 0)) {
        errors.push({ field: "shared_l3.capacity_tokens", message: "容量必须是页大小的非负整数倍；0 表示关闭" });
      }
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
  const capacity = kv.l2_capacity_tokens;
  if (!Number.isInteger(capacity) || capacity < 0 || (pageSize > 0 && capacity % pageSize !== 0)) {
    errors.push({ field: "inference.kv.l2_capacity_tokens", message: "容量必须是页大小的非负整数倍；0 表示关闭该层" });
  }
  const l2Enabled = kv.l2_capacity_tokens > 0;
  const l3Enabled = shared?.mode === "manual" && (shared.capacity_tokens ?? 0) > 0;
  if (l3Enabled && !l2Enabled) errors.push({ field: "shared_l3.capacity_tokens", message: "启用 L3 前必须先启用 L2" });
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
  for (const field of ["l2_latency_ms", "l3_latency_ms"] as const) {
    validateFiniteNumber(errors, `inference.kv.${field}`, kv[field], 0);
  }

  return errors;
}
