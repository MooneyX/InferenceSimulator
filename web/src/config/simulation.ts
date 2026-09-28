import type { LengthDistribution, SimConfig } from "../api/types";

export type ConfigField =
  | "workload.mode"
  | "workload.synthetic"
  | "workload.synthetic.num_requests"
  | "workload.synthetic.arrival_rate"
  | "workload.synthetic.prompt_len.value"
  | "workload.synthetic.prompt_len.mean"
  | "workload.synthetic.prompt_len.sigma"
  | "workload.synthetic.output_len.value"
  | "workload.synthetic.output_len.mean"
  | "workload.synthetic.output_len.sigma"
  | "workload.synthetic.seed"
  | "engine.max_batch_size"
  | "engine.kv_token_budget"
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
        arrival_rate: 50,
        prompt_len: {
          dist: "lognormal",
          mean: 512,
          sigma: 1,
        },
        output_len: {
          dist: "poisson",
          mean: 128,
        },
        seed: 0,
      },
    },
    engine: {
      max_batch_size: 32,
      kv_token_budget: 65_536,
    },
    perf_model: {
      prefill_c1: 0.05,
      prefill_c2: 0.000_01,
      decode_d1: 8,
      decode_d2: 0.002,
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
  field: "workload.synthetic.prompt_len" | "workload.synthetic.output_len",
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
    validateFiniteNumber(
      errors,
      "workload.synthetic.arrival_rate",
      synthetic.arrival_rate,
      0,
      true,
    );
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

    if (!Number.isInteger(synthetic.seed) || synthetic.seed < 0) {
      errors.push({
        field: "workload.synthetic.seed",
        message: "必须是大于或等于 0 的整数",
      });
    }
  }

  validatePositiveInteger(errors, "engine.max_batch_size", engine.max_batch_size);
  validatePositiveInteger(errors, "engine.kv_token_budget", engine.kv_token_budget);
  validateFiniteNumber(errors, "perf_model.prefill_c1", perfModel.prefill_c1, 0);
  validateFiniteNumber(errors, "perf_model.prefill_c2", perfModel.prefill_c2, 0);
  validateFiniteNumber(errors, "perf_model.decode_d1", perfModel.decode_d1, 0);
  validateFiniteNumber(errors, "perf_model.decode_d2", perfModel.decode_d2, 0);

  return errors;
}
