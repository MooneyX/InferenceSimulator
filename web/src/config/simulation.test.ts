import { describe, expect, it } from "vitest";

import type { LengthDistribution, SimConfig } from "../api/types";
import { createDefaultSimConfig, validateSimConfig } from "./simulation";

describe("createDefaultSimConfig", () => {
  it("creates a complete synthetic configuration", () => {
    expect(createDefaultSimConfig()).toEqual({
      workload: {
        mode: "synthetic",
        synthetic: {
          num_requests: 1000,
          arrival_rate: 64,
          arrival_mode: "open_loop",
          prompt_len: { dist: "poisson", mean: 2048 },
          turn_increment_len: { dist: "poisson", mean: 256 },
          output_len: { dist: "poisson", mean: 128 },
          max_new_tokens: 1024,
          seed: 42,
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
    });
    expect(createDefaultSimConfig().workload.synthetic).not.toHaveProperty("session");
  });

  it("returns independent objects", () => {
    const first = createDefaultSimConfig();
    const second = createDefaultSimConfig();

    expect(first).not.toBe(second);
    expect(first.workload.synthetic).not.toBe(second.workload.synthetic);
    expect(first.workload.synthetic?.prompt_len).not.toBe(
      second.workload.synthetic?.prompt_len,
    );
    expect(first.workload.synthetic?.turn_increment_len).not.toBe(second.workload.synthetic?.turn_increment_len);
    expect(first.engine).not.toBe(second.engine);
  });
});

describe("validateSimConfig", () => {
  it("accepts the default configuration", () => {
    expect(validateSimConfig(createDefaultSimConfig())).toEqual([]);
  });

  it("accepts a synthetic request without an optional session", () => {
    const config = createDefaultSimConfig();
    delete config.workload.synthetic!.session;
    expect(validateSimConfig(config)).toEqual([]);
  });

  it("validates the child increment length distribution", () => {
    const config = createDefaultSimConfig();
    config.workload.synthetic!.turn_increment_len = { dist: "fixed", value: 0 };
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.synthetic.turn_increment_len.value",
    ]);
  });

  it("validates closed-loop session inputs without requiring a request QPS", () => {
    const config = createDefaultSimConfig();
    config.workload.synthetic = {
      ...config.workload.synthetic!, arrival_mode: "session", arrival_rate: undefined,
      turn_increment_len: undefined, session_rate: 1.28,
      session: { num_sessions: 20, topology: "linear", think_time_ms: 200,
        target_ideal_prefix_hit_rate: 0 },
    };
    expect(validateSimConfig(config)).toEqual([]);
    config.workload.synthetic.session_rate = 0;
    config.workload.synthetic.session!.num_sessions = 1001;
    config.workload.synthetic.session!.think_time_ms = -1;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.synthetic.session_rate", "workload.synthetic.session.num_sessions",
      "workload.synthetic.session.think_time_ms",
    ]);
  });

  it("rejects non-page capacity rather than rounding it down", () => {
    const config = createDefaultSimConfig();
    config.engine.max_total_tokens = 128;
    expect(validateSimConfig(config)).toEqual([]);
    config.engine.max_total_tokens = 130;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "engine.max_total_tokens",
    ]);
    config.engine.max_total_tokens = 63;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "engine.max_total_tokens",
    ]);
    config.engine.max_total_tokens = 128;
    config.engine.page_size = 1;
    config.engine.pd_mutually_exclusive = false;
    expect(validateSimConfig(config)).toEqual([]);
  });

  it("requires whole-page L2/L3 capacities and L2 before L3", () => {
    const config = createDefaultSimConfig();
    config.engine.l2_capacity_tokens = 65;
    config.engine.l3_capacity_tokens = 130;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "engine.l2_capacity_tokens",
      "engine.l3_capacity_tokens",
      "perf_model.kv_bytes_per_token",
      "perf_model.l2_to_l1_bandwidth_bytes_per_ms",
      "perf_model.l3_to_l2_bandwidth_bytes_per_ms",
    ]);
    config.engine.l2_capacity_tokens = 0;
    config.engine.l3_capacity_tokens = 128;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "engine.l3_capacity_tokens",
      "perf_model.l3_to_l2_bandwidth_bytes_per_ms",
    ]);
  });

  it("validates only enabled tier links as required and accepts a complete three-tier config", () => {
    const config = createDefaultSimConfig();
    config.engine.l2_capacity_tokens = 128;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "perf_model.kv_bytes_per_token",
      "perf_model.l2_to_l1_bandwidth_bytes_per_ms",
    ]);
    config.perf_model.kv_bytes_per_token = 4096;
    config.perf_model.l2_to_l1_bandwidth_bytes_per_ms = 1024;
    expect(validateSimConfig(config)).toEqual([]);
    config.engine.l3_capacity_tokens = 64;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "perf_model.l3_to_l2_bandwidth_bytes_per_ms",
    ]);
    config.perf_model.l3_to_l2_bandwidth_bytes_per_ms = 128;
    config.perf_model.l2_latency_ms = 0.5;
    config.perf_model.l3_latency_ms = 1;
    expect(validateSimConfig(config)).toEqual([]);
    config.perf_model.kv_bytes_per_token = 1.5;
    config.perf_model.l2_to_l1_bandwidth_bytes_per_ms = 0;
    config.perf_model.l3_latency_ms = -1;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "perf_model.kv_bytes_per_token",
      "perf_model.l2_to_l1_bandwidth_bytes_per_ms",
      "perf_model.l3_latency_ms",
    ]);
  });

  it("rejects the unsupported trace mode", () => {
    const config: SimConfig = {
      ...createDefaultSimConfig(),
      workload: { mode: "trace", synthetic: null },
    };

    expect(validateSimConfig(config)).toEqual([
      { field: "workload.mode", message: "当前仅支持 synthetic 模式" },
    ]);
  });

  it("requires a payload in synthetic mode", () => {
    const config: SimConfig = {
      ...createDefaultSimConfig(),
      workload: { mode: "synthetic", synthetic: null },
    };

    expect(validateSimConfig(config)).toEqual([
      {
        field: "workload.synthetic",
        message: "synthetic 模式必须提供负载配置",
      },
    ]);
  });

  it("validates integer and finite numeric constraints", () => {
    const config = createDefaultSimConfig();
    const synthetic = config.workload.synthetic!;
    synthetic.num_requests = 1.5;
    synthetic.arrival_rate = Number.NaN;
    synthetic.seed = -1;
    synthetic.max_new_tokens = 0;
    config.engine.max_running_requests = 0;
    config.engine.max_total_tokens = 2.5;
    config.engine.max_prefill_tokens = 0;
    config.engine.page_size = 0;
    config.engine.pd_mutually_exclusive = null as unknown as boolean;
    config.perf_model.prefill_c1 = -0.1;
    config.perf_model.prefill_c2 = Number.POSITIVE_INFINITY;
    config.perf_model.decode_d1 = -1;
    config.perf_model.decode_d2 = Number.NaN;

    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.synthetic.num_requests",
      "workload.synthetic.arrival_rate",
      "workload.synthetic.max_new_tokens",
      "workload.synthetic.seed",
      "engine.max_running_requests",
      "engine.max_total_tokens",
      "engine.max_prefill_tokens",
      "engine.page_size",
      "engine.pd_mutually_exclusive",
      "perf_model.prefill_c1",
      "perf_model.prefill_c2",
      "perf_model.decode_d1",
      "perf_model.decode_d2",
    ]);
  });

  it.each<{
    distribution: LengthDistribution;
    invalidFields: string[];
  }>([
    {
      distribution: { dist: "fixed", value: 0 },
      invalidFields: ["workload.synthetic.prompt_len.value"],
    },
    {
      distribution: { dist: "poisson", mean: 0 },
      invalidFields: ["workload.synthetic.prompt_len.mean"],
    },
    {
      distribution: { dist: "lognormal", mean: -1, sigma: 0 },
      invalidFields: [
        "workload.synthetic.prompt_len.mean",
        "workload.synthetic.prompt_len.sigma",
      ],
    },
  ])("validates $distribution.dist length parameters", ({ distribution, invalidFields }) => {
    const config = createDefaultSimConfig();
    config.workload.synthetic!.prompt_len = distribution;

    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(invalidFields);
  });
});
