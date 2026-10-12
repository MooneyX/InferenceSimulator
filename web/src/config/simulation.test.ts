import { describe, expect, it } from "vitest";

import type { LengthDistribution, SimConfig } from "../api/types";
import { createDefaultSimConfig, createDefaultPhysicalSimConfig, createDefaultEmpiricalTiming, validateSimConfig } from "./simulation";

describe("createDefaultSimConfig", () => {
  it("creates a complete synthetic configuration", () => {
    expect(createDefaultSimConfig()).toEqual({
      num_instances: 1,
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
      scheduler: { max_running_requests: 32, max_prefill_tokens: 16_384,
        l3_prefetch_min_tokens: 256, pd_mutually_exclusive: true },
      inference: { mode: "manual", page_size: 64,
        kv: { max_total_tokens: 65_536, l2_capacity_tokens: 0, l3_capacity_tokens: 0,
          kv_bytes_per_token: null, l2_to_l1_bandwidth_bytes_per_ms: null,
          l3_to_l2_bandwidth_bytes_per_ms: null, l2_latency_ms: 0, l3_latency_ms: 0,
          l3_lookup_latency_ms: 0 },
        timing: { mode: "empirical", prefill_c1: 0.05, prefill_c2: 0.000_01, decode_d1: 8, decode_d2: 0.002 },
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
    expect(first.scheduler).not.toBe(second.scheduler);
    expect(first.inference.kv).not.toBe(second.inference.kv);
  });
});

describe("validateSimConfig", () => {
  it("uses physical specifications without manual capacity or analytical coefficient placeholders", () => {
    const config = createDefaultPhysicalSimConfig();
    expect(config.inference).not.toHaveProperty("kv");
    expect(config.inference.timing).toEqual({ mode: "analytical" });
    expect(validateSimConfig(config)).toEqual([]);
    config.inference.timing = { ...createDefaultEmpiricalTiming(), prefill_c1: NaN };
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(["inference.timing.prefill_c1"]);
  });

  it("validates trace replay settings even when the plan must be reuploaded", () => {
    const config = createDefaultSimConfig();
    config.workload = { mode: "trace", trace: { mode: "open_loop", session_rate: 0, seed: -1, target_qps: 0, plan_id: null } };
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.trace.session_rate", "workload.trace.seed", "workload.trace.target_qps",
    ]);
  });

  it("accepts the default configuration", () => {
    expect(validateSimConfig(createDefaultSimConfig())).toEqual([]);
  });

  it("accepts a synthetic request without an optional session", () => {
    const config = createDefaultSimConfig();
    delete config.workload.synthetic!.session;
    expect(validateSimConfig(config)).toEqual([]);
  });

  it("validates trace engine settings without requiring synthetic distributions", () => {
    const config = createDefaultSimConfig();
    config.workload = { mode: "trace", trace: { mode: "open_loop", session_rate: 1, seed: 0 } };
    expect(validateSimConfig(config)).toEqual([]);
    config.inference.page_size = 0;
    expect(validateSimConfig(config).some(({ field }) => field === "inference.page_size")).toBe(true);
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
      turn_increment_len: { dist: "fixed", value: 32 }, session_rate: 1.28,
      session: { num_sessions: 20, topology: "linear", think_time_ms: 200 },
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
    config.inference.kv.max_total_tokens = 128;
    expect(validateSimConfig(config)).toEqual([]);
    config.inference.kv.max_total_tokens = 130;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "inference.kv.max_total_tokens",
    ]);
    config.inference.kv.max_total_tokens = 63;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "inference.kv.max_total_tokens",
    ]);
    config.inference.kv.max_total_tokens = 128;
    config.inference.page_size = 1;
    config.scheduler.pd_mutually_exclusive = false;
    expect(validateSimConfig(config)).toEqual([]);
  });

  it("requires whole-page L2/L3 capacities and L2 before L3", () => {
    const config = createDefaultSimConfig();
    config.inference.kv.l2_capacity_tokens = 65;
    config.inference.kv.l3_capacity_tokens = 130;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "inference.kv.l2_capacity_tokens",
      "inference.kv.l3_capacity_tokens",
      "inference.kv.kv_bytes_per_token",
      "inference.kv.l2_to_l1_bandwidth_bytes_per_ms",
      "inference.kv.l3_to_l2_bandwidth_bytes_per_ms",
    ]);
    config.inference.kv.l2_capacity_tokens = 0;
    config.inference.kv.l3_capacity_tokens = 128;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "inference.kv.l3_capacity_tokens",
      "inference.kv.l3_to_l2_bandwidth_bytes_per_ms",
    ]);
  });

  it("validates only enabled tier links as required and accepts a complete three-tier config", () => {
    const config = createDefaultSimConfig();
    config.inference.kv.l2_capacity_tokens = 128;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "inference.kv.kv_bytes_per_token",
      "inference.kv.l2_to_l1_bandwidth_bytes_per_ms",
    ]);
    config.inference.kv.kv_bytes_per_token = 4096;
    config.inference.kv.l2_to_l1_bandwidth_bytes_per_ms = 1024;
    expect(validateSimConfig(config)).toEqual([]);
    config.inference.kv.l3_capacity_tokens = 64;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "inference.kv.l3_to_l2_bandwidth_bytes_per_ms",
    ]);
    config.inference.kv.l3_to_l2_bandwidth_bytes_per_ms = 128;
    config.inference.kv.l2_latency_ms = 0.5;
    config.inference.kv.l3_latency_ms = 1;
    config.inference.kv.l3_lookup_latency_ms = 2.5;
    config.scheduler.l3_prefetch_min_tokens = 0;
    expect(validateSimConfig(config)).toEqual([]);
    config.inference.kv.kv_bytes_per_token = 1.5;
    config.inference.kv.l2_to_l1_bandwidth_bytes_per_ms = 0;
    config.inference.kv.l3_latency_ms = -1;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "inference.kv.kv_bytes_per_token",
      "inference.kv.l2_to_l1_bandwidth_bytes_per_ms",
      "inference.kv.l3_latency_ms",
    ]);
  });

  it("accepts trace mode independently of any synthetic payload", () => {
    const config: SimConfig = {
      ...createDefaultSimConfig(),
      workload: { mode: "trace", trace: { mode: "session", session_rate: 2, seed: 42 } },
    };
    expect(validateSimConfig(config)).toEqual([]);
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
    config.scheduler.max_running_requests = 0;
    config.inference.kv.max_total_tokens = 2.5;
    config.scheduler.max_prefill_tokens = 0;
    config.scheduler.l3_prefetch_min_tokens = -1;
    config.inference.page_size = 0;
    config.scheduler.pd_mutually_exclusive = null as unknown as boolean;
    config.inference.timing.prefill_c1 = -0.1;
    config.inference.timing.prefill_c2 = Number.POSITIVE_INFINITY;
    config.inference.timing.decode_d1 = -1;
    config.inference.timing.decode_d2 = Number.NaN;
    config.inference.kv.l3_lookup_latency_ms = -1;

    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.synthetic.num_requests",
      "workload.synthetic.arrival_rate",
      "workload.synthetic.max_new_tokens",
      "workload.synthetic.seed",
      "scheduler.max_running_requests",
      "scheduler.max_prefill_tokens",
      "scheduler.l3_prefetch_min_tokens",
      "inference.page_size",
      "scheduler.pd_mutually_exclusive",
      "inference.timing.prefill_c1",
      "inference.timing.prefill_c2",
      "inference.timing.decode_d1",
      "inference.timing.decode_d2",
      "inference.kv.max_total_tokens",
      "inference.kv.l3_lookup_latency_ms",
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
