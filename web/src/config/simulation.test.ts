import { describe, expect, it } from "vitest";

import type { LengthDistribution, SimConfig } from "../api/types";
import { createDefaultSimConfig, createDefaultEmpiricalTiming, validateSimConfig } from "./simulation";

describe("createDefaultSimConfig", () => {
  it("creates a complete synthetic configuration", () => {
    expect(createDefaultSimConfig()).toEqual({
      num_instances: 1,
      workload: {
        mode: "synthetic",
        arrival_mode: "open_loop",
        target_qps: 64,
        seed: 0,
        synthetic: {
          num_requests: 1000,
          prompt_len: { dist: "poisson", mean: 2048 },
          turn_increment_len: { dist: "poisson", mean: 256 },
          output_len: { dist: "poisson", mean: 128 },
        },
      },
      scheduler: { max_running_requests: 32, max_prefill_tokens: 16_384,
        l3_prefetch_min_tokens: 256, pd_mutually_exclusive: true },
      shared_l3: { mode: "physical", available_gb: 40_000, lookup_latency_ms: 0 },
      inference: { mode: "physical", page_size: 64, max_output_tokens: 65_536,
        model: expect.any(Object), hardware: expect.any(Object), storage: expect.any(Object),
        tp_size: 1, ep_size: 1, timing: { mode: "analytical" },
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
    expect(first.inference.model).not.toBe(second.inference.model);
    expect(first.inference.model.attention_groups).not.toBe(second.inference.model.attention_groups);
    expect(first.inference.hardware).not.toBe(second.inference.hardware);
    expect(first.inference.storage).not.toBe(second.inference.storage);
  });
});

describe("validateSimConfig", () => {
  it("uses physical specifications without manual capacity or analytical coefficient placeholders", () => {
    const config = createDefaultSimConfig();
    expect(config.inference).not.toHaveProperty("kv");
    expect(config.inference.timing).toEqual({ mode: "analytical" });
    expect(validateSimConfig(config)).toEqual([]);
    config.inference.timing = { ...createDefaultEmpiricalTiming(), prefill_c1: NaN };
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(["inference.timing.prefill_c1"]);
  });

  it("validates trace replay settings even when the plan must be reuploaded", () => {
    const config = createDefaultSimConfig();
    config.workload = { mode: "trace", arrival_mode: "open_loop", session_rate: 0,
      seed: -1, target_qps: 0, trace: { plan_id: null } };
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.session_rate", "workload.target_qps", "workload.seed",
    ]);
  });

  it("accepts the default configuration", () => {
    expect(validateSimConfig(createDefaultSimConfig())).toEqual([]);
  });

  it("enforces synthetic open-loop rate and session exclusions", () => {
    const config = createDefaultSimConfig();
    config.workload.target_qps = null;
    config.workload.session_rate = 1;
    config.workload.synthetic!.session = { num_sessions: 2, topology: "linear", think_time_ms: 1 };
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.target_qps", "workload.session_rate", "workload.synthetic.session.num_sessions",
      "workload.synthetic.session.think_time_ms",
    ]);
  });

  it("allows closed-loop default session counts but rejects target QPS", () => {
    const config = createDefaultSimConfig();
    config.workload.arrival_mode = "closed_loop";
    config.workload.session_rate = 1;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(["workload.target_qps"]);
    delete config.workload.target_qps;
    expect(validateSimConfig(config)).toEqual([]);
    config.workload.session_rate = null;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(["workload.session_rate"]);
  });

  it.each(["open_loop", "closed_loop"] as const)("requires trace session rates for %s", (arrivalMode) => {
    const config = createDefaultSimConfig();
    config.workload = { mode: "trace", arrival_mode: arrivalMode, seed: 0, session_rate: null, trace: {} };
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(["workload.session_rate"]);
    config.workload.session_rate = 1;
    expect(validateSimConfig(config)).toEqual([]);
    config.workload.target_qps = 2;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(
      arrivalMode === "closed_loop" ? ["workload.target_qps"] : [],
    );
  });

  it("accepts a synthetic request without an optional session", () => {
    const config = createDefaultSimConfig();
    delete config.workload.synthetic!.session;
    expect(validateSimConfig(config)).toEqual([]);
  });

  it("validates trace engine settings without requiring synthetic distributions", () => {
    const config = createDefaultSimConfig();
    config.workload = { mode: "trace", arrival_mode: "open_loop", session_rate: 1, seed: 0, trace: {} };
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
    config.workload.arrival_mode = "closed_loop";
    delete config.workload.target_qps;
    config.workload.session_rate = 1.28;
    config.workload.synthetic = {
      ...config.workload.synthetic!, turn_increment_len: { dist: "fixed", value: 32 },
      session: { num_sessions: 20, topology: "linear", think_time_ms: 200 },
    };
    expect(validateSimConfig(config)).toEqual([]);
    config.workload.session_rate = 0;
    config.workload.synthetic.session!.num_sessions = 1001;
    config.workload.synthetic.session!.think_time_ms = -1;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.session_rate", "workload.synthetic.session.num_sessions",
      "workload.synthetic.session.think_time_ms",
    ]);
  });

  it("validates page size while resolving KV capacity from physical specifications", () => {
    const config = createDefaultSimConfig();
    expect(config.inference).not.toHaveProperty("kv");
    config.inference.page_size = 0;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(["inference.page_size"]);
    config.inference.page_size = 1;
    config.scheduler.pd_mutually_exclusive = false;
    expect(validateSimConfig(config)).toEqual([]);
  });

  it("requires L2 and its transfer link before enabling shared physical L3", () => {
    const config = createDefaultSimConfig();
    config.inference.storage.l2_available_gb = 0;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(["shared_l3.available_gb"]);
    config.inference.storage.l2_available_gb = 2048;
    config.inference.storage.l3_to_l2_gb_s = 0;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual(["shared_l3.available_gb"]);
    config.inference.storage.l3_to_l2_gb_s = 50;
    expect(validateSimConfig(config)).toEqual([]);
  });

  it("accepts disabled L3 and validates physical shared capacity and latency", () => {
    const config = createDefaultSimConfig();
    config.inference.storage.l2_available_gb = 0;
    config.inference.storage.l3_to_l2_gb_s = 0;
    config.shared_l3!.available_gb = 0;
    config.scheduler.l3_prefetch_min_tokens = 0;
    expect(validateSimConfig(config)).toEqual([]);
    config.shared_l3!.available_gb = -1;
    config.shared_l3!.lookup_latency_ms = -1;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "shared_l3.lookup_latency_ms", "shared_l3.available_gb",
    ]);
  });

  it("accepts trace mode independently of any synthetic payload", () => {
    const config: SimConfig = {
      ...createDefaultSimConfig(),
      workload: { mode: "trace", arrival_mode: "closed_loop", session_rate: 2, seed: 0, trace: {} },
    };
    expect(validateSimConfig(config)).toEqual([]);
  });

  it("requires a payload in synthetic mode", () => {
    const config: SimConfig = {
      ...createDefaultSimConfig(),
      workload: { mode: "synthetic", arrival_mode: "open_loop", seed: 0, target_qps: 1, synthetic: null },
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
    config.workload.target_qps = Number.NaN;
    config.workload.seed = -1;
    config.inference.max_output_tokens = 0;
    config.scheduler.max_running_requests = 0;
    config.scheduler.max_prefill_tokens = 0;
    config.scheduler.l3_prefetch_min_tokens = -1;
    config.inference.page_size = 0;
    config.scheduler.pd_mutually_exclusive = null as unknown as boolean;
    config.inference.timing = { ...createDefaultEmpiricalTiming(), prefill_c1: -0.1 };
    config.inference.timing.prefill_c2 = Number.POSITIVE_INFINITY;
    config.inference.timing.decode_d1 = -1;
    config.inference.timing.decode_d2 = Number.NaN;
    config.shared_l3!.lookup_latency_ms = -1;

    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.synthetic.num_requests",
      "workload.target_qps",
      "workload.seed",
      "scheduler.max_running_requests",
      "scheduler.max_prefill_tokens",
      "scheduler.l3_prefetch_min_tokens",
      "inference.page_size",
      "inference.max_output_tokens",
      "scheduler.pd_mutually_exclusive",
      "inference.timing.prefill_c1",
      "inference.timing.prefill_c2",
      "inference.timing.decode_d1",
      "inference.timing.decode_d2",
      "shared_l3.lookup_latency_ms",
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
