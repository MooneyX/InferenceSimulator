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
          arrival_rate: 50,
          prompt_len: { dist: "lognormal", mean: 512, sigma: 1 },
          output_len: { dist: "poisson", mean: 128 },
          max_new_tokens: 128,
          seed: 0,
          session: {
            num_sessions: 1000,
            topology: "linear",
            think_time_ms: 0,
            target_ideal_prefix_hit_rate: 0,
          },
        },
      },
      engine: {
        max_running_requests: 32,
        max_total_tokens: 65_536,
        max_prefill_tokens: 16_384,
        page_size: 64,
        pd_mutually_exclusive: true,
      },
      perf_model: {
        prefill_c1: 0.05,
        prefill_c2: 0.000_01,
        decode_d1: 8,
        decode_d2: 0.002,
      },
    });
    const session = createDefaultSimConfig().workload.synthetic!.session!;
    expect(session).not.toHaveProperty("allow_cross_session_shared_prefix");
    expect(JSON.parse(JSON.stringify(session))).not.toHaveProperty("allow_cross_session_shared_prefix");
  });

  it("returns independent objects", () => {
    const first = createDefaultSimConfig();
    const second = createDefaultSimConfig();

    expect(first).not.toBe(second);
    expect(first.workload.synthetic).not.toBe(second.workload.synthetic);
    expect(first.workload.synthetic?.prompt_len).not.toBe(
      second.workload.synthetic?.prompt_len,
    );
    expect(first.workload.synthetic?.session).not.toBe(second.workload.synthetic?.session);
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

  it("validates session boundaries and finite values", () => {
    const config = createDefaultSimConfig();
    const session = config.workload.synthetic!.session!;
    config.workload.synthetic!.num_requests = 2;
    session.num_sessions = 3;
    session.think_time_ms = -1;
    session.target_ideal_prefix_hit_rate = 1.1;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.synthetic.session.num_sessions",
      "workload.synthetic.session.think_time_ms",
      "workload.synthetic.session.target_ideal_prefix_hit_rate",
    ]);
    session.num_sessions = 1;
    session.think_time_ms = 0;
    session.target_ideal_prefix_hit_rate = 1;
    expect(validateSimConfig(config)).toEqual([]);
    session.num_sessions = Number.NaN;
    session.think_time_ms = Number.POSITIVE_INFINITY;
    session.target_ideal_prefix_hit_rate = Number.NaN;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.synthetic.session.num_sessions",
      "workload.synthetic.session.think_time_ms",
      "workload.synthetic.session.target_ideal_prefix_hit_rate",
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
