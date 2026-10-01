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
        max_batch_size: 32,
        kv_token_budget: 65_536,
        prefix_cache: {
          block_size_tokens: 1,
          cache_salt: "",
          eviction_policy: "lru",
          hierarchy_policy: "exclusive_write_back",
          kv_bytes_per_token: 1,
          model_id: "default-model",
          tokenizer_id: "default-tokenizer",
          kv_dtype: "default-dtype",
          adapter_id: "default-adapter",
          multimodal_transform_id: "default-transform",
        },
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
    expect(first.engine.prefix_cache).not.toBe(second.engine.prefix_cache);
  });
});

describe("validateSimConfig", () => {
  it("accepts the default configuration", () => {
    expect(validateSimConfig(createDefaultSimConfig())).toEqual([]);
  });

  it("accepts legacy requests without session or prefix cache", () => {
    const config = createDefaultSimConfig();
    delete config.workload.synthetic!.session;
    delete config.engine.prefix_cache;
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

  it("requires a valid block size and derives one L1 budget", () => {
    const config = createDefaultSimConfig();
    config.engine.kv_token_budget = 9;
    config.engine.prefix_cache!.block_size_tokens = 4;
    expect(validateSimConfig(config)).toEqual([]);
    config.engine.prefix_cache!.l1 = { capacity_blocks: 2 };
    expect(validateSimConfig(config)).toEqual([]);
    config.engine.prefix_cache!.l1.capacity_blocks = 3;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "engine.kv_token_budget",
    ]);
    delete config.engine.prefix_cache!.l1;
    config.engine.prefix_cache!.block_size_tokens = 10;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "engine.prefix_cache.block_size_tokens",
    ]);
    config.engine.prefix_cache!.block_size_tokens = 0;
    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "engine.prefix_cache.block_size_tokens",
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
    config.engine.max_batch_size = 0;
    config.engine.kv_token_budget = 2.5;
    config.perf_model.prefill_c1 = -0.1;
    config.perf_model.prefill_c2 = Number.POSITIVE_INFINITY;
    config.perf_model.decode_d1 = -1;
    config.perf_model.decode_d2 = Number.NaN;

    expect(validateSimConfig(config).map(({ field }) => field)).toEqual([
      "workload.synthetic.num_requests",
      "workload.synthetic.arrival_rate",
      "workload.synthetic.seed",
      "engine.max_batch_size",
      "engine.kv_token_budget",
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
