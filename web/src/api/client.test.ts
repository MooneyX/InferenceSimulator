import { afterEach, describe, expect, it, vi } from "vitest";

import { simulate, SimulationApiError } from "./client";
import type { SimConfig, SimResult } from "./types";

const config: SimConfig = {
  workload: {
    mode: "synthetic",
    synthetic: {
      num_requests: 1,
      arrival_rate: 2,
      prompt_len: { dist: "fixed", value: 16 },
      output_len: { dist: "fixed", value: 8 },
      seed: 42,
    },
  },
  engine: {
    max_batch_size: 4,
    kv_token_budget: 4096,
  },
  perf_model: {
    prefill_c1: 0.1,
    prefill_c2: 0,
    decode_d1: 0.2,
    decode_d2: 0.001,
  },
};

const result: SimResult = {
  meta: {
    config,
    num_requests: 1,
    num_engine_steps: 8,
    sim_start: 0,
    sim_end: 3,
  },
  metrics: {
    ttft_mean: 1,
    ttft_p50: 1,
    ttft_p99: 1,
    tpot_mean: 0.25,
    throughput: 333.33,
    makespan: 3,
    target_ideal_prefix_hit_rate: 0,
    constructed_ideal_prefix_hit_rate: 0,
    ideal_hit_quantization_error: 0,
    actual_prefix_hit_rate: 0,
    requests_with_hit_ratio: 0,
    requests_fully_cached_ratio: 0,
    total_input_tokens: 16,
    ideal_hit_tokens: 0,
    matched_cache_tokens: 0,
    actual_hit_tokens: 0,
    l1_hit_tokens: 0,
    l2_hit_tokens: 0,
    l3_hit_tokens: 0,
    restore_bytes: 0,
    restore_time_ms: 0,
    promotions: 0,
    demotions: 0,
    actual_prefill_tokens: 16,
    saved_prefill_tokens: 0,
    l1_evictions: 0,
    l1_peak_resident_blocks: 24,
    l1_capacity_blocks: 4096,
    l1_resident_blocks: 24,
    l1_active_blocks: 0,
    l1_evictable_blocks: 24,
    l1_free_blocks: 4072,
    l1_reserved_future_blocks: 0,
    l1_internal_fragmentation_tokens: 0,
  },
  requests: [
    {
      id: 0,
      arrival: 0,
      prefill_start: 0,
      prefill_end: 1,
      finish: 3,
      prompt_len: 16,
      output_len: 8,
      session_id: 0,
      ideal_reuse_tokens: 0,
      matched_cache_tokens: 0,
      actual_hit_tokens: 0,
      actual_prefill_tokens: 16,
      logical_context_tokens: 24,
      l1_hit_tokens: 0,
      l2_hit_tokens: 0,
      l3_hit_tokens: 0,
    },
  ],
  warnings: [],
};

function jsonResponse(status: number, payload: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(payload),
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("simulate", () => {
  it("posts a typed config and returns the simulation result", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(200, result));
    vi.stubGlobal("fetch", fetchMock);

    await expect(simulate(config)).resolves.toEqual(result);
    expect(fetchMock).toHaveBeenCalledWith("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
      signal: undefined,
    });
  });

  it("maps a 422 response to validation details", async () => {
    const detail = [
      {
        loc: ["body", "engine", "max_batch_size"],
        msg: "Input should be greater than 0",
        type: "greater_than",
      },
    ];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(422, { detail })));

    const error = await simulate(config).catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(SimulationApiError);
    expect(error).toMatchObject({
      kind: "validation",
      status: 422,
      validationErrors: detail,
    });
  });

  it("shows L1 capacity failures as an actionable 422 message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(422, { detail: "请求输入超过 L1 容量" })),
    );

    await expect(simulate(config)).rejects.toMatchObject({
      kind: "validation",
      status: 422,
      message: "请求输入超过 L1 容量",
      validationErrors: [],
    });
  });

  it("uses the service detail for an unimplemented workload", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(501, { detail: "trace workload 未实现" })),
    );

    await expect(simulate(config)).rejects.toMatchObject({
      kind: "not_implemented",
      status: 501,
      message: "trace workload 未实现",
    });
  });

  it("provides an HTTP fallback when an error body is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: vi.fn().mockRejectedValue(new SyntaxError("invalid JSON")),
    } as unknown as Response));

    await expect(simulate(config)).rejects.toMatchObject({
      kind: "http",
      status: 503,
      message: "仿真请求失败（HTTP 503）",
    });
  });

  it("reports a malformed successful response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: vi.fn().mockRejectedValue(new SyntaxError("invalid JSON")),
    } as unknown as Response));

    await expect(simulate(config)).rejects.toMatchObject({
      kind: "invalid_response",
      status: 200,
    });
  });

  it("maps fetch failures to a network error and preserves the cause", async () => {
    const cause = new TypeError("fetch failed");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(cause));

    const error = await simulate(config).catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(SimulationApiError);
    expect(error).toMatchObject({
      kind: "network",
      message: "无法连接仿真服务",
      cause,
    });
  });
});
