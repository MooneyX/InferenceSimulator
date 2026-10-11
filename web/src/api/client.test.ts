import { afterEach, describe, expect, it, vi } from "vitest";

import { resolveAutomatic, simulate, simulateAutomatic, simulateReplayPlan, simulateReplayPlanAutomatic, SimulationApiError } from "./client";
import type { ResolvedRequest, SimConfig, SimResult } from "./types";

const config: SimConfig = {
  num_instances: 1,
  workload: {
    mode: "synthetic",
    synthetic: {
      num_requests: 1,
      arrival_rate: 2,
      arrival_mode: "open_loop",
      prompt_len: { dist: "fixed", value: 16 },
      turn_increment_len: { dist: "fixed", value: 8 },
      output_len: { dist: "fixed", value: 8 },
      max_new_tokens: 128,
      seed: 42,
    },
  },
  engine: {
    max_running_requests: 4,
    max_total_tokens: 4096,
    l2_capacity_tokens: 0,
    l3_capacity_tokens: 0,
    l3_prefetch_min_tokens: 256,
    max_prefill_tokens: 16_384,
    page_size: 64,
    pd_mutually_exclusive: true,
  },
  perf_model: {
    prefill_c1: 0.1,
    prefill_c2: 0,
    decode_d1: 0.2,
    decode_d2: 0.001,
    kv_bytes_per_token: null,
    l2_to_l1_bandwidth_bytes_per_ms: null,
    l3_to_l2_bandwidth_bytes_per_ms: null,
    l2_latency_ms: 0,
    l3_latency_ms: 0,
    l3_lookup_latency_ms: 0,
  },
};

const result: SimResult = {
  instances: [],
  meta: {
    config,
    num_requests: 1,
    num_completed: 1,
    num_aborted: 0,
    num_skipped: 0,
    num_engine_steps: 8,
    sim_start: 0,
    sim_end: 3,
    wall_clock_ms: 42.5,
  },
  metrics: {
    ttft_mean: 1,
    ttft_p50: 1,
    ttft_p99: 1,
    tpot_mean: 0.25,
    throughput: 333.33,
    makespan: 3,
    num_sessions: 1,
    root_request_ratio: 1,
    constructed_ideal_prefix_hit_rate: 0,
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
    replayed_prefill_tokens: 0,
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
    l2_evictions: 0,
    l2_peak_resident_blocks: 0,
    l2_capacity_blocks: 0,
    l2_resident_blocks: 0,
    l2_active_blocks: 0,
    l2_evictable_blocks: 0,
    l2_free_blocks: 0,
    l3_evictions: 0,
    l3_peak_resident_blocks: 0,
    l3_capacity_blocks: 0,
    l3_resident_blocks: 0,
    l3_active_blocks: 0,
    l3_evictable_blocks: 0,
    l3_free_blocks: 0,
  },
  requests: [
    {
      id: 0,
      instance_id: 0,
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
      replayed_prefill_tokens: 0,
      retry_count: 0,
      logical_context_tokens: 24,
      l1_hit_tokens: 0,
      l2_hit_tokens: 0,
      l3_hit_tokens: 0,
    },
  ],
  aborted_requests: [],
  skipped_requests: [],
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
        loc: ["body", "engine", "max_running_requests"],
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

describe("trace replay requests", () => {
  it("sends target QPS to the manual replay endpoint only when enabled", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(200, result));
    vi.stubGlobal("fetch", fetchMock);
    await simulateReplayPlan("plan-1", config, "open_loop", 2, 7, 12);
    expect(fetchMock).toHaveBeenLastCalledWith("/api/replay-plans/plan-1/simulate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, mode: "open_loop", session_rate: 2, seed: 7, target_qps: 12 }),
    });
    await simulateReplayPlan("plan-1", config, "session", 2, 7);
    expect(fetchMock).toHaveBeenLastCalledWith("/api/replay-plans/plan-1/simulate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, mode: "session", session_rate: 2, seed: 7 }),
    });
  });
});

describe("automatic resolution errors", () => {
  const request = { config } as ResolvedRequest;

  it("posts the complete resolved trace request to the plan endpoint", async () => {
    const body = { result, resolution: { config }, analysis: null };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(200, body));
    vi.stubGlobal("fetch", fetchMock);
    await expect(simulateReplayPlanAutomatic("plan-1", request, "session", 2, 7)).resolves.toEqual(body);
    expect(fetchMock).toHaveBeenCalledWith("/api/replay-plans/plan-1/simulate/resolved", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request, mode: "session", session_rate: 2, seed: 7 }),
    });
    await simulateReplayPlanAutomatic("plan-1", request, "open_loop", 2, 7, 8);
    expect(fetchMock).toHaveBeenLastCalledWith("/api/replay-plans/plan-1/simulate/resolved", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request, mode: "open_loop", session_rate: 2, seed: 7, target_qps: 8 }),
    });
  });

  it("reports unsupported multi-GPU empirical trace timing as not implemented", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(501, { detail: "四系数不支持 TP" })));
    await expect(simulateReplayPlanAutomatic("plan-1", request, "session", 2, 7)).rejects.toMatchObject({
      kind: "not_implemented", status: 501, message: "四系数不支持 TP",
    });
  });

  it("preserves field paths and business messages for both 422 formats", async () => {
    const detail = [{ loc: ["body", "model", "num_experts"], msg: "专家数错误", type: "value_error" }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(jsonResponse(422, { detail }))
      .mockResolvedValueOnce(jsonResponse(422, { detail: "TP 数必须等于 GPU 卡数" })));
    await expect(resolveAutomatic(request)).rejects.toMatchObject({
      kind: "validation", status: 422, validationErrors: detail,
    });
    await expect(simulateAutomatic(request)).rejects.toMatchObject({
      kind: "validation", status: 422, message: "TP 数必须等于 GPU 卡数", validationErrors: [],
    });
  });

  it("recognizes unimplemented resolved run modes", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(501, { detail: "四系数不支持 EP" })));
    await expect(simulateAutomatic(request)).rejects.toMatchObject({
      kind: "not_implemented", status: 501, message: "四系数不支持 EP",
    });
  });
});
