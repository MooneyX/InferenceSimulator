import { afterEach, describe, expect, it, vi } from "vitest";

import { clearSimulationCache, getJobResult, resolveConfig, simulate, SimulationApiError } from "./client";
import type { ResolutionReport, SimConfig, SimResult } from "./types";
import { createDefaultSimConfig } from "../config/simulation";

const config: SimConfig = {
  num_instances: 1,
  shared_l2_enabled: true,
  routing_policy: "least_load",
  workload: {
    mode: "synthetic",
    arrival_mode: "open_loop",
    target_qps: 2,
    seed: 0,
    synthetic: {
      num_requests: 1,
      prompt_len: { dist: "fixed", value: 16 },
      turn_increment_len: { dist: "fixed", value: 8 },
      output_len: { dist: "fixed", value: 8 },
    },
  },
  scheduler: {
    max_running_requests: 4,
    l3_prefetch_min_tokens: 256,
    max_prefill_tokens: 16_384,
    pd_mutually_exclusive: true,
  },
  inference: createDefaultSimConfig().inference,
};

const result: SimResult = {
  resolution: null,
  analysis: null,
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
    l1_prefix_hit_rate: 0,
    l2_prefix_hit_rate: 0,
    l3_prefix_hit_rate: 0,
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
    headers: new Headers(),
    json: vi.fn().mockResolvedValue(payload),
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("simulate", () => {
  it("posts a typed config and returns the simulation result", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(202, { id: "job-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(simulate(config)).resolves.toEqual({ id: "job-1" });
    expect(fetchMock).toHaveBeenCalledWith("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
  });

  it("maps a 422 response to validation details", async () => {
    const detail = [
      {
        loc: ["body", "scheduler", "max_running_requests"],
        msg: "Input should be greater than 0",
        type: "greater_than",
      },
    ];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(422, { detail })));

    const error = await simulate(config).catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(SimulationApiError);
    expect(error).toMatchObject({
      validationErrors: detail,
    });
  });

  it("shows L1 capacity failures as an actionable 422 message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(422, { detail: "请求输入超过 L1 容量" })),
    );

    await expect(simulate(config)).rejects.toMatchObject({
      message: "请求输入超过 L1 容量",
      validationErrors: [],
    });
  });

  it("provides an HTTP fallback when an error body is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: vi.fn().mockRejectedValue(new SyntaxError("invalid JSON")),
    } as unknown as Response));

    await expect(simulate(config)).rejects.toMatchObject({
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
      message: "仿真服务返回了无效的 JSON 响应",
    });
  });

  it("maps fetch failures to a network error and preserves the cause", async () => {
    const cause = new TypeError("fetch failed");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(cause));

    const error = await simulate(config).catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(SimulationApiError);
    expect(error).toMatchObject({
      message: "无法连接仿真服务",
      cause,
    });
  });
});

describe("result cache", () => {
  it("reads hit metadata and sends the force query", async () => {
    const receipt = jsonResponse(202, { id: "job-1" });
    const response = jsonResponse(200, result);
    response.headers.set("X-Simulation-Cache", "hit");
    response.headers.set("X-Simulation-Cache-Lookup-Ms", "1.25");
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValueOnce(receipt).mockResolvedValue(response);
    vi.stubGlobal("fetch", fetchMock);
    await expect(simulate(config, true)).resolves.toEqual({ id: "job-1" });
    expect(fetchMock).toHaveBeenCalledWith("/api/simulate?force=true", expect.objectContaining({
      method: "POST", body: JSON.stringify(config),
    }));
    await expect(getJobResult("job-1")).resolves.toEqual({
      result, cache: { status: "hit", lookupMs: 1.25 },
    });
    response.headers.set("X-Simulation-Cache", "shared");
    expect((await getJobResult("job-1")).cache.status).toBe("shared");
  });

  it("clears the cache with DELETE without reading an empty JSON body", async () => {
    const response = jsonResponse(204, undefined);
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(response);
    vi.stubGlobal("fetch", fetchMock);
    await clearSimulationCache();
    expect(fetchMock).toHaveBeenCalledWith("/api/simulation-cache", { method: "DELETE" });
    expect(response.json).not.toHaveBeenCalled();
  });

  it("reports cache clearing failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(500, { detail: "清理失败" })));
    await expect(clearSimulationCache()).rejects.toThrow("清理失败");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    await expect(clearSimulationCache()).rejects.toThrow("无法连接仿真服务");
  });
});

describe("trace replay requests", () => {
  it("posts trace settings and the plan ID through the single simulation endpoint", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(200, result));
    vi.stubGlobal("fetch", fetchMock);
    const traceConfig: SimConfig = { ...config, workload: { mode: "trace", arrival_mode: "open_loop",
      session_rate: 2, seed: 7, target_qps: 12, trace: { plan_id: "plan-1" } } };
    await simulate(traceConfig);
    expect(fetchMock).toHaveBeenLastCalledWith("/api/simulate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(traceConfig),
    });
    const closedLoopConfig: SimConfig = { ...traceConfig, workload: { mode: "trace",
      arrival_mode: "closed_loop", session_rate: 2, seed: 7, trace: { plan_id: "plan-1" } } };
    await simulate(closedLoopConfig);
    expect(fetchMock).toHaveBeenLastCalledWith("/api/simulate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(closedLoopConfig),
    });
  });
});

describe("resolveConfig", () => {
  it("posts the same configuration without wrapping it in a separate request", async () => {
    const previewConfig = createDefaultSimConfig();
    const report: ResolutionReport = {
      max_total_tokens: 4096, l2_capacity_tokens: 0, l3_capacity_tokens: 0,
      runnable: true,
      capacity: {
        weights_resident_bytes: 16_400_000_000, kv_bytes_per_token: 147456,
        kv_bytes_per_page: 9_437_184, estimated_total_parameters: 8_200_000_000,
        estimated_active_parameters: 8_200_000_000, estimated_expert_parameters: 0,
        l1_budget_bytes: 200_000_000_000, l1_kv_budget_bytes: 183_600_000_000,
        l2_budget_bytes: 0, l3_budget_bytes: 0,
        l1_capacity_pages: 64, l2_capacity_pages: 0, l3_capacity_pages: 0,
        rank_reports: [],
      },
    };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(200, report));
    vi.stubGlobal("fetch", fetchMock);
    await expect(resolveConfig(previewConfig)).resolves.toEqual(report);
    expect(fetchMock).toHaveBeenCalledWith("/api/resolve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(previewConfig),
    });
  });

  it("preserves nested field paths and business messages for both 422 formats", async () => {
    const detail = [{ loc: ["body", "inference", "model", "num_experts"],
      msg: "专家数错误", type: "value_error" }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(jsonResponse(422, { detail }))
      .mockResolvedValueOnce(jsonResponse(422, { detail: "TP 数必须等于 GPU 卡数" })));
    await expect(resolveConfig(createDefaultSimConfig())).rejects.toMatchObject({ validationErrors: detail });
    await expect(simulate(config)).rejects.toMatchObject({
      message: "TP 数必须等于 GPU 卡数", validationErrors: [],
    });
  });

  it("recognizes unsupported performance configurations", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(501, { detail: "性能模型暂不支持该注意力结构" })));
    await expect(simulate(config)).rejects.toMatchObject({ message: "性能模型暂不支持该注意力结构" });
  });
});
