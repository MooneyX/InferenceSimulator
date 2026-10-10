import { describe, expect, it } from "vitest";

import type { ResolvedRequest } from "../api/types";
import { initialHardware, initialModel, initialStorage } from "../components/AutomaticConfig";
import { createDefaultSimConfig } from "./simulation";
import { parseResolvedRequestJson, parseSimConfigJson, serializeResolvedRequest, serializeSimConfig } from "./parameterTransfer";

describe("parameter JSON transfer", () => {
  it("round-trips the complete API config without result data", () => {
    const config = createDefaultSimConfig();
    config.workload.synthetic!.prompt_len = { dist: "fixed", value: 256 };
    config.workload.synthetic!.output_len = { dist: "lognormal", mean: 128, sigma: 0.5 };
    config.workload.synthetic!.num_requests = 20;
    config.workload.synthetic!.turn_increment_len = { dist: "fixed", value: 24 };

    const json = serializeSimConfig(config);
    expect(parseSimConfigJson(json)).toEqual(config);
    expect(JSON.parse(json)).toHaveProperty("workload.synthetic.prompt_len", {
      dist: "fixed", value: 256,
    });
    expect(JSON.parse(json)).not.toHaveProperty("metrics");
    expect(JSON.parse(json)).not.toHaveProperty("requests");
  });

  it("round-trips session mode without a request QPS", () => {
    const config = createDefaultSimConfig();
    config.workload.synthetic = {
      num_requests: 20, arrival_mode: "session", session_rate: 2,
      session: { num_sessions: 5, topology: "linear", think_time_ms: 200 },
      prompt_len: { dist: "fixed", value: 256 },
      turn_increment_len: { dist: "fixed", value: 32 },
      output_len: { dist: "fixed", value: 16 },
      max_new_tokens: 128, seed: 42,
    };
    const json = serializeSimConfig(config);
    expect(parseSimConfigJson(json)).toEqual(config);
    expect(JSON.parse(json).workload.synthetic).not.toHaveProperty("arrival_rate");
    expect(JSON.parse(json).workload.synthetic).not.toHaveProperty("session.target_ideal_prefix_hit_rate");
  });

  it.each(["", "{", "null", "[]", '"hello"'])
  ("rejects non-config JSON: %s", (json) => {
    expect(() => parseSimConfigJson(json)).toThrow();
  });

  it.each([
    [{}, "缺少参数：workload"],
    [{ ...createDefaultSimConfig(), metrics: {} }, "不支持的参数：metrics"],
    [{ ...createDefaultSimConfig(), workload: { mode: "trace", synthetic: null } }, "仅支持 synthetic"],
    [{ ...createDefaultSimConfig(), workload: { mode: "synthetic" } }, "缺少参数：workload.synthetic"],
    [{ ...createDefaultSimConfig(), engine: { ...createDefaultSimConfig().engine, capacity_pages: 10 } },
      "不支持的参数：engine.capacity_pages"],
    [{ ...createDefaultSimConfig(), engine: { ...createDefaultSimConfig().engine, page_size: "64" } },
      "engine.page_size 必须是数值"],
    [{ ...createDefaultSimConfig(), engine: { ...createDefaultSimConfig().engine, max_total_tokens: 130 } },
      "engine.max_total_tokens"],
    [{ ...createDefaultSimConfig(), workload: {
      ...createDefaultSimConfig().workload,
      synthetic: { ...createDefaultSimConfig().workload.synthetic!, session: null },
    } }, "不支持的参数：workload.synthetic.session"],
    [{ ...createDefaultSimConfig(), workload: {
      ...createDefaultSimConfig().workload,
      synthetic: { ...createDefaultSimConfig().workload.synthetic!, prompt_len: { dist: "other" } },
    } }, "prompt_len.dist 不支持"],
    [{ ...createDefaultSimConfig(), workload: {
      ...createDefaultSimConfig().workload,
      synthetic: { ...createDefaultSimConfig().workload.synthetic!, output_len: { dist: "fixed", value: 0 } },
    } }, "output_len.value"],
  ])("rejects invalid or unsupported configs", (value, message) => {
    expect(() => parseSimConfigJson(JSON.stringify(value))).toThrow(message as string);
  });

  it("defaults legacy automatic requests to EP1 and rejects invalid EP widths", () => {
    const request = {
      config: createDefaultSimConfig(), model: { name: "toy", num_layers: 1,
        hidden_size: 16, num_attention_heads: 2, num_kv_heads: 1,
        head_dim: 8, ffn_intermediate_size: 32, vocab_size: 16 },
      hardware: { name: "GPU", gpu_count: 1, memory_bytes_per_gpu: 1000000,
        effective_fp16_tflops: 1, effective_hbm_gb_s: 1 },
      storage: { l1_available_gb: 0.001, l2_available_gb: 0, l3_available_gb: 0 },
    };
    expect(parseResolvedRequestJson(JSON.stringify(request)).ep_size).toBe(1);
    expect(() => parseResolvedRequestJson(JSON.stringify({ ...request, ep_size: 0 })))
      .toThrow("ep_size 必须是正整数");
  });

  it("rejects malformed physical specs and non-finite automatic drafts", () => {
    const base: ResolvedRequest = {
      config: createDefaultSimConfig(),
      model: { ...initialModel, name: "toy", num_layers: 2, hidden_size: 16,
        num_attention_heads: 2, num_kv_heads: 1, head_dim: 8,
        ffn_intermediate_size: 32, vocab_size: 16 },
      hardware: { ...initialHardware, name: "GPU", gpu_count: 1,
        memory_bytes_per_gpu: 1_000_000, effective_fp16_tflops: 1, effective_hbm_gb_s: 1,
        ep_communication_latency_ms: null },
      storage: { ...initialStorage, l1_available_gb: 0.001 }, tp_size: 1, ep_size: 1,
      timing: "analytical",
    };
    expect(parseResolvedRequestJson(serializeResolvedRequest(base))).toEqual(base);
    expect(() => parseResolvedRequestJson(JSON.stringify({ ...base,
      model: { ...base.model, total_parameters: 10_000 },
    }))).toThrow("不支持的参数：model.total_parameters");
    const invalid = [
      [{ ...base, model: { ...base.model, moe_layers: -1 } }, "model.moe_layers"],
      [{ ...base, hardware: { ...base.hardware, extra: 2 } }, "不支持的参数：hardware.extra"],
      [{ ...base, hardware: { ...base.hardware, ep_communication_latency_ms: -1 } }, "hardware.ep_communication_latency_ms"],
      [{ ...base, storage: { ...base.storage, l1_available_gb: -1 } }, "storage.l1_available_gb"],
      [{ ...base, ep_size: 2 }, "Dense 模型"],
      [{ ...base, model: { ...base.model, weight_dtype: "fp8" } }, "BF16/FP16"],
      [{ ...base, model: { ...base.model, moe_layers: 1 } }, "MoE 需要"],
      [{ ...base, unexpected: true }, "不支持的参数：自动解析参数.unexpected"],
    ] as const;
    for (const [request, message] of invalid) {
      expect(() => parseResolvedRequestJson(JSON.stringify(request))).toThrow(message);
    }
    expect(() => serializeResolvedRequest({ ...base, hardware: {
      ...base.hardware, ep_communication_latency_ms: Number.POSITIVE_INFINITY,
    } })).toThrow("ep_communication_latency_ms");
    expect(() => serializeResolvedRequest({ ...base, model: {
      ...base.model, num_experts: Number.NaN,
    } })).toThrow("num_experts");
  });

  it("round-trips a three-tier config including link speeds and latency", () => {
    const config = createDefaultSimConfig();
    config.engine.l2_capacity_tokens = 128;
    config.engine.l3_capacity_tokens = 64;
    config.perf_model.kv_bytes_per_token = 2048;
    config.perf_model.l2_to_l1_bandwidth_bytes_per_ms = 100;
    config.perf_model.l3_to_l2_bandwidth_bytes_per_ms = 50;
    config.perf_model.l2_latency_ms = 1;
    config.perf_model.l3_latency_ms = 2;
    expect(parseSimConfigJson(serializeSimConfig(config))).toEqual(config);
  });

  it("rejects missing, extra and mistyped KV3 fields rather than silently defaulting", () => {
    const base = createDefaultSimConfig();
    const { l3_capacity_tokens: _omitted, ...missing } = base.engine;
    expect(() => parseSimConfigJson(JSON.stringify({ ...base, engine: missing })))
      .toThrow("缺少参数：engine.l3_capacity_tokens");
    expect(() => parseSimConfigJson(JSON.stringify({
      ...base, perf_model: { ...base.perf_model, unknown_link: 1 },
    }))).toThrow("不支持的参数：perf_model.unknown_link");
    expect(() => parseSimConfigJson(JSON.stringify({
      ...base, perf_model: { ...base.perf_model, kv_bytes_per_token: "2048" },
    }))).toThrow("perf_model.kv_bytes_per_token 必须是数值或 null");
    expect(() => parseSimConfigJson(JSON.stringify({
      ...base, engine: { ...base.engine, l3_capacity_tokens: 64 },
    }))).toThrow("engine.l3_capacity_tokens");
  });

  it("refuses to export invalid drafts instead of silently converting non-finite values to null", () => {
    const config = createDefaultSimConfig();
    config.workload.synthetic!.arrival_rate = Number.NaN;
    expect(() => serializeSimConfig(config)).toThrow("arrival_rate");
    config.workload.synthetic!.arrival_rate = 1;
    config.perf_model.kv_bytes_per_token = Number.POSITIVE_INFINITY;
    expect(() => serializeSimConfig(config)).toThrow("kv_bytes_per_token");
    config.perf_model.kv_bytes_per_token = null;
    config.perf_model.l2_to_l1_bandwidth_bytes_per_ms = Number.POSITIVE_INFINITY;
    expect(() => serializeSimConfig(config)).toThrow("l2_to_l1_bandwidth_bytes_per_ms");
  });
});
