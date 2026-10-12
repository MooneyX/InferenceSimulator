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

  it.each(["bf16", "fp16", "fp32"] as const)("round-trips complete auxiliary weight settings in %s", (dtype) => {
    const request: ResolvedRequest = {
      config: createDefaultSimConfig(), model: { ...initialModel, name: "custom", num_layers: 2,
        moe_layers: 2, num_experts: 8, experts_per_token: 2, expert_intermediate_size: 512,
        norms: { kind: "layer", block_per_layer: 3, final_norm: false, dtype },
        router: { dtype, correction_bias: true, correction_dtype: dtype },
        ihc: { mult: 2, dtype },
        attention_layout: "grouped", attention_groups: [{ kind: "mla_dsa", layers: 2,
          num_attention_heads: 32, q_rank: 2048, kv_rank: 512, rope_dim: 64,
          nope_dim: 192, value_dim: 256, index_heads: 32, index_dim: 128,
          topk: 2048, full_indexer_layers: 1, output_gate: true,
          qk_norm: false, norm_dtype: dtype, index_norm: "rms", index_norm_dtype: dtype, sink_dtype: dtype,
        }] }, hardware: initialHardware, storage: initialStorage, timing: "analytical", tp_size: 1, ep_size: 1,
    };
    expect(parseResolvedRequestJson(serializeResolvedRequest(request))).toEqual(request);
    const disabled: ResolvedRequest = { ...request, model: { ...request.model,
      norms: { kind: "none", block_per_layer: 0, final_norm: false, dtype: null },
      router: { dtype: null, correction_bias: false, correction_dtype: "fp32" }, ihc: null,
      attention_groups: [{ ...request.model.attention_groups![0], qk_norm: true,
        norm_dtype: null, index_norm: "none", index_norm_dtype: null, sink_dtype: null }],
    } };
    expect(parseResolvedRequestJson(serializeResolvedRequest(disabled))).toEqual(disabled);
  });

  it("rejects invalid auxiliary settings on import and export", () => {
    const base: ResolvedRequest = { config: createDefaultSimConfig(), model: { ...initialModel,
      attention_layout: "grouped", attention_groups: [{ kind: "gqa", layers: initialModel.num_layers,
        num_attention_heads: 32, num_kv_heads: 8, head_dim: 128, output_gate: false,
        qk_norm: true, index_norm: "layer" }],
    }, hardware: initialHardware, storage: initialStorage, timing: "analytical", tp_size: 1, ep_size: 1 };
    const invalid = [
      [{ norms: null }, "model.norms"], [{ norms: [] }, "model.norms"],
      [{ norms: { kind: "unknown" } }, "model.norms.kind"],
      [{ norms: { block_per_layer: -1 } }, "model.norms.block_per_layer"],
      [{ norms: { block_per_layer: 1.5 } }, "model.norms.block_per_layer"],
      [{ norms: { block_per_layer: Number.NaN } }, "model.norms.block_per_layer"],
      [{ norms: { final_norm: null } }, "model.norms.final_norm"],
      [{ norms: { dtype: "fp8" } }, "model.norms.dtype"],
      [{ norms: { extra: 1 } }, "model.norms.extra"],
      [{ router: null }, "model.router"], [{ router: { dtype: "fp8" } }, "model.router.dtype"],
      [{ router: { correction_bias: "true" } }, "model.router.correction_bias"],
      [{ router: { correction_dtype: null } }, "model.router.correction_dtype"],
      [{ router: { linear_bias: true } }, "model.router.linear_bias"],
      [{ router: { dtype: "fp32" } }, "Router 辅助配置需要 MoE 层"],
      [{ router: { correction_bias: true } }, "Router 辅助配置需要 MoE 层"],
      [{ router: { correction_dtype: "bf16" } }, "Router 辅助配置需要 MoE 层"],
      [{ ihc: {} }, "model.ihc.mult"], [{ ihc: { mult: 0 } }, "model.ihc.mult"],
      [{ ihc: { mult: 1.5 } }, "model.ihc.mult"],
      [{ ihc: { mult: Number.POSITIVE_INFINITY } }, "model.ihc.mult"],
      [{ ihc: { mult: 4, dtype: null } }, "model.ihc.dtype"],
      [{ ihc: { mult: 4, extra: 1 } }, "model.ihc.extra"],
    ] as const;
    for (const [patch, message] of invalid) {
      const request = { ...base, model: { ...base.model, ...patch } } as unknown as ResolvedRequest;
      expect(() => parseResolvedRequestJson(JSON.stringify(request))).toThrow(message);
      expect(() => serializeResolvedRequest(request)).toThrow(message);
    }
    for (const [patch, message] of [
      [{ qk_norm: null }, "qk_norm"], [{ norm_dtype: "fp8" }, "norm_dtype"],
      [{ index_norm: "unknown" }, "index_norm"], [{ index_norm_dtype: "fp8" }, "index_norm_dtype"],
      [{ sink_dtype: false }, "sink_dtype"], [{ index_norm: "none" }, "仅支持 DSA"],
      [{ index_norm_dtype: "fp32" }, "仅支持 DSA"],
    ] as const) {
      const request = { ...base, model: { ...base.model,
        attention_groups: [{ ...base.model.attention_groups![0], ...patch }],
      } } as unknown as ResolvedRequest;
      expect(() => parseResolvedRequestJson(JSON.stringify(request))).toThrow(message);
      expect(() => serializeResolvedRequest(request)).toThrow(message);
    }
    const legacy = { ...base, model: { ...base.model, norms: undefined, router: undefined, ihc: undefined } };
    const parsed = parseResolvedRequestJson(serializeResolvedRequest(legacy));
    expect(parsed.model).not.toHaveProperty("norms");
    expect(parsed.model).not.toHaveProperty("router");
    expect(parsed.model).not.toHaveProperty("ihc");
    const gqaSink = { ...base, model: { ...base.model, attention_groups: [{
      ...base.model.attention_groups![0], sink_dtype: "fp32" as const,
    }] } };
    expect(parseResolvedRequestJson(serializeResolvedRequest(gqaSink))).toEqual(gqaSink);
  });

  it("round-trips trace parameters without serializing the temporary plan ID", () => {
    const config = { ...createDefaultSimConfig(), workload: { mode: "trace" as const } };
    expect(parseSimConfigJson(serializeSimConfig(config))).toEqual(config);
    const resolved: ResolvedRequest = {
      config, model: initialModel, hardware: initialHardware, storage: initialStorage,
      timing: "analytical", tp_size: 1, ep_size: 1,
    };
    expect(parseResolvedRequestJson(serializeResolvedRequest(resolved))).toEqual(resolved);
    expect(JSON.stringify(config)).not.toContain("plan_id");
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
    [{}, "缺少参数：num_instances"],
    [{ ...createDefaultSimConfig(), metrics: {} }, "不支持的参数：metrics"],
    [{ ...createDefaultSimConfig(), workload: { mode: "trace", synthetic: null } }, "不支持的参数：workload.synthetic"],
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

  it("round-trips mixed MLA and MLA+DSA groups and rejects invalid coverage", () => {
    const request: ResolvedRequest = {
      config: createDefaultSimConfig(), model: { ...initialModel, num_layers: 2,
        attention_layout: "grouped", attention_groups: [
          { kind: "mla", layers: 1, num_attention_heads: 32, output_gate: false,
            qk_norm: true, index_norm: "layer",
            q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256 },
          { kind: "mla_dsa", layers: 1, num_attention_heads: 32, output_gate: true,
            qk_norm: true, index_norm: "layer",
            q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256,
            topk: 2048, index_heads: 32, index_dim: 128, full_indexer_layers: 1 },
        ] },
      hardware: initialHardware, storage: initialStorage,
      timing: "analytical", tp_size: 1, ep_size: 1,
    };
    expect(parseResolvedRequestJson(serializeResolvedRequest(request))).toEqual(request);
    expect(() => parseResolvedRequestJson(JSON.stringify({ ...request, model: {
      ...request.model, attention_groups: [{ ...request.model.attention_groups![0], layers: 2 },
        request.model.attention_groups![1]],
    } }))).toThrow("注意力组层数之和");
  });

  it.each(["grouped", "hy4_preview"] as const)("round-trips editable HY4 parameters with %s layout and rejects only invalid structures", (attention_layout) => {
    const request: ResolvedRequest = {
      config: createDefaultSimConfig(),
      model: { ...initialModel, name: "tencent/Hy4-preview",
        revision: "705d81ee51566a186d645b74c974d642ef2828fe",
        num_layers: 78, hidden_size: 6144, num_attention_heads: 64,
        num_kv_heads: 8, head_dim: 256, ffn_intermediate_size: 18432,
        vocab_size: 120832, moe_layers: 77, num_experts: 256,
        experts_per_token: 8, expert_intermediate_size: 2048,
        num_shared_experts: 1, shared_expert_intermediate_size: 2048,
        norms: { kind: "rms", block_per_layer: 2, final_norm: true, dtype: null },
        router: { dtype: "fp32", correction_bias: true, correction_dtype: "fp32" },
        ihc: { mult: 4, dtype: "fp32" },
        attention_layout, attention_groups: [{
          kind: "mla_dsa", layers: 78, num_attention_heads: 64,
          q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256,
          index_heads: 32, index_dim: 128, topk: 2048, full_indexer_layers: 21, output_gate: true,
          qk_norm: true, norm_dtype: null, index_norm: "layer", index_norm_dtype: "fp32", sink_dtype: "fp32",
        }] },
      hardware: { ...initialHardware, gpu_count: 8 },
      storage: initialStorage, timing: "analytical", tp_size: 8, ep_size: 8,
    };
    expect(parseResolvedRequestJson(serializeResolvedRequest(request))).toEqual(request);
    const group = request.model.attention_groups![0];
    const changed: ResolvedRequest = { ...request, model: { ...request.model,
      name: "edited-hy4", revision: "another-revision", num_layers: 80, hidden_size: 8192,
      weight_dtype: "fp16", kv_dtype: "fp16", tie_word_embeddings: true,
      num_shared_experts: 2, shared_expert_intermediate_size: 1024,
      norms: { kind: "layer", block_per_layer: 3, final_norm: false, dtype: "bf16" },
      router: { dtype: "bf16", correction_bias: false, correction_dtype: "fp16" },
      ihc: { mult: 2, dtype: "bf16" }, attention_groups: [{ ...group, layers: 79,
        num_attention_heads: 32, q_rank: 1024, kv_rank: 256, rope_dim: 128, nope_dim: 128, value_dim: 128,
        index_heads: 16, index_dim: 256, topk: 1024, full_indexer_layers: 20, output_gate: false,
        qk_norm: false, norm_dtype: "fp32", index_norm: "rms", index_norm_dtype: "bf16", sink_dtype: null },
        { kind: "gqa", layers: 1, num_attention_heads: 32, num_kv_heads: 8, head_dim: 128,
          output_gate: false, qk_norm: true, index_norm: "layer" }],
    } };
    expect(parseResolvedRequestJson(serializeResolvedRequest(changed))).toEqual(changed);
    for (const patch of [{ revision: null }, { num_shared_experts: 0, shared_expert_intermediate_size: null },
      { norms: undefined, router: undefined, ihc: undefined }, { ihc: null }]) {
      const edited = { ...request, model: { ...request.model, ...patch } };
      expect(parseResolvedRequestJson(serializeResolvedRequest(edited)))
        .toEqual(JSON.parse(JSON.stringify(edited)));
    }
    for (const patch of [{ output_gate: undefined }, { output_gate: false }, { qk_norm: false },
      { norm_dtype: "fp32" as const }, { index_norm: "rms" as const },
      { index_norm_dtype: undefined }, { sink_dtype: undefined }, { index_norm_dtype: null }, { sink_dtype: null }]) {
      const edited = { ...request, model: { ...request.model, attention_groups: [{ ...group, ...patch }] } };
      const parsed = parseResolvedRequestJson(JSON.stringify(edited));
      expect(parseResolvedRequestJson(serializeResolvedRequest(parsed)))
        .toEqual(JSON.parse(JSON.stringify(edited)));
    }
    for (const [patch, message] of [
      [{ attention_groups: undefined }, "必须提供注意力组"], [{ attention_groups: null }, "必须提供注意力组"],
      [{ attention_groups: [] }, "必须包含至少一个"],
      [{ attention_groups: [{ ...group, layers: 79 }] }, "注意力组层数之和"],
      [{ attention_groups: [{ ...group, index_dim: 192 }] }, "MLA/DSA 参数无效"],
      [{ attention_groups: [{ ...group, full_indexer_layers: 79 }] }, "MLA/DSA 参数无效"],
      [{ attention_groups: [{ ...group, num_attention_heads: 0 }] }, "num_attention_heads"],
      [{ moe_layers: 79 }, "MoE 层数不符合模型结构"],
      [{ experts_per_token: 257 }, "MoE 需要"],
      [{ shared_expert_intermediate_size: null }, "启用共享专家需要"],
      [{ weight_dtype: "fp8" }, "BF16/FP16"],
    ] as const) {
      const invalid = { ...request, model: { ...request.model, ...patch } } as unknown as ResolvedRequest;
      expect(() => parseResolvedRequestJson(JSON.stringify(invalid))).toThrow(message);
      expect(() => serializeResolvedRequest(invalid)).toThrow(message);
    }
  });

  it.each(["mla", "mla_dsa"] as const)("round-trips %s output gates with shared experts", (kind) => {
    const request: ResolvedRequest = {
      config: createDefaultSimConfig(), model: { ...initialModel, num_layers: 2,
        moe_layers: 2, num_experts: 8, experts_per_token: 2, expert_intermediate_size: 512,
        num_shared_experts: 2, shared_expert_intermediate_size: 1024,
        attention_layout: "grouped", attention_groups: [{ kind, layers: 2, num_attention_heads: 32,
          q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256,
          ...(kind === "mla_dsa" ? { index_heads: 32, index_dim: 128, topk: 2048, full_indexer_layers: 1 } : {}),
          output_gate: true, qk_norm: true, index_norm: "layer",
        }] }, hardware: initialHardware, storage: initialStorage,
      timing: "analytical", tp_size: 1, ep_size: 1,
    };
    expect(parseResolvedRequestJson(serializeResolvedRequest(request))).toEqual(request);
    const disabled = { ...request, model: { ...request.model,
      num_shared_experts: 0, shared_expert_intermediate_size: null,
      attention_groups: [{ ...request.model.attention_groups![0], output_gate: false }],
    } };
    expect(parseResolvedRequestJson(serializeResolvedRequest(disabled))).toEqual(disabled);
    for (const output_gate of [null, 1, "true"]) {
      expect(() => parseResolvedRequestJson(JSON.stringify({ ...request, model: { ...request.model,
        attention_groups: [{ ...request.model.attention_groups![0], output_gate }],
      } }))).toThrow("output_gate 必须是布尔值");
    }
    const gqa = { ...request, model: { ...request.model, attention_groups: [{
      kind: "gqa" as const, layers: 2, num_attention_heads: 32, num_kv_heads: 8, head_dim: 128,
      output_gate: true, qk_norm: true, index_norm: "layer" as const,
    }] } };
    expect(() => parseResolvedRequestJson(JSON.stringify(gqa))).toThrow("output_gate 仅支持 MLA/DSA");
    expect(() => serializeResolvedRequest(gqa)).toThrow("output_gate 仅支持 MLA/DSA");
    const gqaWithoutGate = { ...gqa, model: { ...gqa.model, attention_groups: [{
      ...gqa.model.attention_groups[0], output_gate: false,
    }] } };
    expect(parseResolvedRequestJson(serializeResolvedRequest(gqaWithoutGate))).toEqual(gqaWithoutGate);
  });

  it("rejects invalid shared expert dimensions and missing MoE layers on import and export", () => {
    const base: ResolvedRequest = { config: createDefaultSimConfig(),
      model: { ...initialModel, moe_layers: 1, num_experts: 8, experts_per_token: 2,
        expert_intermediate_size: 512, num_shared_experts: 1, shared_expert_intermediate_size: 1024 },
      hardware: initialHardware, storage: initialStorage, timing: "analytical", tp_size: 1, ep_size: 1 };
    const invalid = [
      [{ ...base.model, num_shared_experts: -1 }, "model.num_shared_experts"],
      [{ ...base.model, num_shared_experts: 0.5 }, "model.num_shared_experts"],
      [{ ...base.model, num_shared_experts: null }, "model.num_shared_experts"],
      [{ ...base.model, shared_expert_intermediate_size: 0 }, "model.shared_expert_intermediate_size"],
      [{ ...base.model, shared_expert_intermediate_size: 1.5 }, "model.shared_expert_intermediate_size"],
      [{ ...base.model, shared_expert_intermediate_size: "1024" }, "model.shared_expert_intermediate_size"],
      [{ ...base.model, shared_expert_intermediate_size: null }, "启用共享专家需要"],
      [{ ...base.model, shared_expert_intermediate_size: undefined }, "启用共享专家需要"],
      [{ ...base.model, moe_layers: 0 }, "启用共享专家需要"],
      [{ ...base.model, num_shared_experts: 0 }, "关闭共享专家时"],
    ] as const;
    for (const [model, message] of invalid) {
      expect(() => parseResolvedRequestJson(JSON.stringify({ ...base, model }))).toThrow(message);
      expect(() => serializeResolvedRequest({ ...base, model } as unknown as ResolvedRequest)).toThrow(message);
    }
    for (const model of [
      { ...base.model, num_shared_experts: Number.NaN },
      { ...base.model, shared_expert_intermediate_size: Number.POSITIVE_INFINITY },
    ]) {
      expect(() => serializeResolvedRequest({ ...base, model })).toThrow("有限数值");
    }
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
        ffn_intermediate_size: 32, vocab_size: 16,
        attention_layout: "full_gqa", attention_groups: null },
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
      [{ ...base, storage: { ...base.storage, l3_lookup_latency_ms: -1 } }, "storage.l3_lookup_latency_ms"],
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
    config.perf_model.l3_lookup_latency_ms = 0.75;
    config.engine.l3_prefetch_min_tokens = 0;
    expect(parseSimConfigJson(serializeSimConfig(config))).toEqual(config);
  });

  it("rejects missing, extra and mistyped KV3 fields rather than silently defaulting", () => {
    const base = createDefaultSimConfig();
    const { l3_capacity_tokens: _omitted, ...missing } = base.engine;
    expect(() => parseSimConfigJson(JSON.stringify({ ...base, engine: missing })))
      .toThrow("缺少参数：engine.l3_capacity_tokens");
    const { l3_prefetch_min_tokens: _noThreshold, ...withoutThreshold } = base.engine;
    expect(() => parseSimConfigJson(JSON.stringify({ ...base, engine: withoutThreshold })))
      .toThrow("缺少参数：engine.l3_prefetch_min_tokens");
    const { l3_lookup_latency_ms: _noLookup, ...withoutLookup } = base.perf_model;
    expect(() => parseSimConfigJson(JSON.stringify({ ...base, perf_model: withoutLookup })))
      .toThrow("缺少参数：perf_model.l3_lookup_latency_ms");
    expect(() => parseSimConfigJson(JSON.stringify({
      ...base, perf_model: { ...base.perf_model, unknown_link: 1 },
    }))).toThrow("不支持的参数：perf_model.unknown_link");
    expect(() => parseSimConfigJson(JSON.stringify({
      ...base, perf_model: { ...base.perf_model, kv_bytes_per_token: "2048" },
    }))).toThrow("perf_model.kv_bytes_per_token 必须是数值或 null");
    expect(() => parseSimConfigJson(JSON.stringify({
      ...base, engine: { ...base.engine, l3_capacity_tokens: 64 },
    }))).toThrow("engine.l3_capacity_tokens");
    expect(() => parseSimConfigJson(JSON.stringify({
      ...base, engine: { ...base.engine, l3_prefetch_min_tokens: -1 },
    }))).toThrow("engine.l3_prefetch_min_tokens");
    expect(() => parseSimConfigJson(JSON.stringify({
      ...base, perf_model: { ...base.perf_model, l3_lookup_latency_ms: -1 },
    }))).toThrow("perf_model.l3_lookup_latency_ms");
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
    config.perf_model.l2_to_l1_bandwidth_bytes_per_ms = null;
    config.perf_model.l3_lookup_latency_ms = Number.NaN;
    expect(() => serializeSimConfig(config)).toThrow("l3_lookup_latency_ms");
  });
});
