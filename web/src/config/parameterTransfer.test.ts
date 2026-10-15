import { describe, expect, it } from "vitest";

import type { PhysicalInferenceConfig } from "../api/types";
import { initialHardware, initialModel, initialStorage, createDefaultSimConfig, createDefaultPhysicalSimConfig } from "./simulation";
import { parseSimConfigJson, serializeSimConfig } from "./parameterTransfer";

function parsePhysicalJson(json: string): PhysicalInferenceConfig {
  const config = parseSimConfigJson(JSON.stringify({ ...createDefaultPhysicalSimConfig(), shared_l3: null, inference: JSON.parse(json) }));
  if (config.inference.mode !== "physical") throw new Error("预期物理配置");
  return config.inference;
}

function serializePhysical(inference: PhysicalInferenceConfig): string {
  const json = serializeSimConfig({ ...createDefaultPhysicalSimConfig(), shared_l3: null, inference });
  return JSON.stringify(JSON.parse(json).inference);
}

describe("parameter JSON transfer", () => {
  it("rejects legacy root layouts and resolved wrappers without compatibility", () => {
    const old = { num_instances: 1, workload: createDefaultSimConfig().workload, engine: {}, perf_model: {} };
    expect(() => parseSimConfigJson(JSON.stringify(old))).toThrow("缺少参数：scheduler");
    expect(() => parseSimConfigJson(JSON.stringify({ config: createDefaultSimConfig(), model: initialModel }))).toThrow("缺少参数：num_instances");
    expect(() => parseSimConfigJson(JSON.stringify({ ...createDefaultSimConfig(), engine: {} }))).toThrow("不支持的参数：engine");
  });

  it("imports minimal manual settings using schema defaults rather than UI trial values", () => {
    const config = parseSimConfigJson(JSON.stringify({ num_instances: 1,
      workload: { session_rate: 1, synthetic: { num_requests: 3,
        prompt_len: { dist: "fixed", value: 64 }, turn_increment_len: { dist: "fixed", value: 16 },
        output_len: { dist: "fixed", value: 8 } } },
      scheduler: { max_running_requests: 2 },
      inference: { mode: "manual", kv: { max_total_tokens: 128 },
        timing: { prefill_c1: 1, decode_d1: 2, decode_d2: 0.1 } },
    }));
    expect(config.scheduler).toEqual({ max_running_requests: 2, max_prefill_tokens: 16_384, l3_prefetch_min_tokens: 256, pd_mutually_exclusive: true });
    expect(config.inference).toMatchObject({ mode: "manual", page_size: 64,
      kv: { max_total_tokens: 128, l2_capacity_tokens: 0 },
      timing: { mode: "empirical", prefill_c2: 0 } });
    expect(config.workload).toMatchObject({ mode: "synthetic", arrival_mode: "closed_loop", seed: 0, session_rate: 1 });
    expect(config.inference).toHaveProperty("max_output_tokens", 65_536);
    expect(config.workload.synthetic).toMatchObject({ output_len: { dist: "fixed", value: 8 },
      session: { num_sessions: null, topology: "linear", think_time_ms: 0 } });
  });

  it("requires an explicit synthetic output length distribution on import", () => {
    const config = createDefaultSimConfig();
    const { output_len: _outputLength, ...synthetic } = config.workload.synthetic!;
    const workload = { ...config.workload, synthetic };
    expect(() => parseSimConfigJson(JSON.stringify({ ...config, workload })))
      .toThrow("缺少参数：workload.synthetic.output_len");
  });

  it("imports physical defaults and a trace with nullable QPS without a reusable plan", () => {
    const config = parseSimConfigJson(JSON.stringify({ num_instances: 1, scheduler: { max_running_requests: 2 },
      workload: { mode: "trace", arrival_mode: "open_loop", session_rate: 1, target_qps: null, trace: {} },
      inference: { mode: "physical", model: { name: "toy", num_layers: 1, hidden_size: 16,
        num_attention_heads: 2, num_kv_heads: 1, head_dim: 8, ffn_intermediate_size: 32, vocab_size: 16 },
        hardware: { name: "GPU", memory_bytes_per_gpu: 1_000_000, effective_fp16_tflops: 1, effective_hbm_gb_s: 1 },
        storage: { l1_available_gb: 0.001 } },
    }));
    expect(config.inference).toMatchObject({ page_size: 64, tp_size: 1, ep_size: 1,
      timing: { mode: "analytical" }, model: { weight_dtype: "bf16", moe_layers: 0, attention_layout: "full_gqa" },
      hardware: { gpu_count: 1, kernel_overhead_ms: 0 }, storage: { l2_available_gb: 0 } });
    expect(config.workload).toMatchObject({ seed: 0, target_qps: null, trace: { plan_id: null } });
  });

  it("keeps analytical timing free of empirical placeholders", () => {
    const config = createDefaultPhysicalSimConfig();
    expect(parseSimConfigJson(serializeSimConfig(config)).inference.timing).toEqual({ mode: "analytical" });
    expect(() => parseSimConfigJson(JSON.stringify({ ...config, inference: { ...config.inference,
      timing: { mode: "analytical", prefill_c1: 0.05 } } }))).toThrow("inference.timing.prefill_c1");
  });

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
    const request: PhysicalInferenceConfig = {
      ...createDefaultPhysicalSimConfig().inference, model: { ...initialModel, name: "custom", num_layers: 2,
        moe_layers: 2, num_experts: 8, experts_per_token: 2, expert_intermediate_size: 512,
        norms: { kind: "layer", block_per_layer: 3, final_norm: false, dtype },
        router: { dtype, correction_bias: true, correction_dtype: dtype },
        ihc: { mult: 2, dtype },
        attention_layout: "grouped", attention_groups: [{ kind: "mla_dsa", layers: 2,
          num_attention_heads: 32, q_rank: 2048, kv_rank: 512, rope_dim: 64,
          nope_dim: 192, value_dim: 256, index_heads: 32, index_dim: 128,
          topk: 2048, full_indexer_layers: 1, output_gate: true,
          qk_norm: false, norm_dtype: dtype, index_norm: "rms", index_norm_dtype: dtype, sink_dtype: dtype,
        }] }, hardware: initialHardware, storage: initialStorage, timing: { mode: "analytical" as const }, tp_size: 1, ep_size: 1,
    };
    expect(parsePhysicalJson(serializePhysical(request))).toEqual(request);
    const disabled: PhysicalInferenceConfig = { ...request, model: { ...request.model,
      norms: { kind: "none", block_per_layer: 0, final_norm: false, dtype: null },
      router: { dtype: null, correction_bias: false, correction_dtype: "fp32" }, ihc: null,
      attention_groups: [{ ...request.model.attention_groups![0], qk_norm: true,
        norm_dtype: null, index_norm: "none", index_norm_dtype: null, sink_dtype: null }],
    } };
    expect(parsePhysicalJson(serializePhysical(disabled))).toEqual(disabled);
  });

  it("rejects invalid auxiliary settings on import and export", () => {
    const base: PhysicalInferenceConfig = { ...createDefaultPhysicalSimConfig().inference, model: { ...initialModel,
      attention_layout: "grouped", attention_groups: [{ kind: "gqa", layers: initialModel.num_layers,
        num_attention_heads: 32, num_kv_heads: 8, head_dim: 128, output_gate: false,
        qk_norm: true, index_norm: "layer" }],
    }, hardware: initialHardware, storage: initialStorage, timing: { mode: "analytical" as const }, tp_size: 1, ep_size: 1 };
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
      const request = { ...base, model: { ...base.model, ...patch } } as unknown as PhysicalInferenceConfig;
      expect(() => parsePhysicalJson(JSON.stringify(request))).toThrow(message);
      expect(() => serializePhysical(request)).toThrow(message);
    }
    for (const [patch, message] of [
      [{ qk_norm: null }, "qk_norm"], [{ norm_dtype: "fp8" }, "norm_dtype"],
      [{ index_norm: "unknown" }, "index_norm"], [{ index_norm_dtype: "fp8" }, "index_norm_dtype"],
      [{ sink_dtype: false }, "sink_dtype"], [{ index_norm: "none" }, "仅支持 DSA"],
      [{ index_norm_dtype: "fp32" }, "仅支持 DSA"],
    ] as const) {
      const request = { ...base, model: { ...base.model,
        attention_groups: [{ ...base.model.attention_groups![0], ...patch }],
      } } as unknown as PhysicalInferenceConfig;
      expect(() => parsePhysicalJson(JSON.stringify(request))).toThrow(message);
      expect(() => serializePhysical(request)).toThrow(message);
    }
    const legacy = { ...base, model: { ...base.model, norms: undefined, router: undefined, ihc: undefined } };
    const parsed = parsePhysicalJson(serializePhysical(legacy));
    expect(parsed.model).not.toHaveProperty("norms");
    expect(parsed.model).not.toHaveProperty("router");
    expect(parsed.model).not.toHaveProperty("ihc");
    const gqaSink = { ...base, model: { ...base.model, attention_groups: [{
      ...base.model.attention_groups![0], sink_dtype: "fp32" as const,
    }] } };
    expect(parsePhysicalJson(serializePhysical(gqaSink))).toEqual(gqaSink);
  });

  it.each([createDefaultSimConfig, createDefaultPhysicalSimConfig])("round-trips trace settings without reusing a temporary plan ID", (createConfig) => {
    const config = { ...createConfig(), workload: { mode: "trace" as const,
      arrival_mode: "open_loop" as const, session_rate: 2, seed: 7, target_qps: 7.5,
      trace: { plan_id: "temporary-plan" } } };
    const json = serializeSimConfig(config);
    expect(json).not.toContain("temporary-plan");
    expect(parseSimConfigJson(json)).toEqual({ ...config, workload: { ...config.workload,
      trace: { ...config.workload.trace, plan_id: null } } });
    expect(config.workload.trace.plan_id).toBe("temporary-plan");
  });

  it("round-trips closed_loop mode without a request QPS", () => {
    const config = createDefaultSimConfig();
    config.workload.arrival_mode = "closed_loop";
    delete config.workload.target_qps;
    config.workload.session_rate = 2;
    config.workload.synthetic = {
      num_requests: 20,
      session: { num_sessions: 5, topology: "linear", think_time_ms: 200 },
      prompt_len: { dist: "fixed", value: 256 },
      turn_increment_len: { dist: "fixed", value: 32 },
      output_len: { dist: "fixed", value: 16 },
    };
    const json = serializeSimConfig(config);
    expect(parseSimConfigJson(json)).toEqual(config);
    expect(JSON.parse(json).workload.synthetic).not.toHaveProperty("arrival_rate");
    expect(JSON.parse(json).workload.synthetic).not.toHaveProperty("session.target_ideal_prefix_hit_rate");
  });

  it.each(["arrival_mode", "arrival_rate", "session_rate", "seed", "max_new_tokens"])("rejects removed synthetic field %s", (field) => {
    const config = createDefaultSimConfig();
    const workload = { ...config.workload, synthetic: { ...config.workload.synthetic, [field]: 1 } };
    expect(() => parseSimConfigJson(JSON.stringify({ ...config, workload })))
      .toThrow(`不支持的参数：workload.synthetic.${field}`);
  });

  it.each(["mode", "session_rate", "seed", "target_qps"])("rejects removed trace field %s", (field) => {
    const config = createDefaultSimConfig();
    const workload = { mode: "trace", arrival_mode: "open_loop", session_rate: 1,
      trace: { plan_id: null, [field]: 1 } };
    expect(() => parseSimConfigJson(JSON.stringify({ ...config, workload })))
      .toThrow(`不支持的参数：workload.trace.${field}`);
  });

  it.each(["synthetic", "trace"])("rejects the old session arrival mode for %s", (source) => {
    const config = createDefaultSimConfig();
    const workload = source === "synthetic"
      ? { ...config.workload, arrival_mode: "session" }
      : { mode: "trace", arrival_mode: "session", session_rate: 1, trace: {} };
    expect(() => parseSimConfigJson(JSON.stringify({ ...config, workload })))
      .toThrow("仅支持 open_loop 或 closed_loop");
  });

  it.each(["", "{", "null", "[]", '"hello"'])
  ("rejects non-config JSON: %s", (json) => {
    expect(() => parseSimConfigJson(json)).toThrow();
  });

  it.each([
    [{}, "缺少参数：num_instances"],
    [{ ...createDefaultSimConfig(), metrics: {} }, "不支持的参数：metrics"],
    [{ ...createDefaultSimConfig(), workload: { mode: "trace", arrival_mode: "closed_loop", session_rate: 1, seed: 0, trace: {}, synthetic: createDefaultSimConfig().workload.synthetic } }, "Trace 负载不能同时配置 workload.synthetic"],
    [{ ...createDefaultSimConfig(), workload: { mode: "synthetic" } }, "workload.synthetic 必须为 JSON 对象"],
    [{ ...createDefaultSimConfig(), scheduler: { ...createDefaultSimConfig().scheduler, capacity_pages: 10 } },
      "不支持的参数：scheduler.capacity_pages"],
    [{ ...createDefaultSimConfig(), inference: { ...createDefaultSimConfig().inference, page_size: "64" } },
      "inference.page_size 必须是数值"],
    [{ ...createDefaultSimConfig(), inference: { ...createDefaultSimConfig().inference,
      kv: { ...createDefaultSimConfig().inference.kv, max_total_tokens: 130 } } },
      "inference.kv.max_total_tokens"],
    [{ ...createDefaultSimConfig(), workload: {
      ...createDefaultSimConfig().workload,
      synthetic: { ...createDefaultSimConfig().workload.synthetic!, session: null },
    } }, "workload.synthetic.session 必须为 JSON 对象"],
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
    const request: PhysicalInferenceConfig = {
      ...createDefaultPhysicalSimConfig().inference, model: { ...initialModel, num_layers: 2,
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
      timing: { mode: "analytical" as const }, tp_size: 1, ep_size: 1,
    };
    expect(parsePhysicalJson(serializePhysical(request))).toEqual(request);
    expect(() => parsePhysicalJson(JSON.stringify({ ...request, model: {
      ...request.model, attention_groups: [{ ...request.model.attention_groups![0], layers: 2 },
        request.model.attention_groups![1]],
    } }))).toThrow("注意力组层数之和");
  });

  it.each(["grouped", "hy4_preview"] as const)("round-trips editable HY4 parameters with %s layout and rejects only invalid structures", (attention_layout) => {
    const request: PhysicalInferenceConfig = {
      ...createDefaultPhysicalSimConfig().inference,
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
      storage: initialStorage, timing: { mode: "analytical" as const }, tp_size: 8, ep_size: 8,
    };
    expect(parsePhysicalJson(serializePhysical(request))).toEqual(request);
    const group = request.model.attention_groups![0];
    const changed: PhysicalInferenceConfig = { ...request, model: { ...request.model,
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
    expect(parsePhysicalJson(serializePhysical(changed))).toEqual(changed);
    for (const patch of [{ revision: null }, { num_shared_experts: 0, shared_expert_intermediate_size: null },
      { norms: undefined, router: undefined, ihc: undefined }, { ihc: null }]) {
      const edited = { ...request, model: { ...request.model, ...patch } };
      expect(parsePhysicalJson(serializePhysical(edited)))
        .toEqual(JSON.parse(JSON.stringify(edited)));
    }
    for (const patch of [{ output_gate: undefined }, { output_gate: false }, { qk_norm: false },
      { norm_dtype: "fp32" as const }, { index_norm: "rms" as const },
      { index_norm_dtype: undefined }, { sink_dtype: undefined }, { index_norm_dtype: null }, { sink_dtype: null }]) {
      const edited = { ...request, model: { ...request.model, attention_groups: [{ ...group, ...patch }] } };
      const parsed = parsePhysicalJson(JSON.stringify(edited));
      expect(parsePhysicalJson(serializePhysical(parsed)))
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
      const invalid = { ...request, model: { ...request.model, ...patch } } as unknown as PhysicalInferenceConfig;
      expect(() => parsePhysicalJson(JSON.stringify(invalid))).toThrow(message);
      expect(() => serializePhysical(invalid)).toThrow(message);
    }
  });

  it.each(["mla", "mla_dsa"] as const)("round-trips %s output gates with shared experts", (kind) => {
    const request: PhysicalInferenceConfig = {
      ...createDefaultPhysicalSimConfig().inference, model: { ...initialModel, num_layers: 2,
        moe_layers: 2, num_experts: 8, experts_per_token: 2, expert_intermediate_size: 512,
        num_shared_experts: 2, shared_expert_intermediate_size: 1024,
        attention_layout: "grouped", attention_groups: [{ kind, layers: 2, num_attention_heads: 32,
          q_rank: 2048, kv_rank: 512, rope_dim: 64, nope_dim: 192, value_dim: 256,
          ...(kind === "mla_dsa" ? { index_heads: 32, index_dim: 128, topk: 2048, full_indexer_layers: 1 } : {}),
          output_gate: true, qk_norm: true, index_norm: "layer",
        }] }, hardware: initialHardware, storage: initialStorage,
      timing: { mode: "analytical" as const }, tp_size: 1, ep_size: 1,
    };
    expect(parsePhysicalJson(serializePhysical(request))).toEqual(request);
    const disabled = { ...request, model: { ...request.model,
      num_shared_experts: 0, shared_expert_intermediate_size: null,
      attention_groups: [{ ...request.model.attention_groups![0], output_gate: false }],
    } };
    expect(parsePhysicalJson(serializePhysical(disabled))).toEqual(disabled);
    for (const output_gate of [null, 1, "true"]) {
      expect(() => parsePhysicalJson(JSON.stringify({ ...request, model: { ...request.model,
        attention_groups: [{ ...request.model.attention_groups![0], output_gate }],
      } }))).toThrow("output_gate 必须是布尔值");
    }
    const gqa = { ...request, model: { ...request.model, attention_groups: [{
      kind: "gqa" as const, layers: 2, num_attention_heads: 32, num_kv_heads: 8, head_dim: 128,
      output_gate: true, qk_norm: true, index_norm: "layer" as const,
    }] } };
    expect(() => parsePhysicalJson(JSON.stringify(gqa))).toThrow("output_gate 仅支持 MLA/DSA");
    expect(() => serializePhysical(gqa)).toThrow("output_gate 仅支持 MLA/DSA");
    const gqaWithoutGate = { ...gqa, model: { ...gqa.model, attention_groups: [{
      ...gqa.model.attention_groups[0], output_gate: false,
    }] } };
    expect(parsePhysicalJson(serializePhysical(gqaWithoutGate))).toEqual(gqaWithoutGate);
  });

  it("rejects invalid shared expert dimensions and missing MoE layers on import and export", () => {
    const base: PhysicalInferenceConfig = { ...createDefaultPhysicalSimConfig().inference,
      model: { ...initialModel, moe_layers: 1, num_experts: 8, experts_per_token: 2,
        expert_intermediate_size: 512, num_shared_experts: 1, shared_expert_intermediate_size: 1024 },
      hardware: initialHardware, storage: initialStorage, timing: { mode: "analytical" as const }, tp_size: 1, ep_size: 1 };
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
      expect(() => parsePhysicalJson(JSON.stringify({ ...base, model }))).toThrow(message);
      expect(() => serializePhysical({ ...base, model } as unknown as PhysicalInferenceConfig)).toThrow(message);
    }
    for (const model of [
      { ...base.model, num_shared_experts: Number.NaN },
      { ...base.model, shared_expert_intermediate_size: Number.POSITIVE_INFINITY },
    ]) {
      expect(() => serializePhysical({ ...base, model })).toThrow("有限数值");
    }
  });

  it("keeps explicit physical parallelism and rejects invalid EP widths", () => {
    const request = {
      ...createDefaultPhysicalSimConfig().inference, model: { name: "toy", num_layers: 1,
        hidden_size: 16, num_attention_heads: 2, num_kv_heads: 1,
        head_dim: 8, ffn_intermediate_size: 32, vocab_size: 16 },
      hardware: { name: "GPU", gpu_count: 1, memory_bytes_per_gpu: 1000000,
        effective_fp16_tflops: 1, effective_hbm_gb_s: 1 },
      storage: { l1_available_gb: 0.001, l2_available_gb: 0 },
    };
    expect(parsePhysicalJson(JSON.stringify(request)).ep_size).toBe(1);
    expect(() => parsePhysicalJson(JSON.stringify({ ...request, ep_size: 0 })))
      .toThrow("ep_size 必须是正整数");
  });

  it("rejects malformed physical specs and non-finite automatic drafts", () => {
    const base: PhysicalInferenceConfig = {
      ...createDefaultPhysicalSimConfig().inference,
      model: { ...initialModel, name: "toy", num_layers: 2, hidden_size: 16,
        num_attention_heads: 2, num_kv_heads: 1, head_dim: 8,
        ffn_intermediate_size: 32, vocab_size: 16,
        attention_layout: "full_gqa", attention_groups: null },
      hardware: { ...initialHardware, name: "GPU", gpu_count: 1,
        memory_bytes_per_gpu: 1_000_000, effective_fp16_tflops: 1, effective_hbm_gb_s: 1,
        ep_communication_latency_ms: null },
      storage: { ...initialStorage, l1_available_gb: 0.001 }, tp_size: 1, ep_size: 1,
      timing: { mode: "analytical" as const },
    };
    expect(parsePhysicalJson(serializePhysical(base))).toEqual(base);
    expect(() => parsePhysicalJson(JSON.stringify({ ...base,
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
      [{ ...base, unexpected: true }, "不支持的参数：inference.unexpected"],
    ] as const;
    for (const [request, message] of invalid) {
      expect(() => parsePhysicalJson(JSON.stringify(request))).toThrow(message);
    }
    expect(() => serializePhysical({ ...base, hardware: {
      ...base.hardware, ep_communication_latency_ms: Number.POSITIVE_INFINITY,
    } })).toThrow("ep_communication_latency_ms");
    expect(() => serializePhysical({ ...base, model: {
      ...base.model, num_experts: Number.NaN,
    } })).toThrow("num_experts");
  });

  it("round-trips a three-tier config including link speeds and latency", () => {
    const config = createDefaultSimConfig();
    config.inference.kv.l2_capacity_tokens = 128;
    config.shared_l3!.capacity_tokens = 64;
    config.inference.kv.kv_bytes_per_token = 2048;
    config.inference.kv.l2_to_l1_bandwidth_bytes_per_ms = 100;
    config.inference.kv.l3_to_l2_bandwidth_bytes_per_ms = 50;
    config.inference.kv.l2_latency_ms = 1;
    config.inference.kv.l3_latency_ms = 2;
    config.shared_l3!.lookup_latency_ms = 0.75;
    config.scheduler.l3_prefetch_min_tokens = 0;
    expect(parseSimConfigJson(serializeSimConfig(config))).toEqual(config);
  });

  it("accepts defaults but rejects extra and mistyped KV3 fields", () => {
    const base = createDefaultSimConfig();
    const { shared_l3: _omitted, ...withoutL3 } = base;
    for (const [kv, message] of [
      [{ ...base.inference.kv, unknown_link: 1 }, "不支持的参数：inference.kv.unknown_link"],
      [{ ...base.inference.kv, kv_bytes_per_token: "2048" }, "inference.kv.kv_bytes_per_token 必须是数值或 null"],
      [{ ...base.inference.kv, l3_capacity_tokens: 64 }, "inference.kv.l3_capacity_tokens"],
      [{ ...base.inference.kv, l3_lookup_latency_ms: -1 }, "inference.kv.l3_lookup_latency_ms"],
    ] as const) {
      expect(() => parseSimConfigJson(JSON.stringify({ ...base, inference: { ...base.inference, kv } }))).toThrow(message);
    }
    const { l3_prefetch_min_tokens: _noThreshold, ...withoutThreshold } = base.scheduler;
    expect(parseSimConfigJson(JSON.stringify({ ...base, scheduler: withoutThreshold })).scheduler.l3_prefetch_min_tokens).toBe(256);
    expect(parseSimConfigJson(JSON.stringify(withoutL3)).shared_l3).toBeNull();
    expect(parseSimConfigJson(JSON.stringify({ ...base, shared_l3: { mode: "manual" } })).shared_l3)
      .toEqual({ mode: "manual", capacity_tokens: 0, lookup_latency_ms: 0 });
    expect(() => parseSimConfigJson(JSON.stringify({ ...base, scheduler: { ...base.scheduler, l3_prefetch_min_tokens: -1 } })))
      .toThrow("scheduler.l3_prefetch_min_tokens");
  });

  it("refuses to export invalid drafts instead of silently converting non-finite values to null", () => {
    const config = createDefaultSimConfig();
    config.workload.target_qps = Number.NaN;
    expect(() => serializeSimConfig(config)).toThrow("target_qps");
    config.workload.target_qps = 1;
    config.inference.kv.kv_bytes_per_token = Number.POSITIVE_INFINITY;
    expect(() => serializeSimConfig(config)).toThrow("kv_bytes_per_token");
    config.inference.kv.kv_bytes_per_token = null;
    config.inference.kv.l2_to_l1_bandwidth_bytes_per_ms = Number.POSITIVE_INFINITY;
    expect(() => serializeSimConfig(config)).toThrow("l2_to_l1_bandwidth_bytes_per_ms");
    config.inference.kv.l2_to_l1_bandwidth_bytes_per_ms = null;
    config.shared_l3!.lookup_latency_ms = Number.NaN;
    expect(() => serializeSimConfig(config)).toThrow("shared_l3.lookup_latency_ms");
  });
});
