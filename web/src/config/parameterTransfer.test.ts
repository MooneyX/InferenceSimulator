import { describe, expect, it } from "vitest";

import { createDefaultSimConfig } from "./simulation";
import { parseSimConfigJson, serializeSimConfig } from "./parameterTransfer";

describe("parameter JSON transfer", () => {
  it("round-trips the complete API config without result data", () => {
    const config = createDefaultSimConfig();
    config.workload.synthetic!.prompt_len = { dist: "fixed", value: 256 };
    config.workload.synthetic!.output_len = { dist: "lognormal", mean: 128, sigma: 0.5 };
    config.workload.synthetic!.num_requests = 20;
    config.workload.synthetic!.session!.num_sessions = 5;

    const json = serializeSimConfig(config);
    expect(parseSimConfigJson(json)).toEqual(config);
    expect(JSON.parse(json)).toHaveProperty("workload.synthetic.prompt_len", {
      dist: "fixed", value: 256,
    });
    expect(JSON.parse(json)).not.toHaveProperty("metrics");
    expect(JSON.parse(json)).not.toHaveProperty("requests");
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

  it("refuses to export invalid drafts instead of silently converting NaN to null", () => {
    const config = createDefaultSimConfig();
    config.workload.synthetic!.arrival_rate = Number.NaN;
    expect(() => serializeSimConfig(config)).toThrow("arrival_rate");
  });
});
