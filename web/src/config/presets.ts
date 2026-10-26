import type { PresetCatalog } from "../api/types";

export const hardwarePresetFields = [
  "name", "memory_bytes_per_gpu", "effective_fp16_tflops", "effective_fp8_tflops",
  "effective_hbm_gb_s", "effective_interconnect_gb_s", "collective_latency_ms",
  "ep_communication_latency_ms", "kernel_overhead_ms",
] as const;

export function isPresetAxis(key: string): boolean {
  return key === "inference.model" || key === "inference.hardware";
}

export function presetEntries(catalog: PresetCatalog | null, key: string) {
  return key === "inference.model" ? catalog?.models ?? []
    : key === "inference.hardware" ? catalog?.hardware ?? [] : [];
}

export function parsePresetNames(text: string): string[] {
  let names: unknown;
  try { names = JSON.parse(text); } catch { throw new Error("预设档位必须为名称数组"); }
  if (!Array.isArray(names) || names.some((name) => typeof name !== "string")) {
    throw new Error("预设档位必须为名称数组");
  }
  if (new Set(names).size !== names.length) throw new Error("预设档位不能重复");
  return names;
}
