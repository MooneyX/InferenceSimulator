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

export type HardwareScanLevel = { name: string; parallelism: number };

export function parseHardwareLevels(text: string): HardwareScanLevel[] {
  let entries: unknown;
  try { entries = JSON.parse(text); } catch { throw new Error("硬件档位必须包含名称和每实例 GPU / TP 数"); }
  if (!Array.isArray(entries)) throw new Error("硬件档位必须为数组");
  const names = new Set<string>();
  return entries.map((entry) => {
    if (!entry || typeof entry !== "object" || typeof entry.name !== "string" ||
        Object.keys(entry).some((key) => !["name", "parallelism"].includes(key))) {
      throw new Error("硬件档位必须包含名称和每实例 GPU / TP 数");
    }
    if (!Number.isSafeInteger(entry.parallelism) || entry.parallelism < 1) {
      throw new Error(`${entry.name} 的每实例 GPU / TP 数必须为正整数`);
    }
    if (names.has(entry.name)) throw new Error("每种硬件只能设置一个并行数量");
    names.add(entry.name);
    return { name: entry.name, parallelism: entry.parallelism };
  });
}

export function hardwareLevelLabel(value: unknown, fallbackParallelism?: number): string | null {
  if (value && typeof value === "object" && "name" in value && "parallelism" in value) {
    return `${value.name} · TP${value.parallelism}`;
  }
  if (typeof value === "string") return fallbackParallelism == null ? value : `${value} · TP${fallbackParallelism}`;
  return null;
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
