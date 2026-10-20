import type { JobSummary, ScanPoint } from "../api/types";

export type Role = "x" | "color" | "shape" | "line";
export type Roles = Record<Role, string>;
export interface ScanData { job: JobSummary; points: ScanPoint[] }
export const palette = ["#6c63ff", "#f59e0b", "#10b981", "#ef4444", "#06b6d4", "#8b5cf6", "#84cc16"];
export const symbols = ["circle", "rect", "triangle", "diamond", "pin", "arrow", "roundRect", "emptyCircle"];
export const lineStyles: ("solid" | "dashed" | "dotted" | number[])[] = [
  "solid", "dashed", "dotted", [10, 4, 2, 4],
];

export function valueOf(point: ScanPoint, key: string): unknown {
  if (!key) return null;
  if (key in point.values) return point.values[key];
  const config = point.config;
  if (!config) return null;
  if (key === "inference.parallelism") return config.inference.tp_size;
  let value: unknown = config;
  for (const segment of key.split(".")) value = (value as Record<string, unknown> | null)?.[segment];
  if (key.endsWith("_len") && value && typeof value === "object") {
    const length = value as { dist: string; value?: number; mean?: number };
    value = length.dist === "fixed" ? length.value : length.mean;
  }
  if (key === "shared_l3.available_gb" && typeof value === "number") value /= 1000;
  if ((key === "inference.hardware.memory_bytes_per_gpu" ||
       key === "inference.storage.runtime_reserve_bytes") && typeof value === "number") value /= 1e9;
  return value ?? null;
}

export function availableKeys(data: ScanData[]): string[] {
  return [...new Set(data.flatMap(({ job }) =>
    Object.values(job.axes ?? {}).map((axis) => axis.key)))];
}

export function metricsOf(data: ScanData[]): string[] {
  return [...new Set(data.flatMap(({ points }) => points.flatMap((point) =>
    Object.keys(point.metrics ?? {}).filter((key) => typeof point.metrics?.[key] === "number"))))].sort();
}

export function buildChart(data: ScanData[], roles: Roles, metric: string) {
  const candidates = data.flatMap(({ job, points }) => points.map((point) => ({ job, point })));
  const seen = new Map<string, { job: JobSummary; point: ScanPoint; sources: string[] }>();
  for (const { job, point } of candidates) {
    const key = point.signature ?? `${job.id}:${JSON.stringify(point.values)}`;
    const prior = seen.get(key);
    if (prior) prior.sources.push(job.id);
    else seen.set(key, { job, point, sources: [job.id] });
  }
  const entries = [...seen.values()];
  const values = (key: string): unknown[] => [...new Set<unknown>(
    entries.map(({ point }) => valueOf(point, key)).filter((v) => v != null),
  )];
  const xValues = values(roles.x).sort((a, b) => typeof a === "number" && typeof b === "number"
    ? a - b : String(a).localeCompare(String(b), "zh-CN"));
  const colors = values(roles.color);
  const shapes = values(roles.shape);
  const lines = values(roles.line);
  const hidden = availableKeys(data).filter((key) => !Object.values(roles).includes(key));
  const seriesMap = new Map<string, { group: string; name: string; color: number; shape: number; line: number;
    points: Map<string, number | null>; sources: string[] }>();
  for (const { job, point, sources } of entries) {
    const x = valueOf(point, roles.x);
    if (x == null) continue;
    const color = valueOf(point, roles.color);
    const shape = valueOf(point, roles.shape);
    const line = valueOf(point, roles.line);
    const hiddenValues = hidden.map((key) => [key, valueOf(point, key)]);
    const group = JSON.stringify([job.id, color, shape, line, hiddenValues]);
    let item = seriesMap.get(group);
    if (!item) {
      const name = [`扫描 ${job.id.slice(0, 6)}`,
        roles.color && `${roles.color}=${String(color)}`,
        roles.shape && `${roles.shape}=${String(shape)}`,
        roles.line && `${roles.line}=${String(line)}`,
        ...hiddenValues.map(([key, value]) => `${key}=${String(value)}`)].filter(Boolean).join(" · ");
      item = { group, name, color: Math.max(0, colors.indexOf(color)),
        shape: Math.max(0, shapes.indexOf(shape)), line: Math.max(0, lines.indexOf(line)),
        points: new Map(), sources };
      seriesMap.set(group, item);
    }
    const measured = point.metrics?.[metric];
    const y = point.status === "completed" && typeof measured === "number" ? measured : null;
    item.points.set(String(x), y);
  }
  const series = [...seriesMap.values()].map((item) => ({
    name: item.name, type: "line" as const, smooth: true, connectNulls: false, showSymbol: true,
    symbol: symbols[item.shape % symbols.length], symbolSize: 9,
    itemStyle: { color: palette[item.color % palette.length] },
    lineStyle: { color: palette[item.color % palette.length], type: lineStyles[item.line % lineStyles.length] },
    data: xValues.map((x) => item.points.get(String(x)) ?? null),
  }));
  return { xValues: xValues.map(String), entries, series, deduplicated: candidates.length - entries.length };
}
