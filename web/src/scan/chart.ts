import type { JobSummary, ScanParameter, ScanPoint } from "../api/types";

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

function configPaths(config: unknown, prefix = ""): string[] {
  if (config === null || typeof config !== "object" || Array.isArray(config)) return prefix ? [prefix] : [];
  return Object.entries(config).flatMap(([key, value]) => configPaths(value, prefix ? `${prefix}.${key}` : key));
}

export function availableKeys(data: ScanData[], parameters: ScanParameter[] = []): string[] {
  const axes = data.flatMap(({ job }) => (job.axes ?? []).map((axis) => axis.key));
  const configs = data.map(({ points }) => points.find((point) => point.config)?.config).filter(Boolean);
  const paths = new Set(configs.flatMap((config) => configPaths(config)));
  for (const key of [...axes, ...parameters.map((parameter) => parameter.key)]) {
    if (key.endsWith("_len")) {
      paths.delete(`${key}.value`);
      paths.delete(`${key}.mean`);
    }
  }
  if (paths.has("inference.tp_size")) {
    paths.delete("inference.tp_size");
    paths.delete("inference.hardware.gpu_count");
    paths.delete("inference.ep_size");
    paths.add("inference.parallelism");
  }
  const candidates = [...new Set([...axes, ...parameters.map((parameter) => parameter.key), ...paths])];
  const points = data.flatMap(({ points }) => points);
  const varying = candidates.filter((key) => new Set(points.filter((point) =>
    point.config || key in point.values).map((point) => JSON.stringify(valueOf(point, key)))).size > 1);
  return varying.length ? varying : points.length ? axes.slice(0, 1) : [];
}

export function metricsOf(data: ScanData[]): string[] {
  return [...new Set(data.flatMap(({ points }) => points.flatMap((point) =>
    Object.keys(point.metrics ?? {}).filter((key) => typeof point.metrics?.[key] === "number"))))].sort();
}

export function metricUnit(metric: string): string {
  if (["ttft_mean", "ttft_p50", "ttft_p99", "tpot_mean", "makespan", "restore_time_ms"].includes(metric)) return "ms";
  if (["throughput", "arrival_qps"].includes(metric)) return "请求/秒";
  if (metric.endsWith("_rate") || metric.endsWith("_ratio")) return "%";
  if (metric.endsWith("_tokens")) return "token";
  if (metric.endsWith("_blocks")) return "页";
  if (metric === "restore_bytes") return "字节";
  if (["num_sessions", "promotions", "demotions"].includes(metric) || metric.endsWith("_evictions")) return "次";
  return `${metric}（单位未知）`;
}

export type StyleChannel = "color" | "shape" | "line";
export function metricChannel(roles: Roles, count: number): StyleChannel | null {
  if (count < 2) return null;
  if (!roles.line && count <= lineStyles.length) return "line";
  if (!roles.shape && count <= symbols.length) return "shape";
  if (!roles.color && count <= palette.length) return "color";
  return null;
}

export function metricCharts(metrics: string[], roles: Roles): string[][] {
  const groups = new Map<string, string[]>();
  for (const metric of metrics) {
    const unit = metricUnit(metric);
    groups.set(unit, [...(groups.get(unit) ?? []), metric]);
  }
  const units = [...groups.values()];
  const charts = units.length <= 2 ? [metrics] : units;
  return charts.flatMap((items) => items.length > 1 && !metricChannel(roles, items.length)
    ? items.map((item) => [item]) : [items]);
}

export function buildChart(data: ScanData[], roles: Roles, metrics: string | string[],
  keys: string[] = availableKeys(data), styleSlots: Record<string, number> = {}) {
  const selectedMetrics = typeof metrics === "string" ? [metrics] : metrics;
  const channel = metricChannel(roles, selectedMetrics.length);
  const capacity = channel === "line" ? lineStyles.length : channel === "shape" ? symbols.length : palette.length;
  const occupied = new Set<number>();
  const slots = Object.fromEntries(selectedMetrics.map((metric) => {
    const preferred = styleSlots[metric];
    const slot = preferred !== undefined && preferred < capacity && !occupied.has(preferred)
      ? preferred : Array.from({ length: capacity }, (_, index) => index).find((index) => !occupied.has(index)) ?? 0;
    occupied.add(slot);
    return [metric, slot];
  }));
  const units = [...new Set(selectedMetrics.map(metricUnit))];
  const candidates = data.flatMap(({ job, points }) => points.map((point) => ({ job, point })));
  const seen = new Map<string, { job: JobSummary; point: ScanPoint; sources: string[]; copies: ScanPoint[] }>();
  for (const { job, point } of candidates) {
    const key = point.signature ?? `${job.id}:${JSON.stringify(point.values)}`;
    const prior = seen.get(key);
    if (prior) { prior.sources.push(job.id); prior.copies.push(point); }
    else seen.set(key, { job, point, sources: [job.id], copies: [point] });
  }
  const entries = [...seen.values()];
  const token = (value: unknown) => JSON.stringify(value ?? null);
  const label = (value: unknown) => value !== null && typeof value === "object"
    ? JSON.stringify(value) : String(value);
  const values = (key: string): unknown[] => [...new Map(entries.map(({ point }) => valueOf(point, key))
    .filter((value) => value != null).map((value) => [token(value), value])).values()];
  const xValues = values(roles.x).sort((a, b) => typeof a === "number" && typeof b === "number"
    ? a - b : label(a).localeCompare(label(b), "zh-CN"));
  const colors = values(roles.color);
  const shapes = values(roles.shape);
  const lines = values(roles.line);
  const hidden = keys.filter((key) => !Object.values(roles).includes(key));
  const seriesMap = new Map<string, { name: string; metric: string; tooltipValues: string[];
    color: number; shape: number; line: number; points: Map<string, number | null> }>();
  for (const { job, point, copies } of entries) {
    const x = valueOf(point, roles.x);
    if (x == null) continue;
    const color = valueOf(point, roles.color);
    const shape = valueOf(point, roles.shape);
    const line = valueOf(point, roles.line);
    const hiddenValues = hidden.map((key) => [key, valueOf(point, key)]);
    for (const metric of selectedMetrics) {
      const group = JSON.stringify([job.id, color, shape, line, hiddenValues, metric]);
      let item = seriesMap.get(group);
      if (!item) {
        const name = [selectedMetrics.length > 1 && metric, `扫描 ${job.id.slice(0, 6)}`,
          roles.color && `${roles.color}=${label(color)}`,
          roles.shape && `${roles.shape}=${label(shape)}`,
          roles.line && `${roles.line}=${label(line)}`,
          ...hiddenValues.map(([key, value]) => `${key}=${label(value)}`)].filter(Boolean).join(" · ");
        const tooltipValues = [selectedMetrics.length > 1 ? `${metric} (${metricUnit(metric)})` : "",
          roles.color && color, roles.shape && shape, roles.line && line]
          .filter((value) => value !== "" && value != null).map(label);
        const style = slots[metric];
        item = { name, metric, tooltipValues,
          color: channel === "color" ? style : Math.max(0, colors.findIndex((v) => token(v) === token(color))),
          shape: channel === "shape" ? style : Math.max(0, shapes.findIndex((v) => token(v) === token(shape))),
          line: channel === "line" ? style : Math.max(0, lines.findIndex((v) => token(v) === token(line))),
          points: new Map() };
        seriesMap.set(group, item);
      }
      const source = copies.find((entry) => entry.status === "completed"
        && typeof entry.metrics?.[metric] === "number");
      const measured = source?.metrics?.[metric];
      const y = typeof measured === "number" ? measured * (metricUnit(metric) === "%" ? 100 : 1) : null;
      item.points.set(token(x), y);
    }
  }
  const series = [...seriesMap.values()].map((item) => ({
    name: item.name, metric: item.metric, tooltipValues: item.tooltipValues,
    yAxisIndex: units.indexOf(metricUnit(item.metric)),
    type: "line" as const, smooth: true, connectNulls: false, showSymbol: true,
    symbol: symbols[item.shape % symbols.length], symbolSize: 9,
    itemStyle: { color: palette[item.color % palette.length] },
    lineStyle: { color: palette[item.color % palette.length], type: lineStyles[item.line % lineStyles.length] },
    data: xValues.map((x) => item.points.get(token(x)) ?? null),
  }));
  return { xValues: xValues.map(label), entries, series, deduplicated: candidates.length - entries.length };
}
