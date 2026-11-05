type StatDef = [key: string, label: string];
interface MetricDef { label: string; stats: StatDef[]; keywords?: string }

const single = (key: string, label: string, keywords?: string): MetricDef =>
  ({ label, stats: [[key, ""]], keywords });

const tierBlocks = (tier: "l1" | "l2" | "l3"): MetricDef[] => {
  const name = tier.toUpperCase();
  return [
    single(`${tier}_capacity_blocks`, `${name} 容量`),
    single(`${tier}_resident_blocks`, `${name} 驻留`, "结束时占用"),
    single(`${tier}_peak_resident_blocks`, `${name} 峰值驻留`),
    single(`${tier}_active_blocks`, `${name} 活跃`),
    single(`${tier}_evictable_blocks`, `${name} 可驱逐`),
    single(`${tier}_free_blocks`, `${name} 空闲`),
    single(`${tier}_evictions`, `${name} 驱逐`),
    ...(tier === "l1" ? [single("l1_reserved_future_blocks", "L1 预留未来页"),
      single("l1_internal_fragmentation_tokens", "L1 内部碎片")] : []),
  ];
};

const catalog: { name: string; items: MetricDef[] }[] = [
  { name: "延迟", items: [
    { label: "TTFT", keywords: "首 token 延迟", stats: [["ttft_mean", "均值"], ["ttft_p50", "P50"], ["ttft_p99", "P99"]] },
    { label: "TPOT", keywords: "decode 每 token 耗时", stats: [["tpot_mean", "均值"], ["tpot_p99", "P99"]] },
    { label: "ITL", keywords: "输出 token 间隔 卡顿",
      stats: [["itl_mean", "均值"], ["itl_p50", "P50"], ["itl_p99", "P99"], ["itl_max", "最大值"]] },
    single("makespan", "Makespan", "总时长 仿真时间"),
  ] },
  { name: "吞吐与负载", items: [
    single("throughput", "完成 QPS", "吞吐"),
    single("arrival_qps", "实际请求 QPS", "到达 发送"),
    single("num_sessions", "新建会话数"),
    single("root_request_ratio", "根请求占比"),
    single("mean_prompt_tokens", "完成请求平均输入"),
    single("mean_output_tokens", "完成请求平均输出"),
  ] },
  { name: "前缀命中率", items: [
    single("constructed_ideal_prefix_hit_rate", "构造理想命中率"),
    single("actual_prefix_hit_rate", "实际前缀命中率"),
    single("l1_prefix_hit_rate", "L1 前缀命中率"),
    single("l2_prefix_hit_rate", "L2 前缀命中率"),
    single("l3_prefix_hit_rate", "L3 前缀命中率"),
    single("child_ideal_prefix_hit_rate", "子轮理想命中率"),
    single("child_actual_prefix_hit_rate", "子轮实际命中率"),
    single("requests_with_hit_ratio", "有命中请求比例"),
    single("requests_fully_cached_ratio", "完整块全命中请求比例"),
  ] },
  { name: "Token 统计", items: [
    single("total_input_tokens", "总输入"),
    single("ideal_hit_tokens", "理想命中"),
    single("matched_cache_tokens", "缓存匹配"),
    single("actual_hit_tokens", "总实际命中"),
    single("l1_hit_tokens", "L1 命中"),
    single("l2_hit_tokens", "L2 命中"),
    single("l3_hit_tokens", "L3 命中"),
    single("actual_prefill_tokens", "实际 Prefill"),
    single("replayed_prefill_tokens", "重算 Prefill"),
    single("saved_prefill_tokens", "节省 Prefill"),
  ] },
  { name: "KV 迁移", items: [
    single("promotions", "提升到 L1"),
    single("demotions", "逐级降级"),
    single("restore_bytes", "Restore 传输量"),
    single("restore_time_ms", "Restore 耗时"),
  ] },
  { name: "L3 读取", items: [
    single("l3_read_steps", "L3 读取步数"),
    { label: "L3 停顿", stats: [["l3_stall_ms", "累计"], ["l3_stall_p50", "P50"], ["l3_stall_p99", "P99"],
      ["l3_stall_max", "最大值"], ["l3_stall_fraction", "占比"]] },
  ] },
  { name: "L1 KV 页", items: tierBlocks("l1") },
  { name: "L2 KV 页", items: tierBlocks("l2") },
  { name: "L3 KV 页", items: tierBlocks("l3") },
];

const byKey = new Map(catalog.flatMap(({ name, items }) => items.flatMap((def) =>
  def.stats.map(([key, stat]) => [key, { group: name, def, stat }] as const))));

export function metricUnit(metric: string): string {
  if (["ttft_mean", "ttft_p50", "ttft_p99", "tpot_mean", "tpot_p99", "makespan", "restore_time_ms"].includes(metric)
    || metric.startsWith("itl_") || (metric.startsWith("l3_stall_") && metric !== "l3_stall_fraction")) return "ms";
  if (["throughput", "arrival_qps"].includes(metric)) return "请求/秒";
  if (metric.endsWith("_rate") || metric.endsWith("_ratio") || metric.endsWith("_fraction")) return "%";
  if (metric.endsWith("_tokens")) return "token";
  if (metric.endsWith("_blocks")) return "页";
  if (metric === "restore_bytes") return "字节";
  if (["num_sessions", "promotions", "demotions", "l3_read_steps"].includes(metric) || metric.endsWith("_evictions")) return "次";
  return `${metric}（单位未知）`;
}

export function metricLabel(metric: string): string {
  const entry = byKey.get(metric);
  if (!entry) return metric;
  return entry.stat ? `${entry.def.label} ${entry.stat}` : entry.def.label;
}

export interface MetricStatOption { key: string; label: string; unit: string }
export interface MetricOption {
  id: string; label: string; unit: string; multi: boolean; stats: MetricStatOption[]; search: string;
}
export interface MetricGroup { name: string; items: MetricOption[] }

export function metricGroups(available: string[]): MetricGroup[] {
  const present = new Set(available);
  const option = (group: string, def: MetricDef): MetricOption | null => {
    const stats = def.stats.filter(([key]) => present.has(key))
      .map(([key, label]) => ({ key, label, unit: metricUnit(key) }));
    if (!stats.length) return null;
    const units = new Set(stats.map((stat) => stat.unit));
    return { id: def.label, label: def.label, unit: units.size === 1 ? stats[0].unit : "",
      multi: def.stats.length > 1, stats,
      search: [group, def.label, def.keywords ?? "", ...stats.flatMap((stat) => [stat.key, stat.label])]
        .join(" ").toLowerCase() };
  };
  const groups = catalog.map(({ name, items }) => ({ name,
    items: items.map((def) => option(name, def)).filter((item): item is MetricOption => item !== null) }));
  const others = available.filter((key) => !byKey.has(key));
  if (others.length) groups.push({ name: "其他", items: others.map((key) => option("其他", single(key, key))!) });
  return groups.filter((group) => group.items.length);
}

export function filterMetricGroups(groups: MetricGroup[], query: string): MetricGroup[] {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return groups;
  return groups.map((group) => ({ ...group,
    items: group.items.filter((item) => terms.every((term) => item.search.includes(term))) }))
    .filter((group) => group.items.length);
}
