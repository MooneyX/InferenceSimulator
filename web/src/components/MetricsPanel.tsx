import type { SimResult } from "../api/types";
import "./MetricsPanel.css";

type Metrics = SimResult["metrics"];

export interface MetricsPanelProps {
  metrics: Metrics;
}

interface MetricCardProps {
  label: string;
  value: number | null | undefined;
  unit: "ms" | "请求/秒" | "%" | "token" | "blocks";
  description: string;
}

const numberFormatter = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 2,
});

function formatMetric(value: number | null | undefined, unit: MetricCardProps["unit"]): string {
  if (value === null || value === undefined) {
    return "未定义";
  }

  return `${numberFormatter.format(unit === "%" ? value * 100 : value)} ${unit}`;
}

function MetricCard({ label, value, unit, description }: MetricCardProps) {
  return (
    <div className="metric-card">
      <dt>{label}</dt>
      <dd>{formatMetric(value, unit)}</dd>
      <p>{description}</p>
    </div>
  );
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  return (
    <section className="metrics-panel" aria-labelledby="metrics-heading">
      <div className="metrics-heading">
        <div>
          <p className="section-kicker">Metrics</p>
          <h2 id="metrics-heading">仿真指标</h2>
        </div>
        <p>指标由仿真服务统一计算</p>
      </div>

      <dl className="metrics-grid">
        <MetricCard
          label="TTFT 均值"
          value={metrics.ttft_mean}
          unit="ms"
          description="请求从到达到完成 prefill 的平均耗时"
        />
        <MetricCard
          label="TTFT P50"
          value={metrics.ttft_p50}
          unit="ms"
          description="首 token 延迟中位数"
        />
        <MetricCard
          label="TTFT P99"
          value={metrics.ttft_p99}
          unit="ms"
          description="首 token 延迟第 99 百分位"
        />
        <MetricCard
          label="TPOT 均值"
          value={metrics.tpot_mean}
          unit="ms"
          description="Decode 阶段平均每 token 耗时"
        />
        <MetricCard
          label="端到端吞吐"
          value={metrics.throughput}
          unit="请求/秒"
          description="仿真时间范围内完成的请求数"
        />
        <MetricCard
          label="Makespan"
          value={metrics.makespan}
          unit="ms"
          description="首根请求到达至末请求完成的总时长"
        />
        <MetricCard label="目标理想命中率" value={metrics.target_ideal_prefix_hit_rate} unit="%" description="用户设置的所有请求输入 token 加权目标" />
        <MetricCard label="构造理想命中率" value={metrics.constructed_ideal_prefix_hit_rate} unit="%" description="按计划继承块数和完整块粒度量化的理想前缀" />
        <MetricCard label="量化误差" value={metrics.ideal_hit_quantization_error} unit="%" description="构造理想命中率与目标之差" />
        <MetricCard label="实际前缀命中率" value={metrics.actual_prefix_hit_rate} unit="%" description="真正跳过 Prefill 的输入 token 比例" />
        <MetricCard label="有命中请求比例" value={metrics.requests_with_hit_ratio} unit="%" description="至少一个输入 token 真正复用的请求" />
        <MetricCard label="完整块全命中请求比例" value={metrics.requests_fully_cached_ratio} unit="%" description="有完整输入块且全部物理命中的请求" />
        <MetricCard label="实际 Prefill" value={metrics.actual_prefill_tokens} unit="token" description="本场真实计算的输入 token 数" />
        <MetricCard label="节省 Prefill" value={metrics.saved_prefill_tokens} unit="token" description="命中而跳过的输入 token 数" />
        <MetricCard label="L1 容量" value={metrics.l1_capacity_blocks} unit="blocks" description="由 KV token 预算与块大小推导的物理容量" />
        <MetricCard label="L1 驱逐" value={metrics.l1_evictions} unit="blocks" description="仿真中发生的 LRU 驱逐次数" />
        <MetricCard label="L1 峰值驻留" value={metrics.l1_peak_resident_blocks} unit="blocks" description="仿真中物理块驻留的峰值" />
      </dl>
    </section>
  );
}
