import type { SimResult } from "../api/types";
import "./MetricsPanel.css";

type Metrics = SimResult["metrics"];

export interface MetricsPanelProps {
  metrics: Metrics;
}

interface MetricCardProps {
  label: string;
  value: number | null | undefined;
  unit: "ms" | "请求/秒";
  description: string;
}

const numberFormatter = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 2,
});

function formatMetric(value: number | null | undefined, unit: MetricCardProps["unit"]): string {
  if (value === null || value === undefined) {
    return "未定义";
  }

  return `${numberFormatter.format(value)} ${unit}`;
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
          description="首请求到达至末请求完成的总时长"
        />
      </dl>
    </section>
  );
}
