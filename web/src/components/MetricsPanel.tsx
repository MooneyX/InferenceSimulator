import type { SimResult } from "../api/types";
import "./MetricsPanel.css";

type Metrics = SimResult["metrics"];

export interface MetricsPanelProps {
  metrics: Metrics;
}

interface MetricCardProps {
  label: string;
  value: number | null | undefined;
  unit: "ms" | "请求/秒" | "%" | "token" | "页" | "次" | "字节";
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
          description="最早已发送请求到达至末成功请求完成的时间"
        />
        <MetricCard label="目标理想命中率" value={metrics.target_ideal_prefix_hit_rate} unit="%" description="用户设置的所有请求输入 token 加权目标" />
        <MetricCard label="构造理想命中率" value={metrics.constructed_ideal_prefix_hit_rate} unit="%" description="按计划继承块数和完整块粒度量化的理想前缀" />
        <MetricCard label="量化误差" value={metrics.ideal_hit_quantization_error} unit="%" description="构造理想命中率与目标之差" />
        <MetricCard label="实际前缀命中率" value={metrics.actual_prefix_hit_rate} unit="%" description="真正跳过 Prefill 的输入 token 比例" />
        <MetricCard label="有命中请求比例" value={metrics.requests_with_hit_ratio} unit="%" description="至少一个输入 token 真正复用的请求" />
        <MetricCard label="完整块全命中请求比例" value={metrics.requests_fully_cached_ratio} unit="%" description="有完整输入块且全部物理命中的请求" />
        <MetricCard label="实际 Prefill" value={metrics.actual_prefill_tokens} unit="token" description="成功请求首次准入计算的输入 token 数" />
        <MetricCard label="重算 Prefill" value={metrics.replayed_prefill_tokens} unit="token" description="回撤恢复的重复计算，包含后续失败请求" />
        <MetricCard label="节省 Prefill" value={metrics.saved_prefill_tokens} unit="token" description="命中而跳过的输入 token 数" />
        <MetricCard label="总实际命中" value={metrics.actual_hit_tokens} unit="token" description="首次准入实际跳过的输入 token，等于 L1 + L2 + L3 命中" />
        <MetricCard label="L1 命中" value={metrics.l1_hit_tokens} unit="token" description="首次准入时来源为 L1 的输入 token" />
        <MetricCard label="L2 命中" value={metrics.l2_hit_tokens} unit="token" description="首次准入时来源为 L2 的输入 token，不重复计入 L1" />
        <MetricCard label="L3 命中" value={metrics.l3_hit_tokens} unit="token" description="首次准入时来源为 L3 的输入 token，不重复计入 L1/L2" />
        <MetricCard label="提升到 L1" value={metrics.promotions} unit="次" description="实际提升的唯一 KV 页数" />
        <MetricCard label="逐级降级" value={metrics.demotions} unit="次" description="实际逐级下移的 KV 页次数" />
        <MetricCard label="Restore 传输量" value={metrics.restore_bytes} unit="字节" description="L2→L1 与 L3→L2 上行链路传输字节数之和" />
        <MetricCard label="Restore 耗时" value={metrics.restore_time_ms} unit="ms" description="恢复 KV 页的传输及固定延迟累计耗时" />
        <MetricCard label="L1 容量" value={metrics.l1_capacity_blocks} unit="页" description="max_total_tokens / page_size，实际 KV 页池容量" />
        <MetricCard label="L1 驻留" value={metrics.l1_resident_blocks} unit="页" description="仿真结束时 L1 的 KV 页占用" />
        <MetricCard label="L1 空闲" value={metrics.l1_free_blocks} unit="页" description="仿真结束时 L1 的空闲页数" />
        <MetricCard label="L1 驱逐" value={metrics.l1_evictions} unit="次" description="离开 L1 的 KV 页数，含降级到 L2" />
        <MetricCard label="L1 峰值驻留" value={metrics.l1_peak_resident_blocks} unit="页" description="仿真中实际物理 KV 页驻留峰值" />
        <MetricCard label="L2 容量" value={metrics.l2_capacity_blocks} unit="页" description="L2 的整页容量；0 表示关闭" />
        <MetricCard label="L2 驻留" value={metrics.l2_resident_blocks} unit="页" description="仿真结束时 L2 的 KV 页占用" />
        <MetricCard label="L2 空闲" value={metrics.l2_free_blocks} unit="页" description="仿真结束时 L2 的空闲页数" />
        <MetricCard label="L2 驱逐" value={metrics.l2_evictions} unit="次" description="离开 L2 的 KV 页数，含降级到 L3" />
        <MetricCard label="L2 峰值驻留" value={metrics.l2_peak_resident_blocks} unit="页" description="仿真中 L2 驻留页数峰值" />
        <MetricCard label="L3 容量" value={metrics.l3_capacity_blocks} unit="页" description="L3 的整页容量；0 表示关闭" />
        <MetricCard label="L3 驻留" value={metrics.l3_resident_blocks} unit="页" description="仿真结束时 L3 的 KV 页占用" />
        <MetricCard label="L3 空闲" value={metrics.l3_free_blocks} unit="页" description="仿真结束时 L3 的空闲页数" />
        <MetricCard label="L3 驱逐" value={metrics.l3_evictions} unit="次" description="最终删除的 L3 KV 页数" />
        <MetricCard label="L3 峰值驻留" value={metrics.l3_peak_resident_blocks} unit="页" description="仿真中 L3 驻留页数峰值" />
      </dl>
    </section>
  );
}
