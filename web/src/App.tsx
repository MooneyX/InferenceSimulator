import { lazy, Suspense, useState } from "react";

import { simulate, simulateAutomatic, SimulationApiError } from "./api/client";
import type { ResolvedRequest, ResolvedRun, SimConfig, SimResult } from "./api/types";
import { MetricsPanel } from "./components/MetricsPanel";
import { SimulationForm } from "./components/SimulationForm";
import { WarningsPanel } from "./components/WarningsPanel";
import "./App.css";

const RequestGantt = lazy(async () => {
  const module = await import("./components/RequestGantt");
  return { default: module.RequestGantt };
});

const navigation = [
  { id: "workload-heading", label: "基础参数" },
  { id: "engine-heading", label: "引擎配置" },
  { id: "performance-heading", label: "性能模型" },
  { id: "results-heading", label: "仿真结果" },
];

interface PresentationError {
  message: string;
  details: string[];
}

function presentError(error: unknown): PresentationError {
  if (error instanceof SimulationApiError) {
    return {
      message: error.message,
      details: error.validationErrors.map(
        ({ loc, msg }) => `${loc.join(" → ")}：${msg}`,
      ),
    };
  }

  return {
    message: error instanceof Error ? error.message : "运行仿真时发生未知错误",
    details: [],
  };
}

export default function App() {
  const [result, setResult] = useState<SimResult | null>(null);
  const [automatic, setAutomatic] = useState<ResolvedRun | null>(null);
  const [error, setError] = useState<PresentationError | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [activeSection, setActiveSection] = useState(() =>
    navigation.some(({ id }) => `#${id}` === window.location.hash)
      ? window.location.hash.slice(1)
      : navigation[0].id,
  );

  async function handleSubmit(config: SimConfig): Promise<void> {
    setIsRunning(true);
    setError(null);
    setResult(null);
    setAutomatic(null);

    try {
      setResult(await simulate(config));
    } catch (cause) {
      setError(presentError(cause));
    } finally {
      setIsRunning(false);
    }
  }

  async function handleSubmitAuto(request: ResolvedRequest): Promise<void> {
    setIsRunning(true);
    setError(null);
    setResult(null);
    setAutomatic(null);
    try {
      const response = await simulateAutomatic(request);
      setAutomatic(response);
      setResult(response.result);
    } catch (cause) {
      setError(presentError(cause));
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#overview"><span>KV Cache</span> 推理服务仿真器</a>
        <span className="topbar-caption">单实例 · 分层 KV 仿真工作台</span>
      </header>

      <nav className="sidebar" aria-label="建模模块">
        <p className="nav-title">建模模块</p>
        {navigation.map(({ id, label }, index) => (
          <a
            key={id}
            className={`nav-item${activeSection === id ? " active" : ""}`}
            href={`#${id}`}
            aria-current={activeSection === id ? "location" : undefined}
            onClick={() => setActiveSection(id)}
          >
            <span className="nav-icon" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
            {label}
          </a>
        ))}
      </nav>

      <main className="app-shell" id="overview">
        <header className="hero">
          <div>
            <p className="eyebrow">基础参数 / 单实例推理</p>
            <h1>推理服务仿真器</h1>
            <p className="hero-description">选择手动配置或自动解析，设置仿真参数并查看请求生命周期。</p>
          </div>
          <div className="hero-badges" aria-label="当前仿真范围">
            <span>单实例 · TP 可选</span>
            <span>Synthetic workload</span>
            <span>FCFS 调度</span>
          </div>
        </header>

        <SimulationForm onSubmit={handleSubmit} onSubmitAuto={handleSubmitAuto} disabled={isRunning} />

        <div className="result-stack" aria-live="polite">
          <div className="results-heading" id="results-heading">
            <div>
              <p className="section-kicker">Simulation output</p>
              <h2>仿真结果</h2>
            </div>
            <p>运行当前配置后更新 · 指标由仿真服务计算</p>
          </div>
          {isRunning ? (
            <section className="simulation-status" role="status">
              <span className="status-indicator" aria-hidden="true" />
              正在运行仿真…
            </section>
          ) : null}

          {error ? (
            <section className="simulation-error" role="alert" aria-labelledby="error-heading">
              <h2 id="error-heading">仿真失败</h2>
              <p>{error.message}</p>
              {error.details.length > 0 ? (
                <ul>
                  {error.details.map((detail) => (
                    <li key={detail}>{detail}</li>
                  ))}
                </ul>
              ) : null}
            </section>
          ) : null}

          {result ? (
            <>
              <section className="simulation-status" aria-label="请求结果状态">
                已完成 {result.meta.num_completed} · 失败 {result.meta.num_aborted} · 跳过 {result.meta.num_skipped}
                {(result.aborted_requests?.length ?? 0) > 0 ? (
                  <ul aria-label="失败请求原因">
                    {result.aborted_requests?.map((item) => (
                      <li key={item.id}>请求 {item.id}：{item.reason}</li>
                    ))}
                  </ul>
                ) : null}
                {(result.skipped_requests?.length ?? 0) > 0 ? (
                  <ul aria-label="跳过请求原因">
                    {result.skipped_requests?.map((item) => (
                      <li key={item.id}>请求 {item.id}（父请求 {item.parent_request_id}）：{item.reason}</li>
                    ))}
                  </ul>
                ) : null}
              </section>
              {automatic && <section className="simulation-status" aria-label="解析配置与成本">
                <h3>自动解析 · TP{automatic.resolution.tp_size}{automatic.resolution.model.moe_layers ? ` + EP${automatic.resolution.ep_size}` : ""} · {automatic.resolution.timing === "analytical" ? "未经校准的 Roofline/平均负载估计" : "经验计时"}</h3>
                <p>模型 {automatic.resolution.model.name} · GPU {automatic.resolution.hardware.name} ·
                  {automatic.resolution.tp_size > 1 ? "整组权重" : "权重"} {(automatic.resolution.capacity.weights_resident_bytes / 1e9).toFixed(2)} GB</p>
                <p>每 token KV {automatic.resolution.capacity.kv_bytes_per_token.toLocaleString()} bytes ·
                  L1 {automatic.resolution.config.engine.max_total_tokens.toLocaleString()} token ·
                  L2 {automatic.resolution.config.engine.l2_capacity_tokens.toLocaleString()} token ·
                  L3 {automatic.resolution.config.engine.l3_capacity_tokens.toLocaleString()} token</p>
                {(automatic.resolution.capacity.rank_reports ?? []).length > 0 &&
                  <div aria-label="逐卡容量账单">
                    {(automatic.resolution.capacity.rank_reports ?? []).map((rank) => <p key={rank.rank}>
                      GPU {rank.rank}：TP 分片 {(rank.sharded_weights_bytes / 1e9).toFixed(3)} GB ·
                      复制权重 {(rank.replicated_weights_bytes / 1e9).toFixed(3)} GB ·
                      {automatic.resolution.model.moe_layers ? `专家驻留 ${((rank.expert_weights_bytes ?? 0) / 1e9).toFixed(3)} GB ·` : ""}
                      KV {rank.kv_bytes_per_token.toLocaleString()} bytes/token ·
                      L1 {rank.l1_capacity_pages.toLocaleString()} 页
                    </p>)}
                  </div>}
                {automatic.analysis && <p>解析步骤 {automatic.analysis.steps} ·
                  {automatic.resolution.tp_size > 1 ? "单 rank FLOPs" : "FLOPs"} {automatic.analysis.flops.toExponential(3)} ·
                  {automatic.resolution.tp_size > 1 ? "单 rank HBM" : "HBM"} { (automatic.analysis.hbm_bytes / 1e9).toFixed(3) } GB ·
                  GPU 计时 {automatic.analysis.device_ms.toFixed(2)} ms ·
                  KV 传输 {automatic.analysis.restore_ms.toFixed(2)} ms
                  {automatic.resolution.tp_size > 1 && <>
                    {" · "}TP 同步 {(automatic.analysis.tp_communication_ms ?? automatic.analysis.communication_ms).toFixed(2)} ms
                    （已含于 GPU 计时） · {automatic.analysis.collectives} 次 ·
                    单 rank 流量 {((automatic.analysis.tp_communication_bytes ?? automatic.analysis.communication_bytes) / 1e9).toFixed(3)} GB
                  </>}
                  {automatic.resolution.model.moe_layers > 0 && <>
                    {" · "}EP dispatch {(automatic.analysis.ep_dispatch_ms ?? 0).toFixed(2)} ms / combine {(automatic.analysis.ep_combine_ms ?? 0).toFixed(2)} ms
                    （已含于 GPU 计时） · {automatic.analysis.ep_exchanges ?? 0} 次 ·
                    单 rank 逻辑流量 {(((automatic.analysis.ep_dispatch_bytes ?? 0) + (automatic.analysis.ep_combine_bytes ?? 0)) / 1e9).toFixed(3)} GB
                  </>}
                </p>}
                {automatic.resolution.model.moe_layers > 0 && <p>平均负载假设：均匀选专家、无复制或重叠；normal 按不同目标 rank 去重，low_latency 按专家选择计逻辑传输量（含逻辑元数据）。使用假设带宽，未经校准，非 DeepEP 实际链路流量。</p>}
              </section>}
              <MetricsPanel metrics={result.metrics} wallClockMs={result.meta.wall_clock_ms} />
              <WarningsPanel warnings={result.warnings} />
              {(result.requests?.length ?? 0) > 0 ? (
                <Suspense
                  fallback={
                    <section className="simulation-status" role="status">
                      <span className="status-indicator" aria-hidden="true" />
                      正在加载请求生命周期图…
                    </section>
                  }
                >
                  <RequestGantt requests={result.requests ?? []} />
                </Suspense>
              ) : (
                <section className="simulation-status">本次没有已完成请求，无法绘制生命周期图。</section>
              )}
            </>
          ) : null}

          {!isRunning && !error && !result ? (
            <section className="results-placeholder" aria-labelledby="placeholder-heading">
              <h2 id="placeholder-heading">等待仿真结果</h2>
              <p>调整参数并运行仿真后，此处将展示指标、告警和请求生命周期。</p>
            </section>
          ) : null}
        </div>
      </main>
    </>
  );
}
