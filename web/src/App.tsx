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
        <span className="topbar-caption">单 GPU · 分层 KV 仿真工作台</span>
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
            <p className="hero-description">配置合成负载、KV 页池和性能系数，运行仿真并查看请求生命周期。</p>
          </div>
          <div className="hero-badges" aria-label="当前仿真范围">
            <span>单 GPU</span>
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
                <h3>自动解析 · {automatic.resolution.timing === "analytical" ? "未经校准的 Roofline 估计" : "经验计时"}</h3>
                <p>模型 {automatic.resolution.model.name} · GPU {automatic.resolution.hardware.name} ·
                  权重 {(automatic.resolution.capacity.weights_resident_bytes / 1e9).toFixed(2)} GB</p>
                <p>每 token KV {automatic.resolution.capacity.kv_bytes_per_token.toLocaleString()} bytes ·
                  L1 {automatic.resolution.config.engine.max_total_tokens.toLocaleString()} token ·
                  L2 {automatic.resolution.config.engine.l2_capacity_tokens.toLocaleString()} token ·
                  L3 {automatic.resolution.config.engine.l3_capacity_tokens.toLocaleString()} token</p>
                {automatic.analysis && <p>解析步骤 {automatic.analysis.steps} · FLOPs {automatic.analysis.flops.toExponential(3)} ·
                  HBM { (automatic.analysis.hbm_bytes / 1e9).toFixed(3) } GB ·
                  GPU 计时 {automatic.analysis.device_ms.toFixed(2)} ms ·
                  KV 传输 {automatic.analysis.restore_ms.toFixed(2)} ms</p>}
              </section>}
              <MetricsPanel metrics={result.metrics} />
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
