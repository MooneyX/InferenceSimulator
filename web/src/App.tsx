import { lazy, Suspense, useState } from "react";

import { simulate, SimulationApiError } from "./api/client";
import type { SimConfig, SimResult } from "./api/types";
import { MetricsPanel } from "./components/MetricsPanel";
import { SimulationForm } from "./components/SimulationForm";
import { WarningsPanel } from "./components/WarningsPanel";
import "./App.css";

const RequestGantt = lazy(async () => {
  const module = await import("./components/RequestGantt");
  return { default: module.RequestGantt };
});

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
  const [error, setError] = useState<PresentationError | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  async function handleSubmit(config: SimConfig): Promise<void> {
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      setResult(await simulate(config));
    } catch (cause) {
      setError(presentError(cause));
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <p className="eyebrow">InferenceSimulator · M3</p>
        <h1>推理服务仿真器</h1>
        <p className="hero-description">
          配置单 GPU 推理负载，并查看仿真指标与请求生命周期。
        </p>
        <div className="hero-badges" aria-label="当前仿真范围">
          <span>单 GPU</span>
          <span>Synthetic workload</span>
          <span>FCFS 调度</span>
        </div>
      </header>

      <SimulationForm onSubmit={handleSubmit} disabled={isRunning} />

      <div className="result-stack" aria-live="polite">
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
          <section className="results-placeholder" aria-labelledby="results-heading">
            <h2 id="results-heading">等待仿真结果</h2>
            <p>调整参数并运行仿真后，此处将展示指标、告警和请求生命周期。</p>
          </section>
        ) : null}
      </div>
    </main>
  );
}
