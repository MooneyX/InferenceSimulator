import { lazy, Suspense, useState } from "react";

import { clearSimulationCache, SimulationApiError, type SimulationCacheInfo } from "./api/client";
import type { ScanRequest, SimConfig, SimResult } from "./api/types";
import { useJobs } from "./useJobs";
import { MetricsPanel } from "./components/MetricsPanel";
import { ParameterExport } from "./components/ParameterExport";
import { RequestProgress } from "./components/RequestProgress";
import { SimulationForm } from "./components/SimulationForm";
import { ScanTaskList } from "./components/ScanTaskList";
import { WarningsPanel } from "./components/WarningsPanel";
import "./App.css";

const RequestGantt = lazy(async () => {
  const module = await import("./components/RequestGantt");
  return { default: module.RequestGantt };
});

const ScanResults = lazy(async () => {
  const module = await import("./components/ScanResults");
  return { default: module.ScanResults };
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
  const [selectedInstanceId, setSelectedInstanceId] = useState<number | null>(null);
  const [error, setError] = useState<PresentationError | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [cacheInfo, setCacheInfo] = useState<SimulationCacheInfo | null>(null);
  const [resultSource, setResultSource] = useState<string | null>(null);
  const [loadingResult, setLoadingResult] = useState(false);
  const [isClearingCache, setIsClearingCache] = useState(false);
  const [cacheNotice, setCacheNotice] = useState<{ message: string; error: boolean } | null>(null);
  const selectedInstance = result?.instances.find((item) => item.instance_id === selectedInstanceId);
  const inference = result?.meta.config.inference;
  const resolution = result?.resolution;
  const analysis = result?.analysis;
  const [activeSection, setActiveSection] = useState(() =>
    navigation.some(({ id }) => `#${id}` === window.location.hash)
      ? window.location.hash.slice(1)
      : navigation[0].id,
  );
  const tasks = useJobs((response, source) => {
    setResult(response.result);
    setCacheInfo(response.cache);
    setResultSource(source);
    setError(null);
    setLoadingResult(false);
    setSelectedInstanceId(null);
    setIsRunning(false);
  }, (cause) => {
    setLoadingResult(false);
    setError(presentError(cause));
  });

  async function handleSubmit(config: SimConfig, force = false): Promise<void> {
    setIsRunning(true);
    setError(null);
    setCacheNotice(null);
    try { await tasks.submitSingle(config, force); }
    catch (cause) { setError(presentError(cause)); }
    finally { setIsRunning(false); }
  }

  function handleSelectResult(id: string, index?: number): void {
    setResult(null);
    setCacheInfo(null);
    setResultSource(null);
    setError(null);
    setLoadingResult(true);
    if (index === undefined) tasks.select(id);
    else tasks.selectPoint(id, index);
    document.getElementById("results-heading")?.scrollIntoView?.({ behavior: "smooth" });
  }

  async function handleScanSubmit(request: ScanRequest): Promise<void> {
    setError(null);
    try { await tasks.submitSweep(request); }
    catch (cause) { setError(presentError(cause)); }
  }

  async function handleClearCache(): Promise<void> {
    if (!window.confirm("清空本地所有仿真结果缓存？已显示的结果不会删除。")) return;
    setIsClearingCache(true);
    setCacheNotice(null);
    try {
      await clearSimulationCache();
      setCacheNotice({ message: "结果缓存已清空，下一次运行将重新计算。", error: false });
    } catch (cause) {
      setCacheNotice({ message: presentError(cause).message, error: true });
    } finally {
      setIsClearingCache(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#overview"><span>KV Cache</span> 推理服务仿真器</a>
        <span className="topbar-caption">多实例 · 独立 KV 仿真工作台</span>
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
            <p className="eyebrow">基础参数 / 多实例推理</p>
            <h1>推理服务仿真器</h1>
            <p className="hero-description">根据模型、硬件和存储预算解析容量，设置仿真参数并查看请求生命周期。</p>
          </div>
          <div className="hero-badges" aria-label="当前仿真范围">
            <span>多实例 · TP 可选</span>
            <span>Synthetic / Trace workload</span>
            <span>FCFS 调度</span>
          </div>
        </header>

        <SimulationForm onSubmit={handleSubmit} onScanSubmit={(request) => void handleScanSubmit(request)}
          disabled={isClearingCache} />

        {tasks.jobs.length > 0 && <section className="simulation-info job-list" aria-label="后台任务">
          <h2>后台任务</h2>
          <ul>{tasks.jobs.map((job) => <li key={job.id}>
            <strong>{job.kind === "scan" ? "扫描" : "仿真"} {job.id.slice(0, 8)}</strong>
            <span className={`job-status job-status-${job.status}`}>{job.status}</span>
            <span className={`job-code-version job-code-version-${job.code_version_status}`}
              title={job.code_fingerprint || "首次升级前的任务，无法追溯代码版本"}>
              {job.code_version_status === "current" ? "当前代码版本"
                : job.code_version_status === "stale" ? "旧代码版本" : "代码版本未知"}</span>
            {job.kind === "scan"
              ? <span className="job-progress">已结束组合 {job.completed + job.failed}/{job.total}</span>
              : <RequestProgress progress={job.request_progress} status={job.status} />}
            {job.kind === "scan" && <ScanTaskList job={job} onSelect={handleSelectResult} />}
            {job.failed > 0 && job.kind === "scan" && ` · ${job.failed} 点失败`}
            {job.error && (job.kind !== "scan" || job.status === "interrupted") && ` · ${job.error}`}
            {job.kind === "single" && job.status === "completed" &&
              <button type="button" className="secondary-button secondary-button-small" onClick={() => handleSelectResult(job.id)}>
                查看结果</button>}
            {job.kind === "single" && job.status === "completed" && job.code_version_status !== "current" &&
              <a className="secondary-button secondary-button-small job-download" href={`/api/jobs/${encodeURIComponent(job.id)}/result`}
                target="_blank" rel="noopener noreferrer">查看原始 JSON</a>}
            {job.kind === "single" && job.status === "completed" &&
              <a className="secondary-button secondary-button-small job-download" href={`/api/jobs/${encodeURIComponent(job.id)}/result`}
                download={`simulation-${job.id}.json`}>导出 JSON</a>}
            {(job.status === "queued" || job.status === "running") &&
              <button type="button" className="secondary-button secondary-button-small" onClick={() =>
                void tasks.stop(job.id).catch((cause: unknown) => setError(presentError(cause)))}>
                取消任务</button>}
          </li>)}</ul>
        </section>}
        {tasks.jobs.some((job) => job.kind === "scan") && <Suspense fallback={<p>正在加载扫描图表…</p>}>
          <ScanResults jobs={tasks.jobs} />
        </Suspense>}

        <div className="result-stack" aria-live="polite">
          <div className="results-heading" id="results-heading">
            <div>
              <p className="section-kicker">Simulation output</p>
              <h2>仿真结果</h2>
              {result && resultSource && <p className="result-source">{resultSource}</p>}
            </div>
            <div className="cache-controls">
              <span>相同参数自动复用结果</span>
              <button className="secondary-button secondary-button-small" type="button" onClick={() => void handleClearCache()}
                disabled={isRunning || isClearingCache}>
                {isClearingCache ? "正在清空…" : "清空结果缓存"}
              </button>
            </div>
          </div>
          {result && <ParameterExport key={tasks.selectedId} config={result.meta.config} />}
          {cacheNotice && <section className={cacheNotice.error ? "simulation-error" : "simulation-status"}
            role={cacheNotice.error ? "alert" : "status"}>{cacheNotice.message}</section>}
          {cacheInfo && result && <section className={cacheInfo.status === "miss" ? "simulation-info" : "simulation-status"} aria-label="结果缓存状态">
            {cacheInfo.status === "hit" ? "结果缓存命中" : cacheInfo.status === "shared"
              ? "已复用相同参数的进行中计算" : "本次结果为重新计算"}
            {" · "}缓存查找/读取 {cacheInfo.lookupMs.toFixed(2)} ms
            {" · "}原始仿真计算 {result.meta.wall_clock_ms.toFixed(2)} ms
          </section>}
          {loadingResult && <section className="simulation-status" role="status">
            <span className="status-indicator" aria-hidden="true" />正在读取仿真结果…
          </section>}
          {isRunning ? (
            <section className="simulation-status" role="status">
              <span className="status-indicator" aria-hidden="true" />
              正在提交后台任务…
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
              <section className="simulation-info" aria-label="请求结果状态">
                已完成 {result.meta.num_completed} · 失败 {result.meta.num_aborted} · 跳过 {result.meta.num_skipped}
                <p>引擎输出上限：{(inference?.max_output_tokens ?? 65_536).toLocaleString()} token；
                  每请求生成上限按实际输出长度自动向上取二次幂。</p>
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
              {inference && resolution && <section className="simulation-info" aria-label="解析配置与成本">
                <h3>自动解析 · TP{inference.tp_size}{inference.model.moe_layers ? ` + EP${inference.ep_size}` : ""} · {(inference.timing?.mode ?? "analytical") === "analytical" ? "未经校准的 Roofline/平均负载估计" : "经验计时"}</h3>
                <p>模型 {inference.model.name} · GPU {inference.hardware.name} ·
                  {result.meta.config.num_instances} 实例 × 每实例 {inference.hardware.gpu_count} GPU = 总计 {result.meta.config.num_instances * inference.hardware.gpu_count} GPU ·
                  {inference.tp_size > 1 ? "整组权重" : "权重"} {(resolution.capacity.weights_resident_bytes / 1e9).toFixed(2)} GB</p>
                <p>每 token KV {resolution.capacity.kv_bytes_per_token.toLocaleString()} bytes ·
                  L1 {resolution.max_total_tokens.toLocaleString()} token ·
                  L2 {resolution.l2_capacity_tokens.toLocaleString()} token ·
                  全场共享 L3 {resolution.l3_capacity_tokens.toLocaleString()} token</p>
                {(resolution.capacity.rank_reports ?? []).length > 0 &&
                  <div aria-label="逐卡容量账单">
                    {(resolution.capacity.rank_reports ?? []).map((rank) => <p key={rank.rank}>
                      GPU {rank.rank}：TP 分片 {(rank.sharded_weights_bytes / 1e9).toFixed(3)} GB ·
                      复制权重 {(rank.replicated_weights_bytes / 1e9).toFixed(3)} GB ·
                      {inference.model.moe_layers ? `专家驻留 ${((rank.expert_weights_bytes ?? 0) / 1e9).toFixed(3)} GB ·` : ""}
                      KV {rank.kv_bytes_per_token.toLocaleString()} bytes/token ·
                      L1 {rank.l1_capacity_pages.toLocaleString()} 页
                    </p>)}
                  </div>}
                {analysis && <p>解析步骤 {analysis.steps} ·
                  {inference.tp_size > 1 ? "单 rank FLOPs" : "FLOPs"} {analysis.flops.toExponential(3)} ·
                  {inference.tp_size > 1 ? "单 rank HBM" : "HBM"} {(analysis.hbm_bytes / 1e9).toFixed(3)} GB ·
                  GPU 计时 {analysis.device_ms.toFixed(2)} ms ·
                  KV 传输 {analysis.restore_ms.toFixed(2)} ms
                  {inference.tp_size > 1 && <>
                    {" · "}TP 同步 {(analysis.tp_communication_ms ?? analysis.communication_ms).toFixed(2)} ms
                    （已含于 GPU 计时） · {analysis.collectives} 次 ·
                    单 rank 流量 {((analysis.tp_communication_bytes ?? analysis.communication_bytes) / 1e9).toFixed(3)} GB
                  </>}
                  {inference.model.moe_layers > 0 && <>
                    {" · "}EP dispatch {(analysis.ep_dispatch_ms ?? 0).toFixed(2)} ms / combine {(analysis.ep_combine_ms ?? 0).toFixed(2)} ms
                    （已含于 GPU 计时） · {analysis.ep_exchanges ?? 0} 次 ·
                    单 rank 逻辑流量 {(((analysis.ep_dispatch_bytes ?? 0) + (analysis.ep_combine_bytes ?? 0)) / 1e9).toFixed(3)} GB
                  </>}
                </p>}
                {inference.model.moe_layers > 0 && <p>平均负载假设：均匀选专家、无复制或重叠；normal 按不同目标 rank 去重，low_latency 按专家选择计逻辑传输量（含逻辑元数据）。使用假设带宽，未经校准，非 DeepEP 实际链路流量。</p>}
              </section>}
              <section className="simulation-info instance-filter" aria-label="实例结果筛选">
                <label htmlFor="instance-filter">查看范围：</label>
                <select id="instance-filter" value={selectedInstanceId ?? "all"}
                  onChange={(event) => setSelectedInstanceId(event.target.value === "all" ? null : Number(event.target.value))}>
                  <option value="all">全部实例</option>
                  {result.instances.map((instance) => <option key={instance.instance_id} value={instance.instance_id}>
                    实例 {instance.instance_id}
                  </option>)}
                </select>
                {selectedInstance && <p>完成 {selectedInstance.num_completed} ·
                  失败 {selectedInstance.num_aborted} · 跳过 {selectedInstance.num_skipped} ·
                  引擎步 {selectedInstance.num_engine_steps}</p>}
              </section>
              <MetricsPanel metrics={selectedInstance?.metrics ?? result.metrics}
                wallClockMs={result.meta.wall_clock_ms}
                peakAggregated={selectedInstanceId === null && result.instances.length > 1}
                targetQps={selectedInstanceId === null ? result.meta.config.workload.target_qps ?? undefined : undefined}
                targetSessionRate={selectedInstanceId === null ? result.meta.config.workload.session_rate ?? undefined : undefined} />
              <WarningsPanel warnings={result.warnings} />
              {(result.requests ?? []).some((item) => selectedInstanceId === null || item.instance_id === selectedInstanceId) ? (
                <Suspense
                  fallback={
                    <section className="simulation-status" role="status">
                      <span className="status-indicator" aria-hidden="true" />
                      正在加载请求生命周期图…
                    </section>
                  }
                >
                  <RequestGantt requests={selectedInstanceId === null ? result.requests :
                    (result.requests ?? []).filter((item) => item.instance_id === selectedInstanceId)} />
                </Suspense>
              ) : (
                <section className="simulation-info">本次没有已完成请求，无法绘制生命周期图。</section>
              )}
            </>
          ) : null}

          {!isRunning && !loadingResult && !error && !result ? (
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
