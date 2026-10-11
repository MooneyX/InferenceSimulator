import { useEffect, useState } from "react";

import {
  downloadTracePlan, traceSessions, traceStatus, uploadReplayPlan, uploadTrace,
  type TraceJob, type TraceSessionSummary,
} from "../api/client";
import "./TracePanel.css";

export interface ReplaySelection {
  id: string;
  mode: "open_loop" | "session";
  rate: number;
  seed: number;
  targetQps?: number;
}

interface TracePanelProps {
  disabled: boolean;
  onChange: (selection: ReplaySelection | null) => void;
}

export function TracePanel({ disabled, onChange }: TracePanelProps) {
  const [job, setJob] = useState<TraceJob | null>(null);
  const [sessions, setSessions] = useState<TraceSessionSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [planId, setPlanId] = useState<string | null>(null);
  const [mode, setMode] = useState<"open_loop" | "session">("open_loop");
  const [rate, setRate] = useState(1);
  const [scaleToTarget, setScaleToTarget] = useState(false);
  const [targetQps, setTargetQps] = useState(64);
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [problem, setProblem] = useState("");

  useEffect(() => {
    const scale = mode === "open_loop" && scaleToTarget;
    onChange(planId && Number.isFinite(rate) && rate > 0 && Number.isInteger(seed) && seed >= 0
      && (!scale || (Number.isFinite(targetQps) && targetQps > 0)) && !busy
      ? { id: planId, mode, rate, seed, ...(scale ? { targetQps } : {}) } : null);
  }, [planId, mode, rate, seed, scaleToTarget, targetQps, busy, onChange]);

  useEffect(() => {
    if (!job || job.status !== "processing") return;
    let active = true;
    const timer = window.setInterval(() => {
      void traceStatus(job.id).then((next) => {
        if (active) setJob(next);
      }).catch((error: unknown) => {
        if (active) setProblem(error instanceof Error ? error.message : "查询扫描状态失败");
      });
    }, 1500);
    return () => { active = false; window.clearInterval(timer); };
  }, [job]);

  useEffect(() => {
    if (!job || job.status !== "ready") return;
    let active = true;
    void traceSessions(job.id, offset).then((page) => {
      if (active) { setSessions(page.sessions); setTotal(page.total); }
    }).catch((error: unknown) => {
      if (active) setProblem(error instanceof Error ? error.message : "无法加载会话摘要");
    });
    return () => { active = false; };
  }, [job, offset]);

  async function receiveTrace(file: File): Promise<void> {
    setBusy(true); setProblem(""); setMessage(""); setPlanId(null); setJob(null);
    setSessions([]); setSelected(new Set()); setOffset(0);
    try {
      const response = await uploadTrace(file);
      setJob(response);
      setMessage("轨迹已上传，正在后台扫描；大文件建议先用本地处理器生成计划。");
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "轨迹上传失败");
    } finally { setBusy(false); }
  }

  async function receivePlan(file: File): Promise<void> {
    setBusy(true); setProblem(""); setMessage(""); setPlanId(null);
    try {
      const receipt = await uploadReplayPlan(file);
      setPlanId(receipt.id);
      setMessage(`计划已校验：${receipt.sessions} 个会话、${receipt.requests} 个请求。`);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "计划上传失败");
    } finally { setBusy(false); }
  }

  async function exportSelected(): Promise<void> {
    if (!job || selected.size === 0) return;
    setBusy(true); setProblem(""); setPlanId(null);
    try {
      const blob = await downloadTracePlan(job.id, [...selected]);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url; anchor.download = "trace-plan.json"; anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      const receipt = await uploadReplayPlan(blob);
      setPlanId(receipt.id);
      setMessage(`已导出并载入计划：${receipt.requests} 个请求，可直接运行。`);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "计划导出失败");
    } finally { setBusy(false); }
  }

  const locked = disabled || busy;
  return (
    <section className="trace-panel" aria-labelledby="trace-heading">
      <div className="trace-intro">
        <div><p className="section-kicker">Trace replay</p><h3 id="trace-heading">轨迹回放</h3></div>
        <p>上传原始 JSONL 预览并选择会话，或直接上传本地处理器导出的计划。模型与性能由当前配置方式决定。</p>
      </div>
      <div className="trace-upload-grid">
        <div className="form-field trace-upload-field">
          <label htmlFor="trace-file">原始轨迹 JSONL</label>
          <input id="trace-file" type="file" accept=".jsonl" disabled={locked}
            onChange={(event) => { const file = event.target.files?.[0]; if (file) void receiveTrace(file); }} />
          <small>扫描会话摘要后，可选择需要回放的会话。</small>
        </div>
        <div className="form-field trace-upload-field">
          <label htmlFor="trace-plan-file">已导出的计划 JSON</label>
          <input id="trace-plan-file" type="file" accept=".json" disabled={locked}
            onChange={(event) => { const file = event.target.files?.[0]; if (file) void receivePlan(file); }} />
          <small>已有计划可直接上传，不必再次上传原始轨迹。</small>
        </div>
      </div>
      {job && <p className="trace-notice" role="status">扫描状态：{job.status} · 会话 {job.session_count}
        {job.error ? ` · ${job.error}` : ""}</p>}
      {job?.status === "ready" && <div className="trace-sessions">
        <div className="trace-sessions-heading">
          <div><p className="section-kicker">Session selection</p><h4>会话摘要</h4></div>
          <span>已选 {selected.size} / {total} 个会话</span>
        </div>
        <p className="trace-hint">命中率为原始轨迹前缀上界，不是仿真后的实际命中。</p>
        <div className="trace-table-wrap"><table className="trace-table">
          <thead><tr><th scope="col">选择</th><th scope="col">会话 ID</th><th scope="col">请求</th><th scope="col">输入 / 输出 token</th>
            <th scope="col">开环 QPS</th><th scope="col">前缀命中</th><th scope="col">依赖 / 定时根</th></tr></thead>
          <tbody>{sessions.map((item) => <tr key={item.id}>
            <td><input type="checkbox" aria-label={`选择会话 ${item.id}`} checked={selected.has(item.id)}
              disabled={locked} onChange={(event) => {
                const next = new Set(selected);
                if (event.target.checked) next.add(item.id); else next.delete(item.id);
                setSelected(next);
                setPlanId(null);
              }} /></td>
            <td title={item.id}><code>{item.id.slice(0, 12)}…</code></td><td>{item.requests}</td>
            <td>{item.input_tokens.toLocaleString()} / {item.output_tokens.toLocaleString()}</td>
            <td>{item.open_loop_qps?.toFixed(2) ?? "—"}</td>
            <td>{(item.prefix_hit_rate * 100).toFixed(1)}%</td>
            <td>{item.inferred_dependencies} / {item.timed_roots}</td>
          </tr>)}</tbody>
        </table></div>
        <div className="trace-actions">
          <div className="trace-pagination">
            <button className="trace-button trace-button-secondary" type="button" disabled={locked || offset === 0}
              onClick={() => setOffset(Math.max(0, offset - 50))}>上一页</button>
            <span>{offset + 1}–{Math.min(offset + 50, total)} / {total}</span>
            <button className="trace-button trace-button-secondary" type="button" disabled={locked || offset + 50 >= total}
              onClick={() => setOffset(offset + 50)}>下一页</button>
          </div>
          <button className="trace-button trace-button-primary" type="button" disabled={locked || selected.size === 0}
            onClick={() => void exportSelected()}>导出并载入选中会话计划</button>
        </div>
      </div>}
      <div className="trace-settings">
        <div className="trace-sessions-heading">
          <div><p className="section-kicker">Arrival settings</p><h4>回放参数</h4></div>
        </div>
        <div className="form-grid">
          <div className="form-field"><label htmlFor="trace-arrival-mode">投放模式</label>
            <select id="trace-arrival-mode" value={mode} disabled={locked}
              onChange={(event) => setMode(event.target.value as typeof mode)}>
              <option value="open_loop">开环 · 按轨迹时间投放</option>
              <option value="session">闭环 · 父完成后投放</option>
            </select></div>
          <div className="form-field"><label htmlFor="trace-session-rate">会话/秒</label>
            <input id="trace-session-rate" type="number" min="0.000001" step="any" value={rate} disabled={locked}
              onChange={(event) => setRate(event.target.valueAsNumber)} /></div>
          <div className="form-field"><label htmlFor="trace-seed">随机种子</label>
            <input id="trace-seed" type="number" min="0" step="1" value={seed} disabled={locked}
              onChange={(event) => setSeed(event.target.valueAsNumber)} /></div>
        </div>
        {mode === "open_loop" && <div className="trace-target-settings">
          <label htmlFor="trace-scale-target">
            <input id="trace-scale-target" type="checkbox" checked={scaleToTarget} disabled={locked}
              onChange={(event) => setScaleToTarget(event.target.checked)} />
            按目标 QPS 等比例缩放到达间隔
          </label>
          {scaleToTarget && <div className="form-field">
            <label htmlFor="trace-target-qps">目标发送 QPS（请求/秒）</label>
            <input id="trace-target-qps" type="number" min="0.000001" step="any"
              value={Number.isNaN(targetQps) ? "" : targetQps} disabled={locked}
              aria-invalid={!Number.isFinite(targetQps) || targetQps <= 0}
              onChange={(event) => setTargetQps(event.target.valueAsNumber)} />
          </div>}
          <p className="trace-hint">先按会话/秒合流，再缩放整条时间线；保留原有突发和空闲比例，不保证均匀投放。</p>
        </div>}
        {planId && !busy && <p className="trace-notice" role="status">计划已就绪，请使用下方「运行仿真」按钮。</p>}
      </div>
      {message && <p className="trace-notice" role="status">{message}</p>}
      {problem && <p role="alert" className="trace-notice trace-notice-error field-error">{problem}</p>}
    </section>
  );
}
