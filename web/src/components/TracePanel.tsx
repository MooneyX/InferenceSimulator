import { useEffect, useState } from "react";

import {
  downloadTracePlan, traceSessions, traceStatus, uploadReplayPlan, uploadTrace,
  type TraceJob, type TraceSessionSummary,
} from "../api/client";
import "./TracePanel.css";

interface TracePanelProps {
  disabled: boolean;
  onRun: (id: string, mode: "open_loop" | "session", rate: number, seed: number) => void;
}

export function TracePanel({ disabled, onRun }: TracePanelProps) {
  const [job, setJob] = useState<TraceJob | null>(null);
  const [sessions, setSessions] = useState<TraceSessionSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [planId, setPlanId] = useState<string | null>(null);
  const [mode, setMode] = useState<"open_loop" | "session">("open_loop");
  const [rate, setRate] = useState(1);
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [problem, setProblem] = useState("");

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
        <p>上传原始 JSONL 预览并选择会话，或直接上传本地处理器导出的计划。模型与性能由当前手动配置决定。</p>
      </div>
      <div className="trace-upload-grid">
        <label>原始轨迹 JSONL <input type="file" accept=".jsonl" disabled={locked}
          onChange={(event) => { const file = event.target.files?.[0]; if (file) void receiveTrace(file); }} /></label>
        <label>已导出的计划 JSON <input type="file" accept=".json" disabled={locked}
          onChange={(event) => { const file = event.target.files?.[0]; if (file) void receivePlan(file); }} /></label>
      </div>
      {job && <p role="status">扫描状态：{job.status} · 会话 {job.session_count}
        {job.error ? ` · ${job.error}` : ""}</p>}
      {job?.status === "ready" && <>
        <p>已选 {selected.size} / {total} 个会话 · 命中率为原始轨迹前缀上界，不是仿真后的实际命中。</p>
        <div className="trace-table-wrap"><table className="trace-table">
          <thead><tr><th>选择</th><th>会话 ID</th><th>请求</th><th>输入 / 输出 token</th>
            <th>开环 QPS</th><th>前缀命中</th><th>依赖 / 定时根</th></tr></thead>
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
          <button type="button" disabled={locked || offset === 0}
            onClick={() => setOffset(Math.max(0, offset - 50))}>上一页</button>
          <span>{offset + 1}–{Math.min(offset + 50, total)} / {total}</span>
          <button type="button" disabled={locked || offset + 50 >= total}
            onClick={() => setOffset(offset + 50)}>下一页</button>
          <button type="button" disabled={locked || selected.size === 0}
            onClick={() => void exportSelected()}>导出并载入选中会话计划</button>
        </div>
      </>}
      <div className="trace-actions trace-settings">
        <label>投放模式 <select value={mode} disabled={locked}
          onChange={(event) => setMode(event.target.value as typeof mode)}>
          <option value="open_loop">开环 · 原轨迹相对时间</option>
          <option value="session">闭环 · 父完成后投放</option>
        </select></label>
        <label>会话/秒 <input type="number" min="0.000001" step="any" value={rate} disabled={locked}
          onChange={(event) => setRate(event.target.valueAsNumber)} /></label>
        <label>随机种子 <input type="number" min="0" step="1" value={seed} disabled={locked}
          onChange={(event) => setSeed(event.target.valueAsNumber)} /></label>
        <button type="button" disabled={locked || !planId || !(rate > 0) || !Number.isInteger(seed) || seed < 0}
          onClick={() => { if (planId) onRun(planId, mode, rate, seed); }}>运行计划</button>
      </div>
      {message && <p role="status">{message}</p>}
      {problem && <p role="alert" className="field-error">{problem}</p>}
    </section>
  );
}
