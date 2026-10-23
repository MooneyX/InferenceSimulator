import type { JobSummary, RequestProgress as Progress } from "../api/types";

export function RequestProgress({ progress, status }: {
  progress?: Progress | null;
  status: JobSummary["status"] | "not_run";
}) {
  if (status === "queued") return <span className="job-progress">等待执行</span>;
  if (!progress) return <span className="job-progress">
    {status === "running" ? "正在准备请求…" : status === "not_run" ? "—" : "请求进度未知"}
  </span>;

  const processed = progress.completed + progress.aborted + progress.skipped;
  const finished = processed === progress.total;
  const percent = progress.total === 0 ? 100 : Math.floor(processed / progress.total * 100);
  const stopped = status === "cancelled" || status === "interrupted" || status === "failed";
  const format = (value: number) => value.toLocaleString("zh-CN");
  const label = `已处理 ${format(processed)} / ${format(progress.total)} 请求`;

  return <div className={`request-progress request-progress-${status}`}
    title="按成功、失败和跳过的请求数统计，不代表耗时进度；运行中每秒更新">
    <div className="request-progress-summary">
      <span>{label}</span><strong>{percent}%</strong>
    </div>
    <progress aria-label="请求处理进度" aria-valuetext={`${label}，${percent}%`}
      max={progress.total || 1} value={progress.total === 0 ? 1 : processed} />
    <div className="request-progress-details">
      <span>成功 {format(progress.completed)} · 失败 {format(progress.aborted)} · 跳过 {format(progress.skipped)}</span>
      {status === "running" && <span>{finished ? "正在汇总结果…" : "处理中…"}</span>}
      {stopped && <span>最后记录的进度</span>}
    </div>
  </div>;
}
