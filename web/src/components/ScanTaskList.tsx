import { useEffect, useRef, useState } from "react";

import { getScanParameters, getScanTasks } from "../api/client";
import type { JobSummary, ScanTaskPage } from "../api/types";
import { RequestProgress } from "./RequestProgress";

const PAGE_SIZE = 20;
const statusLabels = {
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
  not_run: "未执行",
};

export function ScanTaskList({ job, onSelect }: {
  job: JobSummary;
  onSelect: (id: string, index: number) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [page, setPage] = useState(0);
  const [data, setData] = useState<ScanTaskPage | null>(null);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const refreshPage = useRef<(() => void) | null>(null);
  const total = job.point_archive_version === 1 ? job.total : job.completed + job.failed;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount - 1);
  function changePage(next: number) {
    setData(null);
    setError("");
    setPage(next);
  }

  useEffect(() => {
    if (!expanded) return;
    let active = true;
    void getScanParameters().then((items) => {
      if (active) setLabels(Object.fromEntries(items.map((item) => [item.key, item.label])));
    }).catch(() => { if (active) setLabels({}); });
    return () => { active = false; };
  }, [expanded]);

  useEffect(() => {
    if (!expanded) return;
    let active = true;
    let inFlight = false;
    let dirty = false;
    let lastStarted: number | null = null;
    let timer: number | undefined;

    async function refresh() {
      dirty = true;
      if (inFlight || timer !== undefined) return;
      const delay = lastStarted === null ? 0 : Math.max(0, 1000 - (Date.now() - lastStarted));
      if (delay > 0) {
        timer = window.setTimeout(() => { timer = undefined; void refresh(); }, delay);
        return;
      }
      dirty = false;
      inFlight = true;
      lastStarted = Date.now();
      try {
        const response = await getScanTasks(job.id, currentPage * PAGE_SIZE, PAGE_SIZE);
        if (active) { setData(response); setError(""); }
      } catch (cause) {
        if (active) setError(cause instanceof Error ? cause.message : "子任务加载失败");
      } finally {
        inFlight = false;
        if (active && dirty) void refresh();
      }
    }

    refreshPage.current = () => { void refresh(); };
    return () => {
      active = false;
      window.clearTimeout(timer);
      refreshPage.current = null;
    };
  }, [expanded, job.id, currentPage]);

  useEffect(() => { refreshPage.current?.(); }, [expanded, job.id, currentPage, job.version]);

  return <div className="scan-task-list">
    <button type="button" className="secondary-button secondary-button-small"
      aria-expanded={expanded} aria-controls={`scan-tasks-${job.id}`}
      onClick={() => setExpanded((value) => !value)}>{expanded ? "收起组合" : "展开组合"}</button>
    {expanded && <div id={`scan-tasks-${job.id}`} className="scan-task-content">
      {job.point_archive_version !== 1 && <p className="scan-task-note">
        历史扫描仅显示已记录的组合（按完成顺序）；未保存完整结果，可在扫描图表查看指标。
      </p>}
      {error && <p role="alert" className="scan-task-error">{error}</p>}
      {!data && !error && <p className="scan-task-note">正在加载组合…</p>}
      {data && <>
        {data.items.length === 0 && <p className="scan-task-note">暂无已记录的组合。</p>}
        <ol className="scan-task-items" start={currentPage * PAGE_SIZE + 1}>
          {data.items.map((item) => <li key={item.index}>
            <span className="scan-task-number">#{item.index + 1}</span>
            <span className="scan-task-values">{Object.entries(item.values).map(([key, value]) =>
              `${labels[key] ?? key}: ${value === null ? "null" : String(value)}`).join(" · ")}</span>
            <span className={`job-status job-status-${item.status}`}>{statusLabels[item.status]}</span>
            <RequestProgress progress={item.request_progress} status={item.status} />
            {item.error && <span className="scan-task-error">{item.error}</span>}
            {item.result_available && <>
              <button type="button" className="secondary-button secondary-button-small"
                onClick={() => onSelect(job.id, item.index)}>查看结果</button>
              {job.code_version_status !== "current" &&
                <a className="secondary-button secondary-button-small job-download"
                  href={`/api/jobs/${encodeURIComponent(job.id)}/points/${item.index}/result`}
                  target="_blank" rel="noopener noreferrer">查看原始 JSON</a>}
              <a className="secondary-button secondary-button-small job-download"
                href={`/api/jobs/${encodeURIComponent(job.id)}/points/${item.index}/result`}
                download={`scan-${job.id}-point-${item.index + 1}.json`}>导出 JSON</a>
            </>}
          </li>)}
        </ol>
        <nav className="scan-task-pages" aria-label={`扫描 ${job.id.slice(0, 8)} 组合分页`}>
          <button type="button" disabled={currentPage === 0} onClick={() => changePage(currentPage - 1)}>上一页</button>
          <span>第 {currentPage + 1} / {pageCount} 页 · 共 {data.total} 项</span>
          <button type="button" disabled={currentPage + 1 >= pageCount} onClick={() => changePage(currentPage + 1)}>下一页</button>
        </nav>
      </>}
    </div>}
  </div>;
}
