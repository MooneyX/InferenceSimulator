import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import { CanvasRenderer } from "echarts/renderers";
import { DataZoomComponent, GridComponent, LegendComponent, ToolboxComponent, TooltipComponent } from "echarts/components";

import { getScanParameters, getScanPoints } from "../api/client";
import type { JobSummary, ScanParameter, ScanPoint } from "../api/types";
import { availableKeys, buildChart, metricsOf, palette, symbols, lineStyles,
  type Roles, type Role, type ScanData } from "../scan/chart";
import { resolveSelection, textOf, type BackgroundValues, type SelectedValues } from "../scan/selection";
import { formatScanTooltip } from "../scan/tooltip.js";
import "./ScanResults.css";

echarts.use([LineChart, CanvasRenderer, DataZoomComponent, GridComponent, LegendComponent,
  ToolboxComponent, TooltipComponent]);

function Curve({ data, roles, metric, complete }: { data: ScanData[]; roles: Roles; metric: string; complete: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<ReturnType<typeof echarts.init> | null>(null);
  const model = useMemo(() => buildChart(data, roles, metric), [data, roles, metric]);
  useEffect(() => {
    if (!host.current) return;
    const instance = echarts.init(host.current);
    chart.current = instance;
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(host.current);
    return () => { observer.disconnect(); instance.dispose(); chart.current = null; };
  }, []);
  useEffect(() => {
    chart.current?.setOption({
      animation: false,
      backgroundColor: "transparent",
      textStyle: { color: "#9ca0b0" },
      grid: { top: 42, left: 66, right: 35, bottom: 105 },
      legend: { type: "scroll", bottom: 0, textStyle: { color: "#9ca0b0", fontSize: 10 } },
      tooltip: { trigger: "axis", backgroundColor: "#242837", borderColor: "#2e3347",
        textStyle: { color: "#e4e4e7" },
        extraCssText: "max-width:calc(100vw - 32px);white-space:normal;box-sizing:border-box;",
        formatter: (params: unknown) => formatScanTooltip(params, model.series) },
      toolbox: { iconStyle: { borderColor: "#9ca0b0" }, feature: { saveAsImage: {}, restore: {} } },
      dataZoom: [{ type: "inside" }, { type: "slider", bottom: 52, borderColor: "#2e3347",
        textStyle: { color: "#9ca0b0" }, fillerColor: "rgba(108,99,255,.18)" }],
      xAxis: { type: "category", name: roles.x, data: model.xValues,
        nameTextStyle: { color: "#9ca0b0" }, axisLabel: { color: "#9ca0b0" },
        axisLine: { lineStyle: { color: "#2e3347" } } },
      yAxis: { type: "value", name: metric, scale: true,
        nameTextStyle: { color: "#9ca0b0" }, axisLabel: { color: "#9ca0b0" },
        splitLine: { lineStyle: { color: "#2e3347" } } }, series: model.series,
    }, true);
  }, [model, roles.x, metric]);
  const used = data.reduce((sum, batch) => sum + batch.points.length, 0);
  const levels = (key: string) => new Set(model.entries.map(({ point }) => key && point.values[key])).size;
  return <><div ref={host} className="scan-chart" role="img" aria-label="参数敏感性扫描折线图" />
    <p>{complete ? <>独立配置点 {model.entries.length}，相同输入去重 {model.deduplicated} / {used}。
      缺失点断线，不插值、不平均。</> : "背景参数尚未选齐，图表暂不显示。"}</p>
    {(levels(roles.color) > palette.length || levels(roles.shape) > symbols.length ||
      levels(roles.line) > lineStyles.length) &&
      <p role="status">部分视觉样式循环复用，建议减少相应档位以便区分。</p>}</>;
}

export function ScanResults({ jobs }: { jobs: JobSummary[] }) {
  const scans = jobs.filter((job) => job.kind === "scan");
  const [selected, setSelected] = useState<string[] | null>(null);
  const [points, setPoints] = useState<Record<string, ScanPoint[]>>({});
  const [parameters, setParameters] = useState<ScanParameter[]>([]);
  const [roles, setRoles] = useState<Roles>({ x: "", color: "", shape: "", line: "" });
  const [metric, setMetric] = useState("throughput");
  const [background, setBackground] = useState<BackgroundValues>({});
  const [choosingBackground, setChoosingBackground] = useState(false);
  const [selectedValues, setSelectedValues] = useState<SelectedValues>({});
  const [error, setError] = useState("");
  const chosen = scans.filter((job) => selected === null || selected.includes(job.id));
  const finished = chosen.filter((job) => job.status === "completed" || job.status === "cancelled");
  const ids = finished.map((job) => job.id).join(",");

  useEffect(() => {
    let active = true;
    void getScanParameters().then((items) => { if (active) setParameters(items); })
      .catch(() => { if (active) setError("无法加载扫描参数目录"); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    let active = true;
    void Promise.all(ids.split(",").filter(Boolean).map(async (id) =>
      [id, await getScanPoints(id)] as const)).then((updates) => {
      if (active) setPoints((current) => ({ ...current, ...Object.fromEntries(updates) }));
    }).catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "结果获取失败"); });
    return () => { active = false; };
  }, [ids]);
  const data = useMemo(() => finished.filter((job) => points[job.id]).map((job) =>
    ({ job, points: points[job.id] })), [finished.map((job) => job.id).join(","), points]);
  const keys = availableKeys(data);
  const measures = metricsOf(data);
  const chosenRoles: Roles = { x: roles.x && keys.includes(roles.x) ? roles.x : keys[0] ?? "",
    color: keys.includes(roles.color) ? roles.color : "",
    shape: keys.includes(roles.shape) ? roles.shape : "",
    line: keys.includes(roles.line) ? roles.line : "" };
  const currentMetric = measures.includes(metric) ? metric : measures[0] ?? "";
  const selection = resolveSelection(data, chosenRoles, currentMetric,
    background, selectedValues, choosingBackground);
  const nameOf = (key: string) => parameters.find((item) => item.key === key)?.label ?? key;

  async function exportHtml() {
    try {
      const { downloadReport } = await import("../scan/report");
      downloadReport(data, parameters);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "报告生成失败"); }
  }
  if (!scans.length) return null;
  return <section className="scan-results" aria-labelledby="scan-results-heading">
    <div className="section-heading"><div><p className="section-kicker">Scan results</p>
      <h2 id="scan-results-heading">扫描结果组合</h2></div></div>
    <div className="scan-batches">{scans.map((job) => <label key={job.id}>
      <input type="checkbox" checked={selected === null || selected.includes(job.id)}
        onChange={(event) => { const all = selected ?? scans.map((item) => item.id);
          setSelected(event.target.checked ? [...all, job.id] : all.filter((id) => id !== job.id)); }} />
      扫描 {job.id.slice(0, 8)} · {job.status} · {job.completed + job.failed}/{job.total}
      {job.failed > 0 && ` · ${job.failed} 点失败`}</label>)}</div>
    {keys.length > 0 && <>
      {(["x", "color", "shape", "line"] as Role[]).map((role) =>
        <div className="scan-role-row" key={role}>
          <label>{({ x: "横轴", color: "颜色", shape: "节点形状", line: "线样式" })[role]}
            <select value={chosenRoles[role]} onChange={(event) => {
              setRoles({ ...chosenRoles, [role]: event.target.value });
              setBackground({}); setChoosingBackground(false); setSelectedValues({});
            }}>
              {role !== "x" && <option value="">无</option>}
              {keys.map((key) => <option key={key} value={key}
                disabled={key !== chosenRoles[role] && Object.values(chosenRoles).includes(key)}>
                {nameOf(key)}</option>)}
            </select></label>
          <div className="scan-role-values" aria-label={`${({ x: "横轴", color: "颜色", shape: "节点形状", line: "线样式" })[role]}参数值`}>
            {!chosenRoles[role] ? <span className="scan-role-empty">未选择参数</span> : selection.roleChoices[role].map((choice) =>
              <label key={choice.token} className={choice.disabled ? "unavailable" : ""}>
                <input type="checkbox" checked={choice.checked} disabled={choice.disabled}
                  onChange={(event) => setSelectedValues({ ...selection.selected,
                    [chosenRoles[role]]: event.target.checked
                      ? [...selection.selected[chosenRoles[role]], choice.token]
                      : selection.selected[chosenRoles[role]].filter((token) => token !== choice.token) })} />
                {textOf(choice.value)}</label>)}
          </div>
        </div>)}
      <div className="scan-view-controls">
        <label>纵轴指标 <select value={currentMetric} onChange={(event) => setMetric(event.target.value)}>
          {measures.map((key) => <option key={key} value={key}>{key}</option>)}</select></label>
        <button className="scan-secondary-button" type="button" onClick={() => setSelectedValues({})}>重置变化参数值</button>
        <button type="button" onClick={() => void exportHtml()} disabled={!data.length}>
          导出离线可操作 HTML</button></div>
      {selection.hidden.length > 0 && <div className="scan-background">
        <div className="scan-background-heading"><h3>背景参数</h3>
          <button className="scan-secondary-button" type="button" onClick={() => {
            setBackground({}); setChoosingBackground(true); setSelectedValues({});
          }}>重新选择背景值</button></div>
        <div className="scan-background-groups">{selection.hidden.map((key) =>
          <fieldset key={key}><legend>{nameOf(key)}</legend><div className="scan-background-options">
            {selection.backgroundChoices[key].map((choice) => <label key={choice.token}
              className={choice.disabled ? "unavailable" : ""}>
              <input type="radio" name={`background-${key}`} checked={choice.checked}
                disabled={choice.disabled} onChange={() => {
                  const next = { ...selection.background, [key]: choice.token };
                  setBackground(next); setSelectedValues({});
                  setChoosingBackground(selection.hidden.some((other) => next[other] === undefined));
                }} />{textOf(choice.value)}</label>)}
          </div></fieldset>)}</div>
      </div>}
      <p className="scan-selection-notice" role="status">{selection.notice}</p>
      <Curve data={selection.visible} roles={chosenRoles} metric={currentMetric} complete={selection.complete} />
    </>}
    {error && <p role="alert" className="field-error">{error}</p>}
  </section>;
}
