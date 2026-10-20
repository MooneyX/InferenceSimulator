import echartsSource from "echarts/dist/echarts.min.js?raw";
import reportRuntime from "./reportRuntime.js?raw";

import type { ScanParameter } from "../api/types";
import type { ScanData } from "./chart";

const safeScript = (value: string) => value.replace(/<\/script/gi, "<\\/script");
const embeddedData = (value: unknown) => JSON.stringify(value).replace(/</g, "\\u003c")
  .replace(/>/g, "\\u003e").replace(/&/g, "\\u0026");

export function renderReport(data: ScanData[], parameters: ScanParameter[]): string {
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>参数组合扫描报告</title><style>
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#172c42;background:#f2f5fa}
*{box-sizing:border-box}body{max-width:1560px;margin:auto;padding:24px}
h1{font-size:28px;margin:0}p{line-height:1.6;color:#516479}.panel{background:white;border:1px solid #dce5ed;
 border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 8px 20px #142a4410}
.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center}.controls label{display:flex;gap:8px;align-items:center}
select,button{padding:9px 12px;border:1px solid #afc1d4;border-radius:8px;background:white;color:#172c42}
button{cursor:pointer;background:#e8f2fd}button:hover{background:#d7eafc}#chart{height:520px;width:100%}
.table-wrap{overflow:auto;max-height:540px}table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{border-bottom:1px solid #e5ecf3;text-align:left;padding:8px;white-space:nowrap}
 th{position:sticky;top:0;background:#eff5fb}td:last-child{max-width:400px;overflow:hidden;text-overflow:ellipsis}
</style></head><body>
<header><h1>参数组合扫描报告</h1><p>跨批次筛选与参数透视 · 离线可用 · 不插值、不补算、不对不同配置求平均</p></header>
<section class="panel"><div id="batches" class="controls"></div><div class="controls">
<label>横轴 <select id="role-x"></select></label><label>颜色 <select id="role-color"></select></label>
<label>节点形状 <select id="role-shape"></select></label><label>线样式 <select id="role-line"></select></label>
<label>纵轴指标 <select id="metric"></select></label></div>
<h3>参数值筛选</h3><div id="filters" class="controls"></div></section>
<section class="panel"><div id="chart"></div><p id="summary"></p></section>
<section class="panel"><div class="controls"><h2>点级数据</h2><button id="csv" type="button">导出筛选数据 CSV</button></div>
<div class="table-wrap"><table><thead><tr><th>来源扫描</th><th>横轴</th><th>颜色</th><th>形状</th><th>线型</th><th>指标</th>
<th>状态</th><th>错误</th><th>实际输入配置</th></tr></thead><tbody id="data-body"></tbody></table></div></section>
<script>${safeScript(echartsSource)}</script>
<script type="application/json" id="scan-data">${embeddedData({ data, parameters })}</script>
<script>${safeScript(reportRuntime)}</script></body></html>`;
}

export function downloadReport(data: ScanData[], parameters: ScanParameter[]): void {
  const blob = new Blob([renderReport(data, parameters)], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "sensitivity-scan.html";
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
