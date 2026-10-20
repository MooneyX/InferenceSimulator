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
:root{--bg:#0f1117;--surface:#1a1d27;--surface2:#242837;--border:#2e3347;--accent:#6c63ff;
 --text:#e4e4e7;--text-dim:#9ca0b0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 color:var(--text);background:var(--bg);color-scheme:dark}
*{box-sizing:border-box}body{max-width:1560px;margin:auto;padding:20px 24px;line-height:1.5}
h1{font-size:1.25rem;margin:0}h2,h3{font-size:1.05rem;margin:0 0 12px}
p{line-height:1.6;color:var(--text-dim);font-size:.75rem}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px;margin:16px 0}
.controls{display:flex;gap:8px 12px;flex-wrap:wrap;align-items:center;margin:12px 0}
.controls label{display:inline-flex;gap:6px;align-items:center;color:var(--text-dim);font-size:.75rem}
select{min-width:100px;max-width:100%;padding:6px 10px;border:1px solid var(--border);border-radius:5px;
 background:var(--surface2);color:var(--text);font-size:.8rem}
select:focus{border-color:var(--accent);outline:none}input[type=checkbox],input[type=radio]{accent-color:var(--accent)}
.role-row{display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap;margin:10px 0}
.role-row>label{display:flex;align-items:center;gap:6px;min-width:185px;font-size:.8rem}
.role-values,.background-options{display:flex;flex-wrap:wrap;gap:6px;max-height:150px;overflow:auto;flex:1;min-width:min(100%,250px)}
.role-values label,.background-options label{display:inline-flex;align-items:center;gap:5px;padding:4px 8px;
 border:1px solid var(--border);border-radius:5px;background:var(--surface2);font-size:.75rem;cursor:pointer}
.role-values label:hover,.background-options label:hover{border-color:var(--accent)}
.role-values label:has(input:disabled),.background-options label:has(input:disabled){opacity:.4;cursor:not-allowed}
#selection-notice{margin:8px 0}#background{margin-top:16px}
.background-values{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:10px}
.background-group{min-width:0;margin:0;padding:8px 10px 10px;border:1px solid var(--border);border-radius:6px}
.background-group legend{padding:0 4px;font-size:.75rem;font-weight:600;overflow-wrap:anywhere}
button{padding:7px 16px;border:0;border-radius:6px;background:var(--accent);color:#fff;
 font-size:.82rem;font-weight:600;cursor:pointer}button:hover{background:#7f78ff}
button:disabled{opacity:.4;cursor:not-allowed}
#chart{width:100%;min-height:320px;height:45vh;max-height:500px}
.table-wrap{overflow:auto;max-height:540px}table{border-collapse:collapse;width:100%;font-size:.75rem}
th,td{border-bottom:1px solid var(--border);text-align:left;padding:8px;white-space:nowrap}
th{position:sticky;top:0;background:var(--surface2)}td:last-child{max-width:400px;overflow:hidden;text-overflow:ellipsis}
@media(max-width:700px){body{padding:14px 16px}.panel{padding:16px 14px}#chart{height:36vh;max-height:340px}}
</style></head><body>
<header><h1>参数组合扫描报告</h1><p>跨批次筛选与参数透视 · 离线可用 · 不插值、不补算、不对不同配置求平均</p></header>
<section class="panel"><div id="batches" class="controls"></div>
<div class="role-row"><label>横轴 <select id="role-x"></select></label><div id="values-x" class="role-values"></div></div>
<div class="role-row"><label>颜色 <select id="role-color"></select></label><div id="values-color" class="role-values"></div></div>
<div class="role-row"><label>节点形状 <select id="role-shape"></select></label><div id="values-shape" class="role-values"></div></div>
<div class="role-row"><label>线样式 <select id="role-line"></select></label><div id="values-line" class="role-values"></div></div>
<div class="controls"><label>纵轴指标 <select id="metric"></select></label>
<button id="reset-values" type="button">重置变化参数值</button></div>
<div id="background"><div class="controls"><h3>背景参数（每项选择一个值）</h3>
<button id="reset-background" type="button">重新选择背景值</button></div>
<div id="background-values" class="background-values"></div></div><p id="selection-notice" role="status"></p></section>
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
