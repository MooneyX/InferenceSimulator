import { useEffect, useRef, useState } from "react";

import { getScanParameters, uploadReplayPlan } from "../api/client";
import type { ScanParameter, ScanRequest, SimConfig } from "../api/types";
import { parseSimConfigJson, serializeSimConfig } from "../config/parameterTransfer";
import { validateSimConfig } from "../config/simulation";
import "./ScanPanel.css";

type AxisDraft = { id: number; key: string; text: string; search: string };

function initialValue(config: SimConfig, key: string, fallbackQps = 64): string {
  if (key === "inference.parallelism") return String(config.inference.tp_size);
  if (key.startsWith("shared_l3.") && !config.shared_l3) return "0";
  if (key === "workload.target_qps" && config.workload.target_qps == null) return String(fallbackQps);
  const path = key.split(".");
  let value: unknown = config;
  for (const part of path) value = (value as Record<string, unknown> | null)?.[part];
  if (path.at(-1)?.endsWith("_len") && typeof value === "object" && value) {
    const length = value as { dist: string; value?: number; mean?: number };
    value = length.dist === "fixed" ? length.value : length.mean;
  }
  if (key === "shared_l3.available_gb" && typeof value === "number") value /= 1000;
  if ((key === "inference.hardware.memory_bytes_per_gpu" ||
       key === "inference.storage.runtime_reserve_bytes") && typeof value === "number") value /= 1e9;
  return value === undefined || value === null ? "null" : String(value);
}

export function parseLevels(text: string, kind: string): (number | boolean | null)[] {
  const input = text.trim();
  if (!input) throw new Error("请输入扫描档位");
  let entries: string[];
  const range = input.match(/^(-?[\d.]+):(-?[\d.]+):(-?[\d.]+)$/);
  if (range && kind !== "boolean") {
    const [start, end, step] = range.slice(1).map(Number);
    if (![start, end, step].every(Number.isFinite) || !step ||
        (end - start) * step < 0) throw new Error("范围的起止值或步长无效");
    const count = Math.floor((end - start) / step + 1e-9) + 1;
    if (!Number.isSafeInteger(count) || count < 1) throw new Error("范围档位数不可表示");
    entries = Array.from({ length: count }, (_, i) => String(Number((start + i * step).toPrecision(12))));
  } else {
    entries = input.split(/[,，\s]+/).filter(Boolean);
  }
  return entries.map((entry) => {
    if (entry === "null") return null;
    if (kind === "boolean") {
      if (entry !== "true" && entry !== "false") throw new Error("布尔档位只能填 true 或 false");
      return entry === "true";
    }
    const value = Number(entry);
    if (!Number.isFinite(value) || (kind === "integer" && !Number.isSafeInteger(value))) {
      throw new Error(`无效的${kind === "integer" ? "整数" : "数值"}档位：${entry}`);
    }
    return value;
  });
}

function scanDimensions(config: SimConfig, entries: Pick<AxisDraft, "key" | "text">[], catalog: ScanParameter[],
  mode: "synthetic" | "trace", arrival: "open_loop" | "closed_loop", blockSize: number | null = null): ScanRequest["axes"] {
  if (!entries.length) throw new Error("请至少选择一个扫描参数");
  const available = catalog.filter((item) => item.modes.includes(`${mode}/${arrival}`));
  const seen = new Set<string>();
  return entries.map((axis) => {
    const item = available.find((candidate) => candidate.key === axis.key);
    if (!item) throw new Error(`当前负载模式不支持扫描参数：${axis.key || "未选择"}`);
    if (seen.has(axis.key)) throw new Error(`扫描参数重复：${axis.key}`);
    seen.add(axis.key);
    const values = parseLevels(axis.text, item.kind);
    if (item.key === "inference.parallelism" && config.inference.timing?.mode === "empirical"
        && values.some((value) => value !== 1)) {
      throw new Error("经验计时仅支持每实例 GPU / TP 数为 1");
    }
    if (item.key.startsWith("workload.synthetic.") && item.key.endsWith("_len")) {
      const name = item.key.split(".")[2] as "prompt_len" | "turn_increment_len" | "output_len";
      if (config.workload.synthetic?.[name].dist === "fixed" &&
          values.some((value) => typeof value !== "number" || !Number.isSafeInteger(value))) {
        throw new Error("固定长度扫描只能填写整数档位");
      }
    }
    if (mode === "trace" && blockSize != null && item.key === "inference.page_size"
        && values.some((value) => value !== blockSize)) {
      throw new Error(`Trace 每页 token 数必须等于计划块大小 ${blockSize}`);
    }
    return { key: item.key, values };
  });
}

interface Props {
  base: SimConfig;
  onSubmit: (request: ScanRequest) => void;
  onImportBase: (config: SimConfig) => void;
  disabled?: boolean;
}

export function ScanPanel({ base, onSubmit, onImportBase, disabled = false }: Props) {
  const [mode, setMode] = useState<"synthetic" | "trace">("synthetic");
  const [arrival, setArrival] = useState<"open_loop" | "closed_loop">(base.workload.arrival_mode);
  const [planId, setPlanId] = useState<string | null>(null);
  const [blockSize, setBlockSize] = useState<number | null>(null);
  const [planNotice, setPlanNotice] = useState("");
  const [planBusy, setPlanBusy] = useState(false);
  const [catalog, setCatalog] = useState<ScanParameter[]>([]);
  const [axes, setAxes] = useState<AxisDraft[]>([{ id: 0, key: "", text: "", search: "" }]);
  const nextId = useRef(1);
  const [focused, setFocused] = useState<number | null>(null);
  const [problem, setProblem] = useState("");
  const [scanJson, setScanJson] = useState("");
  const [transferNote, setTransferNote] = useState<{ message: string; error: boolean } | null>(null);
  const scanInputRef = useRef<HTMLInputElement>(null);
  const [planKey, setPlanKey] = useState(0);
  const [importedTraceWorkload, setImportedTraceWorkload] = useState<SimConfig["workload"] | null>(null);
  useEffect(() => {
    let active = true;
    void getScanParameters().then((items) => { if (active) setCatalog(items); })
      .catch(() => { if (active) setProblem("无法加载扫描参数目录"); });
    return () => { active = false; };
  }, []);

  const context = `${mode}/${arrival}`;
  const available = catalog.filter((item) => item.modes.includes(context));
  const selected = (axis: AxisDraft) => available.find((item) => item.key === axis.key);
  const usable = (id: number) => available.filter((item) =>
    !axes.some((other) => other.id !== id && other.key === item.key));
  function updateAxis(id: number, update: Partial<AxisDraft>) {
    setAxes((current) => current.map((axis) => axis.id === id ? { ...axis, ...update } : axis));
  }
  function changeContext(nextMode: "synthetic" | "trace", nextArrival: "open_loop" | "closed_loop") {
    setMode(nextMode); setArrival(nextArrival);
    const valid = new Set(catalog.filter((item) => item.modes.includes(`${nextMode}/${nextArrival}`)).map((item) => item.key));
    setAxes((current) => current.map((axis) => valid.has(axis.key)
      ? { ...axis, search: "" } : { id: axis.id, key: "", text: "", search: "" }));
    setProblem("");
  }

  async function receivePlan(file: File) {
    setPlanBusy(true); setPlanId(null); setBlockSize(null); setPlanNotice(""); setProblem("");
    try {
      const receipt = await uploadReplayPlan(file);
      setPlanId(receipt.id);
      setBlockSize(receipt.block_size ?? null);
      setPlanNotice(`${file.name} · ${receipt.sessions} 个会话 · ${receipt.requests} 个请求`
        + (receipt.block_size ? ` · 每块 ${receipt.block_size} token` : ""));
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "回放计划上传失败");
    } finally { setPlanBusy(false); }
  }

  function makeBase(): SimConfig {
    const config = structuredClone(base);
    if (mode === "trace") {
      const source = importedTraceWorkload ?? base.workload;
      config.workload = { mode: "trace", arrival_mode: arrival, seed: source.seed,
        session_rate: source.session_rate ?? 1,
        ...(arrival === "open_loop" && importedTraceWorkload?.target_qps != null
          ? { target_qps: importedTraceWorkload.target_qps } : {}),
        trace: { plan_id: planId } };
    } else {
      const source = base.workload.synthetic;
      if (!source) throw new Error("请先在普通配置中提供合成负载参数");
      const synthetic = structuredClone(source);
      if (arrival === "open_loop") {
        synthetic.session = { num_sessions: null, think_time_ms: 0, topology: "linear" };
        config.workload = { mode: "synthetic", arrival_mode: arrival,
          seed: base.workload.seed, synthetic, target_qps: base.workload.target_qps ?? 64 };
      } else {
        synthetic.session = { num_sessions: synthetic.session?.num_sessions ?? Math.min(20, synthetic.num_requests),
          think_time_ms: synthetic.session?.think_time_ms ?? 10_000, topology: "linear" };
        config.workload = { mode: "synthetic", arrival_mode: arrival,
          seed: base.workload.seed, synthetic, session_rate: base.workload.session_rate ?? 1 };
      }
    }
    return config;
  }

  function run() {
    try {
      if (mode === "trace" && !planId) throw new Error("请上传回放计划 JSON");
      const config = makeBase();
      const errors = validateSimConfig(config);
      if (errors.length) throw new Error(`基础配置不合法：${errors[0].message}`);
      const dimensions = scanDimensions(config, axes, catalog, mode, arrival, blockSize);
      if (mode === "trace" && blockSize != null &&
          !dimensions.some((axis) => axis.key === "inference.page_size") &&
          config.inference.page_size !== blockSize) {
        throw new Error(`当前每页 token 数必须等于 Trace 计划块大小 ${blockSize}`);
      }
      setProblem("");
      onSubmit({ base: config, axes: dimensions });
    } catch (error) { setProblem(error instanceof Error ? error.message : "扫描配置无效"); }
  }
  function exportScan() {
    try {
      const config = makeBase();
      scanDimensions(config, axes, catalog, mode, arrival);
      const json = JSON.stringify({ version: 1, base: JSON.parse(serializeSimConfig(config)),
        axes: axes.map(({ key, text }) => ({ key, text })) });
      setScanJson(json);
      const message = mode === "trace"
        ? "已生成扫描方案 JSON；再次使用时请重新上传回放计划"
        : "已生成扫描方案 JSON，可手动复制";
      setTransferNote({ message, error: false });
      scanInputRef.current?.focus();
      scanInputRef.current?.select();
      if (navigator.clipboard?.writeText) {
        try {
          void navigator.clipboard.writeText(json).then(() => {
            setTransferNote((note) => note?.message === message
              ? { message: `已生成并复制扫描方案 JSON${mode === "trace" ? "；再次使用时请重新上传回放计划" : ""}`, error: false }
              : note);
          }).catch(() => {});
        } catch {}
      }
    } catch (error) {
      setTransferNote({ message: `扫描方案导出失败：${error instanceof Error ? error.message : "未知错误"}`, error: true });
    }
  }

  function importScan() {
    try {
      let parsed: unknown;
      try { parsed = JSON.parse(scanJson.trim()); } catch { throw new Error("JSON 格式不正确"); }
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("扫描方案必须为 JSON 对象");
      const record = parsed as Record<string, unknown>;
      if (record.version !== 1 || Object.keys(record).some((key) => !["version", "base", "axes"].includes(key))) {
        throw new Error("不支持的扫描方案格式或版本");
      }
      const imported = parseSimConfigJson(JSON.stringify(record.base));
      const importedMode = imported.workload.mode;
      const importedArrival = imported.workload.arrival_mode;
      if (!Array.isArray(record.axes) || !record.axes.length || record.axes.some((axis: unknown) =>
        axis === null || typeof axis !== "object" || Array.isArray(axis) ||
        typeof (axis as Record<string, unknown>).key !== "string" ||
        typeof (axis as Record<string, unknown>).text !== "string" ||
        Object.keys(axis).some((key) => !["key", "text"].includes(key)))) {
        throw new Error("扫描维度必须包含参数 key 和档位 text");
      }
      const importedAxes = record.axes as { key: string; text: string }[];
      scanDimensions(imported, importedAxes, catalog, importedMode, importedArrival);
      if (importedMode === "trace") imported.workload.trace = { plan_id: null };
      onImportBase(imported);
      setImportedTraceWorkload(importedMode === "trace" ? imported.workload : null);
      setMode(importedMode); setArrival(importedArrival);
      setAxes(importedAxes.map((axis, index) => ({ ...axis, id: index, search: "" })));
      nextId.current = importedAxes.length;
      setPlanId(null); setBlockSize(null); setPlanNotice(""); setPlanKey((key) => key + 1);
      setFocused(null); setProblem("");
      setTransferNote({ message: importedMode === "trace"
        ? "扫描方案已导入，基础配置已更新；请重新上传回放计划"
        : "扫描方案已导入，基础配置已更新", error: false });
    } catch (error) {
      setTransferNote({ message: `扫描方案导入失败：${error instanceof Error ? error.message : "未知错误"}`, error: true });
    }
  }

  const total = axes.reduce((count, axis) => {
    if (!selected(axis)) return count;
    const text = axis.text.trim();
    const range = text.match(/^(-?[\d.]+):(-?[\d.]+):(-?[\d.]+)$/);
    if (range) {
      const [start, end, step] = range.slice(1).map(Number);
      const length = Math.floor((end - start) / step + 1e-9) + 1;
      return count * BigInt(Number.isSafeInteger(length) && length > 0 ? length : 0);
    }
    return count * BigInt(text.split(/[,，\s]+/).filter(Boolean).length);
  }, 1n);

  return <section className="scan-panel" aria-labelledby="scan-heading">
    <div className="section-heading"><div><p className="section-kicker">Sensitivity sweep</p>
      <h2 id="scan-heading">参数组合扫描</h2></div>
      <p>按需添加扫描参数；结果中最多选择四个可变参数，其余作为固定背景参数。</p></div>
    <div className="parameter-transfer scan-transfer" role="group" aria-label="扫描方案导入导出">
      <button type="button" className="transfer-button" onClick={exportScan} disabled={disabled || planBusy || !catalog.length}>
        导出扫描方案</button>
      <button type="button" className="transfer-button" onClick={importScan} disabled={disabled || planBusy || !catalog.length}>
        导入扫描方案</button>
      <label className="transfer-label" htmlFor="scan-json">扫描方案 JSON</label>
      <input id="scan-json" ref={scanInputRef} type="text" value={scanJson}
        placeholder="扫描方案 JSON 在此粘贴/生成" disabled={disabled}
        onChange={(event) => { setScanJson(event.currentTarget.value); setTransferNote(null); }}
        aria-invalid={transferNote?.error || undefined}
        aria-describedby={transferNote ? "scan-transfer-note" : undefined} />
      {transferNote && <span id="scan-transfer-note"
        className={transferNote.error ? "transfer-note transfer-note-error" : "transfer-note"}
        role={transferNote.error ? "alert" : "status"}>{transferNote.message}</span>}
    </div>
    <p className="scan-hint">扫描方案包含当前基础配置和扫描档位；导入会覆盖上方基础配置与当前扫描设置，Trace 回放计划需重新上传。</p>
    <div className="form-grid">
      <div className="form-field"><label htmlFor="scan-mode">负载模式</label>
        <select id="scan-mode" value={mode} onChange={(event) =>
          changeContext(event.target.value as typeof mode, arrival)}>
          <option value="synthetic">合成负载</option><option value="trace">数据集 Trace 回放</option>
        </select></div>
      <div className="form-field"><label htmlFor="scan-arrival">请求到达模式</label>
        <select id="scan-arrival" value={arrival} onChange={(event) =>
          changeContext(mode, event.target.value as typeof arrival)}>
          <option value="open_loop">开环</option><option value="closed_loop">闭环</option>
        </select></div>
      {mode === "trace" && <div className="form-field">
        <label htmlFor="scan-plan">回放计划 JSON 文件</label>
        <input id="scan-plan" key={planKey} type="file" accept=".json" disabled={disabled || planBusy}
          onChange={(event) => { const file = event.target.files?.[0]; if (file) void receivePlan(file); }} />
        {planNotice && <small>{planNotice}</small>}</div>}
    </div>
    <div className="scan-axes">{axes.map((axis, index) => {
      const options = usable(axis.id).filter((item) =>
        `${item.label} ${item.key}`.toLowerCase().includes(axis.search.trim().toLowerCase()));
      return <div className="scan-axis" key={axis.id}>
        <div className="scan-axis-parameter">
          <label htmlFor={`scan-search-${axis.id}`}>扫描参数 {index + 1}</label>
          <input id={`scan-search-${axis.id}`} role="combobox" aria-expanded={focused === axis.id}
            aria-controls={`scan-options-${axis.id}`} autoComplete="off" placeholder="输入中文名或字段路径搜索"
            value={focused === axis.id ? axis.search : (selected(axis)?.label ?? axis.search)}
            onFocus={() => { setFocused(axis.id); updateAxis(axis.id, { search: "" }); }}
            onBlur={() => window.setTimeout(() => setFocused((current) => current === axis.id ? null : current), 140)}
            onChange={(event) => updateAxis(axis.id, { search: event.target.value, key: "", text: "" })}
            onKeyDown={(event) => {
              if (event.key === "Escape") setFocused(null);
              if (event.key === "Enter" && options.length) {
                event.preventDefault(); updateAxis(axis.id, { key: options[0].key,
                  text: initialValue(makeBase(), options[0].key), search: "" });
                setFocused(null);
              }
            }} />
          {focused === axis.id && <div id={`scan-options-${axis.id}`} className="scan-options" role="listbox">
            {options.map((item) => <button type="button" role="option" aria-selected={axis.key === item.key}
              key={item.key} onMouseDown={(event) => event.preventDefault()}
              onClick={() => { updateAxis(axis.id, { key: item.key,
                text: initialValue(makeBase(), item.key), search: "" }); setFocused(null); }}>
              {item.label}<small>{item.key}</small></button>)}
            {!options.length && <p>没有匹配参数</p>}</div>}
        </div>
        <div className="scan-axis-levels">
          <label htmlFor={`scan-values-${axis.id}`}>扫描档位 {index + 1}</label>
          <input id={`scan-values-${axis.id}`} value={axis.text} disabled={!selected(axis)}
            placeholder="逗号分隔，或 起点:终点:步长"
            onChange={(event) => updateAxis(axis.id, { text: event.target.value })} />
        </div>
        <button type="button" className="secondary-button secondary-button-small scan-remove-axis" disabled={axes.length === 1}
          aria-label={`删除扫描参数 ${index + 1}`}
          onClick={() => { setAxes((current) => current.filter((row) => row.id !== axis.id)); setFocused(null); }}>
          删除</button>
      </div>;
    })}</div>
    <button type="button" className="secondary-button scan-add-axis"
      onClick={() => { setAxes((current) => [...current, { id: nextId.current++, key: "", text: "", search: "" }]);
        setFocused(null); }}>添加参数</button>
    <p className="scan-hint">选定负载与到达模式后，适用的请求参数可在上方选择为扫描维度；未扫描的参数沿用当前基础配置，切换模式时缺失的投放速率使用默认值。Trace 开环扫描目标 QPS 时会缩放到达时间线。</p>
    <p className="scan-hint">输入离散档位（逗号分隔）或数值范围（起点:终点:步长）；布尔档位使用 true,false。
      固定长度使用整数，随机长度使用分布均值，保留原分布类型。</p>
    <div className="form-actions"><button type="button" className="run-button" onClick={run}
      disabled={disabled || planBusy || (mode === "trace" && !planId) || !catalog.length}>提交后台扫描</button>
      <span>组合数按所选维度档位的乘积计算；不设置扫描规模上限。{total > 0n ? ` 当前约 ${total.toString()} 点。` : ""}</span></div>
    {problem && <p className="field-error" role="alert">{problem}</p>}
  </section>;
}
