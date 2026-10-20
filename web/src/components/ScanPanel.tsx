import { useEffect, useState } from "react";

import { getScanParameters, uploadReplayPlan } from "../api/client";
import type { ScanParameter, ScanRequest, SimConfig } from "../api/types";
import { validateSimConfig } from "../config/simulation";
import "./ScanPanel.css";

type Role = "x" | "color" | "shape" | "line";
type AxisDraft = { key: string; text: string };
const roles: { id: Role; label: string }[] = [
  { id: "x", label: "横轴" }, { id: "color", label: "颜色" },
  { id: "shape", label: "节点形状" }, { id: "line", label: "线样式" },
];
const empty = (): Record<Role, AxisDraft> => ({
  x: { key: "", text: "" }, color: { key: "", text: "" },
  shape: { key: "", text: "" }, line: { key: "", text: "" },
});

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

interface Props {
  base: SimConfig;
  onSubmit: (request: ScanRequest) => void;
}

export function ScanPanel({ base, onSubmit }: Props) {
  const [mode, setMode] = useState<"synthetic" | "trace">("synthetic");
  const [arrival, setArrival] = useState<"open_loop" | "closed_loop">(base.workload.arrival_mode);
  const [planId, setPlanId] = useState<string | null>(null);
  const [blockSize, setBlockSize] = useState<number | null>(null);
  const [planNotice, setPlanNotice] = useState("");
  const [planBusy, setPlanBusy] = useState(false);
  const [catalog, setCatalog] = useState<ScanParameter[]>([]);
  const [axes, setAxes] = useState<Record<Role, AxisDraft>>(empty);
  const [search, setSearch] = useState<Record<Role, string>>({ x: "", color: "", shape: "", line: "" });
  const [focused, setFocused] = useState<Role | null>(null);
  const [problem, setProblem] = useState("");
  useEffect(() => {
    let active = true;
    void getScanParameters().then((items) => { if (active) setCatalog(items); })
      .catch(() => { if (active) setProblem("无法加载扫描参数目录"); });
    return () => { active = false; };
  }, []);

  const context = `${mode}/${arrival}`;
  const available = catalog.filter((item) => item.modes.includes(context));
  const selected = (role: Role) => available.find((item) => item.key === axes[role].key);
  const usable = (role: Role) => available.filter((item) =>
    !roles.some((other) => other.id !== role && axes[other.id].key === item.key));
  function changeContext(nextMode: "synthetic" | "trace", nextArrival: "open_loop" | "closed_loop") {
    setMode(nextMode); setArrival(nextArrival);
    const valid = new Set(catalog.filter((item) => item.modes.includes(`${nextMode}/${nextArrival}`)).map((item) => item.key));
    setAxes((current) => Object.fromEntries(roles.map(({ id }) => [id,
      valid.has(current[id].key) ? current[id] : { key: "", text: "" }])) as Record<Role, AxisDraft>);
    setSearch({ x: "", color: "", shape: "", line: "" });
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
      config.workload = { mode: "trace", arrival_mode: arrival, seed: base.workload.seed,
        session_rate: base.workload.session_rate ?? 1, trace: { plan_id: planId } };
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
      if (!selected("x")) throw new Error("请先选择横轴参数");
      if (mode === "trace" && !planId) throw new Error("请上传回放计划 JSON");
      const config = makeBase();
      const errors = validateSimConfig(config);
      if (errors.length) throw new Error(`基础配置不合法：${errors[0].message}`);
      const dimensions: ScanRequest["axes"] = {};
      for (const { id } of roles) {
        const item = selected(id);
        if (!item) continue;
        const values = parseLevels(axes[id].text, item.kind);
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
        dimensions[id] = { key: item.key, values };
      }
      if (mode === "trace" && blockSize != null &&
          !Object.values(dimensions).some((axis) => axis.key === "inference.page_size") &&
          config.inference.page_size !== blockSize) {
        throw new Error(`当前每页 token 数必须等于 Trace 计划块大小 ${blockSize}`);
      }
      setProblem("");
      onSubmit({ base: config, axes: dimensions });
    } catch (error) { setProblem(error instanceof Error ? error.message : "扫描配置无效"); }
  }
  const total = roles.reduce((count, { id }) => {
    if (!selected(id)) return count;
    const text = axes[id].text.trim();
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
      <p>独立于普通仿真；横轴与颜色、形状、线型构成参数组合。</p></div>
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
        <input id="scan-plan" type="file" accept=".json" disabled={planBusy}
          onChange={(event) => { const file = event.target.files?.[0]; if (file) void receivePlan(file); }} />
        {planNotice && <small>{planNotice}</small>}</div>}
    </div>
    <div className="scan-axes">{roles.map(({ id, label }) => {
      const options = usable(id).filter((item) =>
        `${item.label} ${item.key}`.toLowerCase().includes(search[id].trim().toLowerCase()));
      return <div className="scan-axis" key={id}>
        <label htmlFor={`scan-search-${id}`}>{label}参数{id === "x" ? "（必选）" : "（可选）"}</label>
        <input id={`scan-search-${id}`} role="combobox" aria-expanded={focused === id}
          aria-controls={`scan-options-${id}`} autoComplete="off" placeholder="输入中文名或字段路径搜索"
          value={focused === id ? search[id] : (selected(id)?.label ?? search[id])}
          onFocus={() => { setFocused(id); setSearch((current) => ({ ...current, [id]: "" })); }}
          onBlur={() => window.setTimeout(() => setFocused((current) => current === id ? null : current), 140)}
          onChange={(event) => { setSearch((current) => ({ ...current, [id]: event.target.value }));
            setAxes((current) => ({ ...current, [id]: { key: "", text: "" } })); }}
          onKeyDown={(event) => {
            if (event.key === "Escape") setFocused(null);
            if (event.key === "Enter" && options.length) {
              event.preventDefault(); setAxes((current) => ({ ...current, [id]: {
                key: options[0].key, text: initialValue(makeBase(), options[0].key) } }));
              setFocused(null);
            }
          }} />
        {focused === id && <div id={`scan-options-${id}`} className="scan-options" role="listbox">
          {id !== "x" && <button type="button" role="option" aria-selected={!axes[id].key}
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => { setAxes((current) => ({ ...current, [id]: { key: "", text: "" } })); setFocused(null); }}>
            不使用此维度</button>}
          {options.map((item) => <button type="button" role="option" aria-selected={axes[id].key === item.key}
            key={item.key} onMouseDown={(event) => event.preventDefault()}
            onClick={() => { setAxes((current) => ({ ...current, [id]: {
              key: item.key, text: initialValue(makeBase(), item.key) } })); setFocused(null); }}>
            {item.label}<small>{item.key}</small></button>)}
          {!options.length && <p>没有匹配参数</p>}</div>}
        {selected(id) && <><label htmlFor={`scan-values-${id}`}>{label}档位</label>
          <input id={`scan-values-${id}`} value={axes[id].text} placeholder="逗号分隔，或 起点:终点:步长"
            onChange={(event) => setAxes((current) => ({ ...current, [id]: {
              ...current[id], text: event.target.value } }))} /></>}
      </div>;
    })}</div>
    <p className="scan-hint">选定负载与到达模式后，适用的请求参数可在上方选择为扫描维度；未扫描的参数沿用当前基础配置，切换模式时缺失的投放速率使用默认值。Trace 开环扫描目标 QPS 时会缩放到达时间线。</p>
    <p className="scan-hint">输入离散档位（逗号分隔）或数值范围（起点:终点:步长）；布尔档位使用 true,false。
      固定长度使用整数，随机长度使用分布均值，保留原分布类型。</p>
    <div className="form-actions"><button type="button" className="run-button" onClick={run}
      disabled={planBusy || (mode === "trace" && !planId) || !catalog.length}>提交后台扫描</button>
      <span>组合数按所选维度档位的乘积计算；不设置扫描规模上限。{total > 0n ? ` 当前约 ${total.toString()} 点。` : ""}</span></div>
    {problem && <p className="field-error" role="alert">{problem}</p>}
  </section>;
}
