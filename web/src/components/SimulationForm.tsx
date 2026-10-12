import { useRef, useState, type FormEvent } from "react";

import type { ManualInferenceConfig, SimConfig } from "../api/types";
import { parseSimConfigJson, serializeSimConfig } from "../config/parameterTransfer";
import {
  createDefaultSimConfig,
  createDefaultPhysicalSimConfig,
  createDefaultEmpiricalTiming,
  validateSimConfig,
  type ManualSimConfig,
  type PhysicalSimConfig,
  type ConfigField,
} from "../config/simulation";
import { AutomaticConfig } from "./AutomaticConfig";
import { LengthDistributionEditor } from "./LengthDistributionEditor";
import { TracePanel, type ReplaySelection } from "./TracePanel";
import "./SimulationForm.css";

export interface SimulationFormProps {
  onSubmit: (config: SimConfig) => void;
  disabled?: boolean;
}

interface NumberFieldProps {
  id: string;
  name: ConfigField;
  label: string;
  value: number;
  error?: string;
  integer?: boolean;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
}

function inputValue(value: number): number | "" {
  return Number.isNaN(value) ? "" : value;
}

function NumberField({
  id,
  name,
  label,
  value,
  error,
  integer = false,
  min = 0,
  max,
  onChange,
}: NumberFieldProps) {
  const errorId = `${id}-error`;

  return (
    <div className="form-field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        name={name}
        type="number"
        min={min}
        max={max}
        step={integer ? 1 : "any"}
        value={inputValue(value)}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? errorId : undefined}
        onChange={(event) => onChange(event.currentTarget.valueAsNumber)}
      />
      {error ? (
        <span id={errorId} className="field-error" role="alert">
          {error}
        </span>
      ) : null}
    </div>
  );
}

export function SimulationForm({ onSubmit, disabled = false }: SimulationFormProps) {
  const [mode, setMode] = useState<"manual" | "automatic">("automatic");
  const [workloadMode, setWorkloadMode] = useState<"synthetic" | "trace">("synthetic");
  const [replay, setReplay] = useState<ReplaySelection | null>(null);
  const [traceKey, setTraceKey] = useState(0);
  const [traceDraft, setTraceDraft] = useState({ mode: "open_loop" as "open_loop" | "session", session_rate: 1, seed: 0, target_qps: undefined as number | undefined });
  const [manualConfig, setManualConfig] = useState<ManualSimConfig>(createDefaultSimConfig);
  const [automaticConfig, setAutomaticConfig] = useState<PhysicalSimConfig>(createDefaultPhysicalSimConfig);
  const [empiricalDraft, setEmpiricalDraft] = useState(createDefaultEmpiricalTiming);
  const config: ManualSimConfig | PhysicalSimConfig = mode === "automatic" ? automaticConfig : manualConfig;
  function setConfig(next: SimConfig) {
    if (next.inference.mode === "manual") setManualConfig({ ...next, inference: next.inference });
    else setAutomaticConfig({ ...next, inference: { ...next.inference, timing: next.inference.timing ?? { mode: "analytical" } } });
  }
  const { model, hardware, tp_size: tpSize, ep_size: epSize } = automaticConfig.inference;
  const timing = config.inference.timing.mode;
  const empiricalTiming = config.inference.timing.mode === "empirical" ? config.inference.timing : null;
  function updatePhysical(patch: Partial<PhysicalSimConfig["inference"]>) {
    setAutomaticConfig((current) => ({ ...current, inference: { ...current.inference, ...patch } }));
  }
  function setTiming(value: PhysicalSimConfig["inference"]["timing"]["mode"]) {
    if (automaticConfig.inference.timing.mode === "empirical") setEmpiricalDraft(automaticConfig.inference.timing);
    updatePhysical({ timing: value === "analytical" ? { mode: "analytical" } : empiricalDraft });
  }
  function updateKv(patch: Partial<ManualInferenceConfig["kv"]>) {
    setManualConfig((current) => ({ ...current, inference: { ...current.inference, kv: { ...current.inference.kv, ...patch } } }));
  }
  function updateTiming(patch: Partial<ManualInferenceConfig["timing"]>) {
    if (config.inference.timing.mode !== "empirical") return;
    setConfig({ ...config, inference: { ...config.inference, timing: { ...config.inference.timing, ...patch } } });
  }
  const [showValidation, setShowValidation] = useState(false);
  const [paramsJson, setParamsJson] = useState("");
  const [transferNote, setTransferNote] = useState<{ message: string; error: boolean } | null>(null);
  const paramsInputRef = useRef<HTMLInputElement>(null);
  const submissionConfig: SimConfig = workloadMode === "trace"
    ? { ...config, workload: { mode: "trace", trace: replay ? {
      mode: replay.mode, session_rate: replay.rate, seed: replay.seed, plan_id: replay.id,
      ...(replay.targetQps === undefined ? {} : { target_qps: replay.targetQps }),
    } : { ...traceDraft, plan_id: null } } }
    : config;
  const validationErrors = showValidation ? validateSimConfig(submissionConfig) : [];
  const errorByField = new Map(
    validationErrors.map(({ field, message }) => [field, message]),
  );
  const needsTpCommunication = tpSize > 1;
  const needsEpCommunication = model.moe_layers > 0 && epSize > 1;
  const automaticRunUnavailable = mode === "automatic" && (
    (timing === "empirical" && (model.moe_layers > 0 || needsTpCommunication)) ||
    tpSize !== hardware.gpu_count ||
    (model.moe_layers > 0 ? epSize !== tpSize : epSize !== 1) ||
    (needsTpCommunication && (hardware.effective_interconnect_gb_s == null ||
      hardware.collective_latency_ms == null)) ||
    (needsEpCommunication && hardware.ep_communication_latency_ms == null)
  );
  const synthetic = config.workload.synthetic!;
  function changeArrivalMode(arrivalMode: "open_loop" | "session"): void {
    const current = config.workload.synthetic!;
    const nextSynthetic = arrivalMode === "session"
      ? {
          num_requests: current.num_requests,
          arrival_mode: "session" as const,
          session_rate: 1.28,
          prompt_len: current.prompt_len,
          turn_increment_len: current.turn_increment_len,
          output_len: current.output_len,
          max_new_tokens: current.max_new_tokens,
          seed: current.seed,
          session: { num_sessions: Math.min(20, current.num_requests), topology: "linear" as const,
            think_time_ms: 10_000 },
        }
      : {
          num_requests: current.num_requests,
          arrival_mode: "open_loop" as const,
          arrival_rate: 64,
          prompt_len: current.prompt_len,
          turn_increment_len: current.turn_increment_len,
          output_len: current.output_len,
          max_new_tokens: current.max_new_tokens,
          seed: current.seed,
        };
    setConfig({ ...config, workload: { ...config.workload, synthetic: nextSynthetic } });
    setShowValidation(false);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setShowValidation(true);

    if (validateSimConfig(submissionConfig).length > 0 || automaticRunUnavailable) return;
    if (workloadMode === "trace" && !replay) return;
    onSubmit(submissionConfig);
  }

  function handleExport(): void {
    try {
      const json = serializeSimConfig(submissionConfig);
      setParamsJson(json);
      const exportMessage = workloadMode === "trace" ? "已生成参数 JSON，可手动复制；再次使用时请重新上传请求计划" : "已生成参数 JSON，可手动复制";
      setTransferNote({ message: exportMessage, error: false });
      paramsInputRef.current?.focus();
      paramsInputRef.current?.select();
      if (navigator.clipboard?.writeText) {
        try {
          void navigator.clipboard.writeText(json).then(() => {
            setTransferNote((note) => note?.message === exportMessage
              ? { message: workloadMode === "trace" ? "已生成并复制参数 JSON；再次使用时请重新上传请求计划" : "已生成并复制参数 JSON", error: false } : note);
          }).catch(() => {});
        } catch {
          setTransferNote({ message: "已生成参数 JSON，可手动复制", error: false });
        }
      }
    } catch (error) {
      setTransferNote({
        message: `参数导出失败：${error instanceof Error ? error.message : "未知错误"}`,
        error: true,
      });
    }
  }

  function handleImport(): void {
    try {
      const imported = parseSimConfigJson(paramsJson.trim());
      if (imported.inference.mode === "physical") {
        const inference = { ...imported.inference, timing: imported.inference.timing ?? { mode: "analytical" as const } };
        setAutomaticConfig({ ...imported, inference,
          workload: imported.workload.mode === "trace" ? automaticConfig.workload : imported.workload });
        if (inference.timing.mode === "empirical") setEmpiricalDraft(inference.timing);
        setMode("automatic");
      } else {
        setManualConfig({ ...imported, inference: imported.inference,
          workload: imported.workload.mode === "trace" ? manualConfig.workload : imported.workload });
        setMode("manual");
      }
      setWorkloadMode(imported.workload.mode);
      if (imported.workload.mode === "trace" && imported.workload.trace) {
        const trace = imported.workload.trace;
        setTraceDraft({ mode: trace.mode, session_rate: trace.session_rate, seed: trace.seed, target_qps: trace.target_qps ?? undefined });
      }
      setReplay(null);
      setTraceKey((key) => key + 1);
      setShowValidation(false);
      setTransferNote({ message: imported.workload.mode === "trace"
        ? "参数已导入，回放设置已保留；请重新上传请求计划" : "参数已导入，可检查后运行仿真", error: false });
    } catch (error) {
      setTransferNote({
        message: `参数导入失败：${error instanceof Error ? error.message : "未知错误"}`,
        error: true,
      });
    }
  }

  return (
    <form className="simulation-form" noValidate onSubmit={handleSubmit}>
      <fieldset className="configuration-mode" disabled={disabled}>
        <legend>配置方式</legend>
        <label className={mode === "automatic" ? "mode-option selected" : "mode-option"}>
          <input type="radio" name="configuration-mode" value="automatic" checked={mode === "automatic"}
            onChange={() => { setMode("automatic"); setShowValidation(false); setTransferNote(null); }} />
          <span><strong>自动解析</strong><small>根据模型与硬件推算容量，支持解析或经验计时</small></span>
        </label>
        <label className={mode === "manual" ? "mode-option selected" : "mode-option"}>
          <input type="radio" name="configuration-mode" value="manual" checked={mode === "manual"}
            onChange={() => { setMode("manual"); setShowValidation(false); setTransferNote(null); }} />
          <span><strong>手动配置</strong><small>直接填写容量与经验计时参数</small></span>
        </label>
      </fieldset>
      <fieldset className="configuration-panel" disabled={disabled} aria-label={mode === "automatic" ? "自动解析参数" : "手动配置参数"}>
      <section className="form-section" aria-labelledby="workload-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Workload</p>
            <h2 id="workload-heading">负载配置</h2>
          </div>
          <p>{workloadMode === "trace" ? "数据集请求计划 · 开环或依赖闭环投放"
            : synthetic.arrival_mode === "open_loop" ? "泊松开环投放 · 完成的会话优先续写"
              : "闭环会话 · 完成后按思考时间续写"}</p>
        </div>

        <div className="parameter-transfer" role="group" aria-label="参数导入导出">
          <button type="button" className="transfer-button" onClick={handleExport} disabled={disabled}>
            导出参数
          </button>
          <button type="button" className="transfer-button" onClick={handleImport} disabled={disabled}>
            导入参数
          </button>
          <label className="transfer-label" htmlFor="params-json">参数 JSON</label>
          <input
            id="params-json"
            ref={paramsInputRef}
            type="text"
            value={paramsJson}
            placeholder="参数 JSON 在此粘贴/生成"
            onChange={(event) => {
              setParamsJson(event.currentTarget.value);
              setTransferNote(null);
            }}
            aria-invalid={transferNote?.error || undefined}
            aria-describedby={transferNote ? "parameter-transfer-note" : undefined}
          />
          {transferNote ? (
            <span
              id="parameter-transfer-note"
              className={transferNote.error ? "transfer-note transfer-note-error" : "transfer-note"}
              role={transferNote.error ? "alert" : "status"}
            >
              {transferNote.message}
            </span>
          ) : null}
        </div>

        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="workload-mode">负载模式</label>
            <select id="workload-mode" value={workloadMode}
              onChange={(event) => {
                setWorkloadMode(event.currentTarget.value as "synthetic" | "trace");
                setReplay(null);
                setShowValidation(false);
              }}>
              <option value="synthetic">合成负载</option>
              <option value="trace">数据集 Trace 回放</option>
            </select>
          </div>
        </div>
        {workloadMode === "trace" ? <TracePanel key={traceKey} disabled={disabled}
          initialReplay={{ mode: traceDraft.mode,
            rate: traceDraft.session_rate, seed: traceDraft.seed, targetQps: traceDraft.target_qps }}
          onChange={setReplay} /> : <>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="arrival-mode">请求投放方式</label>
            <select id="arrival-mode" value={synthetic.arrival_mode}
              onChange={(event) => changeArrivalMode(event.currentTarget.value as "open_loop" | "session")}>
              <option value="open_loop">开环：按目标请求 QPS 投放</option>
              <option value="session">闭环：按新会话频率投放</option>
            </select>
          </div>
          <NumberField
            id="num-requests"
            name="workload.synthetic.num_requests"
            label="请求数量"
            value={synthetic.num_requests}
            integer
            min={1}
            error={errorByField.get("workload.synthetic.num_requests")}
            onChange={(numRequests) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: {
                    ...synthetic,
                    num_requests: numRequests,
                  },
                },
              })
            }
          />
          {synthetic.arrival_mode === "open_loop" ? <NumberField
            id="arrival-rate"
            name="workload.synthetic.arrival_rate"
            label="目标发送 QPS（请求/秒，泊松到达）"
            value={synthetic.arrival_rate ?? NaN}
            error={errorByField.get("workload.synthetic.arrival_rate")}
            onChange={(arrivalRate) => setConfig({ ...config, workload: { ...config.workload,
              synthetic: { ...synthetic, arrival_rate: arrivalRate } } })}
          /> : <>
            <NumberField id="num-sessions" name="workload.synthetic.session.num_sessions"
              label="会话数量（根请求数）" integer min={1}
              value={synthetic.session?.num_sessions ?? NaN}
              error={errorByField.get("workload.synthetic.session.num_sessions")}
              onChange={(numSessions) => setConfig({ ...config, workload: { ...config.workload,
                synthetic: { ...synthetic, session: { ...synthetic.session, topology: "linear",
                  think_time_ms: synthetic.session?.think_time_ms ?? 0,
                  num_sessions: numSessions } } } })}
            />
            <NumberField id="session-rate" name="workload.synthetic.session_rate"
              label="新会话投放频率（会话/秒，泊松到达）"
              value={synthetic.session_rate ?? NaN}
              error={errorByField.get("workload.synthetic.session_rate")}
              onChange={(sessionRate) => setConfig({ ...config, workload: { ...config.workload,
                synthetic: { ...synthetic, session_rate: sessionRate } } })}
            />
            <NumberField id="think-time" name="workload.synthetic.session.think_time_ms"
              label="轮次间平均思考时间（ms，指数分布）"
              value={synthetic.session?.think_time_ms ?? NaN}
              error={errorByField.get("workload.synthetic.session.think_time_ms")}
              onChange={(thinkTime) => setConfig({ ...config, workload: { ...config.workload,
                synthetic: { ...synthetic, session: { ...synthetic.session, topology: "linear",
                  num_sessions: synthetic.session?.num_sessions,
                  think_time_ms: thinkTime } } } })}
            />
          </>}
          <NumberField
            id="seed"
            name="workload.synthetic.seed"
            label="随机种子"
            value={synthetic.seed}
            integer
            error={errorByField.get("workload.synthetic.seed")}
            onChange={(seed) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: { ...synthetic, seed },
                },
              })
            }
          />
        </div>

        <div className="form-grid" style={{ marginTop: 18 }}>
          <NumberField
            id="max-new-tokens"
            name="workload.synthetic.max_new_tokens"
            label="每请求生成上限（token）"
            value={synthetic.max_new_tokens}
            integer
            min={1}
            error={errorByField.get("workload.synthetic.max_new_tokens")}
            onChange={(maxNewTokens) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: { ...synthetic, max_new_tokens: maxNewTokens },
                },
              })
            }
          />
          <p>{synthetic.arrival_mode === "open_loop"
            ? "没有就绪子轮时自动创建新会话；理想前缀命中率由累积上下文决定。"
            : "闭环子轮须等待父轮完成及思考间隔；实际请求 QPS 由仿真结果计算。"}</p>
        </div>

        <div className="distribution-grid">
          <LengthDistributionEditor
            name="prompt_len"
            label="新会话根输入长度"
            value={synthetic.prompt_len}
            errors={{
              value: errorByField.get("workload.synthetic.prompt_len.value"),
              mean: errorByField.get("workload.synthetic.prompt_len.mean"),
              sigma: errorByField.get("workload.synthetic.prompt_len.sigma"),
            }}
            onChange={(promptLength) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: { ...synthetic, prompt_len: promptLength },
                },
              })
            }
          />
          <LengthDistributionEditor
            name="turn_increment_len"
            label="子轮新增内容长度"
            value={synthetic.turn_increment_len!}
            errors={{
              value: errorByField.get("workload.synthetic.turn_increment_len.value"),
              mean: errorByField.get("workload.synthetic.turn_increment_len.mean"),
              sigma: errorByField.get("workload.synthetic.turn_increment_len.sigma"),
            }}
            onChange={(incrementLength) => setConfig({
              ...config,
              workload: { ...config.workload, synthetic: { ...synthetic, turn_increment_len: incrementLength } },
            })}
          />
          <LengthDistributionEditor
            name="output_len"
            label="输出长度"
            value={synthetic.output_len}
            errors={{
              value: errorByField.get("workload.synthetic.output_len.value"),
              mean: errorByField.get("workload.synthetic.output_len.mean"),
              sigma: errorByField.get("workload.synthetic.output_len.sigma"),
            }}
            onChange={(outputLength) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: { ...synthetic, output_len: outputLength },
                },
              })
            }
          />
        </div>
        </>}
      </section>

      {mode === "automatic" && <AutomaticConfig config={{ ...automaticConfig, workload: submissionConfig.workload }}
        onInferenceChange={(inference) => setAutomaticConfig((current) => ({ ...current, inference }))} disabled={disabled} />}

      <section className="form-section" aria-labelledby="engine-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Engine</p>
            <h2 id="engine-heading">引擎配置</h2>
          </div>
          <p>{workloadMode === "trace" ? "每实例独占 KV · 同原会话的分支共享实例"
            : "每实例独占 KV · 根请求轮询 · 子请求严格继承父实例"}</p>
        </div>

        <div className="form-grid">
          <NumberField
            id="num-instances"
            name="num_instances"
            label="推理实例数量（每实例独占相同配置）"
            value={config.num_instances}
            integer
            min={1}
            error={errorByField.get("num_instances")}
            onChange={(value) => setConfig({ ...config, num_instances: value })}
          />
          <NumberField
            id="max-running-requests"
            name="scheduler.max_running_requests"
            label="最多同时运行请求"
            value={config.scheduler.max_running_requests}
            integer
            min={1}
            error={errorByField.get("scheduler.max_running_requests")}
            onChange={(value) => setConfig({
              ...config, scheduler: { ...config.scheduler, max_running_requests: value },
            })}
          />
          {config.inference.mode === "manual" && ([
            ["max_total_tokens", "L1 KV 总容量（token）"],
            ["l2_capacity_tokens", "L2 KV 容量（token，0 为关闭）"],
            ["l3_capacity_tokens", "L3 KV 容量（token，0 为关闭）"],
          ] as const).map(([field, label]) => <NumberField key={field} id={field.replaceAll("_", "-")}
            name={`inference.kv.${field}`} label={label} value={manualConfig.inference.kv[field]}
            integer min={field === "max_total_tokens" ? 1 : 0}
            error={errorByField.get(`inference.kv.${field}`)} onChange={(value) => updateKv({ [field]: value })} />)}
          <NumberField
            id="max-prefill-tokens"
            name="scheduler.max_prefill_tokens"
            label="单次 Prefill 额度（token）"
            value={config.scheduler.max_prefill_tokens}
            integer
            min={1}
            error={errorByField.get("scheduler.max_prefill_tokens")}
            onChange={(value) => setConfig({
              ...config, scheduler: { ...config.scheduler, max_prefill_tokens: value },
            })}
          />
          <NumberField
            id="l3-prefetch-min-tokens"
            name="scheduler.l3_prefetch_min_tokens"
            label="L3 异步预取最小长度（token，0 为不设阈值）"
            value={config.scheduler.l3_prefetch_min_tokens}
            integer
            error={errorByField.get("scheduler.l3_prefetch_min_tokens")}
            onChange={(value) => setConfig({
              ...config, scheduler: { ...config.scheduler, l3_prefetch_min_tokens: value },
            })}
          />
          <NumberField
            id="page-size"
            name="inference.page_size"
            label="每页 token 数"
            value={config.inference.page_size}
            integer
            min={1}
            error={errorByField.get("inference.page_size")}
            onChange={(value) => setConfig({
              ...config, inference: { ...config.inference, page_size: value },
            })}
          />
        </div>
        <div className="form-field">
          <label htmlFor="pd-mutually-exclusive">
            <input
              id="pd-mutually-exclusive"
              name="scheduler.pd_mutually_exclusive"
              type="checkbox"
              checked={config.scheduler.pd_mutually_exclusive}
              onChange={(event) => setConfig({
                ...config,
                scheduler: { ...config.scheduler, pd_mutually_exclusive: event.currentTarget.checked },
              })}
            />
            Prefill / Decode 分步执行（关闭后允许融合步）
          </label>
        </div>
        {config.inference.mode === "manual" && <p>
          L1 容量：{Number.isInteger(config.inference.kv.max_total_tokens) &&
          Number.isInteger(config.inference.page_size) && config.inference.page_size > 0 &&
          config.inference.kv.max_total_tokens >= config.inference.page_size &&
          config.inference.kv.max_total_tokens % config.inference.page_size === 0
            ? `${config.inference.kv.max_total_tokens / config.inference.page_size} 页`
            : "请填写至少一页且为整页倍数的容量"}
        </p>}
      </section>

      <section className="form-section" aria-labelledby="performance-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Performance model</p>
            <h2 id="performance-heading">性能模型</h2>
          </div>
          <p>所有时间系数的单位均为毫秒</p>
        </div>
        {mode === "automatic" && <div className="form-grid" style={{ marginBottom: 14 }}>
          <div className="form-field"><label htmlFor="auto-timing">计时模式</label>
            <select id="auto-timing" value={timing} onChange={(event) =>
              setTiming(event.target.value as PhysicalSimConfig["inference"]["timing"]["mode"])}>
              <option value="analytical">逐步 Roofline 估计</option>
              <option value="empirical">原四系数经验模型</option>
            </select></div>
        </div>}
        {mode === "automatic" && timing === "analytical" &&
          <p>使用模型与硬件规格估算计时；KV 容量和链路参数由自动解析生成。</p>}

        {config.inference.mode === "manual" && config.inference.kv.l2_capacity_tokens > 0 && (
          <div className="form-grid" style={{ marginBottom: 14 }}>
            {([
              ["kv_bytes_per_token", "KV 每 token 字节数"],
              ["l2_to_l1_bandwidth_bytes_per_ms", "L2 → L1 带宽（字节/ms）"],
              ["l2_latency_ms", "L2 链路固定延迟（ms）"],
              ...(config.inference.kv.l3_capacity_tokens > 0 ? [
                ["l3_to_l2_bandwidth_bytes_per_ms", "L3 → L2 带宽（字节/ms）"],
                ["l3_latency_ms", "L3 链路固定延迟（ms）"],
                ["l3_lookup_latency_ms", "L3 查找延迟（ms）"],
              ] as const : []),
            ] as const).map(([field, label]) => <NumberField key={field} id={field.replaceAll("_", "-")}
              name={`inference.kv.${field}`} label={label} value={manualConfig.inference.kv[field] ?? NaN}
              integer={field === "kv_bytes_per_token"} min={field === "kv_bytes_per_token" ? 1 : 0}
              error={errorByField.get(`inference.kv.${field}`)} onChange={(value) => updateKv({
                [field]: field === "kv_bytes_per_token" || field.endsWith("bandwidth_bytes_per_ms")
                  ? Number.isNaN(value) ? null : value : value,
              })} />)}
          </div>
        )}
        {empiricalTiming && <div className="form-grid">
          {([ ["prefill_c1", "Prefill C1"], ["prefill_c2", "Prefill C2"],
            ["decode_d1", "Decode D1"], ["decode_d2", "Decode D2"] ] as const).map(([field, label]) =>
            <NumberField key={field} id={field.replaceAll("_", "-")} name={`inference.timing.${field}`}
              label={label} value={empiricalTiming[field]}
              error={errorByField.get(`inference.timing.${field}`)} onChange={(value) => updateTiming({ [field]: value })} />)}
        </div>}
      </section>
      </fieldset>

      <div className="form-actions">
        <button className="run-button" type="submit"
          disabled={disabled || automaticRunUnavailable || (workloadMode === "trace" && !replay)}>
          {disabled ? "正在运行…" : "运行仿真"}
        </button>
        {automaticRunUnavailable && <p>当前配置尚不可运行；请核对 TP/EP 与 GPU 数、计时模式及通信参数。仍可预览容量。</p>}
        {workloadMode === "trace" && !replay &&
          <p>请先选择会话并导出计划，或直接上传请求计划；同时检查回放参数。</p>}
      </div>
    </form>
  );
}
