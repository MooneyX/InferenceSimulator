import { useRef, useState, type FormEvent } from "react";

import type { EmpiricalTimingConfig, ScanRequest, SimConfig } from "../api/types";
import { parseSimConfigJson, serializeSimConfig } from "../config/parameterTransfer";
import {
  createDefaultSimConfig,
  createDefaultEmpiricalTiming,
  initialSharedL3,
  resolveClosedLoopCounts,
  validateSimConfig,
  type ClosedLoopDerivedField,
  type FormSimConfig,
  type ConfigField,
} from "../config/simulation";
import { ModelHardwareConfig } from "./ModelHardwareConfig";
import { ScanPanel } from "./ScanPanel";
import { LengthDistributionEditor } from "./LengthDistributionEditor";
import { TracePanel, type ReplaySelection } from "./TracePanel";
import "./SimulationForm.css";

export interface SimulationFormProps {
  onSubmit: (config: SimConfig, force?: boolean) => void;
  onScanSubmit?: (request: ScanRequest) => void;
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
  readOnly?: boolean;
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
  readOnly = false,
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
        readOnly={readOnly}
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

export function SimulationForm({ onSubmit, onScanSubmit, disabled = false }: SimulationFormProps) {
  const [workloadMode, setWorkloadMode] = useState<"synthetic" | "trace">("synthetic");
  const [replay, setReplay] = useState<ReplaySelection | null>(null);
  const [traceKey, setTraceKey] = useState(0);
  const [traceDraft, setTraceDraft] = useState({ arrival_mode: "open_loop" as "open_loop" | "closed_loop", session_rate: 1, seed: 0, target_qps: undefined as number | undefined });
  const [draftConfig, setConfig] = useState<FormSimConfig>(createDefaultSimConfig);
  const [derivedField, setDerivedField] = useState<ClosedLoopDerivedField>("turns");
  const [turnsDraft, setTurnsDraft] = useState(NaN);
  const draftSynthetic = draftConfig.workload.synthetic!;
  const closedLoopCounts = draftConfig.workload.arrival_mode === "closed_loop"
    ? resolveClosedLoopCounts(draftSynthetic.num_requests,
      draftSynthetic.session?.num_sessions ?? draftSynthetic.num_requests, turnsDraft, derivedField)
    : null;
  const config: FormSimConfig = closedLoopCounts ? { ...draftConfig, workload: { ...draftConfig.workload,
    synthetic: { ...draftSynthetic, num_requests: closedLoopCounts.numRequests,
      session: { ...draftSynthetic.session, topology: "linear",
        think_time_ms: draftSynthetic.session?.think_time_ms ?? 0,
        num_sessions: closedLoopCounts.numSessions } } } } : draftConfig;
  const [empiricalDraft, setEmpiricalDraft] = useState(createDefaultEmpiricalTiming);
  const { model, hardware, tp_size: tpSize, ep_size: epSize } = config.inference;
  const timing = config.inference.timing.mode;
  const empiricalTiming = config.inference.timing.mode === "empirical" ? config.inference.timing : null;
  function updateInference(patch: Partial<FormSimConfig["inference"]>) {
    setConfig((current) => ({ ...current, inference: { ...current.inference, ...patch } }));
  }
  function setTiming(value: FormSimConfig["inference"]["timing"]["mode"]) {
    if (config.inference.timing.mode === "empirical") setEmpiricalDraft(config.inference.timing);
    updateInference({ timing: value === "analytical" ? { mode: "analytical" } : empiricalDraft });
  }
  function updateTiming(patch: Partial<EmpiricalTimingConfig>) {
    if (config.inference.timing.mode !== "empirical") return;
    updateInference({ timing: { ...config.inference.timing, ...patch } });
  }
  const [showValidation, setShowValidation] = useState(false);
  const [paramsJson, setParamsJson] = useState("");
  const [transferNote, setTransferNote] = useState<{ message: string; error: boolean } | null>(null);
  const paramsInputRef = useRef<HTMLInputElement>(null);
  const submissionConfig: SimConfig = workloadMode === "trace"
    ? { ...config, workload: { mode: "trace", ...(replay ? {
      arrival_mode: replay.arrivalMode, session_rate: replay.sessionRate, seed: replay.seed,
      ...(replay.targetQps === undefined ? {} : { target_qps: replay.targetQps }),
    } : traceDraft), trace: { plan_id: replay?.id ?? null } } }
    : config;
  const countErrors = workloadMode === "synthetic" ? closedLoopCounts?.errors ?? [] : [];
  const validationErrors = [...validateSimConfig(submissionConfig), ...countErrors];
  const errorByField = new Map(
    (showValidation ? validationErrors : []).map(({ field, message }) => [field, message]),
  );
  const needsTpCommunication = tpSize > 1;
  const needsEpCommunication = model.moe_layers > 0 && epSize > 1;
  const runUnavailable = (
    (timing === "empirical" && (model.moe_layers > 0 || needsTpCommunication)) ||
    tpSize !== hardware.gpu_count ||
    (model.moe_layers > 0 ? epSize !== tpSize : epSize !== 1) ||
    (needsTpCommunication && (hardware.effective_interconnect_gb_s == null ||
      hardware.collective_latency_ms == null)) ||
    (needsEpCommunication && hardware.ep_communication_latency_ms == null)
  );
  const synthetic = config.workload.synthetic!;
  function changeDerivedField(next: ClosedLoopDerivedField): void {
    setConfig(config);
    if (derivedField === "turns") setTurnsDraft(closedLoopCounts?.turnsPerSession ?? NaN);
    setDerivedField(next);
    setShowValidation(false);
  }
  function changeArrivalMode(arrivalMode: "open_loop" | "closed_loop"): void {
    setDerivedField("turns");
    const { session: _session, ...current } = config.workload.synthetic!;
    const { target_qps: _targetQps, session_rate: _sessionRate, ...workload } = config.workload;
    setConfig({ ...config, workload: { ...workload, arrival_mode: arrivalMode,
      ...(arrivalMode === "closed_loop"
        ? { session_rate: 1.28, synthetic: { ...current,
          session: { num_sessions: Math.min(20, current.num_requests), topology: "linear" as const,
            think_time_ms: 10_000 } } }
        : { target_qps: 64, synthetic: current }),
    } });
    setShowValidation(false);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setShowValidation(true);

    if (validationErrors.length > 0 || runUnavailable) return;
    if (workloadMode === "trace" && !replay) return;
    const force = (event.nativeEvent as SubmitEvent).submitter?.getAttribute("value") === "force";
    if (force) onSubmit(submissionConfig, true);
    else onSubmit(submissionConfig);
  }

  function handleExport(): void {
    try {
      if (countErrors.length) {
        setShowValidation(true);
        throw new Error(countErrors[0].message);
      }
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

  function handleScanBaseImport(imported: SimConfig): void {
    const inference = { ...imported.inference, timing: imported.inference.timing ?? { mode: "analytical" as const } };
    setConfig({ ...imported, inference,
      workload: imported.workload.mode === "trace" ? config.workload : imported.workload });
    if (inference.timing.mode === "empirical") setEmpiricalDraft(inference.timing);
    setWorkloadMode(imported.workload.mode);
    if (imported.workload.mode === "synthetic") setDerivedField("turns");
    if (imported.workload.mode === "trace") {
      setTraceDraft({ arrival_mode: imported.workload.arrival_mode ?? "closed_loop",
        session_rate: imported.workload.session_rate ?? 1, seed: imported.workload.seed ?? 0,
        target_qps: imported.workload.target_qps ?? undefined });
    }
    setReplay(null);
    setTraceKey((key) => key + 1);
    setShowValidation(false);
  }

  function handleImport(): void {
    try {
      const imported = parseSimConfigJson(paramsJson.trim());
      const inference = { ...imported.inference, timing: imported.inference.timing ?? { mode: "analytical" as const } };
      setConfig({ ...imported, inference,
        workload: imported.workload.mode === "trace" ? config.workload : imported.workload });
      if (inference.timing.mode === "empirical") setEmpiricalDraft(inference.timing);
      setWorkloadMode(imported.workload.mode);
      if (imported.workload.mode === "synthetic") setDerivedField("turns");
      if (imported.workload.mode === "trace" && imported.workload.trace) {
        const workload = imported.workload;
        setTraceDraft({ arrival_mode: workload.arrival_mode ?? "closed_loop",
          session_rate: workload.session_rate ?? NaN, seed: workload.seed ?? 0,
          target_qps: workload.target_qps ?? undefined });
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
      <fieldset className="configuration-panel" disabled={disabled} aria-label="仿真参数">
      <section className="form-section" aria-labelledby="workload-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Workload</p>
            <h2 id="workload-heading">负载配置</h2>
          </div>
          <p>{workloadMode === "trace" ? "数据集请求计划 · 开环或依赖闭环投放"
            : config.workload.arrival_mode === "open_loop" ? "泊松开环投放 · 完成的会话优先续写"
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
          initialReplay={{ arrivalMode: traceDraft.arrival_mode,
            sessionRate: traceDraft.session_rate, seed: traceDraft.seed, targetQps: traceDraft.target_qps }}
          onChange={setReplay} /> : <>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="arrival-mode">请求投放方式</label>
            <select id="arrival-mode" value={config.workload.arrival_mode}
              onChange={(event) => changeArrivalMode(event.currentTarget.value as "open_loop" | "closed_loop")}>
              <option value="open_loop">开环：按目标请求 QPS 投放</option>
              <option value="closed_loop">闭环：按新会话频率投放</option>
            </select>
          </div>
          {closedLoopCounts && <div className="form-field">
            <label htmlFor="closed-loop-derived-field">自动推定项</label>
            <select id="closed-loop-derived-field" value={derivedField}
              onChange={(event) => changeDerivedField(event.currentTarget.value as ClosedLoopDerivedField)}>
              <option value="turns">每会话轮次数</option>
              <option value="requests">请求数量</option>
              <option value="sessions">会话数量</option>
            </select>
          </div>}
          <NumberField
            id="num-requests"
            name="workload.synthetic.num_requests"
            label="请求数量"
            value={synthetic.num_requests}
            integer
            min={1}
            readOnly={closedLoopCounts !== null && derivedField === "requests"}
            error={closedLoopCounts && derivedField === "requests" ? undefined : errorByField.get("workload.synthetic.num_requests")}
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
          {config.workload.arrival_mode === "open_loop" ? <NumberField
            id="arrival-rate"
            name="workload.target_qps"
            label="目标发送 QPS（请求/秒，泊松到达）"
            value={config.workload.target_qps ?? NaN}
            error={errorByField.get("workload.target_qps")}
            onChange={(targetQps) => setConfig({ ...config, workload: { ...config.workload,
              target_qps: targetQps } })}
          /> : <>
            <NumberField id="num-sessions" name="workload.synthetic.session.num_sessions"
              label="会话数量（根请求数）" integer min={1}
              value={synthetic.session?.num_sessions ?? NaN}
              readOnly={derivedField === "sessions"}
              error={derivedField === "sessions" ? undefined : errorByField.get("workload.synthetic.session.num_sessions")}
              onChange={(numSessions) => setConfig({ ...config, workload: { ...config.workload,
                synthetic: { ...synthetic, session: { ...synthetic.session, topology: "linear",
                  think_time_ms: synthetic.session?.think_time_ms ?? 0,
                  num_sessions: numSessions } } } })}
            />
            {derivedField === "turns" ? <div className="form-field">
              <label htmlFor="turns-per-session">每会话轮次数（含首轮）</label>
              <input id="turns-per-session" type="text" readOnly
                value={closedLoopCounts?.turnsDisplay ?? ""} />
            </div> : <NumberField id="turns-per-session" name="turns_per_session"
              label="每会话轮次数（含首轮）" integer min={1} value={turnsDraft}
              error={errorByField.get("turns_per_session")} onChange={setTurnsDraft} />}
            <NumberField id="session-rate" name="workload.session_rate"
              label="新会话投放频率（会话/秒，泊松到达）"
              value={config.workload.session_rate ?? NaN}
              error={errorByField.get("workload.session_rate")}
              onChange={(sessionRate) => setConfig({ ...config, workload: { ...config.workload,
                session_rate: sessionRate } })}
            />
            <NumberField id="think-time" name="workload.synthetic.session.think_time_ms"
              label="轮次间平均思考时间（ms，指数分布）"
              value={synthetic.session?.think_time_ms ?? 0}
              error={errorByField.get("workload.synthetic.session.think_time_ms")}
              onChange={(thinkTime) => setConfig({ ...config, workload: { ...config.workload,
                synthetic: { ...synthetic, session: { ...synthetic.session, topology: "linear",
                  num_sessions: synthetic.session?.num_sessions,
                  think_time_ms: thinkTime } } } })}
            />
          </>}
          <NumberField
            id="seed"
            name="workload.seed"
            label="随机种子"
            value={config.workload.seed ?? 0}
            integer
            error={errorByField.get("workload.seed")}
            onChange={(seed) => setConfig({ ...config,
              workload: { ...config.workload, seed } })}
          />
        </div>

        <div className="workload-notes">
          {closedLoopCounts && <p>请求数量、会话数量、每会话轮次数只填写两项，另一项自动推定且只读。
            {derivedField === "sessions" ? "请求数量必须是每会话轮次数的整数倍。"
              : derivedField === "turns" ? "不能整除时显示两种轮次数，余数随机分配给部分会话，各增加一轮。"
                : "请求数量等于会话数量乘以每会话轮次数。"}</p>}
          <p>每请求生成上限由实际输出长度向上取二次幂自动确定，不可编辑；
            输出长度超过引擎输出上限的请求将失败，不会截断输出。</p>
          <p>{config.workload.arrival_mode === "open_loop"
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

      <ModelHardwareConfig config={{ ...config, workload: submissionConfig.workload }}
        onInferenceChange={(inference) => setConfig((current) => ({ ...current, inference }))} disabled={disabled} />

      <section className="form-section" aria-labelledby="shared-l3-heading">
        <div className="section-heading"><div>
          <p className="section-kicker">Shared cache</p><h2 id="shared-l3-heading">共享 L3</h2>
        </div><p>所有实例共享总容量，不乘实例数量；0 表示关闭。带宽按每实例接入链路计算，不模拟存储端总吞吐竞争。</p></div>
        <div className="form-grid">
          <NumberField id="shared-l3-capacity" name="shared_l3.available_gb"
            label="共享 L3 总容量（TB，仅 KV，0 为关闭）"
            value={(config.shared_l3?.available_gb ?? 0) / 1000}
            error={errorByField.get("shared_l3.available_gb")}
            onChange={(value) => setConfig({ ...config, shared_l3: { available_gb: value * 1000,
              lookup_latency_ms: config.shared_l3?.lookup_latency_ms ?? initialSharedL3.lookup_latency_ms } })} />
          <NumberField id="shared-l3-lookup" name="shared_l3.lookup_latency_ms" label="L3 查找延迟（ms）"
            value={config.shared_l3?.lookup_latency_ms ?? initialSharedL3.lookup_latency_ms} error={errorByField.get("shared_l3.lookup_latency_ms")}
            onChange={(value) => setConfig({ ...config, shared_l3: config.shared_l3
              ? { ...config.shared_l3, lookup_latency_ms: value }
              : { available_gb: 0, lookup_latency_ms: value } })} />
        </div>
        <p>L3 查找默认 0.1 ms/次，未命中也计入；假设低延迟共享存储，未经实测校准，可按部署调整，设为 0 可对照理想基线。</p>
      </section>

      <section className="form-section" aria-labelledby="engine-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Engine</p>
            <h2 id="engine-heading">引擎配置</h2>
          </div>
          <p>{(config.shared_l2_enabled ?? true) ? "每实例独占 L1 · 全局共享 L2/L3" : "每实例独占 L1/L2 · 全场共享 L3"}
            {(config.routing_policy ?? "least_load") === "least_load"
              ? " · 每轮最低负载优先，不绑定会话" : " · 新会话轮询，同会话固定实例"}</p>
        </div>

        <div className="form-grid">
          <NumberField
            id="num-instances"
            name="num_instances"
            label="推理实例数量"
            value={config.num_instances}
            integer
            min={1}
            error={errorByField.get("num_instances")}
            onChange={(value) => setConfig({ ...config, num_instances: value })}
          />
          <div className="form-field">
            <label htmlFor="shared-l2-enabled">
              <input id="shared-l2-enabled" name="shared_l2_enabled" type="checkbox"
                checked={config.shared_l2_enabled ?? true}
                onChange={(event) => setConfig({ ...config, shared_l2_enabled: event.currentTarget.checked })} />
              共享 L2
            </label>
          </div>
          <div className="form-field">
            <label htmlFor="routing-policy">实例路由策略</label>
            <select id="routing-policy" name="routing_policy" value={config.routing_policy ?? "least_load"}
              onChange={(event) => setConfig({ ...config,
                routing_policy: event.target.value as SimConfig["routing_policy"] })}>
              <option value="least_load">最低负载优先</option>
              <option value="cache_affinity">缓存亲和（新会话轮询）</option>
            </select>
          </div>
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
          <NumberField id="max-output-tokens" name="inference.max_output_tokens"
            label="引擎输出上限（token）" value={config.inference.max_output_tokens ?? 65_536}
            integer min={1} error={errorByField.get("inference.max_output_tokens")}
            onChange={(value) => setConfig({ ...config,
              inference: { ...config.inference, max_output_tokens: value } })} />
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
      </section>

      <section className="form-section" aria-labelledby="performance-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Performance model</p>
            <h2 id="performance-heading">性能模型</h2>
          </div>
          <p>所有时间系数的单位均为毫秒</p>
        </div>
        <div className="form-grid" style={{ marginBottom: 14 }}>
          <div className="form-field"><label htmlFor="timing-mode">计时模式</label>
            <select id="timing-mode" value={timing} onChange={(event) =>
              setTiming(event.target.value as FormSimConfig["inference"]["timing"]["mode"])}>
              <option value="analytical">逐步 Roofline 估计</option>
              <option value="empirical">四系数经验模型</option>
            </select></div>
        </div>
        {timing === "analytical" &&
          <p>使用模型与硬件规格估算计时；KV 容量和链路参数由自动解析生成。</p>}
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
          disabled={disabled || runUnavailable || (workloadMode === "trace" && !replay)}>
          {disabled ? "正在运行…" : "运行仿真"}
        </button>
        <button className="secondary-button" type="submit" name="cache-action" value="force"
          disabled={disabled || runUnavailable || (workloadMode === "trace" && !replay)}>
          强制重新计算
        </button>
        <p>相同参数自动复用结果缓存；强制重新计算会更新缓存。</p>
        {runUnavailable && <p>当前配置尚不可运行；请核对 TP/EP 与 GPU 数、计时模式及通信参数。仍可预览容量。</p>}
        {workloadMode === "trace" && !replay &&
          <p>请先选择会话并导出计划，或直接上传请求计划；同时检查回放参数。</p>}
      </div>
      {onScanSubmit && <ScanPanel base={config} onSubmit={onScanSubmit}
        closedLoopCountError={closedLoopCounts?.errors[0]?.message}
        onImportBase={handleScanBaseImport} disabled={disabled} />}
    </form>
  );
}
