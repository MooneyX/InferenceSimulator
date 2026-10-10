import { useRef, useState, type FormEvent } from "react";

import type { ResolvedRequest, SimConfig } from "../api/types";
import { parseResolvedRequestJson, parseSimConfigJson, serializeResolvedRequest, serializeSimConfig } from "../config/parameterTransfer";
import {
  createDefaultSimConfig,
  validateSimConfig,
  type ConfigField,
} from "../config/simulation";
import { AutomaticConfig, initialHardware, initialModel, initialStorage } from "./AutomaticConfig";
import { LengthDistributionEditor } from "./LengthDistributionEditor";
import "./SimulationForm.css";

export interface SimulationFormProps {
  onSubmit: (config: SimConfig) => void;
  onSubmitAuto?: (request: ResolvedRequest) => void;
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

const defaultConfig = createDefaultSimConfig();

function configForAutomaticResolution(config: SimConfig, timing: ResolvedRequest["timing"]): SimConfig {
  const perfModel = {
    ...config.perf_model,
    kv_bytes_per_token: null,
    l2_to_l1_bandwidth_bytes_per_ms: null,
    l3_to_l2_bandwidth_bytes_per_ms: null,
    l2_latency_ms: 0,
    l3_latency_ms: 0,
  };
  if (timing === "analytical") {
    for (const field of ["prefill_c1", "prefill_c2", "decode_d1", "decode_d2"] as const) {
      if (!Number.isFinite(perfModel[field]) || perfModel[field] < 0) {
        perfModel[field] = defaultConfig.perf_model[field];
      }
    }
  }
  const pageSize = config.engine.page_size;
  return {
    ...config,
    engine: {
      ...config.engine,
      max_total_tokens: Number.isInteger(pageSize) && pageSize > 0 ? pageSize : defaultConfig.engine.max_total_tokens,
      l2_capacity_tokens: 0,
      l3_capacity_tokens: 0,
    },
    perf_model: perfModel,
  };
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

export function SimulationForm({ onSubmit, onSubmitAuto, disabled = false }: SimulationFormProps) {
  const [mode, setMode] = useState<"manual" | "automatic">(onSubmitAuto ? "automatic" : "manual");
  const [manualConfig, setManualConfig] = useState<SimConfig>(createDefaultSimConfig);
  const [automaticConfig, setAutomaticConfig] = useState<SimConfig>(createDefaultSimConfig);
  const config = mode === "automatic" ? automaticConfig : manualConfig;
  const setConfig = mode === "automatic" ? setAutomaticConfig : setManualConfig;
  const [model, setModel] = useState<ResolvedRequest["model"]>(initialModel);
  const [hardware, setHardware] = useState<ResolvedRequest["hardware"]>(initialHardware);
  const [storage, setStorage] = useState<ResolvedRequest["storage"]>(initialStorage);
  const [timing, setTiming] = useState<ResolvedRequest["timing"]>("analytical");
  const [tpSize, setTpSize] = useState(1);
  const [epSize, setEpSize] = useState(1);
  const [showValidation, setShowValidation] = useState(false);
  const [paramsJson, setParamsJson] = useState("");
  const [transferNote, setTransferNote] = useState<{ message: string; error: boolean } | null>(null);
  const paramsInputRef = useRef<HTMLInputElement>(null);
  const submissionConfig = mode === "automatic" ? configForAutomaticResolution(config, timing) : config;
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
  const synthetic = config.workload.synthetic;

  if (synthetic === undefined || synthetic === null) {
    throw new Error("SimulationForm requires a synthetic workload configuration");
  }
  const session = synthetic.session!;

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setShowValidation(true);

    if (validateSimConfig(submissionConfig).length === 0 && !automaticRunUnavailable) {
      if (mode === "automatic") {
        onSubmitAuto?.({ config: submissionConfig, model, hardware, storage, timing, tp_size: tpSize, ep_size: epSize });
      } else {
        onSubmit(submissionConfig);
      }
    }
  }

  function handleExport(): void {
    try {
      const json = mode === "automatic"
        ? serializeResolvedRequest({ config: submissionConfig, model, hardware, storage,
          timing, tp_size: tpSize, ep_size: epSize })
        : serializeSimConfig(submissionConfig);
      setParamsJson(json);
      setTransferNote({ message: "已生成参数 JSON，可手动复制", error: false });
      paramsInputRef.current?.focus();
      paramsInputRef.current?.select();
      if (navigator.clipboard?.writeText) {
        try {
          void navigator.clipboard.writeText(json).then(() => {
            setTransferNote((note) => note?.message === "已生成参数 JSON，可手动复制"
              ? { message: "已生成并复制参数 JSON", error: false } : note);
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
      const value: unknown = JSON.parse(paramsJson.trim());
      if (value !== null && typeof value === "object" && "config" in value) {
        const imported = parseResolvedRequestJson(paramsJson.trim());
        if (!onSubmitAuto) throw new Error("当前页面不支持自动解析参数");
        setAutomaticConfig(imported.config);
        setModel(imported.model);
        setHardware(imported.hardware);
        setStorage(imported.storage);
        setTiming(imported.timing);
        setTpSize(imported.tp_size);
        setEpSize(imported.ep_size);
        setMode("automatic");
      } else {
        setManualConfig(parseSimConfigJson(paramsJson.trim()));
        setMode("manual");
      }
      setShowValidation(false);
      setTransferNote({ message: "参数已导入，可检查后运行仿真", error: false });
    } catch (error) {
      setTransferNote({
        message: `参数导入失败：${error instanceof Error ? error.message : "未知错误"}`,
        error: true,
      });
    }
  }

  return (
    <form className="simulation-form" noValidate onSubmit={handleSubmit}>
      {onSubmitAuto && <fieldset className="configuration-mode" disabled={disabled}>
        <legend>配置方式</legend>
        <label className={mode === "automatic" ? "mode-option selected" : "mode-option"}>
          <input type="radio" name="configuration-mode" value="automatic" checked={mode === "automatic"}
            onChange={() => { setMode("automatic"); setShowValidation(false); setTransferNote(null); }} />
          <span><strong>自动解析</strong><small>根据模型与硬件推算容量及链路参数</small></span>
        </label>
        <label className={mode === "manual" ? "mode-option selected" : "mode-option"}>
          <input type="radio" name="configuration-mode" value="manual" checked={mode === "manual"}
            onChange={() => { setMode("manual"); setShowValidation(false); setTransferNote(null); }} />
          <span><strong>手动配置</strong><small>直接填写容量与经验计时参数</small></span>
        </label>
      </fieldset>}
      <fieldset className="configuration-panel" disabled={disabled} aria-label={mode === "automatic" ? "自动解析参数" : "手动配置参数"}>
      <section className="form-section" aria-labelledby="workload-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Workload</p>
            <h2 id="workload-heading">负载配置</h2>
          </div>
          <p>泊松到达过程与请求长度分布</p>
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
            <input id="workload-mode" value="synthetic" readOnly />
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
                    session: {
                      ...session,
                      num_sessions:
                        session.num_sessions === synthetic.num_requests &&
                        Number.isInteger(numRequests) && numRequests >= 1
                          ? numRequests
                          : session.num_sessions,
                    },
                  },
                },
              })
            }
          />
          <NumberField
            id="arrival-rate"
            name="workload.synthetic.arrival_rate"
            label="总请求目标 QPS（请求/秒）"
            value={synthetic.arrival_rate}
            error={errorByField.get("workload.synthetic.arrival_rate")}
            onChange={(arrivalRate) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: { ...synthetic, arrival_rate: arrivalRate },
                },
              })
            }
          />
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
          <NumberField
            id="num-sessions"
            name="workload.synthetic.session.num_sessions"
            label="会话数量（根请求数）"
            value={session.num_sessions ?? synthetic.num_requests}
            integer
            min={1}
            max={synthetic.num_requests}
            error={errorByField.get("workload.synthetic.session.num_sessions")}
            onChange={(numSessions) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: { ...synthetic, session: { ...session, num_sessions: numSessions } },
                },
              })
            }
          />
          <NumberField
            id="think-time-ms"
            name="workload.synthetic.session.think_time_ms"
            label="轮次间平均思考时间（ms，指数分布）"
            value={session.think_time_ms}
            error={errorByField.get("workload.synthetic.session.think_time_ms")}
            onChange={(thinkTime) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: { ...synthetic, session: { ...session, think_time_ms: thinkTime } },
                },
              })
            }
          />
          <NumberField
            id="target-ideal-prefix-hit-rate"
            name="workload.synthetic.session.target_ideal_prefix_hit_rate"
            label="目标理想前缀命中率（%）"
            value={session.target_ideal_prefix_hit_rate * 100}
            max={100}
            error={errorByField.has("workload.synthetic.session.target_ideal_prefix_hit_rate")
              ? "必须是 0 到 100 之间的有限数值" : undefined}
            onChange={(targetRate) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: {
                    ...synthetic,
                    session: { ...session, target_ideal_prefix_hit_rate: targetRate / 100 },
                  },
                },
              })
            }
          />
        </div>

        <div className="distribution-grid">
          <LengthDistributionEditor
            name="prompt_len"
            label="Prompt 长度"
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
      </section>

      {mode === "automatic" && <AutomaticConfig config={submissionConfig} model={model} hardware={hardware}
        storage={storage} timing={timing} tpSize={tpSize} epSize={epSize}
        onTpSizeChange={setTpSize} onEpSizeChange={setEpSize} disabled={disabled}
        onModelChange={setModel} onHardwareChange={setHardware} onStorageChange={setStorage} />}

      <section className="form-section" aria-labelledby="engine-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Engine</p>
            <h2 id="engine-heading">引擎配置</h2>
          </div>
          <p>单实例 KV 页池与 Prefill/Decode 执行方式</p>
        </div>

        <div className="form-grid">
          <NumberField
            id="max-running-requests"
            name="engine.max_running_requests"
            label="最多同时运行请求"
            value={config.engine.max_running_requests}
            integer
            min={1}
            error={errorByField.get("engine.max_running_requests")}
            onChange={(value) => setConfig({
              ...config, engine: { ...config.engine, max_running_requests: value },
            })}
          />
          {mode === "manual" && <>
          <NumberField
            id="max-total-tokens"
            name="engine.max_total_tokens"
            label="L1 KV 总容量（token）"
            value={config.engine.max_total_tokens}
            integer
            min={1}
            error={errorByField.get("engine.max_total_tokens")}
            onChange={(value) => setConfig({
              ...config, engine: { ...config.engine, max_total_tokens: value },
            })}
          />
          <NumberField
            id="l2-capacity-tokens"
            name="engine.l2_capacity_tokens"
            label="L2 KV 容量（token，0 为关闭）"
            value={config.engine.l2_capacity_tokens}
            integer
            error={errorByField.get("engine.l2_capacity_tokens")}
            onChange={(value) => setConfig({
              ...config, engine: { ...config.engine, l2_capacity_tokens: value },
            })}
          />
          <NumberField
            id="l3-capacity-tokens"
            name="engine.l3_capacity_tokens"
            label="L3 KV 容量（token，0 为关闭）"
            value={config.engine.l3_capacity_tokens}
            integer
            error={errorByField.get("engine.l3_capacity_tokens")}
            onChange={(value) => setConfig({
              ...config,                 engine: { ...config.engine, l3_capacity_tokens: value },
            })}
          />
          </>}
          <NumberField
            id="max-prefill-tokens"
            name="engine.max_prefill_tokens"
            label="单次 Prefill 额度（token）"
            value={config.engine.max_prefill_tokens}
            integer
            min={1}
            error={errorByField.get("engine.max_prefill_tokens")}
            onChange={(value) => setConfig({
              ...config, engine: { ...config.engine, max_prefill_tokens: value },
            })}
          />
          <NumberField
            id="page-size"
            name="engine.page_size"
            label="每页 token 数"
            value={config.engine.page_size}
            integer
            min={1}
            error={errorByField.get("engine.page_size")}
            onChange={(value) => setConfig({
              ...config, engine: { ...config.engine, page_size: value },
            })}
          />
        </div>
        <div className="form-field">
          <label htmlFor="pd-mutually-exclusive">
            <input
              id="pd-mutually-exclusive"
              name="engine.pd_mutually_exclusive"
              type="checkbox"
              checked={config.engine.pd_mutually_exclusive}
              onChange={(event) => setConfig({
                ...config,
                engine: { ...config.engine, pd_mutually_exclusive: event.currentTarget.checked },
              })}
            />
            Prefill / Decode 分步执行（关闭后允许融合步）
          </label>
        </div>
        {mode === "manual" && <p>
          L1 容量：{Number.isInteger(config.engine.max_total_tokens) &&
          Number.isInteger(config.engine.page_size) && config.engine.page_size > 0 &&
          config.engine.max_total_tokens >= config.engine.page_size &&
          config.engine.max_total_tokens % config.engine.page_size === 0
            ? `${config.engine.max_total_tokens / config.engine.page_size} 页`
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
              setTiming(event.target.value as ResolvedRequest["timing"])}>
              <option value="analytical">逐步 Roofline 估计</option>
              <option value="empirical">原四系数经验模型</option>
            </select></div>
        </div>}
        {mode === "automatic" && timing === "analytical" &&
          <p>使用模型与硬件规格估算计时；KV 容量和链路参数由自动解析生成。</p>}

        {mode === "manual" && config.engine.l2_capacity_tokens > 0 ? (
          <div className="form-grid" style={{ marginBottom: 14 }}>
            <NumberField
              id="kv-bytes-per-token"
              name="perf_model.kv_bytes_per_token"
              label="KV 每 token 字节数"
              value={config.perf_model.kv_bytes_per_token ?? NaN}
              integer
              min={1}
              error={errorByField.get("perf_model.kv_bytes_per_token")}
              onChange={(value) => setConfig({
                ...config,
                perf_model: { ...config.perf_model, kv_bytes_per_token: Number.isNaN(value) ? null : value },
              })}
            />
            <NumberField
              id="l2-to-l1-bandwidth"
              name="perf_model.l2_to_l1_bandwidth_bytes_per_ms"
              label="L2 → L1 带宽（字节/ms）"
              value={config.perf_model.l2_to_l1_bandwidth_bytes_per_ms ?? NaN}
              error={errorByField.get("perf_model.l2_to_l1_bandwidth_bytes_per_ms")}
              onChange={(value) => setConfig({
                ...config,
                perf_model: { ...config.perf_model, l2_to_l1_bandwidth_bytes_per_ms: Number.isNaN(value) ? null : value },
              })}
            />
            <NumberField
              id="l2-latency-ms"
              name="perf_model.l2_latency_ms"
              label="L2 链路固定延迟（ms）"
              value={config.perf_model.l2_latency_ms}
              error={errorByField.get("perf_model.l2_latency_ms")}
              onChange={(value) => setConfig({
                ...config, perf_model: { ...config.perf_model, l2_latency_ms: value },
              })}
            />
            {config.engine.l3_capacity_tokens > 0 ? (
              <>
                <NumberField
                  id="l3-to-l2-bandwidth"
                  name="perf_model.l3_to_l2_bandwidth_bytes_per_ms"
                  label="L3 → L2 带宽（字节/ms）"
                  value={config.perf_model.l3_to_l2_bandwidth_bytes_per_ms ?? NaN}
                  error={errorByField.get("perf_model.l3_to_l2_bandwidth_bytes_per_ms")}
                  onChange={(value) => setConfig({
                    ...config,
                    perf_model: { ...config.perf_model, l3_to_l2_bandwidth_bytes_per_ms: Number.isNaN(value) ? null : value },
                  })}
                />
                <NumberField
                  id="l3-latency-ms"
                  name="perf_model.l3_latency_ms"
                  label="L3 链路固定延迟（ms）"
                  value={config.perf_model.l3_latency_ms}
                  error={errorByField.get("perf_model.l3_latency_ms")}
                  onChange={(value) => setConfig({
                    ...config, perf_model: { ...config.perf_model, l3_latency_ms: value },
                  })}
                />
              </>
            ) : null}
          </div>
        ) : null}
        {(mode === "manual" || timing === "empirical") && <div className="form-grid">
          <NumberField
            id="prefill-c1"
            name="perf_model.prefill_c1"
            label="Prefill C1"
            value={config.perf_model.prefill_c1}
            error={errorByField.get("perf_model.prefill_c1")}
            onChange={(prefillC1) =>
              setConfig({
                ...config,
                perf_model: { ...config.perf_model, prefill_c1: prefillC1 },
              })
            }
          />
          <NumberField
            id="prefill-c2"
            name="perf_model.prefill_c2"
            label="Prefill C2"
            value={config.perf_model.prefill_c2}
            error={errorByField.get("perf_model.prefill_c2")}
            onChange={(prefillC2) =>
              setConfig({
                ...config,
                perf_model: { ...config.perf_model, prefill_c2: prefillC2 },
              })
            }
          />
          <NumberField
            id="decode-d1"
            name="perf_model.decode_d1"
            label="Decode D1"
            value={config.perf_model.decode_d1}
            error={errorByField.get("perf_model.decode_d1")}
            onChange={(decodeD1) =>
              setConfig({
                ...config,
                perf_model: { ...config.perf_model, decode_d1: decodeD1 },
              })
            }
          />
          <NumberField
            id="decode-d2"
            name="perf_model.decode_d2"
            label="Decode D2"
            value={config.perf_model.decode_d2}
            error={errorByField.get("perf_model.decode_d2")}
            onChange={(decodeD2) =>
              setConfig({
                ...config,
                perf_model: { ...config.perf_model, decode_d2: decodeD2 },
              })
            }
          />
        </div>}
      </section>
      </fieldset>

      <div className="form-actions">
        <button className="run-button" type="submit" disabled={disabled || automaticRunUnavailable}>
          {disabled ? "正在运行…" : "运行仿真"}
        </button>
        {automaticRunUnavailable && <p>当前配置尚不可运行；请核对 TP/EP 与 GPU 数、计时模式及通信参数。仍可预览容量。</p>}
      </div>
    </form>
  );
}
