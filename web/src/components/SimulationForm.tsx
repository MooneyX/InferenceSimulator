import { useState, type FormEvent } from "react";

import type { SimConfig } from "../api/types";
import {
  createDefaultSimConfig,
  validateSimConfig,
  type ConfigField,
} from "../config/simulation";
import { LengthDistributionEditor } from "./LengthDistributionEditor";
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
  const [config, setConfig] = useState<SimConfig>(createDefaultSimConfig);
  const [showValidation, setShowValidation] = useState(false);
  const validationErrors = showValidation ? validateSimConfig(config) : [];
  const errorByField = new Map(
    validationErrors.map(({ field, message }) => [field, message]),
  );
  const synthetic = config.workload.synthetic;

  if (synthetic === undefined || synthetic === null) {
    throw new Error("SimulationForm requires a synthetic workload configuration");
  }
  const session = synthetic.session!;
  const prefixCache = config.engine.prefix_cache!;

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setShowValidation(true);

    if (validateSimConfig(config).length === 0) {
      onSubmit(config);
    }
  }

  return (
    <form className="simulation-form" noValidate onSubmit={handleSubmit}>
      <section className="form-section" aria-labelledby="workload-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Workload</p>
            <h2 id="workload-heading">负载配置</h2>
          </div>
          <p>泊松到达过程与请求长度分布</p>
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
            label="轮次间思考时间（ms）"
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
            label="目标理想前缀命中率（0–1）"
            value={session.target_ideal_prefix_hit_rate}
            max={1}
            error={errorByField.get("workload.synthetic.session.target_ideal_prefix_hit_rate")}
            onChange={(targetRate) =>
              setConfig({
                ...config,
                workload: {
                  ...config.workload,
                  synthetic: {
                    ...synthetic,
                    session: { ...session, target_ideal_prefix_hit_rate: targetRate },
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

      <section className="form-section" aria-labelledby="engine-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Engine</p>
            <h2 id="engine-heading">引擎配置</h2>
          </div>
          <p>批处理规模与 KV cache token 预算</p>
        </div>

        <div className="form-grid">
          <NumberField
            id="max-batch-size"
            name="engine.max_batch_size"
            label="最大批大小"
            value={config.engine.max_batch_size}
            integer
            min={1}
            error={errorByField.get("engine.max_batch_size")}
            onChange={(maxBatchSize) =>
              setConfig({
                ...config,
                engine: { ...config.engine, max_batch_size: maxBatchSize },
              })
            }
          />
          <NumberField
            id="kv-token-budget"
            name="engine.kv_token_budget"
            label="L1 KV token 预算"
            value={config.engine.kv_token_budget}
            integer
            min={1}
            error={errorByField.get("engine.kv_token_budget")}
            onChange={(kvTokenBudget) =>
              setConfig({
                ...config,
                engine: { ...config.engine, kv_token_budget: kvTokenBudget },
              })
            }
          />
          <NumberField
            id="block-size-tokens"
            name="engine.prefix_cache.block_size_tokens"
            label="Prefix cache 块大小（token）"
            value={prefixCache.block_size_tokens}
            integer
            min={1}
            error={errorByField.get("engine.prefix_cache.block_size_tokens")}
            onChange={(blockSize) =>
              setConfig({
                ...config,
                engine: {
                  ...config.engine,
                  prefix_cache: { ...prefixCache, block_size_tokens: blockSize },
                },
              })
            }
          />
        </div>
        <p>
          L1 容量由 KV token 预算换算：
          {Number.isInteger(prefixCache.block_size_tokens) && prefixCache.block_size_tokens > 0 &&
          Number.isInteger(config.engine.kv_token_budget) && config.engine.kv_token_budget >= 1
            ? ` ${Math.floor(config.engine.kv_token_budget / prefixCache.block_size_tokens)} blocks`
            : " —"}
          （向下取整，至少 1 block）
        </p>
      </section>

      <section className="form-section" aria-labelledby="performance-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Performance model</p>
            <h2 id="performance-heading">性能模型</h2>
          </div>
          <p>所有时间系数的单位均为毫秒</p>
        </div>

        <div className="form-grid">
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
        </div>
      </section>

      <div className="form-actions">
        <button className="run-button" type="submit" disabled={disabled}>
          {disabled ? "正在运行…" : "运行仿真"}
        </button>
      </div>
    </form>
  );
}
