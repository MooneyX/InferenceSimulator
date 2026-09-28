import type { ChangeEvent } from "react";

import type { LengthDistribution } from "../api/types";
import "./LengthDistributionEditor.css";

type DistributionKind = LengthDistribution["dist"];
type DistributionParameter = "value" | "mean" | "sigma";

export interface LengthDistributionEditorProps {
  name: "prompt_len" | "output_len";
  label: string;
  value: LengthDistribution;
  onChange: (value: LengthDistribution) => void;
  errors?: Partial<Record<DistributionParameter, string>>;
}

function representativeLength(distribution: LengthDistribution): number {
  const value = distribution.dist === "fixed" ? distribution.value : distribution.mean;
  return Number.isFinite(value) && value > 0 ? value : 1;
}

function changeDistributionKind(
  distribution: LengthDistribution,
  kind: DistributionKind,
): LengthDistribution {
  const length = representativeLength(distribution);

  switch (kind) {
    case "fixed":
      return { dist: "fixed", value: Math.max(1, Math.round(length)) };
    case "poisson":
      return { dist: "poisson", mean: length };
    case "lognormal":
      return {
        dist: "lognormal",
        mean: length,
        sigma: distribution.dist === "lognormal" ? distribution.sigma : 1,
      };
  }
}

function inputValue(value: number): number | "" {
  return Number.isNaN(value) ? "" : value;
}

interface NumberFieldProps {
  id: string;
  label: string;
  name: string;
  value: number;
  integer?: boolean;
  error?: string;
  onChange: (value: number) => void;
}

function NumberField({
  id,
  label,
  name,
  value,
  integer = false,
  error,
  onChange,
}: NumberFieldProps) {
  const errorId = `${id}-error`;

  return (
    <div className="length-field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        name={name}
        type="number"
        min={integer ? 1 : 0}
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

export function LengthDistributionEditor({
  name,
  label,
  value,
  onChange,
  errors = {},
}: LengthDistributionEditorProps) {
  const idPrefix = `length-${name}`;

  function handleKindChange(event: ChangeEvent<HTMLSelectElement>): void {
    onChange(changeDistributionKind(value, event.currentTarget.value as DistributionKind));
  }

  return (
    <fieldset className="length-distribution">
      <legend>{label}</legend>

      <div className="length-field">
        <label htmlFor={`${idPrefix}-dist`}>分布类型</label>
        <select
          id={`${idPrefix}-dist`}
          name={`${name}.dist`}
          value={value.dist}
          onChange={handleKindChange}
        >
          <option value="fixed">固定长度</option>
          <option value="poisson">泊松分布</option>
          <option value="lognormal">对数正态分布</option>
        </select>
      </div>

      {value.dist === "fixed" ? (
        <NumberField
          id={`${idPrefix}-value`}
          name={`${name}.value`}
          label="固定长度（token）"
          value={value.value}
          integer
          error={errors.value}
          onChange={(nextValue) => onChange({ dist: "fixed", value: nextValue })}
        />
      ) : (
        <>
          <NumberField
            id={`${idPrefix}-mean`}
            name={`${name}.mean`}
            label="平均长度（token）"
            value={value.mean}
            error={errors.mean}
            onChange={(nextValue) => onChange({ ...value, mean: nextValue })}
          />
          {value.dist === "lognormal" ? (
            <NumberField
              id={`${idPrefix}-sigma`}
              name={`${name}.sigma`}
              label="形状参数（sigma）"
              value={value.sigma}
              error={errors.sigma}
              onChange={(nextValue) => onChange({ ...value, sigma: nextValue })}
            />
          ) : null}
        </>
      )}
    </fieldset>
  );
}
