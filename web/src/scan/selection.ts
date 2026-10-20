import type { JobSummary, ScanPoint } from "../api/types";
import { availableKeys, lineStyles, palette, symbols, valueOf,
  type Role, type Roles, type ScanData } from "./chart";

export const tokenOf = (value: unknown): string => JSON.stringify(value ?? null);
export const textOf = (value: unknown): string => value == null ? "未设置" : String(value);

export type SelectedValues = Record<string, string[]>;
export type BackgroundValues = Record<string, string>;

type Row = { job: JobSummary; point: ScanPoint; tokens: Record<string, string> };

export function resolveSelection(data: ScanData[], roles: Roles, metric: string,
  backgroundDraft: BackgroundValues = {}, selectedDraft: SelectedValues = {}, choosing = false) {
  const keys = availableKeys(data);
  const roleKeys = (Object.values(roles) as string[]).filter(Boolean);
  const hidden = keys.filter((key) => !roleKeys.includes(key));
  const rows: Row[] = data.flatMap(({ job, points }) => points.map((point) => ({ job, point,
    tokens: Object.fromEntries(keys.map((key) => [key, tokenOf(valueOf(point, key))])),
  })));
  const levels = Object.fromEntries(keys.map((key) => {
    const unique = new Map(rows.map((row) => [row.tokens[key], valueOf(row.point, key) ?? null]));
    return [key, [...unique].sort(([, a], [, b]) => a === null ? 1 : b === null ? -1
      : typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b)))];
  })) as Record<string, [string, unknown][]>;
  const goodRows = rows.filter(({ point }) => point.status === "completed"
    && typeof point.metrics?.[metric] === "number" && valueOf(point, roles.x) != null);
  const matchesBackground = (row: Row, values: BackgroundValues) => hidden.every((key) =>
    values[key] === undefined || row.tokens[key] === values[key]);
  let background = { ...backgroundDraft };
  if (choosing && goodRows.length && !goodRows.some((row) => matchesBackground(row, background))) background = {};
  const anchor = goodRows.find((row) => matchesBackground(row, background)) ?? goodRows[0];
  if (!choosing) background = anchor
    ? Object.fromEntries(hidden.map((key) => [key, anchor.tokens[key]])) : {};
  const complete = hidden.every((key) => background[key] !== undefined);
  const eligible = goodRows.filter((row) => matchesBackground(row, background));
  const limits: Record<Role, number> = { x: Infinity, color: palette.length,
    shape: symbols.length, line: lineStyles.length };
  const matchesRoles = (row: Row, selected: SelectedValues) => roleKeys.every((key) =>
    selected[key]?.includes(row.tokens[key]));
  let selected = { ...selectedDraft };
  if (!eligible.some((row) => matchesRoles(row, selected))) {
    selected = Object.fromEntries(roleKeys.map((key) => {
      const role = (Object.keys(roles) as Role[]).find((item) => roles[item] === key)!;
      const available = levels[key].map(([token]) => token).filter((token) =>
        eligible.some((row) => row.tokens[key] === token));
      const first = anchor?.tokens[key];
      return [key, [first, ...available.filter((token) => token !== first)]
        .filter((token): token is string => token !== undefined).slice(0, limits[role])];
    }));
  } else {
    for (const key of roleKeys) {
      selected[key] = selected[key].filter((token) => eligible.some((row) =>
        row.tokens[key] === token && roleKeys.every((other) =>
          other === key || selected[other].includes(row.tokens[other]))));
    }
  }
  const backgroundChoices = Object.fromEntries(hidden.map((key) => [key, levels[key].map(([token, value]) => ({
    token, value, checked: background[key] === token,
    disabled: !goodRows.some((row) => row.tokens[key] === token && matchesRoles(row, selected)
      && hidden.every((other) => other === key || background[other] === undefined
        || row.tokens[other] === background[other])),
  }))]));
  const roleChoices = Object.fromEntries((Object.keys(roles) as Role[]).map((role) => {
    const key = roles[role];
    return [role, key ? levels[key].map(([token, value]) => {
      const checked = selected[key]?.includes(token) ?? false;
      const supported = eligible.some((row) => row.tokens[key] === token && roleKeys.every((other) =>
        other === key || selected[other].includes(row.tokens[other])));
      const canRemove = selected[key].length > 1 && eligible.some((row) => row.tokens[key] !== token
        && matchesRoles(row, selected));
      return { token, value, checked, disabled: !complete || !supported || (checked && !canRemove)
        || (!checked && selected[key].length >= limits[role]) };
    }) : []];
  })) as Record<Role, { token: string; value: unknown; checked: boolean; disabled: boolean }[]>;
  const combinations = new Set(eligible.filter((row) => matchesRoles(row, selected))
    .map((row) => JSON.stringify(roleKeys.map((key) => row.tokens[key]))));
  const total = roleKeys.reduce((count, key) => count * selected[key].length, 1);
  const visible: ScanData[] = data.map(({ job, points }) => ({ job, points: points.filter((point) =>
    complete && hidden.every((key) => tokenOf(valueOf(point, key)) === background[key])
      && roleKeys.every((key) => selected[key].includes(tokenOf(valueOf(point, key))))),
  }));
  const notice = !eligible.length ? "当前批次与指标没有可绘制的数据点；无数据的选项已禁用。"
    : !complete ? "请为每项背景参数选择一个值；全部选齐后显示结果。"
      : `当前背景下 ${combinations.size} 组实际有数据，${total - combinations.size} 组勾选值组合无数据（不补算）。`;
  return { hidden, background, selected, backgroundChoices, roleChoices, complete, visible, notice };
}
