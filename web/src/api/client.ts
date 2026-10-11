import type {
  ErrorResponse,
  HTTPValidationError,
  PresetCatalog,
  ResolvedConfig,
  ResolvedRequest,
  ResolvedRun,
  SimConfig,
  SimResult,
  ValidationError,
  PlanReceipt,
  TraceJob,
  TraceSessionSummary,
} from "./types";

export type { PlanReceipt, TraceJob, TraceSessionSummary } from "./types";

const SIMULATE_ENDPOINT = "/api/simulate";

export type SimulationApiErrorKind =
  | "validation"
  | "not_implemented"
  | "http"
  | "network"
  | "invalid_response";

interface SimulationApiErrorOptions {
  kind: SimulationApiErrorKind;
  status?: number;
  validationErrors?: ValidationError[];
  cause?: unknown;
}

export class SimulationApiError extends Error {
  readonly kind: SimulationApiErrorKind;
  readonly status?: number;
  readonly validationErrors: ValidationError[];

  constructor(message: string, options: SimulationApiErrorOptions) {
    super(message, { cause: options.cause });
    this.name = "SimulationApiError";
    this.kind = options.kind;
    this.status = options.status;
    this.validationErrors = options.validationErrors ?? [];
  }
}

export interface SimulateOptions {
  signal?: AbortSignal;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return undefined;
  }
}

function getErrorDetail(payload: unknown): string | undefined {
  if (
    typeof payload === "object" &&
    payload !== null &&
    "detail" in payload &&
    typeof payload.detail === "string"
  ) {
    return payload.detail;
  }

  return undefined;
}

export async function simulate(
  config: SimConfig,
  options: SimulateOptions = {},
): Promise<SimResult> {
  let response: Response;

  try {
    response = await fetch(SIMULATE_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
      signal: options.signal,
    });
  } catch (cause) {
    throw new SimulationApiError("无法连接仿真服务", {
      kind: "network",
      cause,
    });
  }

  const payload = await readJson(response);

  if (response.ok) {
    if (payload === undefined) {
      throw new SimulationApiError("仿真服务返回了无效的 JSON 响应", {
        kind: "invalid_response",
        status: response.status,
      });
    }

    return payload as SimResult;
  }

  if (response.status === 422) {
    const validationResponse = payload as HTTPValidationError | undefined;
    const detail = validationResponse?.detail;
    throw new SimulationApiError(getErrorDetail(payload) ?? "仿真配置校验失败", {
      kind: "validation",
      status: response.status,
      validationErrors: Array.isArray(detail) ? detail : [],
    });
  }

  if (response.status === 501) {
    const errorResponse = payload as ErrorResponse | undefined;
    throw new SimulationApiError(
      errorResponse?.detail ?? "服务端尚未实现此仿真模式",
      {
        kind: "not_implemented",
        status: response.status,
      },
    );
  }

  throw new SimulationApiError(
    getErrorDetail(payload) ?? `仿真请求失败（HTTP ${response.status}）`,
    {
      kind: "http",
      status: response.status,
    },
  );
}

async function requestAuto<T>(endpoint: string, body?: ResolvedRequest): Promise<T> {
  let response: Response;
  try {
    response = await fetch(endpoint, body ? {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    } : undefined);
  } catch (cause) {
    throw new SimulationApiError("无法连接仿真服务", { kind: "network", cause });
  }
  const payload = await readJson(response);
  if (response.ok && payload !== undefined) return payload as T;
  if (response.status === 422) {
    const detail = (payload as HTTPValidationError | undefined)?.detail;
    throw new SimulationApiError(getErrorDetail(payload) ?? "模型或硬件规格校验失败", {
      kind: "validation", status: 422, validationErrors: Array.isArray(detail) ? detail : [],
    });
  }
  if (response.status === 501) {
    throw new SimulationApiError(getErrorDetail(payload) ?? "此解析运行模式尚未实现", {
      kind: "not_implemented", status: 501,
    });
  }
  throw new SimulationApiError(getErrorDetail(payload) ?? `请求失败（HTTP ${response.status}）`, {
    kind: response.ok ? "invalid_response" : "http", status: response.status,
  });
}

export function getPresets(): Promise<PresetCatalog> {
  return requestAuto<PresetCatalog>("/api/presets");
}

export function resolveAutomatic(request: ResolvedRequest): Promise<ResolvedConfig> {
  return requestAuto<ResolvedConfig>("/api/resolve", request);
}

export function simulateAutomatic(request: ResolvedRequest): Promise<ResolvedRun> {
  return requestAuto<ResolvedRun>("/api/simulate/resolved", request);
}

async function replayRequest(endpoint: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(endpoint, init);
  } catch (cause) {
    throw new SimulationApiError("无法连接轨迹处理服务", { kind: "network", cause });
  }
  if (!response.ok) {
    const detail = getErrorDetail(await readJson(response));
    throw new SimulationApiError(detail ?? `轨迹请求失败（HTTP ${response.status}）`, {
      kind: response.status === 422 ? "validation"
        : response.status === 501 ? "not_implemented" : "http", status: response.status,
    });
  }
  return response;
}

export async function uploadTrace(file: File): Promise<TraceJob> {
  return (await replayRequest("/api/traces", { method: "POST", body: file })).json() as Promise<TraceJob>;
}

export async function traceStatus(id: string): Promise<TraceJob> {
  return (await replayRequest(`/api/traces/${id}`)).json() as Promise<TraceJob>;
}

export async function traceSessions(id: string, offset: number): Promise<{
  total: number; sessions: TraceSessionSummary[];
}> {
  return (await replayRequest(`/api/traces/${id}/sessions?offset=${offset}&limit=50`)).json();
}

export async function downloadTracePlan(id: string, sessions: string[]): Promise<Blob> {
  const response = await replayRequest(`/api/traces/${id}/plans`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sessions }),
  });
  return response.blob();
}

export async function uploadReplayPlan(file: Blob): Promise<PlanReceipt> {
  return (await replayRequest("/api/replay-plans", { method: "POST", body: file })).json() as Promise<PlanReceipt>;
}

export async function simulateReplayPlan(
  id: string, config: SimConfig, mode: "open_loop" | "session", sessionRate: number, seed: number,
): Promise<SimResult> {
  const response = await replayRequest(`/api/replay-plans/${id}/simulate`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config, mode, session_rate: sessionRate, seed }),
  });
  return response.json() as Promise<SimResult>;
}

export async function simulateReplayPlanAutomatic(
  id: string, request: ResolvedRequest, mode: "open_loop" | "session", sessionRate: number, seed: number,
): Promise<ResolvedRun> {
  const response = await replayRequest(`/api/replay-plans/${id}/simulate/resolved`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request, mode, session_rate: sessionRate, seed }),
  });
  return response.json() as Promise<ResolvedRun>;
}
