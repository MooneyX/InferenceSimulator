import type {
  ErrorResponse,
  HTTPValidationError,
  PresetCatalog,
  ResolutionReport,
  SimConfig,
  SimResult,
  ValidationError,
  PlanReceipt,
  TraceJob,
  TraceSessionSummary,
} from "./types";

export type { PlanReceipt, TraceJob, TraceSessionSummary } from "./types";

const SIMULATE_ENDPOINT = "/api/simulate";

interface SimulationApiErrorOptions {
  validationErrors?: ValidationError[];
  cause?: unknown;
}

export class SimulationApiError extends Error {
  readonly validationErrors: ValidationError[];

  constructor(message: string, options: SimulationApiErrorOptions = {}) {
    super(message, { cause: options.cause });
    this.name = "SimulationApiError";
    this.validationErrors = options.validationErrors ?? [];
  }
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

function validationErrors(detail: unknown): ValidationError[] {
  return Array.isArray(detail) ? detail.filter((item): item is ValidationError =>
    typeof item === "object" && item !== null && Array.isArray(item.loc) &&
    typeof item.msg === "string" && typeof item.type === "string") : [];
}

export async function simulate(config: SimConfig): Promise<SimResult> {
  let response: Response;

  try {
    response = await fetch(SIMULATE_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
    });
  } catch (cause) {
    throw new SimulationApiError("无法连接仿真服务", { cause });
  }

  const payload = await readJson(response);

  if (response.ok) {
    if (payload === undefined) {
      throw new SimulationApiError("仿真服务返回了无效的 JSON 响应");
    }

    return payload as SimResult;
  }

  if (response.status === 422) {
    const validationResponse = payload as HTTPValidationError | undefined;
    const detail = validationResponse?.detail;
    throw new SimulationApiError(getErrorDetail(payload) ?? "仿真配置校验失败", {
      validationErrors: validationErrors(detail),
    });
  }

  if (response.status === 501) {
    const errorResponse = payload as ErrorResponse | undefined;
    throw new SimulationApiError(
      errorResponse?.detail ?? "服务端尚未实现此仿真模式",
    );
  }

  throw new SimulationApiError(
    getErrorDetail(payload) ?? `仿真请求失败（HTTP ${response.status}）`,
  );
}

async function requestConfig<T>(endpoint: string, body?: SimConfig): Promise<T> {
  let response: Response;
  try {
    response = await fetch(endpoint, body ? {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    } : undefined);
  } catch (cause) {
    throw new SimulationApiError("无法连接仿真服务", { cause });
  }
  const payload = await readJson(response);
  if (response.ok) {
    if (payload === undefined) throw new SimulationApiError("仿真服务返回了无效的 JSON 响应");
    return payload as T;
  }
  if (response.status === 422) {
    const detail = (payload as HTTPValidationError | undefined)?.detail;
    throw new SimulationApiError(getErrorDetail(payload) ?? "模型或硬件规格校验失败", {
      validationErrors: validationErrors(detail),
    });
  }
  if (response.status === 501) {
    throw new SimulationApiError(getErrorDetail(payload) ?? "此解析运行模式尚未实现");
  }
  throw new SimulationApiError(getErrorDetail(payload) ?? `请求失败（HTTP ${response.status}）`);
}

export function getPresets(): Promise<PresetCatalog> {
  return requestConfig<PresetCatalog>("/api/presets");
}

export function resolveConfig(config: SimConfig): Promise<ResolutionReport> {
  return requestConfig<ResolutionReport>("/api/resolve", config);
}

async function replayRequest(endpoint: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(endpoint, init);
  } catch (cause) {
    throw new SimulationApiError("无法连接轨迹处理服务", { cause });
  }
  if (!response.ok) {
    const detail = getErrorDetail(await readJson(response));
    throw new SimulationApiError(detail ?? `轨迹请求失败（HTTP ${response.status}）`);
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
