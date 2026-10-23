import type {
  HTTPValidationError,
  PresetCatalog,
  ResolutionReport,
  SimConfig,
  SimResult,
  JobReceipt,
  JobSummary,
  ScanRequest,
  ScanPoint,
  ScanTaskPage,
  ScanParameter,
  ValidationError,
  PlanReceipt,
  TraceJob,
  TraceSessionPage,
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

export interface SimulationCacheInfo {
  status: "hit" | "miss" | "shared";
  lookupMs: number;
}

export interface SimulationResponse {
  result: SimResult;
  cache: SimulationCacheInfo;
}

export async function simulate(config: SimConfig, force = false): Promise<JobReceipt> {
  let response: Response;

  try {
    response = await fetch(force ? `${SIMULATE_ENDPOINT}?force=true` : SIMULATE_ENDPOINT, {
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

    return payload as JobReceipt;
  }

  if (response.status === 422) {
    const validationResponse = payload as HTTPValidationError | undefined;
    const detail = validationResponse?.detail;
    throw new SimulationApiError(getErrorDetail(payload) ?? "仿真配置校验失败", {
      validationErrors: validationErrors(detail),
    });
  }

  throw new SimulationApiError(
    getErrorDetail(payload) ?? `仿真请求失败（HTTP ${response.status}）`,
  );
}

export function submitScan(request: ScanRequest): Promise<JobReceipt> {
  return requestConfig<JobReceipt>("/api/scans", request);
}

export function listJobs(): Promise<JobSummary[]> {
  return requestConfig<JobSummary[]>("/api/jobs");
}

export function getJob(id: string): Promise<JobSummary> {
  return requestConfig<JobSummary>(`/api/jobs/${encodeURIComponent(id)}`);
}

export function getScanParameters(): Promise<ScanParameter[]> {
  return requestConfig<ScanParameter[]>("/api/scans/parameters");
}

export function getScanPoints(id: string): Promise<ScanPoint[]> {
  return requestConfig<ScanPoint[]>(`/api/jobs/${encodeURIComponent(id)}/points`);
}

export function getScanTasks(id: string, offset: number, limit: number): Promise<ScanTaskPage> {
  return requestConfig<ScanTaskPage>(`/api/jobs/${encodeURIComponent(id)}/tasks?offset=${offset}&limit=${limit}`);
}

export async function getScanPointResult(id: string, index: number): Promise<SimResult> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(id)}/points/${index}/result`);
  if (!response.ok) throw new SimulationApiError(getErrorDetail(await readJson(response)) ?? "获取扫描点结果失败");
  const result = await readJson(response);
  if (result === undefined) throw new SimulationApiError("仿真服务返回了无效的 JSON 响应");
  return result as SimResult;
}

export async function cancelJob(id: string): Promise<JobSummary> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) throw new SimulationApiError(getErrorDetail(await readJson(response)) ?? "取消任务失败");
  return response.json() as Promise<JobSummary>;
}

export async function getJobResult(id: string): Promise<SimulationResponse> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(id)}/result`);
  if (!response.ok) throw new SimulationApiError(getErrorDetail(await readJson(response)) ?? "获取结果失败");
  const result = await readJson(response);
  if (result === undefined) throw new SimulationApiError("仿真服务返回了无效的 JSON 响应");
  const status = response.headers.get("X-Simulation-Cache");
  return { result: result as SimResult, cache: {
    status: status === "hit" || status === "shared" ? status : "miss",
    lookupMs: Number(response.headers.get("X-Simulation-Cache-Lookup-Ms") ?? 0),
  } };
}

export async function clearSimulationCache(): Promise<void> {
  let response: Response;
  try {
    response = await fetch("/api/simulation-cache", { method: "DELETE" });
  } catch (cause) {
    throw new SimulationApiError("无法连接仿真服务", { cause });
  }
  if (!response.ok) {
    throw new SimulationApiError(
      getErrorDetail(await readJson(response)) ?? `清空结果缓存失败（HTTP ${response.status}）`,
    );
  }
}

async function requestConfig<T>(endpoint: string, body?: SimConfig | ScanRequest): Promise<T> {
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

export async function traceSessions(id: string, offset: number): Promise<TraceSessionPage> {
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
