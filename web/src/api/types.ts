import type { components, operations } from "./schema";

type SimulateOperation = operations["simulate_endpoint_api_simulate_post"];

export type SimConfig = SimulateOperation["requestBody"]["content"]["application/json"];
export type SimResult = SimulateOperation["responses"][200]["content"]["application/json"];
export type HTTPValidationError = SimulateOperation["responses"][422]["content"]["application/json"];
export type ErrorResponse = SimulateOperation["responses"][501]["content"]["application/json"];
export type ValidationError = components["schemas"]["ValidationError"];
export type RequestRecord = components["schemas"]["RequestRecord"];
export type LengthDistribution = components["schemas"]["SyntheticWorkloadConfig"]["prompt_len"];
export type ResolvedRequest = components["schemas"]["ResolvedRequest"];
export type ResolvedConfig = components["schemas"]["ResolvedConfig"];
export type ResolvedRun = components["schemas"]["ResolvedRun"];
export type PresetCatalog = components["schemas"]["PresetCatalog"];
export type TraceSessionSummary = components["schemas"]["SessionSummary"];
export type TraceJob = components["schemas"]["TraceJob"];
export type PlanReceipt = components["schemas"]["PlanReceipt"];
