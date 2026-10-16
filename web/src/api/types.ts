import type { components, operations } from "./schema";

type SimulateOperation = operations["simulate_endpoint_api_simulate_post"];

export type SimConfig = SimulateOperation["requestBody"]["content"]["application/json"];
export type SimResult = SimulateOperation["responses"][200]["content"]["application/json"];
export type HTTPValidationError = SimulateOperation["responses"][422]["content"]["application/json"];
export type ErrorResponse = SimulateOperation["responses"][501]["content"]["application/json"];
export type ValidationError = components["schemas"]["ValidationError"];
export type RequestRecord = components["schemas"]["RequestRecord"];
export type LengthDistribution = components["schemas"]["SyntheticWorkloadConfig"]["prompt_len"];
export type LengthDist = LengthDistribution;
export type ResolutionReport = components["schemas"]["ResolutionReport"];
export type WorkloadConfig = SimConfig["workload"];
export type SyntheticWorkloadConfig = NonNullable<WorkloadConfig["synthetic"]>;
export type TraceWorkloadConfig = NonNullable<WorkloadConfig["trace"]>;
export type SchedulerConfig = SimConfig["scheduler"];
export type InferenceConfig = SimConfig["inference"];
export type TimingConfig = NonNullable<InferenceConfig["timing"]>;
export type EmpiricalTimingConfig = Extract<TimingConfig, { mode: "empirical" }>;
export type AnalyticalTimingConfig = Extract<TimingConfig, { mode: "analytical" }>;
export type PresetCatalog = components["schemas"]["PresetCatalog"];
export type TraceSessionSummary = components["schemas"]["SessionSummary"];
export type TraceJob = components["schemas"]["TraceJob"];
export type PlanReceipt = components["schemas"]["PlanReceipt"];
