import type { InferenceConfig } from "../api/types";

type Model = InferenceConfig["model"];
export type LayerSegment = Model["layers"][number];

/** 层序合计的模型总层数。 */
export function layerCount(model: Pick<Model, "layers">): number {
  return model.layers.reduce((total, segment) => total + segment.count, 0);
}

/** 层序中 MoE 层的数量。 */
export function moeLayerCount(model: Pick<Model, "layers">): number {
  return model.layers.reduce((total, segment) => total + (segment.ffn === "moe" ? segment.count : 0), 0);
}

export function layerSegment(count: number, patch: Partial<LayerSegment> = {}): LayerSegment {
  return { count, attention_group: 0, ffn: "dense", full_indexer: false, ...patch };
}
