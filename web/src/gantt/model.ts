import type { RequestRecord } from "../api/types";

export const GANTT_PHASES = ["queue", "prefill", "decode"] as const;

export type GanttPhase = (typeof GANTT_PHASES)[number];

export interface GanttSegment {
  key: string;
  requestId: number;
  rowIndex: number;
  phase: GanttPhase;
  start: number;
  end: number;
  duration: number;
}

export interface GanttRequestRow extends RequestRecord {
  rowIndex: number;
  ttft: number;
  decodeDuration: number;
}

export interface GanttModel {
  rows: GanttRequestRow[];
  segments: GanttSegment[];
  startTime: number | null;
  endTime: number | null;
}

export interface GanttViewport {
  startTime: number;
  endTime: number;
  firstRow: number;
  lastRow: number;
}

export interface GanttOverscan {
  rows?: number;
  timeRatio?: number;
}

const DEFAULT_ROW_OVERSCAN = 2;
const DEFAULT_TIME_OVERSCAN_RATIO = 0.1;

function createSegment(
  row: GanttRequestRow,
  phase: GanttPhase,
  start: number,
  end: number,
): GanttSegment {
  return {
    key: `${row.id}-${phase}`,
    requestId: row.id,
    rowIndex: row.rowIndex,
    phase,
    start,
    end,
    duration: end - start,
  };
}

export function buildGanttModel(requests: readonly RequestRecord[]): GanttModel {
  const sortedRequests = [...requests].sort(
    (left, right) => left.arrival - right.arrival || left.id - right.id,
  );
  const rows = sortedRequests.map<GanttRequestRow>((request, rowIndex) => ({
    ...request,
    rowIndex,
    ttft: request.prefill_end - request.arrival,
    decodeDuration: request.finish - request.prefill_end,
  }));
  const segments = rows.flatMap((row) => [
    createSegment(row, "queue", row.arrival, row.prefill_start),
    createSegment(row, "prefill", row.prefill_start, row.prefill_end),
    createSegment(row, "decode", row.prefill_end, row.finish),
  ]);

  if (rows.length === 0) {
    return { rows, segments, startTime: null, endTime: null };
  }

  return {
    rows,
    segments,
    startTime: rows[0].arrival,
    endTime: rows.reduce(
      (latestFinish, { finish }) => Math.max(latestFinish, finish),
      rows[0].finish,
    ),
  };
}

export function selectVisibleSegments(
  model: GanttModel,
  viewport: GanttViewport,
  overscan: GanttOverscan = {},
): GanttSegment[] {
  const rowOverscan = Math.max(0, overscan.rows ?? DEFAULT_ROW_OVERSCAN);
  const timeRatio = Math.max(0, overscan.timeRatio ?? DEFAULT_TIME_OVERSCAN_RATIO);
  const visibleDuration = Math.max(0, viewport.endTime - viewport.startTime);
  const timeOverscan = visibleDuration * timeRatio;
  const firstRow = Math.floor(viewport.firstRow - rowOverscan);
  const lastRow = Math.ceil(viewport.lastRow + rowOverscan);
  const startTime = viewport.startTime - timeOverscan;
  const endTime = viewport.endTime + timeOverscan;

  return model.segments.filter(
    (segment) =>
      segment.rowIndex >= firstRow &&
      segment.rowIndex <= lastRow &&
      segment.end >= startTime &&
      segment.start <= endTime,
  );
}
