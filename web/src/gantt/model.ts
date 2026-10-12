import type { RequestRecord } from "../api/types";

export const GANTT_PHASES = ["queue", "prefill", "decode"] as const;

export type GanttPhase = (typeof GANTT_PHASES)[number];

export interface GanttSegment {
  requestId: number;
  rowIndex: number;
  phase: GanttPhase;
  start: number;
  end: number;
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

function createSegment(
  row: GanttRequestRow,
  phase: GanttPhase,
  start: number,
  end: number,
): GanttSegment {
  return {
    requestId: row.id,
    rowIndex: row.rowIndex,
    phase,
    start,
    end,
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
