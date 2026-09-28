import { describe, expect, it } from "vitest";

import type { RequestRecord } from "../api/types";
import { buildGanttModel, selectVisibleSegments } from "./model";

function request(
  id: number,
  arrival: number,
  prefillStart: number,
  prefillEnd: number,
  finish: number,
): RequestRecord {
  return {
    id,
    arrival,
    prefill_start: prefillStart,
    prefill_end: prefillEnd,
    finish,
    prompt_len: 16,
    output_len: 8,
  };
}

describe("buildGanttModel", () => {
  it("sorts requests by arrival and then by ID without mutating the input", () => {
    const requests = [request(9, 5, 6, 7, 10), request(2, 1, 2, 4, 8), request(1, 1, 1, 3, 6)];
    const originalOrder = requests.map(({ id }) => id);

    const model = buildGanttModel(requests);

    expect(model.rows.map(({ id }) => id)).toEqual([1, 2, 9]);
    expect(requests.map(({ id }) => id)).toEqual(originalOrder);
    expect(model.startTime).toBe(1);
    expect(model.endTime).toBe(10);
  });

  it("creates queue, prefill and one merged decode segment per request", () => {
    const model = buildGanttModel([request(4, 2, 5, 11, 19)]);

    expect(model.rows[0]).toMatchObject({
      id: 4,
      rowIndex: 0,
      ttft: 9,
      decodeDuration: 8,
    });
    expect(model.segments).toEqual([
      {
        key: "4-queue",
        requestId: 4,
        rowIndex: 0,
        phase: "queue",
        start: 2,
        end: 5,
        duration: 3,
      },
      {
        key: "4-prefill",
        requestId: 4,
        rowIndex: 0,
        phase: "prefill",
        start: 5,
        end: 11,
        duration: 6,
      },
      {
        key: "4-decode",
        requestId: 4,
        rowIndex: 0,
        phase: "decode",
        start: 11,
        end: 19,
        duration: 8,
      },
    ]);
  });

  it("keeps zero-duration lifecycle phases without producing negative intervals", () => {
    const model = buildGanttModel([request(0, 3, 3, 5, 5)]);

    expect(model.segments.map(({ duration }) => duration)).toEqual([0, 2, 0]);
    expect(model.segments.every(({ duration }) => duration >= 0)).toBe(true);
  });

  it("returns nullable time bounds for an empty result", () => {
    expect(buildGanttModel([])).toEqual({
      rows: [],
      segments: [],
      startTime: null,
      endTime: null,
    });
  });

  it("creates exactly three intervals for each of 1,000 requests", () => {
    const requests = Array.from({ length: 1000 }, (_, id) =>
      request(id, id * 2, id * 2 + 1, id * 2 + 2, id * 2 + 4),
    );

    const model = buildGanttModel(requests);

    expect(model.rows).toHaveLength(1000);
    expect(model.segments).toHaveLength(3000);
  });
});

describe("selectVisibleSegments", () => {
  it("keeps only intervals near the visible row and time range", () => {
    const model = buildGanttModel([
      request(0, 0, 2, 4, 8),
      request(1, 10, 12, 14, 18),
      request(2, 20, 22, 24, 28),
      request(3, 30, 32, 34, 38),
    ]);

    const visible = selectVisibleSegments(
      model,
      { startTime: 11, endTime: 16, firstRow: 1, lastRow: 1 },
      { rows: 0, timeRatio: 0 },
    );

    expect(visible.map(({ key }) => key)).toEqual([
      "1-queue",
      "1-prefill",
      "1-decode",
    ]);
  });

  it("adds row and time overscan around the viewport", () => {
    const model = buildGanttModel([
      request(0, 0, 2, 4, 8),
      request(1, 10, 12, 14, 18),
      request(2, 20, 22, 24, 28),
    ]);

    const visible = selectVisibleSegments(
      model,
      { startTime: 10, endTime: 20, firstRow: 1, lastRow: 1 },
      { rows: 1, timeRatio: 0.2 },
    );

    expect(visible.map(({ key }) => key)).toEqual([
      "0-decode",
      "1-queue",
      "1-prefill",
      "1-decode",
      "2-queue",
      "2-prefill",
    ]);
  });
});
