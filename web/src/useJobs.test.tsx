import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";

import { getJob, getJobResult, listJobs, simulate } from "./api/client";
import { createDefaultSimConfig } from "./config/simulation";
import { useJobs } from "./useJobs";

vi.mock("./api/client", () => ({
  getJob: vi.fn(), getJobResult: vi.fn(), listJobs: vi.fn(), simulate: vi.fn(),
  submitScan: vi.fn(), cancelJob: vi.fn(),
}));

class FakeEvents {
  static current: FakeEvents;
  handlers = new Map<string, (event: MessageEvent) => void>();
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  close = vi.fn();
  constructor(public readonly url: string) { FakeEvents.current = this; }
  addEventListener(name: string, callback: (event: MessageEvent) => void) {
    this.handlers.set(name, callback);
  }
  emit(name: string, payload: object) {
    this.handlers.get(name)?.({ data: JSON.stringify(payload) } as MessageEvent);
  }
}

const queued = { id: "job-1", kind: "single" as const, status: "queued" as const,
  total: 1, completed: 0, failed: 0, error: null, cache: null, lookup_ms: null,
  axes: null, version: 1, code_fingerprint: "engine-1", code_version_status: "current" as const,
  point_archive_version: 0 };
const finished = { ...queued, status: "completed" as const, completed: 1, version: 3 };

beforeEach(() => {
  vi.stubGlobal("EventSource", FakeEvents);
  vi.mocked(listJobs).mockReset().mockResolvedValue([]);
  vi.mocked(simulate).mockReset().mockResolvedValue({ id: "job-1" });
  vi.mocked(getJob).mockReset().mockResolvedValue(queued);
  vi.mocked(getJobResult).mockReset().mockResolvedValue({
    result: { meta: { num_requests: 1 } } as Awaited<ReturnType<typeof getJobResult>>["result"],
    cache: { status: "miss", lookupMs: 0 },
  });
});

it("restores completed jobs after refresh without displaying their results until selected", async () => {
  vi.mocked(listJobs).mockResolvedValue([finished]);
  const onResult = vi.fn();
  const { result, unmount } = renderHook(() => useJobs(onResult, vi.fn()));
  await waitFor(() => expect(result.current.jobs).toHaveLength(1));
  act(() => { FakeEvents.current.emit("snapshot", [finished]); });
  expect(result.current.selectedId).toBeNull();
  expect(getJobResult).not.toHaveBeenCalled();
  expect(onResult).not.toHaveBeenCalled();

  act(() => { result.current.select(finished.id); });
  await waitFor(() => expect(onResult).toHaveBeenCalledTimes(1));
  expect(result.current.selectedId).toBe(finished.id);
  unmount();
});

it("updates request progress without reordering jobs or accepting stale progress", async () => {
  const running = { ...queued, status: "running" as const,
    request_progress: { total: 10, completed: 0, aborted: 0, skipped: 0 } };
  vi.mocked(listJobs).mockResolvedValue([{ ...queued, id: "newer" }, running]);
  const { result } = renderHook(() => useJobs(vi.fn(), vi.fn()));
  await waitFor(() => expect(result.current.jobs).toHaveLength(2));
  const updated = { ...running, version: 2,
    request_progress: { total: 10, completed: 5, aborted: 1, skipped: 1 } };
  act(() => { FakeEvents.current.emit("update", updated); });
  expect(result.current.jobs.map((job) => job.id)).toEqual(["newer", "job-1"]);
  expect(result.current.jobs[1].request_progress).toEqual(updated.request_progress);
  act(() => { FakeEvents.current.emit("update", running); });
  act(() => { FakeEvents.current.emit("snapshot", [{ ...queued, id: "newer" }, running]); });
  expect(result.current.jobs[1].request_progress).toEqual(updated.request_progress);
  act(() => { FakeEvents.current.emit("update", { ...finished, request_progress: {
    total: 10, completed: 8, aborted: 1, skipped: 1,
  } }); });
  act(() => { FakeEvents.current.emit("update", updated); });
  expect(result.current.jobs[1].status).toBe("completed");
  expect(result.current.jobs[1].request_progress?.completed).toBe(8);
});

it("receives SSE completion, loads results and ignores an older status query", async () => {
  const onResult = vi.fn();
  const onError = vi.fn();
  const { result, unmount } = renderHook(() => useJobs(onResult, onError));
  await waitFor(() => expect(listJobs).toHaveBeenCalled());
  expect(FakeEvents.current.url).toBe("/api/jobs/events");
  await act(async () => { await result.current.submitSingle(createDefaultSimConfig(), false); });
  act(() => { FakeEvents.current.emit("update", finished); });
  await waitFor(() => expect(onResult).toHaveBeenCalledTimes(1));
  act(() => { FakeEvents.current.emit("update", queued); });
  expect(result.current.jobs[0].status).toBe("completed");
  expect(onError).not.toHaveBeenCalled();
  unmount();
  expect(FakeEvents.current.close).toHaveBeenCalled();
});
