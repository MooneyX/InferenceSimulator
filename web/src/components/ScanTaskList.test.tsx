import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { getScanParameters, getScanTasks } from "../api/client";
import type { JobSummary, ScanTaskPage } from "../api/types";
import { ScanTaskList } from "./ScanTaskList";

vi.mock("../api/client", () => ({ getScanParameters: vi.fn(), getScanTasks: vi.fn() }));

const job: JobSummary = { id: "scan", kind: "scan", status: "running", total: 21,
  completed: 0, failed: 0, error: null, cache: null, lookup_ms: null,
  axes: [{ key: "num_instances", values: [1, 2] }], version: 1,
  code_fingerprint: "engine", code_version_status: "current", point_archive_version: 1 };

function page(index: number, completed: number): ScanTaskPage {
  return { total: 21, items: [{ index, values: { num_instances: index + 1 }, status: "running",
    error: null, result_available: false,
    request_progress: { total: 10, completed, aborted: 0, skipped: 0 } }] };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(getScanParameters).mockReset().mockResolvedValue([]);
  vi.mocked(getScanTasks).mockReset();
});
afterEach(() => { vi.useRealTimers(); });

it("coalesces frequent updates without discarding a slow page response", async () => {
  let resolve!: (value: ScanTaskPage) => void;
  vi.mocked(getScanTasks).mockImplementationOnce(() => new Promise((done) => { resolve = done; }))
    .mockResolvedValue(page(0, 8));
  const { rerender } = render(<ScanTaskList job={job} onSelect={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "展开组合" }));
  for (let version = 2; version <= 12; version++) {
    rerender(<ScanTaskList job={{ ...job, version }} onSelect={vi.fn()} />);
  }
  expect(getScanTasks).toHaveBeenCalledTimes(1);
  await act(async () => { resolve(page(0, 3)); });
  expect(screen.getByText("已处理 3 / 10 请求")).toBeInTheDocument();
  await act(async () => { await vi.advanceTimersByTimeAsync(999); });
  expect(getScanTasks).toHaveBeenCalledTimes(1);
  await act(async () => { await vi.advanceTimersByTimeAsync(1); });
  expect(getScanTasks).toHaveBeenCalledTimes(2);
  expect(screen.getByText("已处理 8 / 10 请求")).toBeInTheDocument();
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(getScanTasks).toHaveBeenCalledTimes(2);
});

it("ignores an old page response after pagination", async () => {
  let resolve!: (value: ScanTaskPage) => void;
  vi.mocked(getScanTasks).mockResolvedValueOnce(page(0, 1))
    .mockImplementationOnce(() => new Promise((done) => { resolve = done; }))
    .mockResolvedValue(page(20, 2));
  const { rerender } = render(<ScanTaskList job={job} onSelect={vi.fn()} />);
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "展开组合" })); });
  rerender(<ScanTaskList job={{ ...job, version: 2 }} onSelect={vi.fn()} />);
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "下一页" })); });
  expect(getScanTasks).toHaveBeenLastCalledWith("scan", 20, 20);
  expect(screen.getByText("#21")).toBeInTheDocument();
  await act(async () => { resolve(page(0, 9)); });
  expect(screen.queryByText("#1")).not.toBeInTheDocument();
  expect(screen.getByText("已处理 2 / 10 请求")).toBeInTheDocument();
});

it("stops scheduled refreshes while collapsed", async () => {
  vi.mocked(getScanTasks).mockResolvedValue(page(0, 1));
  const { rerender } = render(<ScanTaskList job={job} onSelect={vi.fn()} />);
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "展开组合" })); });
  rerender(<ScanTaskList job={{ ...job, version: 2 }} onSelect={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "收起组合" }));
  rerender(<ScanTaskList job={{ ...job, version: 3 }} onSelect={vi.fn()} />);
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(getScanTasks).toHaveBeenCalledTimes(1);
});
