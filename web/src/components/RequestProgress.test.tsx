import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";

import { RequestProgress } from "./RequestProgress";

it("counts success, failure and skipped requests, rather than successes only", () => {
  render(<RequestProgress status="running" progress={{ total: 1000, completed: 700, aborted: 10, skipped: 10 }} />);
  expect(screen.getByText("已处理 720 / 1,000 请求")).toBeInTheDocument();
  expect(screen.getByText("72%")).toBeInTheDocument();
  expect(screen.getByText("成功 700 · 失败 10 · 跳过 10")).toBeInTheDocument();
  expect(screen.getByText("处理中…")).toBeInTheDocument();
  expect(screen.getByRole("progressbar")).toHaveAttribute("value", "720");
});

it("keeps the task running during result summarization", () => {
  const { rerender } = render(<RequestProgress status="running"
    progress={{ total: 3, completed: 0, aborted: 1, skipped: 2 }} />);
  expect(screen.getByText("100%")).toBeInTheDocument();
  expect(screen.getByText("正在汇总结果…")).toBeInTheDocument();
  rerender(<RequestProgress status="completed" progress={{ total: 3, completed: 0, aborted: 1, skipped: 2 }} />);
  expect(screen.queryByText("正在汇总结果…")).not.toBeInTheDocument();
});

it.each(["cancelled", "interrupted", "failed"] as const)("preserves the last progress for %s", (status) => {
  render(<RequestProgress status={status} progress={{ total: 10, completed: 3, aborted: 0, skipped: 1 }} />);
  expect(screen.getByText("40%")).toBeInTheDocument();
  expect(screen.getByText("最后记录的进度")).toBeInTheDocument();
  expect(screen.queryByText("正在汇总结果…")).not.toBeInTheDocument();
});

it("does not invent progress for queued or historical tasks", () => {
  const { rerender } = render(<RequestProgress status="queued" />);
  expect(screen.getByText("等待执行")).toBeInTheDocument();
  expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  rerender(<RequestProgress status="completed" />);
  expect(screen.getByText("请求进度未知")).toBeInTheDocument();
  expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
});

it("does not round unfinished progress up to 100 percent", () => {
  render(<RequestProgress status="running" progress={{ total: 1000, completed: 999, aborted: 0, skipped: 0 }} />);
  expect(screen.getByText("99%")).toBeInTheDocument();
  expect(screen.queryByText("正在汇总结果…")).not.toBeInTheDocument();
});
