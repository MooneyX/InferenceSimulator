import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { createDefaultSimConfig } from "../config/simulation";
import { parseSimConfigJson } from "../config/parameterTransfer";
import { ParameterExport } from "./ParameterExport";

function clipboard(writeText: ReturnType<typeof vi.fn> | undefined) {
  vi.stubGlobal("navigator", { clipboard: writeText ? { writeText } : undefined });
}
afterEach(() => { vi.unstubAllGlobals(); });

it("copies actual parameters in the existing import format", async () => {
  const config = { ...createDefaultSimConfig(), num_instances: 5 };
  const writeText = vi.fn().mockResolvedValue(undefined);
  clipboard(writeText);
  render(<ParameterExport config={config} />);
  fireEvent.click(screen.getByRole("button", { name: "导出本次参数" }));
  const text = (screen.getByLabelText("本次参数 JSON") as HTMLInputElement).value;
  expect(parseSimConfigJson(text)).toEqual(config);
  expect(writeText).toHaveBeenCalledWith(text);
  await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("已生成并复制本次参数 JSON"));
});

it.each(["unavailable", "rejected", "throws"])("keeps a manual copy when clipboard is %s", async (mode) => {
  clipboard(mode === "unavailable" ? undefined : mode === "rejected"
    ? vi.fn().mockRejectedValue(new Error("denied")) : vi.fn(() => { throw new Error("denied"); }));
  const config = createDefaultSimConfig();
  render(<ParameterExport config={config} />);
  fireEvent.click(screen.getByRole("button", { name: "导出本次参数" }));
  const field = screen.getByLabelText("本次参数 JSON") as HTMLInputElement;
  expect(parseSimConfigJson(field.value)).toEqual(config);
  fireEvent.focus(field);
  expect(field.selectionEnd).toBe(field.value.length);
  await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("可手动复制"));
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("exports all ordered scan levels including duplicate, boolean and null values", () => {
  clipboard(undefined);
  const config = createDefaultSimConfig();
  render(<ParameterExport config={config} label="导出扫描方案" axes={[
    { key: "num_instances", values: [3, 1, 3] },
    { key: "scheduler.pd_mutually_exclusive", values: [false, true] },
    { key: "scheduler.max_prefill_tokens", values: [null, 64] },
  ]} />);
  fireEvent.click(screen.getByRole("button", { name: "导出扫描方案" }));
  expect(JSON.parse((screen.getByLabelText("扫描方案 JSON") as HTMLInputElement).value)).toEqual({
    version: 1, base: config, axes: [
      { key: "num_instances", text: "3,1,3" },
      { key: "scheduler.pd_mutually_exclusive", text: "false,true" },
      { key: "scheduler.max_prefill_tokens", text: "null,64" },
    ],
  });
});

it.each([false, true])("clears only the temporary Trace plan reference for scan=%s", (scan) => {
  clipboard(undefined);
  const config = createDefaultSimConfig();
  config.workload = { mode: "trace", arrival_mode: "open_loop", seed: 7,
    session_rate: 2, target_qps: 9, trace: { plan_id: "archived-plan" } };
  render(<ParameterExport config={config} axes={scan ? [{ key: "num_instances", values: [1, 2] }] : undefined} />);
  fireEvent.click(screen.getByRole("button", { name: "导出本次参数" }));
  const exported = JSON.parse((screen.getByRole("textbox") as HTMLInputElement).value);
  const reusable = scan ? exported.base : exported;
  expect(reusable.workload).toEqual({ ...config.workload, trace: { plan_id: null } });
  expect(config.workload.trace?.plan_id).toBe("archived-plan");
  expect(screen.getByRole("status")).toHaveTextContent("请重新上传请求计划");
});

it("reports incompatible historical parameters without substituting current defaults", () => {
  const config = { ...createDefaultSimConfig(), num_instances: 0 };
  render(<ParameterExport config={config} />);
  fireEvent.click(screen.getByRole("button", { name: "导出本次参数" }));
  expect(screen.getByRole("alert")).toHaveTextContent("导出失败");
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
});
