import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getPresets, resolveAutomatic } from "../api/client";
import type { ResolvedConfig } from "../api/types";
import { createDefaultSimConfig } from "../config/simulation";
import { AutomaticConfig } from "./AutomaticConfig";

vi.mock("../api/client", () => ({
  getPresets: vi.fn(), resolveAutomatic: vi.fn(),
}));

const resolution = {
  config: {
    ...createDefaultSimConfig(),
    engine: { ...createDefaultSimConfig().engine, max_total_tokens: 1024 },
  },
  model: {}, hardware: {}, storage: {}, timing: "analytical",
  capacity: {
    kv_bytes_per_token: 147456, weights_resident_bytes: 16400000000,
    l1_capacity_pages: 16, l2_capacity_pages: 0, l3_capacity_pages: 0,
  },
} as ResolvedConfig;

describe("AutomaticConfig", () => {
  beforeEach(() => {
    vi.mocked(getPresets).mockResolvedValue({ models: [], hardware: [], scenarios: [] });
    vi.mocked(resolveAutomatic).mockResolvedValue(resolution);
  });

  it("previews server-resolved capacity and submits physical specs", async () => {
    const onRun = vi.fn();
    render(<AutomaticConfig config={createDefaultSimConfig()} disabled={false} onRun={onRun} />);
    fireEvent.click(screen.getByRole("button", { name: "预览解析容量" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("1,024 token"));
    fireEvent.click(screen.getByRole("button", { name: "运行自动解析仿真" }));
    expect(onRun).toHaveBeenCalledWith(expect.objectContaining({
      timing: "analytical",
      model: expect.objectContaining({ name: "Qwen/Qwen3-8B", num_layers: 36 }),
      hardware: expect.objectContaining({ gpu_count: 1 }),
      config: expect.objectContaining({ engine: expect.objectContaining({ l2_capacity_tokens: 0 }) }),
    }));
  });

  it("reports imported physical specifications rejected by the service", async () => {
    vi.mocked(resolveAutomatic).mockRejectedValueOnce(new Error("KV 头数必须大于零"));
    render(<AutomaticConfig config={createDefaultSimConfig()} disabled={false} onRun={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "导出自动参数" }));
    const textarea = screen.getByLabelText("自动参数 JSON") as HTMLTextAreaElement;
    const exported = JSON.parse(textarea.value) as { model: { num_kv_heads: number } };
    fireEvent.change(textarea, { target: { value: JSON.stringify({
      ...exported, model: { ...exported.model, num_kv_heads: 0 },
    }) } });
    fireEvent.click(screen.getByRole("button", { name: "导入自动参数" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("KV 头数必须大于零"));
  });

  it("round-trips automatic JSON without replacing the manual capacity", () => {
    const onImportConfig = vi.fn();
    render(<AutomaticConfig config={createDefaultSimConfig()} disabled={false}
      onRun={vi.fn()} onImportConfig={onImportConfig} />);
    fireEvent.click(screen.getByRole("button", { name: "导出自动参数" }));
    const textarea = screen.getByLabelText("自动参数 JSON") as HTMLTextAreaElement;
    const exported = JSON.parse(textarea.value) as {
      model: { name: string }; config: { engine: { max_total_tokens: number } }
    };
    expect(exported.model.name).toBe("Qwen/Qwen3-8B");
    expect(exported.config.engine.max_total_tokens).toBe(65_536);
    fireEvent.change(textarea, { target: { value: JSON.stringify({
      ...exported, model: { ...exported.model, name: "用户自定义" },
    }) } });
    fireEvent.click(screen.getByRole("button", { name: "导入自动参数" }));
    expect(screen.getByLabelText("模型名称")).toHaveValue("用户自定义");
    expect(onImportConfig.mock.calls[0][0].engine.max_total_tokens).toBe(65_536);
  });
});
