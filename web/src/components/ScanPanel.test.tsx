import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getPresets, getScanParameters, uploadReplayPlan } from "../api/client";
import { createDefaultSimConfig, initialHardware, initialModel } from "../config/simulation";
import { ScanPanel, parseLevels } from "./ScanPanel";

vi.mock("../api/client", () => ({ getPresets: vi.fn(), getScanParameters: vi.fn(), uploadReplayPlan: vi.fn() }));
const presetHardware = { ...initialHardware, effective_fp8_tflops: 4500,
  effective_interconnect_gb_s: 600, collective_latency_ms: 0.01, ep_communication_latency_ms: 0.02 };
const presets = {
  models: [initialModel, { ...initialModel, name: "another/model" }],
  hardware: [
    { ...presetHardware, peak_fp16_tflops: 2250, peak_fp8_tflops: 4500,
      peak_hbm_gb_s: 8000, peak_interconnect_gb_s: 1800, performance_note: "未经实测校准" },
    { ...presetHardware, name: "Huawei Ascend 950PR", memory_bytes_per_gpu: 128e9,
      peak_fp16_tflops: 500, peak_fp8_tflops: 1000, peak_hbm_gb_s: 1600,
      peak_interconnect_gb_s: 2000, performance_note: "未经实测校准" },
  ],
};
const catalog = [
  { key: "inference.model", label: "模型", kind: "model_preset",
    modes: ["synthetic/open_loop", "synthetic/closed_loop", "trace/open_loop", "trace/closed_loop"] },
  { key: "inference.hardware", label: "硬件", kind: "hardware_preset",
    modes: ["synthetic/open_loop", "synthetic/closed_loop", "trace/open_loop", "trace/closed_loop"] },
  { key: "workload.synthetic.num_requests", label: "请求数量", kind: "integer",
    modes: ["synthetic/open_loop", "synthetic/closed_loop"] },
  { key: "workload.session_rate", label: "会话投放频率", kind: "number",
    modes: ["synthetic/closed_loop", "trace/open_loop", "trace/closed_loop"] },
  { key: "workload.target_qps", label: "目标 QPS", kind: "number",
    modes: ["synthetic/open_loop", "trace/open_loop"] },
  { key: "workload.synthetic.session.num_sessions", label: "会话数量", kind: "integer",
    modes: ["synthetic/closed_loop"] },
  { key: "workload.synthetic.session.think_time_ms", label: "思考间隔", kind: "number",
    modes: ["synthetic/closed_loop"] },
  { key: "workload.synthetic.prompt_len", label: "平均输入长度", kind: "number",
    modes: ["synthetic/open_loop", "synthetic/closed_loop"] },
  { key: "workload.synthetic.turn_increment_len", label: "平均新增长度", kind: "number",
    modes: ["synthetic/open_loop", "synthetic/closed_loop"] },
  { key: "workload.synthetic.output_len", label: "平均输出长度", kind: "number",
    modes: ["synthetic/open_loop", "synthetic/closed_loop"] },
  { key: "inference.hardware.memory_bytes_per_gpu", label: "单卡显存容量 (GB)", kind: "number",
    modes: ["synthetic/open_loop", "synthetic/closed_loop", "trace/open_loop", "trace/closed_loop"] },
  { key: "inference.parallelism", label: "每实例 GPU / TP / EP 数", kind: "integer",
    modes: ["synthetic/open_loop", "synthetic/closed_loop", "trace/open_loop", "trace/closed_loop"] },
];

beforeEach(() => {
  vi.mocked(getScanParameters).mockReset().mockResolvedValue(catalog);
  vi.mocked(getPresets).mockReset().mockResolvedValue(presets);
  vi.mocked(uploadReplayPlan).mockReset();
});

describe("ScanPanel", () => {
  it("matches parameter names and submits axis levels without changing the ordinary form", async () => {
    const onSubmit = vi.fn();
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={onSubmit} onImportBase={vi.fn()} />);
    expect(screen.getByLabelText("负载模式")).toHaveValue("synthetic");
    await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
    for (const hint of [
      /按需添加扫描参数/, /扫描方案包含当前基础配置/, /选定负载与到达模式后/,
      /模型包含全部模型配置/, /输入离散档位/, /组合数按所选维度档位的乘积计算/,
    ]) {
      expect(screen.queryByText(hint)).not.toBeInTheDocument();
    }
    fireEvent.focus(screen.getByLabelText("扫描参数 1"));
    fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: "请求" } });
    const options = screen.getByRole("listbox");
    fireEvent.click(within(options).getByRole("option", { name: /请求数量/ }));
    fireEvent.change(screen.getByLabelText("扫描档位 1"), { target: { value: "10,20" } });
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      axes: [{ key: "workload.synthetic.num_requests", values: [10, 20] }],
    }));
  });

  it("uses GPU memory in GB as the default memory scan level", async () => {
    const onSubmit = vi.fn();
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={onSubmit} onImportBase={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
    fireEvent.focus(screen.getByLabelText("扫描参数 1"));
    fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: "显存" } });
    fireEvent.click(within(screen.getByRole("listbox")).getByRole("option", { name: /单卡显存容量/ }));
    expect(screen.getByLabelText("扫描档位 1")).toHaveValue("288");
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      axes: [{ key: "inference.hardware.memory_bytes_per_gpu", values: [288] }],
    }));
  });

  it("adds more than four parameters, preserves order and removes unused rows", async () => {
    const onSubmit = vi.fn();
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={onSubmit} onImportBase={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
    const names = ["请求数量", "目标 QPS", "平均输入长度", "平均新增长度", "平均输出长度"];
    for (const [index, name] of names.entries()) {
      if (index) fireEvent.click(screen.getByRole("button", { name: "添加参数" }));
      const search = screen.getByLabelText(`扫描参数 ${index + 1}`);
      fireEvent.focus(search);
      if (index) expect(within(screen.getByRole("listbox"))
        .queryByRole("option", { name: /请求数量/ })).not.toBeInTheDocument();
      fireEvent.change(search, { target: { value: name } });
      fireEvent.click(within(screen.getByRole("listbox")).getByRole("option", { name: new RegExp(name) }));
      fireEvent.change(screen.getByLabelText(`扫描档位 ${index + 1}`),
        { target: { value: index ? "2,4" : "10,20" } });
    }
    expect(screen.queryByText(/当前约 \d+ 点/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit.mock.lastCall?.[0].axes.map((axis: { key: string }) => axis.key)).toEqual([
      "workload.synthetic.num_requests", "workload.target_qps", "workload.synthetic.prompt_len",
      "workload.synthetic.turn_increment_len", "workload.synthetic.output_len",
    ]);
    fireEvent.click(screen.getByRole("button", { name: "删除扫描参数 3" }));
    expect(screen.getByLabelText("扫描参数 4")).toHaveValue("平均输出长度");
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit.mock.lastCall?.[0].axes).toEqual([
      { key: "workload.synthetic.num_requests", values: [10, 20] },
      { key: "workload.target_qps", values: [2, 4] },
      { key: "workload.synthetic.turn_increment_len", values: [2, 4] },
      { key: "workload.synthetic.output_len", values: [2, 4] },
    ]);
  });

  it("offers request parameters as scan axes instead of separate inputs for each mode", async () => {
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={vi.fn()} onImportBase={vi.fn()} />);
    await waitFor(() => expect(getScanParameters).toHaveBeenCalled());
    const search = screen.getByLabelText("扫描参数 1");
    fireEvent.focus(search);
    expect(screen.queryByLabelText("目标发送 QPS")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("会话投放频率（会话/秒）")).not.toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /目标 QPS/ })).toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /平均输入长度/ })).toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).queryByRole("option", { name: /会话数量/ })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("请求到达模式"), { target: { value: "closed_loop" } });
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /会话数量/ })).toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /会话投放频率/ })).toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).queryByRole("option", { name: /目标 QPS/ })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    expect(within(screen.getByRole("listbox")).queryByRole("option", { name: /会话数量/ })).not.toBeInTheDocument();
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /会话投放频率/ })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("请求到达模式"), { target: { value: "open_loop" } });
    expect(within(screen.getByRole("listbox")).getByRole("option", { name: /目标 QPS/ })).toBeInTheDocument();
  });

  it("uses a Trace open-loop target QPS axis without a separate scaling control", async () => {
    const onSubmit = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-scan", requests: 2, sessions: 1 });
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={onSubmit} onImportBase={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    expect(screen.queryByRole("checkbox", { name: "按目标 QPS 缩放 Trace" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("回放计划 JSON 文件"), {
      target: { files: [new File(["{}"], "replay-plan.json")] },
    });
    await screen.findByText(/replay-plan.json/);
    fireEvent.focus(screen.getByLabelText("扫描参数 1"));
    fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: "目标 QPS" } });
    fireEvent.click(screen.getByRole("option", { name: /目标 QPS/ }));
    fireEvent.change(screen.getByLabelText("扫描档位 1"), { target: { value: "8,16" } });
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      base: expect.objectContaining({ workload: expect.not.objectContaining({ target_qps: expect.anything() }) }),
      axes: [{ key: "workload.target_qps", values: [8, 16] }],
    }));
  });

  it("requires an independently uploaded replay plan and removes invalid axes on mode switch", async () => {
    const onSubmit = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-scan", requests: 2, sessions: 1 });
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={onSubmit} onImportBase={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    expect(screen.getByLabelText("请求到达模式")).toHaveValue("open_loop");
    expect(screen.getByLabelText("回放计划 JSON 文件")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("回放计划 JSON 文件"), {
      target: { files: [new File(["{}"], "replay-plan.json")] },
    });
    await screen.findByText(/replay-plan.json/);
    fireEvent.focus(screen.getByLabelText("扫描参数 1"));
    fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: "会话" } });
    fireEvent.click(screen.getByRole("option", { name: /会话投放频率/ }));
    fireEvent.change(screen.getByLabelText("扫描档位 1"), { target: { value: "1,2" } });
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      base: expect.objectContaining({ workload: expect.objectContaining({
        mode: "trace", trace: { plan_id: "plan-scan" } }) }),
    }));
  });

  it("round-trips editable range levels and rejects an unsupported or duplicated axis without changing state", async () => {
    const onImportBase = vi.fn();
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={vi.fn()} onImportBase={onImportBase} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "导出扫描方案" })).toBeEnabled());
    fireEvent.focus(screen.getByLabelText("扫描参数 1"));
    fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: "请求数量" } });
    fireEvent.click(screen.getByRole("option", { name: /请求数量/ }));
    fireEvent.change(screen.getByLabelText("扫描档位 1"), { target: { value: "10:30:10" } });
    fireEvent.click(screen.getByRole("button", { name: "导出扫描方案" }));
    const json = (screen.getByLabelText("扫描方案 JSON") as HTMLInputElement).value;
    expect(JSON.parse(json)).toMatchObject({ version: 1, axes: [{ key: "workload.synthetic.num_requests", text: "10:30:10" }] });
    fireEvent.change(screen.getByLabelText("扫描档位 1"), { target: { value: "5,6" } });
    fireEvent.click(screen.getByRole("button", { name: "导入扫描方案" }));
    expect(screen.getByLabelText("扫描档位 1")).toHaveValue("10:30:10");
    expect(onImportBase).toHaveBeenCalledTimes(1);
    for (const axes of [[{ key: "workload.session_rate", text: "1,2" }],
      [{ key: "workload.synthetic.num_requests", text: "1,2" },
        { key: "workload.synthetic.num_requests", text: "3,4" }]]) {
      fireEvent.change(screen.getByLabelText("扫描方案 JSON"), { target: { value: JSON.stringify({ ...JSON.parse(json), axes }) } });
      fireEvent.click(screen.getByRole("button", { name: "导入扫描方案" }));
      expect(screen.getByRole("alert")).toHaveTextContent("扫描方案导入失败");
      expect(screen.getByLabelText("扫描档位 1")).toHaveValue("10:30:10");
      expect(onImportBase).toHaveBeenCalledTimes(1);
    }
  });

  it("removes the Trace plan on export and requires a new upload after import", async () => {
    const onSubmit = vi.fn();
    const onImportBase = vi.fn();
    vi.mocked(uploadReplayPlan).mockResolvedValue({ id: "plan-scan", requests: 2, sessions: 1 });
    render(<ScanPanel base={createDefaultSimConfig()} onSubmit={onSubmit} onImportBase={onImportBase} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "导出扫描方案" })).toBeEnabled());
    fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
    fireEvent.change(screen.getByLabelText("回放计划 JSON 文件"), {
      target: { files: [new File(["{}"], "replay-plan.json")] },
    });
    await screen.findByText(/replay-plan.json/);
    fireEvent.focus(screen.getByLabelText("扫描参数 1"));
    fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: "目标 QPS" } });
    fireEvent.click(screen.getByRole("option", { name: /目标 QPS/ }));
    fireEvent.change(screen.getByLabelText("扫描档位 1"), { target: { value: "8,16" } });
    fireEvent.click(screen.getByRole("button", { name: "导出扫描方案" }));
    const json = (screen.getByLabelText("扫描方案 JSON") as HTMLInputElement).value;
    expect(JSON.parse(json).base.workload.trace.plan_id).toBeNull();
    expect(json).not.toContain("plan-scan");
    fireEvent.click(screen.getByRole("button", { name: "导入扫描方案" }));
    expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeDisabled();
    expect(screen.getByText(/请重新上传回放计划/)).toBeInTheDocument();
    expect(onImportBase.mock.lastCall?.[0].workload.trace.plan_id).toBeNull();
    fireEvent.change(screen.getByLabelText("回放计划 JSON 文件"), {
      target: { files: [new File(["{}"], "replay-plan.json")] },
    });
    await screen.findByText(/replay-plan.json/);
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(onSubmit.mock.lastCall?.[0]).toMatchObject({ base: { workload: { mode: "trace", trace: { plan_id: "plan-scan" } } },
      axes: [{ key: "workload.target_qps", values: [8, 16] }] });
    expect(uploadReplayPlan).toHaveBeenCalledTimes(2);
  });
});

it("selects preset checkboxes, overrides and round-trips names containing spaces", async () => {
  const base = createDefaultSimConfig();
  const original = structuredClone(base);
  const onSubmit = vi.fn();
  const onImportBase = vi.fn();
  render(<ScanPanel base={base} onSubmit={onSubmit} onImportBase={onImportBase} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
  for (const [index, name] of ["模型", "硬件", "单卡显存容量"].entries()) {
    if (index) fireEvent.click(screen.getByRole("button", { name: "添加参数" }));
    fireEvent.focus(screen.getByLabelText(`扫描参数 ${index + 1}`));
    fireEvent.change(screen.getByLabelText(`扫描参数 ${index + 1}`), { target: { value: name } });
    fireEvent.click(screen.getByRole("option", { name: new RegExp(name) }));
  }
  const models = within(screen.getByRole("group", { name: "模型预设" }));
  const hardware = within(screen.getByRole("group", { name: "硬件预设" }));
  expect(models.getByRole("checkbox", { name: initialModel.name })).toBeChecked();
  expect(hardware.getByRole("checkbox", { name: initialHardware.name })).toBeChecked();
  fireEvent.click(models.getByRole("button", { name: "全选" }));
  expect(screen.getByLabelText("NVIDIA B300 每实例 GPU / TP 数")).toHaveValue(1);
  expect(screen.getByLabelText("Huawei Ascend 950PR 每实例 GPU / TP 数")).toBeDisabled();
  fireEvent.click(hardware.getByRole("button", { name: "全选" }));
  fireEvent.change(screen.getByLabelText("NVIDIA B300 每实例 GPU / TP 数"), { target: { value: "8" } });
  fireEvent.change(screen.getByLabelText("Huawei Ascend 950PR 每实例 GPU / TP 数"), { target: { value: "16" } });
  fireEvent.click(hardware.getByRole("checkbox", { name: "Huawei Ascend 950PR" }));
  expect(screen.getByLabelText("Huawei Ascend 950PR 每实例 GPU / TP 数")).toBeDisabled();
  fireEvent.click(hardware.getByRole("checkbox", { name: "Huawei Ascend 950PR" }));
  expect(screen.getByLabelText("Huawei Ascend 950PR 每实例 GPU / TP 数")).toHaveValue(16);
  fireEvent.change(screen.getByLabelText("扫描档位 3"), { target: { value: "160,192" } });
  expect(screen.queryByText(/当前约 \d+ 点/)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
  const axes = [
    { key: "inference.model", values: presets.models.map((entry) => entry.name) },
    { key: "inference.hardware", values: presets.hardware.map((entry, index) => ({ name: entry.name, parallelism: index ? 16 : 8 })) },
    { key: "inference.hardware.memory_bytes_per_gpu", values: [160, 192] },
  ];
  expect(onSubmit.mock.lastCall?.[0].axes).toEqual(axes);
  expect(base).toEqual(original);
  fireEvent.click(screen.getByRole("button", { name: "导出扫描方案" }));
  const saved = (screen.getByLabelText("扫描方案 JSON") as HTMLInputElement).value;
  expect(JSON.parse(JSON.parse(saved).axes[1].text)).toEqual(axes[1].values);
  fireEvent.click(hardware.getByRole("button", { name: "清空" }));
  fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
  expect(screen.getByRole("alert")).toHaveTextContent("请至少选择一个硬件预设");
  expect(onSubmit).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "导入扫描方案" }));
  expect(hardware.getAllByRole("checkbox").every((input) => (input as HTMLInputElement).checked)).toBe(true);
  expect(onImportBase).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
  expect(onSubmit.mock.lastCall?.[0].axes).toEqual(axes);
  fireEvent.change(screen.getByLabelText("扫描方案 JSON"), { target: { value: JSON.stringify({
    ...JSON.parse(saved), axes: [{ key: "inference.hardware", text: '[{"name":"unknown","parallelism":8}]' }],
  }) } });
  fireEvent.click(screen.getByRole("button", { name: "导入扫描方案" }));
  expect(screen.getByRole("alert")).toHaveTextContent("硬件预设不存在");
  expect(onImportBase).toHaveBeenCalledTimes(1);
});

it("keeps preset selections when changing workload mode and starts custom models unchecked", async () => {
  const base = createDefaultSimConfig();
  base.inference.model.name = "自定义模型";
  render(<ScanPanel base={base} onSubmit={vi.fn()} onImportBase={vi.fn()} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
  fireEvent.focus(screen.getByLabelText("扫描参数 1"));
  fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: "模型" } });
  fireEvent.keyDown(screen.getByLabelText("扫描参数 1"), { key: "Enter" });
  expect(screen.getAllByRole("checkbox").every((input) => !(input as HTMLInputElement).checked)).toBe(true);
  fireEvent.click(screen.getByRole("checkbox", { name: "another/model" }));
  fireEvent.change(screen.getByLabelText("负载模式"), { target: { value: "trace" } });
  expect(screen.getByRole("checkbox", { name: "another/model" })).toBeChecked();
});

it.each(["硬件", "每实例 GPU / TP / EP 数"])("makes %s mutually exclusive with the other parallelism axis", async (first) => {
  render(<ScanPanel base={createDefaultSimConfig()} onSubmit={vi.fn()} onImportBase={vi.fn()} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
  fireEvent.focus(screen.getByLabelText("扫描参数 1"));
  fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: first } });
  fireEvent.click(screen.getByRole("option", { name: new RegExp(first) }));
  fireEvent.click(screen.getByRole("button", { name: "添加参数" }));
  const other = first === "硬件" ? "每实例 GPU / TP / EP 数" : "硬件";
  fireEvent.focus(screen.getByLabelText("扫描参数 2"));
  fireEvent.change(screen.getByLabelText("扫描参数 2"), { target: { value: other } });
  expect(screen.getByRole("option", { name: new RegExp(other) })).toBeDisabled();
  fireEvent.keyDown(screen.getByLabelText("扫描参数 2"), { key: "Enter" });
  expect(screen.getByRole("alert")).toHaveTextContent("不能同时扫描独立并行度");
  fireEvent.click(screen.getByRole("button", { name: "删除扫描参数 1" }));
  fireEvent.focus(screen.getByLabelText("扫描参数 1"));
  fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: other } });
  expect(screen.getByRole("option", { name: new RegExp(other) })).toBeEnabled();
});

it("rejects invalid bound counts and conflicting imported axes without updating the base", async () => {
  const base = createDefaultSimConfig();
  const onSubmit = vi.fn();
  const onImportBase = vi.fn();
  render(<ScanPanel base={base} onSubmit={onSubmit} onImportBase={onImportBase} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
  fireEvent.focus(screen.getByLabelText("扫描参数 1"));
  fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: "硬件" } });
  fireEvent.click(screen.getByRole("option", { name: /硬件/ }));
  for (const value of ["", "0", "-1", "1.5"]) {
    fireEvent.change(screen.getByLabelText("NVIDIA B300 每实例 GPU / TP 数"), { target: { value } });
    fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
    expect(screen.getByRole("alert")).toHaveTextContent("每实例 GPU / TP 数必须为正整数");
  }
  expect(onSubmit).not.toHaveBeenCalled();
  const hardware = { key: "inference.hardware", text: '[{"name":"NVIDIA B300","parallelism":8}]' };
  fireEvent.change(screen.getByLabelText("扫描方案 JSON"), { target: { value: JSON.stringify({ version: 1, base,
    axes: [hardware, { key: "inference.parallelism", text: "4,8" }],
  }) } });
  fireEvent.click(screen.getByRole("button", { name: "导入扫描方案" }));
  expect(screen.getByText(/扫描方案导入失败.*不能同时扫描独立并行度/)).toBeInTheDocument();
  expect(onImportBase).not.toHaveBeenCalled();
});

it("accepts bound multi-GPU presets without timing configuration", async () => {
  const base = createDefaultSimConfig();
  const onSubmit = vi.fn();
  render(<ScanPanel base={base} onSubmit={onSubmit} onImportBase={vi.fn()} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
  fireEvent.focus(screen.getByLabelText("扫描参数 1"));
  fireEvent.change(screen.getByLabelText("扫描参数 1"), { target: { value: "硬件" } });
  fireEvent.click(screen.getByRole("option", { name: /硬件/ }));
  fireEvent.change(screen.getByLabelText("NVIDIA B300 每实例 GPU / TP 数"), { target: { value: "8" } });
  fireEvent.click(screen.getByRole("button", { name: "提交后台扫描" }));
  expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
    base: expect.objectContaining({ inference: base.inference }),
    axes: [{ key: "inference.hardware", values: [{ name: "NVIDIA B300", parallelism: 8 }] }],
  }));
  expect(onSubmit.mock.lastCall![0].base.inference).not.toHaveProperty("timing");
});

it.each(["analytical", "empirical"])("rejects removed %s timing in imported scan bases", async (mode) => {
  const base = createDefaultSimConfig();
  const onImportBase = vi.fn();
  render(<ScanPanel base={base} onSubmit={vi.fn()} onImportBase={onImportBase} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "提交后台扫描" })).toBeEnabled());
  fireEvent.change(screen.getByLabelText("扫描方案 JSON"), { target: { value: JSON.stringify({
    version: 1, base: { ...base, inference: { ...base.inference, timing: { mode } } },
    axes: [{ key: "inference.parallelism", text: "1,4" }],
  }) } });
  fireEvent.click(screen.getByRole("button", { name: "导入扫描方案" }));
  expect(screen.getByText(/扫描方案导入失败.*不支持的参数：inference.timing/)).toBeInTheDocument();
  expect(onImportBase).not.toHaveBeenCalled();
});

it("accepts number, boolean and range levels", () => {
  expect(parseLevels("2:6:2", "integer")).toEqual([2, 4, 6]);
  expect(parseLevels("true,false", "boolean")).toEqual([true, false]);
  expect(() => parseLevels("1.5,2", "integer")).toThrow();
});
