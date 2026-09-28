import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { LengthDistributionEditor } from "./LengthDistributionEditor";

describe("LengthDistributionEditor", () => {
  it("renders only the fixed-length parameter for a fixed distribution", () => {
    render(
      <LengthDistributionEditor
        name="prompt_len"
        label="Prompt 长度"
        value={{ dist: "fixed", value: 128 }}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByRole("group", { name: "Prompt 长度" })).toBeInTheDocument();
    expect(screen.getByLabelText("固定长度（token）")).toHaveValue(128);
    expect(screen.queryByLabelText("平均长度（token）")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("形状参数（sigma）")).not.toBeInTheDocument();
  });

  it("preserves a representative length when changing distribution type", () => {
    const onChange = vi.fn();
    render(
      <LengthDistributionEditor
        name="output_len"
        label="输出长度"
        value={{ dist: "fixed", value: 64 }}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("分布类型"), {
      target: { value: "poisson" },
    });

    expect(onChange).toHaveBeenCalledWith({ dist: "poisson", mean: 64 });
  });

  it("edits both lognormal parameters without changing its discriminator", () => {
    const onChange = vi.fn();
    render(
      <LengthDistributionEditor
        name="prompt_len"
        label="Prompt 长度"
        value={{ dist: "lognormal", mean: 512, sigma: 1 }}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("平均长度（token）"), {
      target: { value: "256" },
    });
    fireEvent.change(screen.getByLabelText("形状参数（sigma）"), {
      target: { value: "0.75" },
    });

    expect(onChange).toHaveBeenNthCalledWith(1, {
      dist: "lognormal",
      mean: 256,
      sigma: 1,
    });
    expect(onChange).toHaveBeenNthCalledWith(2, {
      dist: "lognormal",
      mean: 512,
      sigma: 0.75,
    });
  });

  it("reports field errors through accessible input attributes", () => {
    render(
      <LengthDistributionEditor
        name="output_len"
        label="输出长度"
        value={{ dist: "poisson", mean: Number.NaN }}
        errors={{ mean: "必须是大于 0 的有限数值" }}
        onChange={vi.fn()}
      />,
    );

    const input = screen.getByLabelText("平均长度（token）");
    const error = screen.getByRole("alert");

    expect(input).toHaveValue(null);
    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(input).toHaveAttribute("aria-describedby", error.id);
    expect(error).toHaveTextContent("必须是大于 0 的有限数值");
  });
});
