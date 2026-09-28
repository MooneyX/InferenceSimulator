import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { WarningsPanel } from "./WarningsPanel";

describe("WarningsPanel", () => {
  it.each([undefined, []])("renders nothing for empty warnings", (warnings) => {
    const { container } = render(<WarningsPanel warnings={warnings} />);

    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByRole("heading", { name: "仿真告警" })).not.toBeInTheDocument();
  });

  it("renders every warning returned by the service", () => {
    const warnings = [
      "请求 3 超过 KV token 预算，已强制准入",
      "请求 7 超过 KV token 预算，已强制准入",
    ];

    render(<WarningsPanel warnings={warnings} />);

    expect(screen.getByRole("heading", { name: "仿真告警" })).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByText(warnings[0])).toBeInTheDocument();
    expect(screen.getByText(warnings[1])).toBeInTheDocument();
  });
});
