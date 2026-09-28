import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "./App";

describe("App", () => {
  it("renders the M3 application shell", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { level: 1, name: "推理服务仿真器" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "前端工作区已就绪" }),
    ).toBeInTheDocument();
  });
});
