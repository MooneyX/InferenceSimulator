import { expect, it } from "vitest";
import { hardwareLevelLabel, parseHardwareLevels } from "./presets";

it("parses hardware bindings with normalized property order", () => {
  const entries = parseHardwareLevels('[{"parallelism":8,"name":"NVIDIA B300"}]');
  expect(entries).toEqual([{ name: "NVIDIA B300", parallelism: 8 }]);
  expect(hardwareLevelLabel(entries[0])).toBe("NVIDIA B300 · TP8");
  expect(hardwareLevelLabel("NVIDIA B300", 16)).toBe("NVIDIA B300 · TP16");
});

it.each([
  "bad json", "{}", '["NVIDIA B300"]', '[null]', '[{}]',
  '[{"name":"NVIDIA B300","parallelism":true}]',
  '[{"name":"NVIDIA B300","parallelism":"8"}]',
  '[{"name":"NVIDIA B300","parallelism":8,"gpu_count":16}]',
  '[{"name":"NVIDIA B300","parallelism":8},{"name":"NVIDIA B300","parallelism":16}]',
])("rejects malformed or duplicate hardware bindings: %s", (text) => {
  expect(() => parseHardwareLevels(text)).toThrow();
});
