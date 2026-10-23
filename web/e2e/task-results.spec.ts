import { expect, test } from "@playwright/test";

import { createDefaultSimConfig } from "../src/config/simulation";

for (const width of [1600, 768]) {
  test(`任务结果定位与参数导出入口 ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const base = { ...createDefaultSimConfig(), num_instances: 7 };
    const axes = [{ key: "num_instances", values: [1, 2] }];
    const shared = { status: "completed", failed: 0, error: null, cache: null, lookup_ms: null,
      version: 1, code_fingerprint: "test", code_version_status: "current", point_archive_version: 1 };
    const jobs = [{ ...shared, id: "scan-test", kind: "scan", total: 2, completed: 2, base, axes },
      { ...shared, id: "single-test", kind: "single", total: 1, completed: 1, base: null, axes: null }];
    await page.route("**/api/jobs**", async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path.endsWith("/events")) {
        await route.fulfill({ contentType: "text/event-stream", body: `event: snapshot\ndata: ${JSON.stringify(jobs)}\n\n` });
      } else if (path.endsWith("/tasks")) {
        await route.fulfill({ json: { total: 2, items: [{ index: 0, values: { num_instances: 1 },
          status: "completed", error: null, result_available: true }] } });
      } else if (path.endsWith("/result")) {
        await route.fulfill({ json: { meta: { config: { ...base, num_instances: path.includes("/points/") ? 1 : 3 },
          num_requests: 0, num_completed: 0, num_aborted: 0, num_skipped: 0, wall_clock_ms: 1 },
          instances: [], metrics: {}, requests: [], warnings: [], resolution: null, analysis: null } });
      } else {
        await route.fulfill({ json: jobs });
      }
    });
    await page.route("**/api/scans/parameters", (route) => route.fulfill({ json: [{ key: "num_instances",
      label: "实例数", kind: "integer", modes: ["synthetic/open_loop"] }] }));
    await page.goto("/");
    await expect(page.locator("#routing-policy-help")).toHaveCount(0);
    const tasks = page.getByRole("region", { name: "后台任务" });
    const expand = tasks.getByRole("button", { name: "展开组合" });
    await expand.scrollIntoViewIfNeeded();
    if (width === 1600) {
      const progressBox = (await tasks.getByText("已结束组合 2/2").boundingBox())!;
      const buttonBox = (await expand.boundingBox())!;
      expect(buttonBox.x).toBeGreaterThan(progressBox.x + progressBox.width);
      expect(Math.abs(buttonBox.y + buttonBox.height / 2 - progressBox.y - progressBox.height / 2)).toBeLessThan(2);
    }
    await expand.click();
    const scanRow = tasks.locator("ul > li").filter({ has: page.getByRole("button", { name: "收起组合" }) });
    await expect(scanRow.getByText("实例数: 1")).toBeVisible();
    await expect(tasks.getByRole("button", { name: "导出本次参数" })).toHaveCount(0);
    await scanRow.getByRole("button", { name: "查看结果" }).click();
    await expect(page.locator("#results-heading")).toBeInViewport();
    await page.getByRole("button", { name: "导出本次参数" }).click();
    expect(JSON.parse(await page.getByLabel("本次参数 JSON", { exact: true }).inputValue()).num_instances).toBe(1);

    const singleRow = tasks.locator("ul > li").filter({ hasText: "仿真 single-t" });
    await singleRow.getByRole("button", { name: "查看结果" }).click();
    await expect(page.locator("#results-heading")).toBeInViewport();
    await page.getByRole("button", { name: "导出本次参数" }).click();
    expect(JSON.parse(await page.getByLabel("本次参数 JSON", { exact: true }).inputValue()).num_instances).toBe(3);

    const scans = page.getByRole("region", { name: "扫描结果组合" });
    await scans.getByRole("button", { name: "导出扫描方案" }).click();
    const json = await scans.getByLabel("扫描方案 JSON", { exact: true }).inputValue();
    expect(JSON.parse(json)).toEqual({ version: 1, base, axes: [{ key: "num_instances", text: "1,2" }] });
    await expect(scans.getByRole("checkbox")).not.toBeChecked();
    const planInput = page.locator(".scan-panel").getByLabel("扫描方案 JSON", { exact: true });
    await planInput.fill(json);
    await page.getByRole("button", { name: "导入扫描方案" }).click();
    await expect(page.getByLabel("扫描档位 1")).toHaveValue("1,2");
    await expect(page.getByLabel("推理实例数量")).toHaveValue("7");
    expect(errors).toEqual([]);
  });
}
