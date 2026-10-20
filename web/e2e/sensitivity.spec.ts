import { expect, test } from "@playwright/test";
import { pathToFileURL } from "node:url";
import { readFile } from "node:fs/promises";

test("后台扫描不锁表单，导出的离线 HTML 可重新透视", async ({ page, context }, testInfo) => {
  test.setTimeout(60_000);
  await page.goto("/");
  await page.getByLabel("请求数量", { exact: true }).fill("2");
  const selector = page.getByLabel("横轴参数（必选）");
  await selector.fill("最多同时运行请求");
  await page.getByRole("listbox").getByRole("option", { name: /最多同时运行请求/ }).click();
  await page.getByLabel("横轴档位").fill("1,2,3");
  await page.getByLabel("颜色参数（可选）").fill("Prefill / Decode");
  await page.getByRole("listbox").getByRole("option", { name: /Prefill \/ Decode 分步执行/ }).click();
  await page.getByLabel("颜色档位").fill("true,false");
  const submitted = page.waitForResponse((response) => response.url().endsWith("/api/scans")
    && response.request().method() === "POST");
  await page.getByRole("button", { name: "提交后台扫描" }).click();
  const jobId = ((await (await submitted).json()) as { id: string }).id;
  await expect(page.getByLabel("请求数量", { exact: true })).toBeEnabled();
  await expect(page.getByRole("heading", { name: "扫描结果组合" })).toBeVisible();
  const batches = page.locator(".scan-batches label");
  await expect(batches.filter({ hasText: jobId.slice(0, 8) })).toBeVisible();
  const batchCount = await batches.count();
  for (let index = 0; index < batchCount; index++) {
    const batch = batches.nth(index);
    if (!(await batch.textContent())?.includes(jobId.slice(0, 8))) await batch.locator("input").uncheck();
  }
  await expect(page.getByText(/独立配置点 6/)).toBeVisible({ timeout: 30_000 });
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出离线可操作 HTML" }).click();
  const download = await downloadPromise;
  const htmlPath = testInfo.outputPath("sensitivity-report.html");
  await download.saveAs(htmlPath);
  const report = await context.newPage();
  await report.goto(pathToFileURL(htmlPath).href);
  await expect(report.locator("#chart canvas")).toBeVisible();
  await expect(report.locator("#summary")).toContainText("6 个独立配置点");
  const count = report.locator("#data-body tr");
  const requests = report.locator("#filters fieldset").filter({ hasText: "最多同时运行请求" });
  const pd = report.locator("#filters fieldset").filter({ hasText: "Prefill / Decode 分步执行" });
  await requests.getByRole("checkbox", { name: "2" }).uncheck();
  await expect(count).toHaveCount(4);
  await pd.getByRole("checkbox", { name: "false" }).uncheck();
  await expect(count).toHaveCount(2);
  await requests.getByRole("button", { name: "清空" }).click();
  await expect(count).toHaveCount(0);
  await requests.getByRole("button", { name: "全选" }).click();
  await expect(count).toHaveCount(3);
  await report.locator("#metric").selectOption("ttft_mean");
  await expect(count).toHaveCount(3);
  const csvPromise = report.waitForEvent("download");
  await report.getByRole("button", { name: "导出筛选数据 CSV" }).click();
  const csv = await csvPromise;
  expect((await readFile(await csv.path(), "utf8")).trim().split(/\r?\n/)).toHaveLength(4);
  await report.close();
});
