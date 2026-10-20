import { expect, test } from "@playwright/test";
import { pathToFileURL } from "node:url";

test("后台扫描不锁表单，导出的离线 HTML 可重新透视", async ({ page, context }) => {
  await page.goto("/");
  await page.getByLabel("请求数量", { exact: true }).fill("2");
  const selector = page.getByLabel("横轴参数（必选）");
  await selector.fill("Prefill / Decode");
  await page.getByRole("option", { name: /Prefill \/ Decode 分步执行/ }).click();
  await page.getByLabel("横轴档位").fill("true,false");
  await page.getByRole("button", { name: "提交后台扫描" }).click();
  await expect(page.getByLabel("请求数量", { exact: true })).toBeEnabled();
  await expect(page.getByRole("heading", { name: "扫描结果组合" })).toBeVisible();
  await expect(page.getByText(/独立配置点 2/)).toBeVisible({ timeout: 30_000 });
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出离线可操作 HTML" }).click();
  const download = await downloadPromise;
  const report = await context.newPage();
  await report.goto(pathToFileURL(await download.path()).href);
  await expect(report.locator("#chart canvas")).toBeVisible();
  await expect(report.locator("#summary")).toContainText("2 个独立配置点");
  await report.locator("#metric").selectOption("ttft_mean");
  await expect(report.locator("#data-body tr")).toHaveCount(2);
  await report.close();
});
