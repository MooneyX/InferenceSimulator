import { expect, test } from "@playwright/test";
import { pathToFileURL } from "node:url";
import { readFile } from "node:fs/promises";

test("后台扫描不锁表单，导出的离线 HTML 可重新透视", async ({ page, context }, testInfo) => {
  test.setTimeout(60_000);
  await page.goto("/");
  await page.getByLabel("请求数量", { exact: true }).fill("2");
  await page.getByLabel("扫描参数 1").fill("最多同时运行请求");
  await page.getByRole("listbox").getByRole("option", { name: /最多同时运行请求/ }).click();
  await page.getByLabel("扫描档位 1").fill("1,2,3");
  await page.getByRole("button", { name: "添加参数" }).click();
  await page.getByLabel("扫描参数 2").fill("Prefill / Decode");
  await page.getByRole("listbox").getByRole("option", { name: /Prefill \/ Decode 分步执行/ }).click();
  await page.getByLabel("扫描档位 2").fill("true,false");
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
  const colorRole = page.locator(".scan-role-row").filter({ has: page.locator(".scan-role-values[aria-label='颜色参数值']") }).locator("select");
  await expect(colorRole).toHaveValue("scheduler.pd_mutually_exclusive");
  await colorRole.selectOption("");
  await expect(page.getByText(/独立配置点 3/)).toBeVisible();
  const background = page.locator(".scan-background-groups fieldset").filter({ hasText: "Prefill / Decode 分步执行" });
  await expect(background.getByRole("radio", { name: "true" })).toBeChecked();
  await page.getByRole("button", { name: "重新选择背景值" }).click();
  await expect(page.getByText("背景参数尚未选齐，图表暂不显示。")).toBeVisible();
  await background.getByRole("radio", { name: "false" }).check();
  await expect(page.getByText(/独立配置点 3/)).toBeVisible();
  await colorRole.selectOption("scheduler.pd_mutually_exclusive");
  await expect(page.getByText(/独立配置点 6/)).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出离线可操作 HTML" }).click();
  const download = await downloadPromise;
  const htmlPath = testInfo.outputPath("sensitivity-report.html");
  await download.saveAs(htmlPath);
  const report = await context.newPage();
  await report.goto(pathToFileURL(htmlPath).href);
  await expect(report.locator("#chart canvas")).toBeVisible();
  await report.evaluate(() => {
    const { echarts } = window as unknown as { echarts: { getInstanceByDom(node: HTMLElement): {
      dispatchAction(action: { type: string; seriesIndex: number; dataIndex: number }): void;
    } } };
    echarts.getInstanceByDom(document.getElementById("chart")!).dispatchAction({
      type: "showTip", seriesIndex: 0, dataIndex: 0,
    });
  });
  await expect(report.locator("#chart svg[aria-hidden='true']").first()).toBeVisible();
  const tooltipFits = await report.locator("#chart svg[aria-hidden='true']").first().evaluate((svg) => {
    const text = svg.nextElementSibling as HTMLElement;
    const box = svg.closest("#chart > div") as HTMLElement;
    text.textContent = "很长的扫描参数名称与参数值".repeat(20);
    return getComputedStyle(box).whiteSpace === "normal"
      && box.getBoundingClientRect().width <= window.innerWidth - 20
      && text.scrollWidth <= text.clientWidth + 1
      && text.getBoundingClientRect().right <= box.getBoundingClientRect().right;
  });
  expect(tooltipFits).toBe(true);
  await expect(report.locator("#summary")).toContainText("6 个独立配置点");
  const count = report.locator("#data-body tr");
  const requests = report.locator("#values-x");
  const pd = report.locator("#values-color");
  await requests.getByRole("checkbox", { name: "2" }).uncheck();
  await expect(count).toHaveCount(4);
  await pd.getByRole("checkbox", { name: "false" }).uncheck();
  await expect(count).toHaveCount(2);
  await requests.getByRole("checkbox", { name: "1" }).uncheck();
  await expect(count).toHaveCount(1);
  await expect(requests.getByRole("checkbox", { name: "3" })).toBeDisabled();
  await expect(requests.getByRole("checkbox", { name: "3" })).toBeChecked();
  await requests.getByRole("checkbox", { name: "1" }).check();
  await expect(count).toHaveCount(2);
  await report.locator("#metric").selectOption("ttft_mean");
  await expect(count).toHaveCount(2);
  const csvPromise = report.waitForEvent("download");
  await report.getByRole("button", { name: "导出筛选数据 CSV" }).click();
  const csv = await csvPromise;
  expect((await readFile(await csv.path(), "utf8")).trim().split(/\r?\n/)).toHaveLength(3);
  await report.close();
});
