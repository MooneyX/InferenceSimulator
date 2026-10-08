import { expect, test } from "@playwright/test";

test("runs and renders a 1,000-request simulation", async ({ page }) => {
  test.setTimeout(360_000);
  const browserErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => browserErrors.push(error.message));

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "推理服务仿真器" })).toBeVisible();
  await expect(page.getByLabel("请求数量")).toHaveValue("1000");

  const responsePromise = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/simulate") && response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "运行仿真" }).click();

  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const submittedSession = response.request().postDataJSON().workload.synthetic.session;
  const result = await response.json();
  const returnedSession = result.meta.config.workload.synthetic.session;
  expect(result.meta.num_completed + result.meta.num_aborted + result.meta.num_skipped).toBe(1000);
  await expect(page.getByLabel("请求结果状态")).toContainText(`已完成 ${result.meta.num_completed}`);
  expect(submittedSession).not.toHaveProperty("allow_cross_session_shared_prefix");
  expect(returnedSession).not.toHaveProperty("allow_cross_session_shared_prefix");
  await expect(page.getByRole("heading", { name: "仿真指标" })).toBeVisible();

  const chart = page.getByRole("img", {
    name: `${result.meta.num_completed} 个请求的排队、Prefill 和 Decode 生命周期甘特图`,
  });
  await expect(chart).toBeVisible();
  await expect(chart.locator("canvas")).toHaveCount(1);
  await expect(page.getByText(`共 ${result.meta.num_completed} 个请求，按到达时间排序`)).toBeVisible();
  expect(browserErrors).toEqual([]);
});

test("reports browser response for 100/1k/10k in both P/D modes", async ({ page }) => {
  test.setTimeout(360_000);
  await page.goto("/");
  for (const requests of [100, 1000, 10000]) {
    for (const exclusive of [true, false]) {
      await page.getByLabel("请求数量").fill(String(requests));
      const toggle = page.getByLabel(/Prefill \/ Decode 分步执行/);
      if (exclusive) {
        await toggle.check();
      } else {
        await toggle.uncheck();
      }
      const responsePromise = page.waitForResponse((response) =>
        response.url().endsWith("/api/simulate") && response.request().method() === "POST",
      );
      const start = await page.evaluate(() => performance.now());
      await page.getByRole("button", { name: "运行仿真" }).click();
      const response = await responsePromise;
      const responseMs = (await page.evaluate(() => performance.now())) - start;
      expect(response.status()).toBe(200);
      const result = await response.json();
      expect(result.meta.num_requests).toBe(requests);
      expect(result.meta.num_completed + result.meta.num_aborted + result.meta.num_skipped).toBe(requests);
      await expect(page.getByLabel("请求结果状态")).toContainText(`已完成 ${result.meta.num_completed}`);
      await expect(page.getByRole("img", {
        name: `${result.meta.num_completed} 个请求的排队、Prefill 和 Decode 生命周期甘特图`,
      })).toBeVisible();
      const renderedMs = (await page.evaluate(() => performance.now())) - start;
      console.log(`requests=${requests} pd_mutually_exclusive=${exclusive} response_ms=${responseMs.toFixed(1)} chart_ms=${renderedMs.toFixed(1)}`);
    }
  }
});

test("keeps the configuration page within a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "负载配置" })).toBeVisible();
  await expect(page.getByRole("button", { name: "运行仿真" })).toBeVisible();

  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth + 1,
  );
  expect(hasHorizontalOverflow).toBe(false);
});
