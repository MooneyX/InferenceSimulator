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
  const returnedSession = (await response.json()).meta.config.workload.synthetic.session;
  expect(submittedSession).not.toHaveProperty("allow_cross_session_shared_prefix");
  expect(returnedSession).not.toHaveProperty("allow_cross_session_shared_prefix");
  await expect(page.getByRole("heading", { name: "仿真指标" })).toBeVisible();

  const chart = page.getByRole("img", {
    name: "1000 个请求的排队、Prefill 和 Decode 生命周期甘特图",
  });
  await expect(chart).toBeVisible();
  await expect(chart.locator("canvas")).toHaveCount(1);
  await expect(page.getByText("共 1000 个请求，按到达时间排序")).toBeVisible();
  expect(browserErrors).toEqual([]);
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
