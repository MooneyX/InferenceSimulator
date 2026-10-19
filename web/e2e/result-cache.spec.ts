import { expect, test } from "@playwright/test";

test("结果缓存命中和强制重新计算", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByLabel("请求数量").fill("3");
  const firstResponse = page.waitForResponse((response) =>
    response.url().endsWith("/api/simulate?force=true") && response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "强制重新计算" }).click();
  const first = await firstResponse;
  expect(first.status()).toBe(200);
  expect(first.headers()["x-simulation-cache"]).toBe("miss");
  const original = await first.json();
  await expect(page.getByLabel("结果缓存状态")).toContainText("本次结果为重新计算");
  await expect(page.getByRole("heading", { name: "仿真指标" })).toBeVisible();

  const cachedResponse = page.waitForResponse((response) =>
    response.url().endsWith("/api/simulate") && response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "运行仿真" }).click();
  const cached = await cachedResponse;
  expect(cached.status()).toBe(200);
  expect(cached.headers()["x-simulation-cache"]).toBe("hit");
  expect(await cached.json()).toEqual(original);
  await expect(page.getByLabel("结果缓存状态")).toContainText("结果缓存命中");
  await expect(page.getByLabel("结果缓存状态")).toContainText("缓存查找/读取");
  await expect(page.getByRole("heading", { name: "仿真指标" })).toBeVisible();

  await page.route("**/api/simulation-cache", (route) => route.fulfill({ status: 204 }));
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "清空结果缓存" }).click();
  await expect(page.getByText("结果缓存已清空，下一次运行将重新计算。")).toBeVisible();
  await expect(page.getByRole("heading", { name: "仿真指标" })).toBeVisible();
  expect(errors).toEqual([]);
});
