import { expect, test } from "@playwright/test";

test("runs 10 requests through the unified simulation endpoint", async ({ page }) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("radio", { name: /自动解析|手动配置/ })).toHaveCount(0);
  await page.getByLabel("请求数量").fill("10");
  const responsePromise = page.waitForResponse((response) =>
    response.url().endsWith("/api/simulate") && response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "运行仿真" }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const config = response.request().postDataJSON();
  expect(config.workload.synthetic.num_requests).toBe(10);
  expect(config.inference).not.toHaveProperty("mode");
  expect(config.shared_l3).not.toHaveProperty("mode");
  expect(config.inference.timing).toEqual({ mode: "analytical" });
  expect(config).toHaveProperty("scheduler.max_running_requests");
  expect(config).not.toHaveProperty("engine");
  expect(config).not.toHaveProperty("perf_model");
  const result = await response.json();
  expect(result.meta.config).toMatchObject(config);
  expect(result.meta.num_requests).toBe(10);
  expect(result.meta.num_completed + result.meta.num_aborted + result.meta.num_skipped).toBe(10);
  expect(result.resolution).not.toBeNull();
  expect(result.resolution).not.toHaveProperty("config");
  expect(result.resolution).not.toHaveProperty("model");
  await expect(page.getByRole("heading", { name: "仿真指标" })).toBeVisible();
  await expect(page.getByLabel("解析配置与成本")).toContainText("自动解析");
  await expect(page.getByLabel("请求结果状态")).toContainText(`已完成 ${result.meta.num_completed}`);
  if (result.meta.num_completed > 0) {
    await expect(page.getByRole("img", {
      name: `${result.meta.num_completed} 个请求的排队、Prefill 和 Decode 生命周期甘特图`,
    })).toBeVisible();
  }
  expect(browserErrors).toEqual([]);
});

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
  const submittedConfig = response.request().postDataJSON();
  expect(submittedConfig.inference).not.toHaveProperty("mode");
  expect(submittedConfig).toHaveProperty("scheduler.pd_mutually_exclusive");
  expect(submittedConfig).not.toHaveProperty("engine");
  expect(submittedConfig).not.toHaveProperty("perf_model");
  const submittedSession = submittedConfig.workload.synthetic.session ?? {};
  const result = await response.json();
  expect(result.meta.config).toMatchObject(submittedConfig);
  expect(result.resolution).not.toBeNull();
  const returnedSession = result.meta.config.workload.synthetic.session ?? {};
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

test("edits HY4 preset and exposes complete attention group parameters", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("模型选择").selectOption("tencent/Hy4-preview");
  await expect(page.getByLabel("第 1 组注意力方式")).toHaveValue("mla_dsa");
  await expect(page.getByLabel("第 1 组层数")).toHaveValue("78");
  await expect(page.getByLabel("第 1 组Q 压缩维度")).toBeEnabled();
  await expect(page.getByLabel("第 1 组KV 压缩维度")).toBeEnabled();
  await expect(page.getByLabel("第 1 组选取 token 数")).toBeEnabled();
  await expect(page.getByLabel("共享专家数（0 为关闭）")).toBeEnabled();
  await page.getByLabel("隐藏维度", { exact: true }).fill("4096");
  await page.getByLabel("第 1 组选取 token 数").fill("1024");
  await expect(page.getByLabel("模型选择")).toHaveValue("custom");
  await page.getByText("辅助权重容量配置", { exact: true }).click();
  await expect(page.getByLabel("启用 iHC", { exact: true })).toBeEnabled();
  await expect(page.getByLabel("第 1 组启用 attention sink")).toBeEnabled();
  const responsePromise = page.waitForResponse((response) =>
    response.url().endsWith("/api/resolve") && response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "预览解析容量" }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const submittedConfig = response.request().postDataJSON();
  expect(submittedConfig.inference).not.toHaveProperty("mode");
  expect(submittedConfig).not.toHaveProperty("config");
  const submitted = submittedConfig.inference.model;
  expect(submitted.hidden_size).toBe(4096);
  expect(submitted.attention_groups[0].topk).toBe(1024);
  expect(submitted.attention_layout).toBe("grouped");
  expect(submitted.revision).toBeNull();
  expect(submitted.ihc).toEqual({ mult: 4, dtype: "fp32" });
  expect(submitted.router).toEqual({ dtype: "fp32", correction_bias: true, correction_dtype: "fp32" });
  await expect(page.getByRole("status")).toContainText("服务端单实例解析");
  await page.getByRole("button", { name: "添加注意力组" }).click();
  await expect(page.getByLabel("第 1 组层数")).toHaveValue("77");
  await expect(page.getByLabel("第 2 组注意力方式")).toHaveValue("gqa");
  await expect(page.getByLabel("第 2 组KV 头数")).toBeEnabled();
  await page.getByLabel("第 1 组层数").fill("78");
  await page.getByRole("button", { name: "删除第 2 组" }).click();
  await expect(page.getByLabel("第 2 组注意力方式")).toHaveCount(0);
});

test("shows Qwen attention defaults with consistent auxiliary form styling", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("模型选择")).toHaveValue("Qwen/Qwen3-8B");
  for (const preset of [
    { name: "Qwen/Qwen3-8B", layers: "36", heads: "32", kv: "8" },
    { name: "Qwen/Qwen3-235B-A22B", layers: "94", heads: "64", kv: "4" },
  ]) {
    await page.getByLabel("模型选择").selectOption(preset.name);
    await expect(page.getByLabel("第 1 组注意力方式")).toHaveValue("gqa");
    await expect(page.getByLabel("第 1 组层数")).toHaveValue(preset.layers);
    await expect(page.getByLabel("第 1 组 Query 头数")).toHaveValue(preset.heads);
    await expect(page.getByLabel("第 1 组KV 头数")).toHaveValue(preset.kv);
    await expect(page.getByLabel("第 1 组每头维度")).toHaveValue("128");
    await expect(page.getByLabel("第 1 组注意力输出门控")).not.toBeChecked();
  }
  const auxiliary = page.locator("details.auxiliary-config");
  await expect(auxiliary).not.toHaveAttribute("open", "");
  await auxiliary.locator("summary").click();
  await expect(auxiliary).toHaveAttribute("open", "");
  await expect(page.getByRole("checkbox", { name: "第 1 组 Q/K 归一化", exact: true })).toBeChecked();
  await expect(page.getByLabel("第 1 组启用 attention sink")).not.toBeChecked();
  const title = page.getByRole("heading", { name: "第 1 组注意力", exact: true });
  expect(await auxiliary.locator("summary").evaluate((element) => getComputedStyle(element).color))
    .toBe(await title.evaluate((element) => getComputedStyle(element).color));
  expect(await auxiliary.locator("summary").evaluate((element) => getComputedStyle(element).backgroundColor))
    .toBe(await title.locator("..").evaluate((element) => getComputedStyle(element).backgroundColor));
  await expect(auxiliary.locator("fieldset")).toHaveCSS("border-top-width", "0px");
  await expect(page.getByRole("heading", { name: "第 1 组注意力", exact: true }))
    .toHaveCSS("font-size", "12.8px");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false);
  await auxiliary.locator("summary").click();
  await expect(auxiliary).not.toHaveAttribute("open", "");
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
