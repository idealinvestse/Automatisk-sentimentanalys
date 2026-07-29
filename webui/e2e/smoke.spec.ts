import { test, expect } from "@playwright/test";

import { stubDashboardApi } from "./helpers/mock-api";

/**
 * Smoke tests: each route loads without console errors and renders its H1.
 * Stubs dashboard API routes so CI does not require a live FastAPI process.
 */

const ROUTES = [
  { path: "/", h1: "Översikt" },
  { path: "/analytics", h1: "Analys" },
  { path: "/analysis", h1: "Analysdetaljer" },
  { path: "/agents", h1: "Agentprestanda" },
  { path: "/insights", h1: "Fas 4" },
  { path: "/executive", h1: "Executive Insights" },
  { path: "/edge", h1: "Edge AI" },
  { path: "/transcription", h1: "Transkribering" },
  { path: "/testlab", h1: "Testlabb" },
];

for (const route of ROUTES) {
  test(`${route.path} renders H1 "${route.h1}"`, async ({ page }) => {
    await stubDashboardApi(page);
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    await page.goto(route.path);
    await expect(page.getByRole("heading", { level: 1 })).toContainText(route.h1);
    expect(errors, `Console errors on ${route.path}: ${errors.join("; ")}`).toEqual([]);
  });
}

test("sidebar navigation works", async ({ page }) => {
  await stubDashboardApi(page);
  await page.goto("/");
  await page.getByRole("link", { name: /Analys & Trender/i }).click();
  await expect(page).toHaveURL(/\/analytics/);
});

test("theme toggle switches dark mode", async ({ page }) => {
  await stubDashboardApi(page);
  await page.goto("/");
  const html = page.locator("html");
  const before = await html.getAttribute("class");
  await page.getByRole("button", { name: /theme|tema|mörk|ljus/i }).first().click();
  const after = await html.getAttribute("class");
  expect(before).not.toEqual(after);
});
