import { expect, test } from "@playwright/test";

import { stubDashboardApi } from "./helpers/mock-api";

test.beforeEach(async ({ page }) => {
  await stubDashboardApi(page);
});

test("table navigation opens a call detail", async ({ page }) => {
  await page.goto("/");

  const firstCall = page.getByRole("button", { name: /Öppna samtal/i }).first();
  await expect(firstCall).toBeVisible({ timeout: 15_000 });
  await firstCall.click();

  await expect(page).toHaveURL(/\/calls\/CALL-/, { timeout: 15_000 });
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15_000 });
});

test("unknown call id renders the Swedish not-found page", async ({ page }) => {
  await page.goto("/calls/not-a-call");

  await expect(page.getByRole("heading", { name: "Sidan kunde inte hittas" })).toBeVisible({
    timeout: 15_000,
  });
  await expect(page.getByRole("link", { name: "Till översikten" })).toBeVisible();
});

test("edge text analysis renders backend response", async ({ page }) => {
  await page.goto("/edge");

  await page.locator("textarea").fill("Tack för hjälpen");
  await page.getByRole("button", { name: "Kör edge-analys" }).click();

  await expect(page.getByText("Positiv kundrespons.")).toBeVisible();
  await expect(page.getByText("Intent: resolution")).toBeVisible();
});

test("testlab runs the partial pipeline path", async ({ page }) => {
  await page.goto("/testlab");

  await page.locator("textarea").fill(JSON.stringify([{ text: "Hej", speaker: "Agent" }]));
  await page.getByText(/Partial path/).click();
  await page.getByRole("button", { name: "Kör partial" }).click();

  await expect(page.getByText("Partial metadata")).toBeVisible();
});

test("empty ASR output explains why pipeline analysis was skipped", async ({ page }) => {
  await page.route("**/api/backend/transcribe", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ transcript: { segments: [] }, timestamp: new Date().toISOString() }),
    }),
  );
  await page.goto("/transcription");

  await page.locator("#audio-file").setInputFiles({
    name: "silent.wav",
    mimeType: "audio/wav",
    buffer: Buffer.from("RIFFtest"),
  });
  await page.getByRole("button", { name: "Ladda upp och transkribera" }).click();

  await expect(page.getByText(/innehöll inget tal som kan analyseras/i)).toBeVisible({ timeout: 15_000 });
});

test("upload, transcription, pipeline, and persistence complete as one flow", async ({ page }) => {
  await page.goto("/transcription");

  await page.locator("#audio-file").setInputFiles({
    name: "e2e.wav",
    mimeType: "audio/wav",
    buffer: Buffer.from("RIFFtest"),
  });
  await page.getByRole("button", { name: "Ladda upp och transkribera" }).click();

  await expect(page.getByText(/Transkribering och analys klar/)).toBeVisible({ timeout: 15_000 });
  await page.goto("/");
  await expect(page.getByText("e2e.wav")).toBeVisible({ timeout: 15_000 });
});

test("mobile navigation remains available", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto("/");

  const navigation = page.getByRole("navigation", { name: "Mobilnavigering" });
  await expect(navigation).toBeVisible();
  await navigation.getByRole("link", { name: /Transkribering/i }).click();
  await expect(page).toHaveURL(/\/transcription/);
});
