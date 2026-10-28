// Narration captions, the mode / language pickers (ProfileCmd) and the Spotter accuracy panel, against the live
// server. Copes with no narratives (another component produces them): a Narrative is injected through the demo hook.
import { expect, test } from "@playwright/test";
import { openLive, requireLiveServer } from "./live.ts";

requireLiveServer();

test("captions render narratives; pickers switch mode and language; accuracy panel hidden without its file", async ({ page, request }) => {
  await openLive(page, 600);
  await expect(page.getByTestId("profile-picker")).toBeVisible();
  await expect(page.locator(".pp-btn.on[data-value=casual]")).toBeVisible();
  await expect(page.locator(".pp-btn.on[data-value=en]")).toBeVisible();

  await page.locator(".pp-btn[data-value=analyst]").click();
  await page.locator(".pp-btn[data-value=es]").click();
  await expect(page.locator(".pp-btn.on[data-value=es]")).toBeVisible();
  expect(await page.evaluate(() => window.regolazoDemo!.profile())).toEqual({ mode: "analyst", language: "es" });

  await page.evaluate(() =>
    window.regolazoDemo!.narrate({
      schema_version: "1.0.0",
      id: "nar_test",
      kind: "live_commentary",
      t: 600,
      language: "es",
      mode: "analyst",
      club_perspective: null,
      text: "Lopes encuentra el espacio entre líneas.",
      evidence: [],
      fact_check: { verified: true, attempts: 1, issues: [] },
      source_id: null,
      audio_url: null,
      voice: null,
    }),
  );
  await expect(page.getByTestId("caption")).toHaveClass(/show/);
  await expect(page.getByTestId("caption")).toContainText("Lopes encuentra el espacio");

  const acc = await request.get("/api/files/spotter/accuracy.json");
  if (acc.status() === 404) await expect(page.locator(".acc-btn")).toBeHidden();
  else await expect(page.locator(".acc-btn")).toBeVisible();
});
