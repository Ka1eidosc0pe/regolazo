// Rewind & Rewrite against the real server: open a flagged moment from its timeline marker, pick a teammate,
// commit, check the verdict numbers against the backend, run 1,000 realities and return to live.
// Skipped on CI (needs the simulator); see e2e/live.ts.
import { expect, test } from "@playwright/test";
import { openLive, phase, requireLiveServer } from "./live.ts";

requireLiveServer();
test.use({ viewport: { width: 1600, height: 900 } });

test("rewind a flagged moment, commit, verdict, 1,000 realities, back to live", async ({ page }) => {
  test.setTimeout(180_000);
  await openLive(page, 4448);
  await expect(page.getByTestId("scorebug")).toBeVisible();
  await expect(page.getByTestId("marker-mo_08")).toBeAttached();

  // flagged rewind from the timeline marker
  await page.getByTestId("marker-mo_08").click();
  await expect.poll(() => phase(page), { timeout: 20_000 }).toBe("choose");
  await expect(page.getByTestId("rewind-banner")).toContainText("What would you have done?");
  // probabilities stay hidden until commit (analyst toggle shows them)
  await expect(page.locator(".rw-ring-p").first()).toBeHidden();
  await page.getByTestId("analyst-toggle").click();
  await expect(page.locator(".rw-ring.mate .rw-ring-p").first()).toBeVisible();
  await page.getByTestId("analyst-toggle").click();

  // click a teammate on the pitch
  const mate = await page.evaluate(() => {
    const m = window.regolazoDemo!.moments().find((x) => x.id === "mo_08")!;
    const o = m.options.filter((q) => q.intent.kind === "pass_player" && q.viable && q.p_success >= 0.5);
    o.sort((a, b) => b.value_if_success - a.value_if_success);
    return o[0]!.intent.target_player!;
  });
  await page.getByTestId(`mate-${mate}`).click();
  await expect(page.getByTestId("commit")).toBeEnabled({ timeout: 10_000 });

  const rolloutResp = page.waitForResponse((r) => r.url().includes("/api/rollout") && r.request().method() === "POST");
  const t0 = Date.now();
  await page.getByTestId("commit").click();
  const rollout = (await (await rolloutResp).json()) as { evaluation: { p_success: number }; outcome: { label: string }; frames: unknown[] };
  await expect.poll(() => phase(page), { timeout: 15_000 }).toBe("playing");
  const latency = Date.now() - t0;
  console.log(`commit -> playback ${latency} ms, ${rollout.frames.length} frames, outcome ${rollout.outcome.label}`);
  expect(rollout.frames.length).toBeGreaterThan(10);

  // verdict numbers match the backend
  await expect(page.locator(".rw-card.show")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("verdict")).toContainText(`${Math.round(rollout.evaluation.p_success * 100)}%`);
  await expect(page.locator(".vd-framing")).not.toBeEmpty();

  // 1,000 realities
  const distResp = page.waitForResponse((r) => r.url().includes("/api/distribution"), { timeout: 60_000 });
  await page.getByTestId("run-realities").click();
  const dist = (await (await distResp).json()) as { n: number; p_primary_success: number; elapsed_ms: number };
  await expect(page.getByTestId("distribution")).toHaveAttribute("data-n", "1000", { timeout: 10_000 });
  await expect(page.getByTestId("distribution")).toContainText(`${Math.round(dist.p_primary_success * 100)}%`);
  console.log(`1,000 realities: ${dist.elapsed_ms.toFixed(0)} ms, p_success ${dist.p_primary_success}`);

  await page.getByTestId("back-to-live").click();
  await expect.poll(() => phase(page)).toBe("idle");
});
