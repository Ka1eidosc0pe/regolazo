// The scripted demo path for the video (< 2 min), deterministic against the real server:
// live play -> flagged rewind (mo_08) -> pick the open teammate -> commit -> alternate reality -> verdict ->
// 1,000 realities -> back to live (the real goal plays). Each beat holds on screen for a few seconds.
// Screenshots of every beat go to $REGOLAZO_DATA_DIR/demo/ (else test-results/demo/); the video is copied there too.
//   REGOLAZO_API=http://127.0.0.1:8000 REGOLAZO_DATA_DIR=... npx playwright test e2e/demo.spec.ts --project chromium
// Optional: DEMO_MOMENT=mo_10 (90th-minute header), DEMO_HOLD_MS=3500 (seconds per beat).
import { mkdirSync, copyFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { openLive, phase, requireLiveServer } from "./live.ts";

requireLiveServer();
test.use({
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: 1,
  video: { mode: "on", size: { width: 1920, height: 1080 } },
});

const OUT = join(process.env.REGOLAZO_DATA_DIR ?? "test-results", "demo");
const MOMENT = process.env.DEMO_MOMENT ?? "mo_08";
const HOLD = Number(process.env.DEMO_HOLD_MS ?? 3500);

async function beat(page: Page, name: string, holdMs = HOLD): Promise<void> {
  await page.waitForTimeout(holdMs);
  await page.screenshot({ path: join(OUT, `${name}.jpg`), type: "jpeg", quality: 88 });
}

test("demo: live -> rewind -> rewrite -> verdict -> 1,000 realities -> live", async ({ page }, info) => {
  test.setTimeout(240_000);
  mkdirSync(OUT, { recursive: true });
  const moments = await (await page.request.get("/api/moments")).json() as { id: string; t: number }[];
  const m = moments.find((x) => x.id === MOMENT);
  expect(m, `moment ${MOMENT}`).toBeTruthy();

  // 1. live broadcast, ~12 s before the moment
  await openLive(page, Math.max(0, m!.t - 12));
  await beat(page, "01_live_broadcast", 7000);
  await page.waitForTimeout(3500);

  // 2. the flagged moment: click its marker on the timeline
  await page.getByTestId(`marker-${MOMENT}`).click();
  await expect.poll(() => phase(page), { timeout: 20_000 }).toBe("choose");
  await beat(page, "02_rewind_freeze", HOLD + 1000);

  // 3. pick the open teammate (click the ring on the pitch)
  const target = await page.evaluate(async () => window.regolazoDemo!.pickOpenTeammate());
  expect(target).toBeTruthy();
  await expect(page.getByTestId("commit")).toBeEnabled({ timeout: 10_000 });
  await beat(page, "03_pick_teammate");

  // 4. commit: the alternate reality plays in full colour, reality as cyan ghosts
  await page.getByTestId("commit").click();
  await expect.poll(() => phase(page), { timeout: 15_000 }).toBe("playing");
  await beat(page, "04_alternate_reality", 2500);

  // 5. verdict
  await page.evaluate(() => window.regolazoDemo!.verdict());
  await beat(page, "05_verdict", HOLD + 1000);

  // 6. 1,000 realities
  await page.getByTestId("run-realities").click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(OUT, "06a_realities_running.jpg"), type: "jpeg", quality: 88 });
  await expect(page.getByTestId("distribution")).toHaveAttribute("data-n", "1000", { timeout: 60_000 });
  await beat(page, "06_realities", HOLD + 1500);

  // 7. back to live: reality resumes a few seconds before the moment
  await page.getByTestId("back-to-live").click();
  await expect.poll(() => phase(page)).toBe("idle");
  await beat(page, "07_back_to_live", 9000);

  await page.close();
  const video = page.video();
  if (video) copyFileSync(await video.path(), join(OUT, `regolazo_demo_${MOMENT}.webm`));
  info.annotations.push({ type: "demo", description: OUT });
});
