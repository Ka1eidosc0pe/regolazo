// Visual regression: the dev clip at fixed match times and cameras, compared with approved baselines
// (e2e/__screenshots__/, rendered on the M3 Pro GPU). Update after an approved art change with:
//   npx playwright test --project visual --update-snapshots
import { expect, test } from "@playwright/test";

test.skip(!!process.env.CI, "baselines are GPU renders; CI has only SwiftShader");

const T0 = 1520;
const SHOTS: { name: string; t: number; cam: "broadcast" | "tactical" | "replay"; target?: string }[] = [
  { name: "broadcast-buildup", t: T0 + 6.4, cam: "broadcast" },
  { name: "broadcast-wing", t: T0 + 13.0, cam: "broadcast" },
  { name: "broadcast-header", t: T0 + 16.24, cam: "broadcast" },
  { name: "tactical", t: T0 + 11.5, cam: "tactical" },
  { name: "replay-finish", t: T0 + 18.4, cam: "replay", target: "H08" },
];

for (const s of SHOTS) {
  test(`frame ${s.name}`, async ({ page }) => {
    await page.goto("/?w=1280&h=720&paused=1");
    await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 90_000 });
    const info = await page.evaluate((shot) => {
      const r = window.regolazo!;
      r.setClean(true);
      r.setCamera(shot.cam, shot.target ?? null);
      r.jumpTo(shot.t, 2.5);
      return r.info();
    }, s);
    expect(info.renderer).not.toContain("SwiftShader");
    await expect(page.locator("#stage")).toHaveScreenshot(`${s.name}.png`);
  });
}
