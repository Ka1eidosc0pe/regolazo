// Visual regression: the dev clip at fixed match times and cameras, compared with approved baselines
// (e2e/__screenshots__/, rendered on the M3 Pro GPU). Update after an approved art change with:
//   npx playwright test --project visual --update-snapshots
import { createHash } from "node:crypto";
import { expect, test, type Page } from "@playwright/test";

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
    // record mode: no live loop running before the jump, so the frame depends only on (t, camera)
    await page.goto("/?record=1&w=1280&h=720");
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

async function record(page: Page, n: number): Promise<{ img: string[]; labels: string[] }> {
  await page.evaluate(() => window.regolazo!.recorder.start({ start: 1531, fps: 25, format: "png" }));
  const img: string[] = [];
  const labels: string[] = [];
  for (let i = 0; i < n; i++) {
    const f = await page.evaluate((k) => window.regolazo!.recorder.frame(k), i);
    img.push(createHash("sha256").update(f.image).digest("hex").slice(0, 16));
    labels.push(createHash("sha256").update(JSON.stringify(f.labels)).digest("hex").slice(0, 16));
  }
  return { img, labels };
}

async function fresh(page: Page): Promise<void> {
  await page.goto("/?record=1&w=640&h=360");
  await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 90_000 });
}

test("recording is deterministic: twice in one page, and in two fresh pages, frames and labels are identical", async ({ page, browser }) => {
  await fresh(page);
  const a = await record(page, 10);
  const b = await record(page, 10);
  const other = await browser.newPage({ viewport: { width: 640, height: 360 } });
  await fresh(other);
  const c = await record(other, 10);
  expect(b).toEqual(a);
  expect(c).toEqual(a);
  expect(new Set(a.img).size).toBe(10);
});

test("recording after other playback: labels identical (images may differ in a few GPU edge pixels)", async ({ page, browser }) => {
  await fresh(page);
  // play something else first, so any leaked state (phases, blends, ball spin, net, camera, grain) would show
  await page.evaluate(() => {
    const r = window.regolazo!;
    r.setCamera("replay", "H08");
    r.jumpTo(1540, 3);
  });
  const a = await record(page, 10);
  const other = await browser.newPage({ viewport: { width: 640, height: 360 } });
  await fresh(other);
  const c = await record(other, 10);
  expect(a.labels).toEqual(c.labels);
});

// The turf is rough and weakly specular: from the low broadcast angle the far corner used to mirror a floodlight
// bank and turn white. Render only the pitch and measure the share of near-white (min channel > 150) pitch pixels;
// painted lines are legitimately white but cover only a few percent of the frame.
for (const t of [T0 + 13.5, T0 + 15.0, T0 + 6.4]) {
  test(`pitch does not wash out under the floodlights (broadcast, t=${(t - T0).toFixed(1)})`, async ({ page }) => {
    await page.goto("/?record=1&w=1280&h=720");
    await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 90_000 });
    const share = await page.evaluate((time) => {
      const r = window.regolazo!;
      r.setClean(true);
      r.setCamera("broadcast", null);
      r.jumpTo(time, 2.5);
      const scene = r.app.scene;
      const hidden = scene.children.filter((o) => o.visible && o.name !== "pitch" && o.name !== "lighting");
      for (const o of hidden) o.visible = false;
      r.app.render();
      const c = r.app.renderer.domElement;
      const cx = document.createElement("canvas");
      cx.width = c.width;
      cx.height = c.height;
      const ctx = cx.getContext("2d")!;
      ctx.drawImage(c, 0, 0);
      const d = ctx.getImageData(0, 0, c.width, c.height).data;
      let pitch = 0;
      let white = 0;
      for (let i = 0; i < d.length; i += 16) {
        const [rr, gg, bb] = [d[i]!, d[i + 1]!, d[i + 2]!];
        if (rr + gg + bb < 40) continue; // background (everything but the pitch is hidden)
        pitch++;
        if (Math.min(rr, gg, bb) > 150) white++;
      }
      for (const o of hidden) o.visible = true;
      return white / Math.max(1, pitch);
    }, t);
    expect(share).toBeLessThan(0.05);
  });
}
