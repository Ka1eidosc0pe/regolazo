// Records live-broadcast captures of the running app (vite on :5173 proxying to the server on :8000).
//   node scripts/video/capture.mjs <name> --t 1222 --dur 18 [--mode analyst] [--lang es] [--spotter-at 13]
// Writes data/video/raw/<name>.webm plus <name>.json (seconds from the start of play at which each caption showed).
import { createRequire } from "node:module";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const { chromium } = createRequire(join(root, "web/package.json"))("playwright");

const args = process.argv.slice(2);
const name = args[0];
const opt = (k, d) => (args.includes(`--${k}`) ? args[args.indexOf(`--${k}`) + 1] : d);
const t = Number(opt("t", "1222"));
const dur = Number(opt("dur", "18"));
const mode = opt("mode", null);
const lang = opt("lang", null);
const spotterAt = opt("spotter-at", null);
const base = opt("base", "http://localhost:5173");
const out = join(root, "data/video/raw");
mkdirSync(out, { recursive: true });

const gpu = ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"];
const browser = await chromium.launch({ args: gpu });
const ctx = await browser.newContext({
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: 1,
  recordVideo: { dir: out, size: { width: 1920, height: 1080 } },
});
const page = await ctx.newPage();
const t0 = Date.now();
await page.goto(`${base}/?live=1&paused=1&t=${t}`);
await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 120_000 });
await page.waitForSelector(".pp-btn[data-key=mode]");
if (mode) await page.locator(`.pp-btn[data-value=${mode}]`).click();
if (lang) await page.locator(`.pp-btn[data-value=${lang}]`).click();
await page.waitForTimeout(800);
await page.evaluate((tt) => { window.regolazoDemo.seek(tt); window.regolazoDemo.play(); }, t);
const tPlay = Date.now();
const captions = [];
let shown = false;
let spotterDone = false;
while ((Date.now() - tPlay) / 1000 < dur) {
  const s = (Date.now() - tPlay) / 1000;
  const info = await page.evaluate(() => {
    const c = document.querySelector("[data-testid=caption]");
    return { show: !!c && c.classList.contains("show"), text: c?.querySelector(".cap-text")?.textContent ?? "" };
  });
  if (info.show && !shown) captions.push({ at: +s.toFixed(2), text: info.text });
  shown = info.show;
  if (spotterAt && !spotterDone && s >= Number(spotterAt)) {
    await page.locator(".acc-btn").click().catch(() => undefined);
    spotterDone = true;
  }
  await page.waitForTimeout(100);
}
const video = page.video();
await ctx.close();
await browser.close();
const dest = join(out, `${name}.webm`);
renameSync(await video.path(), dest);
// video clock starts at page creation; play started (tPlay - t0) s later
writeFileSync(join(out, `${name}.json`), JSON.stringify({ t, mode, lang, playStart: +((tPlay - t0) / 1000).toFixed(2), dur, captions }, null, 2));
console.log(name, "playStart", (tPlay - t0) / 1000, "captions", JSON.stringify(captions));
