// Renders the title / architecture / close cards in cards.html to PNG (1920x1080) with Playwright.
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const { chromium } = createRequire(join(root, "web/package.json"))("playwright");
const out = join(root, "data/video/cards");
mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
for (const c of ["hook1", "hook2", "wordmark", "arch", "close"]) {
  await page.goto(`file://${join(here, "cards.html")}?card=${c}`);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(200);
  await page.screenshot({ path: join(out, `${c}.png`) });
  console.log("card", c);
}
await browser.close();
