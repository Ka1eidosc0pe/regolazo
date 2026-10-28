// Renders the split-screen furniture: a base frame (background, headers, divider) and one caption panel per side
// (transparent PNG) from the caption text captured in raw/split_left.json and raw/split_right.json.
import { createRequire } from "node:module";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const { chromium } = createRequire(join(root, "web/package.json"))("playwright");
const out = join(root, "data/video/cards");
mkdirSync(out, { recursive: true });
const fonts = join(root, "web/public/fonts");
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const cap = (n) => JSON.parse(readFileSync(join(root, `data/video/raw/${n}.json`), "utf8")).captions[0];
const left = cap("split_left");
const right = cap("split_right");

const css = `
@font-face{font-family:B;src:url("file://${fonts}/BarlowCondensed-Bold.ttf");font-weight:700}
@font-face{font-family:B;src:url("file://${fonts}/BarlowCondensed-ExtraBold.ttf");font-weight:800}
*{margin:0;box-sizing:border-box}
body{width:1920px;height:1080px;font-family:B,"Arial Narrow",sans-serif;color:#e8ecf4;background:transparent}
.hd{position:absolute;top:0;height:70px;width:960px;display:flex;align-items:center;gap:18px;padding:0 30px;font-weight:800;font-size:46px;letter-spacing:.08em;text-transform:uppercase}
.hd span{color:#05070d;background:#d7ff3a;padding:0 14px;line-height:52px}
.hd i{font-style:normal;color:#9aa6bd;font-size:32px;font-weight:700;letter-spacing:.12em}
.cap{position:absolute;top:810px;height:250px;width:900px;padding:16px 22px;border-left:5px solid #d7ff3a;background:rgba(255,255,255,.05);font-weight:700;font-size:36px;line-height:1.16;color:#e8ecf4}
.cap small{display:block;font-size:22px;letter-spacing:.2em;color:#d7ff3a;text-transform:uppercase;margin-bottom:6px}
.div{position:absolute;left:959px;top:0;width:2px;height:1080px;background:#d7ff3a}
`;
const pages = {
  split_base: `<style>${css} body{background:#05070d}</style>
    <div class="hd" style="left:0"><span>Analyst</span>EN<i>precise, number-led</i></div>
    <div class="hd" style="left:960px"><span>Casual</span>ES<i>plain, conversational</i></div><div class="div"></div>`,
  split_cap_left: `<style>${css}</style><div class="cap" style="left:30px"><small>Analyst &middot; English</small>${esc(left.text)}</div>`,
  split_cap_right: `<style>${css}</style><div class="cap" style="left:990px"><small>Casual &middot; Espa&ntilde;ol</small>${esc(right.text)}</div>`,
};
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
for (const [name, html] of Object.entries(pages)) {
  const f = join(out, `${name}.html`);
  writeFileSync(f, `<!doctype html><meta charset="utf-8">${html}`);
  await page.goto(`file://${f}`);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(150);
  await page.screenshot({ path: join(out, `${name}.png`), omitBackground: name !== "split_base" });
  console.log("rendered", name);
}
await browser.close();
// caption hold mirrors the UI's captionHoldS()
const hold = (t) => Math.min(9, Math.max(3.5, t.split(/\s+/).length * 0.38));
writeFileSync(join(out, "split_timing.json"), JSON.stringify({ left: { at: left.at, hold: hold(left.text) }, right: { at: right.at, hold: hold(right.text) } }));
