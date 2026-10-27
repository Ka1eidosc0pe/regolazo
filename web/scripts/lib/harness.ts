// Shared Playwright harness: a Vite dev server plus Chromium on the real GPU (ANGLE/Metal on macOS).
import { chromium, type Browser, type Page } from "@playwright/test";
import { createServer, type ViteDevServer } from "vite";
import { fileURLToPath } from "node:url";

export const WEB_ROOT = fileURLToPath(new URL("../../", import.meta.url));
export const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));

export const GPU_ARGS =
  process.platform === "darwin"
    ? ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"]
    : ["--ignore-gpu-blocklist", "--enable-gpu", "--use-gl=angle"];

export interface Harness {
  server: ViteDevServer;
  browser: Browser;
  base: string;
  open(query: string, size?: { width: number; height: number }): Promise<Page>;
  close(): Promise<void>;
}

export async function startHarness(opts: { headed?: boolean; port?: number; unthrottled?: boolean } = {}): Promise<Harness> {
  const port = opts.port ?? 5180 + Math.floor(Math.random() * 300);
  const server = await createServer({ root: WEB_ROOT, server: { port, strictPort: true }, logLevel: "error" });
  await server.listen();
  const base = `http://localhost:${port}`;
  // headed Chromium already uses the GPU; the ANGLE/Metal flags are for headless
  const extra = opts.unthrottled ? ["--disable-gpu-vsync", "--disable-frame-rate-limit"] : [];
  const browser = await chromium.launch({ headless: !opts.headed, args: [...(opts.headed ? ["--ignore-gpu-blocklist"] : GPU_ARGS), ...extra] });
  browser.on("disconnected", () => console.log("[harness] browser disconnected"));
  return {
    server,
    browser,
    base,
    async open(query, size = { width: 1920, height: 1080 }) {
      const page = await browser.newPage({ viewport: size, deviceScaleFactor: 1 });
      page.on("console", (m) => {
        if (m.type() === "error" || m.type() === "warning") console.log(`[page ${m.type()}] ${m.text()}`);
      });
      let fail: (e: Error) => void = () => undefined;
      const failed = new Promise<never>((_, reject) => (fail = reject));
      page.on("pageerror", (e) => {
        console.log(`[page error] ${e.message}`);
        fail(e);
      });
      await page.goto(`${base}/?${query}`);
      await Promise.race([page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 120_000 }), failed]);
      return page;
    },
    async close() {
      await browser.close();
      await server.close();
    },
  };
}
