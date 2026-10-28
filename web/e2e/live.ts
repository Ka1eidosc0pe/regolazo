// Shared setup for the e2e specs that need the live engine (`python -m server` with the real simulator).
// The preview server proxies /api and /ws to REGOLAZO_API (default http://127.0.0.1:8000); see vite.config.ts.
import { test, type Page } from "@playwright/test";

const gpuArgs =
  process.platform === "darwin" ? ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"] : ["--ignore-gpu-blocklist"];

/** Skip on CI and whenever the live engine is not reachable through the proxy. */
export function requireLiveServer(): void {
  test.skip(!!process.env.CI, "needs the live engine and the simulator");
  test.use({ launchOptions: { args: gpuArgs } });
  test.beforeEach(async ({ request }) => {
    const ok = await request
      .get("/api/moments", { timeout: 5000 })
      .then((r) => r.ok())
      .catch(() => false);
    test.skip(!ok, "live engine not reachable (start `python -m server` and set REGOLAZO_API)");
  });
}

export async function openLive(page: Page, t: number): Promise<void> {
  await page.goto(`/?live=1&t=${t}`);
  await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 90_000 });
}

export async function phase(page: Page): Promise<string> {
  return page.evaluate(() => window.regolazoDemo?.phase() ?? "none");
}
