import { defineConfig, devices } from "@playwright/test";

// Visual baselines are rendered on the real GPU (ANGLE/Metal on the M3 Pro). They are skipped on CI, where Chromium
// falls back to SwiftShader and the images would differ (see e2e/visual.spec.ts).
const gpuArgs =
  process.platform === "darwin" ? ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"] : ["--ignore-gpu-blocklist"];

export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: "http://localhost:4173" },
  timeout: 120_000,
  expect: { toHaveScreenshot: { maxDiffPixelRatio: 0.01, threshold: 0.15 } },
  snapshotPathTemplate: "{testDir}/__screenshots__/{testFilePath}/{arg}-{platform}{ext}",
  webServer: {
    command: "npm run build && npm run preview -- --port 4173 --strictPort",
    port: 4173,
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: "chromium", testIgnore: /visual\.spec\.ts/, use: { ...devices["Desktop Chrome"] } },
    {
      name: "visual",
      testMatch: /visual\.spec\.ts/,
      use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1, launchOptions: { args: gpuArgs } },
    },
  ],
});
