// Frame-rate check for the live view: 1920 x 1080 canvas, clip playing, broadcast camera.
//   node scripts/fps.ts [--headless] [--seconds 10] [--quality high|low]
// Headed by default (the real compositor, vsync-limited to the display refresh). Reports mean fps and frame-time
// percentiles from requestAnimationFrame timestamps, plus the GPU renderer string.
import { startHarness } from "./lib/harness.ts";

const args = process.argv.slice(2);
const headed = !args.includes("--headless");
const sIdx = args.indexOf("--seconds");
const seconds = sIdx >= 0 ? Number(args[sIdx + 1]) : 10;
const qIdx = args.indexOf("--quality");
const quality = qIdx >= 0 ? args[qIdx + 1] : "high";
const dIdx = args.indexOf("--debug");
const debug = dIdx >= 0 ? (JSON.parse(args[dIdx + 1]!) as { hide?: string[]; shadows?: boolean; bloom?: boolean }) : null;

const h = await startHarness({ headed });
try {
  const page = await h.open(`w=1920&h=1080&quality=${quality}`, { width: 1920, height: 1080 });
  if (debug) await page.evaluate((d) => window.regolazo!.debug(d), debug);
  await page.waitForTimeout(3000); // warm-up: shader compile, texture upload
  const result = await page.evaluate(async (secs: number) => {
    const times: number[] = [];
    await new Promise<void>((resolve) => {
      const start = performance.now();
      const tick = (now: number) => {
        times.push(now);
        if (now - start < secs * 1000) requestAnimationFrame(tick);
        else resolve();
      };
      requestAnimationFrame(tick);
    });
    const dts = times.slice(1).map((t, i) => t - times[i]!).sort((a, b) => a - b);
    const pct = (p: number) => dts[Math.min(dts.length - 1, Math.floor(p * dts.length))]!;
    const total = (times[times.length - 1]! - times[0]!) / 1000;
    const info = window.regolazo!.info();
    return {
      frames: dts.length,
      seconds: +total.toFixed(2),
      mean_fps: +(dts.length / total).toFixed(1),
      p50_ms: +pct(0.5).toFixed(2),
      p95_ms: +pct(0.95).toFixed(2),
      p99_ms: +pct(0.99).toFixed(2),
      canvas: `${info.width}x${info.height}`,
      renderer: info.renderer,
    };
  }, seconds);
  console.log(JSON.stringify({ headed, quality, debug, ...result }));
} finally {
  await h.close();
}
