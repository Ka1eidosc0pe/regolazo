// Recording mode: headless Chromium (Metal GPU) renders a clip deterministically frame by frame with the broadcast
// camera and writes frames + one FrameLabels JSON per frame.
//
//   node scripts/record.ts [--clip /dev/clip_demo.json] [--out <dir>] [--fps 25] [--width 1920 --height 1080]
//                          [--start <t>] [--end <t>] [--max-frames N] [--camera broadcast|tactical|replay] [--png]
//
// Output (default $REGOLAZO_DATA_DIR/renders/<match_id>-<clip name>, REGOLAZO_DATA_DIR defaults to <repo>/data):
//   frames/000000.jpg ...   labels/000000.json ...   manifest.json
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { REPO_ROOT, startHarness } from "./lib/harness.ts";

const args = process.argv.slice(2);
const opt = (name: string, fallback?: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1]! : fallback;
};
const clip = opt("clip", "/dev/clip_demo.json")!;
const fps = Number(opt("fps", "25"));
const width = Number(opt("width", "1920"));
const height = Number(opt("height", "1080"));
const start = opt("start");
const end = opt("end");
const maxFrames = Number(opt("max-frames", "100000"));
const camera = opt("camera", "broadcast");
const png = args.includes("--png");
const dataDir = process.env.REGOLAZO_DATA_DIR ?? join(REPO_ROOT, "data");

const h = await startHarness();
try {
  const page = await h.open(`record=1&w=${width}&h=${height}&clip=${encodeURIComponent(clip)}`, { width, height });
  const info = await page.evaluate(
    (o) => {
      const r = window.regolazo!;
      const plan = r.recorder.start({
        fps: o.fps,
        start: o.start ? Number(o.start) : undefined,
        end: o.end ? Number(o.end) : undefined,
        camera: o.camera as "broadcast",
        format: o.png ? "png" : "jpeg",
      });
      return { ...plan, match_id: r.app.meta.match_id, renderer: r.info().renderer };
    },
    { fps, start, end, camera, png },
  );
  const out = opt("out") ?? join(dataDir, "renders", `${info.match_id}-${basename(clip, ".json")}`);
  mkdirSync(join(out, "frames"), { recursive: true });
  mkdirSync(join(out, "labels"), { recursive: true });
  const n = Math.min(info.frames, maxFrames);
  const t0 = Date.now();
  const digest = createHash("sha256");
  let objects = 0;
  for (let i = 0; i < n; i++) {
    const f = await page.evaluate((k) => window.regolazo!.recorder.frame(k), i);
    const b64 = f.image.slice(f.image.indexOf(",") + 1);
    writeFileSync(join(out, f.labels.image), Buffer.from(b64, "base64"));
    const json = JSON.stringify(f.labels);
    writeFileSync(join(out, "labels", `${String(i).padStart(6, "0")}.json`), json + "\n");
    digest.update(json);
    objects += f.labels.objects.length;
    if (i % 50 === 0) console.log(`frame ${i}/${n} t=${f.labels.t} objects=${f.labels.objects.length}`);
  }
  const secs = (Date.now() - t0) / 1000;
  const manifest = {
    match_id: info.match_id,
    clip,
    fps,
    frames: n,
    start: info.start,
    width,
    height,
    camera_mode: camera,
    image_format: png ? "png" : "jpg",
    renderer: info.renderer,
    labels_sha256: digest.digest("hex"),
    seconds: +secs.toFixed(1),
  };
  writeFileSync(join(out, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log(`wrote ${n} frames (${objects} object labels) to ${out} in ${secs.toFixed(1)} s (${(n / secs).toFixed(1)} frames/s)`);
  console.log(`labels sha256 ${manifest.labels_sha256}`);
} finally {
  await h.close();
}
