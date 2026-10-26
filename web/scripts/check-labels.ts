// Label projection test (Step 5 gate). Records the dev clip in-browser and checks, for every frame:
//  1. calibration: K [R | t] applied to each object's world point (here, in Node) matches the Three.js camera
//     projection of the same point (computed in the page) -> max reprojection error in pixels;
//  2. alignment: the box projected from skinned geometry matches the object's GPU silhouette (solo ID pass)
//     -> IoU for unoccluded objects at least 20 px tall;
//  3. the ball centre lies inside the ball box; the ball-shadow pixel lies below (or at) the ball centre.
// Saves overlay images (boxes, centres, ball shadow) to docs/art/labels/ and exits non-zero on failure.
//   node scripts/check-labels.ts [--every 1] [--overlays 1532.6,1535.0,1536.2,1538.6]
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { CameraCalibration, FrameLabels } from "../src/types.ts";
import { REPO_ROOT, startHarness } from "./lib/harness.ts";

const args = process.argv.slice(2);
const opt = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1]! : fallback;
};
const every = Number(opt("every", "1"));
const overlayTimes = opt("overlays", "1526.4,1534.6,1536.24,1538.62").split(",").map(Number);
const outDir = join(REPO_ROOT, "docs/art/labels");
mkdirSync(outDir, { recursive: true });

function project(cal: CameraCalibration, X: [number, number, number]): [number, number] {
  const r = cal.extrinsics;
  const k = cal.intrinsics;
  const xc = r[0] * X[0] + r[1] * X[1] + r[2] * X[2] + r[3];
  const yc = r[4] * X[0] + r[5] * X[1] + r[6] * X[2] + r[7];
  const zc = r[8] * X[0] + r[9] * X[1] + r[10] * X[2] + r[11];
  return [(k[0] * xc) / zc + k[2], (k[4] * yc) / zc + k[5]];
}

function iou(a: number[], b: number[]): number {
  const ix = Math.max(0, Math.min(a[2]!, b[2]!) - Math.max(a[0]!, b[0]!));
  const iy = Math.max(0, Math.min(a[3]!, b[3]!) - Math.max(a[1]!, b[1]!));
  const inter = ix * iy;
  const area = (q: number[]) => (q[2]! - q[0]!) * (q[3]! - q[1]!);
  return inter / Math.max(1e-6, area(a) + area(b) - inter);
}

const h = await startHarness();
let failed: boolean;
try {
  const page = await h.open("record=1&w=1920&h=1080");
  const plan = await page.evaluate(() => window.regolazo!.recorder.start({ fps: 25 }));
  let maxReproj = 0;
  let sumReproj = 0;
  let nReproj = 0;
  const ious: number[] = [];
  let ballInside = 0;
  let ballFrames = 0;
  let shadowBelow = 0;
  let shadowFrames = 0;
  let worst = { iou: 1, frame: -1, track: "" };
  const wantOverlay = new Set(overlayTimes.map((t) => Math.round((t - plan.start) * plan.fps)));
  for (let i = 0; i < plan.frames; i++) {
    const dbg = (i % every === 0) || wantOverlay.has(i);
    const f = await page.evaluate(([k, d]) => window.regolazo!.recorder.frame(k as number, d as boolean), [i, dbg]);
    if (!dbg || !f.debug) continue;
    const L: FrameLabels = f.labels;
    for (const o of L.objects) {
      const p = project(L.camera, o.world);
      const tc = f.debug.threeCenter[o.track_id]!;
      const e = Math.hypot(p[0] - tc[0], p[1] - tc[1]);
      const e2 = Math.hypot(p[0] - o.center_px[0], p[1] - o.center_px[1]);
      maxReproj = Math.max(maxReproj, e, e2);
      sumReproj += e;
      nReproj++;
      const sil = f.debug.silhouette[o.track_id];
      const hgt = o.box[3] - o.box[1];
      const inside = o.box[0] > 1 && o.box[2] < L.width - 1 && o.box[1] > 1 && o.box[3] < L.height - 1;
      if (sil && o.visible_fraction >= 0.95 && hgt >= 20 && inside && o.cls !== "ball") {
        const v = iou(o.box, sil);
        ious.push(v);
        if (v < worst.iou) worst = { iou: v, frame: i, track: o.track_id };
      }
      if (o.cls === "ball" && o.visible_fraction > 0) {
        ballFrames++;
        const [cx, cy] = o.center_px;
        if (cx >= o.box[0] - 0.5 && cx <= o.box[2] + 0.5 && cy >= o.box[1] - 0.5 && cy <= o.box[3] + 0.5) ballInside++;
        if (L.ball_shadow_px) {
          shadowFrames++;
          if (L.ball_shadow_px[1] >= cy - 0.5) shadowBelow++;
        }
      }
    }
    if (wantOverlay.has(i)) {
      const png = await page.evaluate(
        async ([img, labels, sil]) => {
          const L = labels as FrameLabels;
          const im = new Image();
          im.src = img as string;
          await im.decode();
          const c = document.createElement("canvas");
          c.width = L.width;
          c.height = L.height;
          const ctx = c.getContext("2d")!;
          ctx.drawImage(im, 0, 0);
          ctx.font = "600 13px sans-serif";
          const colors: Record<string, string> = {
            player_home: "#4cc9f0",
            player_away: "#ff4d5e",
            goalkeeper_home: "#2dc653",
            goalkeeper_away: "#c77dff",
            referee: "#d7ff3a",
            ball: "#ffffff",
          };
          for (const o of L.objects) {
            const [x0, y0, x1, y1] = o.box;
            ctx.strokeStyle = colors[o.cls] ?? "#fff";
            ctx.lineWidth = 2;
            ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
            const s = (sil as Record<string, number[] | null>)[o.track_id];
            if (s) {
              ctx.strokeStyle = "rgba(255,255,255,0.55)";
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              ctx.strokeRect(s[0]!, s[1]!, s[2]! - s[0]!, s[3]! - s[1]!);
              ctx.setLineDash([]);
            }
            ctx.fillStyle = "#ff7a1a";
            ctx.beginPath();
            ctx.arc(o.center_px[0], o.center_px[1], 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = colors[o.cls] ?? "#fff";
            ctx.fillText(`${o.track_id} ${o.visible_fraction.toFixed(2)}`, x0, y0 - 4);
          }
          if (L.ball_shadow_px) {
            ctx.strokeStyle = "#ff7a1a";
            ctx.lineWidth = 2;
            const [sx, sy] = L.ball_shadow_px;
            ctx.beginPath();
            ctx.moveTo(sx - 7, sy);
            ctx.lineTo(sx + 7, sy);
            ctx.moveTo(sx, sy - 7);
            ctx.lineTo(sx, sy + 7);
            ctx.stroke();
          }
          ctx.fillStyle = "rgba(5,7,13,0.75)";
          ctx.fillRect(0, L.height - 34, 760, 34);
          ctx.fillStyle = "#e8ecf4";
          ctx.font = "600 15px sans-serif";
          ctx.fillText(`frame ${L.frame}  t=${L.t.toFixed(2)}  tick ${L.tick}  solid: projected box · dashed: GPU silhouette · dot: centre_px · cross: ball shadow`, 12, L.height - 12);
          return c.toDataURL("image/jpeg", 0.9);
        },
        [f.image, f.labels, f.debug.silhouette],
      );
      const file = join(outDir, `labels_frame_${String(i).padStart(4, "0")}.jpg`);
      writeFileSync(file, Buffer.from(png.slice(png.indexOf(",") + 1), "base64"));
      console.log(`overlay ${file}`);
    }
  }
  ious.sort((a, b) => a - b);
  const meanIou = ious.reduce((a, b) => a + b, 0) / Math.max(1, ious.length);
  const report = {
    frames: plan.frames,
    objects_checked: nReproj,
    reprojection_px: { mean: +(sumReproj / Math.max(1, nReproj)).toFixed(4), max: +maxReproj.toFixed(4) },
    silhouette_iou: { n: ious.length, mean: +meanIou.toFixed(4), p05: +(ious[Math.floor(ious.length * 0.05)] ?? 0).toFixed(4), min: +(ious[0] ?? 0).toFixed(4), worst },
    ball_center_in_box: `${ballInside}/${ballFrames}`,
    ball_shadow_below_ball: `${shadowBelow}/${shadowFrames}`,
  };
  console.log(JSON.stringify(report, null, 2));
  writeFileSync(join(outDir, "report.json"), JSON.stringify(report, null, 2) + "\n");
  failed = maxReproj > 0.5 || meanIou < 0.85 || (ious[Math.floor(ious.length * 0.05)] ?? 0) < 0.7 || ballInside !== ballFrames || shadowBelow !== shadowFrames;
  console.log(failed ? "LABEL CHECK FAILED" : "LABEL CHECK PASSED");
} finally {
  await h.close();
}
process.exit(failed ? 1 : 0);
