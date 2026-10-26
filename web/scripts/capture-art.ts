// Reference screenshots for the art direction (docs/art/). Usage:
//   node scripts/capture-art.ts [--out ../docs/art] [--only name1,name2]
import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { CameraMode } from "../src/camera/director.ts";
import type { V3 } from "../src/coords.ts";
import { REPO_ROOT, startHarness } from "./lib/harness.ts";

interface Shot {
  name: string;
  t: number;
  cam?: CameraMode;
  target?: string;
  free?: { pos: V3; look: V3; fov: number };
}

const T0 = 1520;
export const SHOTS: Shot[] = [
  { name: "ref_broadcast_wide", t: T0 + 6.4, cam: "broadcast" },
  { name: "ref_broadcast_attack", t: T0 + 15.6, cam: "broadcast" },
  { name: "ref_tactical", t: T0 + 11.5, cam: "tactical" },
  { name: "ref_closeup_players", t: T0 + 14.2, free: { pos: [34, -31.5, 1.35], look: [40.5, -22.0, 1.0], fov: 26 } },
  { name: "ref_closeup_box", t: T0 + 16.2, free: { pos: [39.5, -6.5, 1.5], look: [47.0, 1.5, 1.6], fov: 30 } },
  { name: "ref_replay_goal", t: T0 + 18.4, cam: "replay", target: "H08" },
  { name: "ref_replay_celebration", t: T0 + 24.5, cam: "replay", target: "H08" },
  { name: "ref_stadium_night", t: T0 + 3, free: { pos: [-30, -40, 9], look: [25, 30, 8], fov: 58 } },
];

const args = process.argv.slice(2);
const outIdx = args.indexOf("--out");
const out = outIdx >= 0 ? args[outIdx + 1]! : join(REPO_ROOT, "docs/art");
const shotsIdx = args.indexOf("--shots");
const shots: Shot[] = shotsIdx >= 0 ? (JSON.parse(readFileSync(args[shotsIdx + 1]!, "utf8")) as Shot[]) : SHOTS;
const onlyIdx = args.indexOf("--only");
const only = onlyIdx >= 0 ? new Set(args[onlyIdx + 1]!.split(",")) : null;
mkdirSync(out, { recursive: true });

const h = await startHarness();
try {
  const page = await h.open("w=1920&h=1080&paused=1");
  for (const s of shots) {
    if (only && !only.has(s.name)) continue;
    await page.evaluate((shot) => {
      const r = window.regolazo!;
      r.setClean(false);
      if (shot.free) r.setFreePose(shot.free.pos, shot.free.look, shot.free.fov);
      else r.setCamera(shot.cam ?? "broadcast", shot.target ?? null);
      r.jumpTo(shot.t, 2.5);
    }, s);
    const file = join(out, `${s.name}.jpg`);
    await page.locator("#app").screenshot({ path: file, type: "jpeg", quality: 90 });
    console.log(`saved ${file}`);
  }
  console.log(JSON.stringify(await page.evaluate(() => window.regolazo!.info())));
} finally {
  await h.close();
}
