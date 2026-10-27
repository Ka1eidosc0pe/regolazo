// The camera never jumps: per-frame movement stays bounded across the whole dev clip, in every mode
// (including the replay camera when the ball goes dead after the goal).
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { CameraDirector, type CameraMode } from "../src/camera/director";
import { buildScriptedClip } from "../src/dev/scriptedClip";
import { StateBuffer } from "../src/source/interpolate";
import type { MatchMeta } from "../src/types";

const meta = JSON.parse(readFileSync(fileURLToPath(new URL("../../shared/fixtures/MatchMeta.json", import.meta.url)), "utf8")) as MatchMeta;
const clip = buildScriptedClip(meta);
const buf = new StateBuffer();
buf.setAll(clip.frames);

function run(mode: CameraMode, target: string | null) {
  const d = new CameraDirector(16 / 9);
  d.replayTarget = target;
  d.setMode(mode, 0);
  const t0 = buf.first!.t;
  const t1 = buf.last!.t;
  let prevPos: THREE.Vector3 | null = null;
  let prevDir: THREE.Vector3 | null = null;
  let maxMove = 0;
  let maxTurn = 0;
  let maxFov = 0;
  let prevFov = 0;
  for (let t = t0; t <= t1; t += 0.04) {
    d.update(buf.sample(t), 0.04);
    const pos = d.camera.position.clone();
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(d.camera.quaternion);
    if (prevPos && prevDir) {
      maxMove = Math.max(maxMove, pos.distanceTo(prevPos));
      maxTurn = Math.max(maxTurn, THREE.MathUtils.radToDeg(dir.angleTo(prevDir)));
      maxFov = Math.max(maxFov, Math.abs(d.camera.fov - prevFov));
    }
    prevPos = pos;
    prevDir = dir;
    prevFov = d.camera.fov;
  }
  return { maxMove, maxTurn, maxFov };
}

describe("camera motion is bounded frame to frame (25 fps)", () => {
  it("broadcast: fixed gantry, pan under 1.5 deg/frame", () => {
    const r = run("broadcast", null);
    expect(r.maxMove).toBe(0);
    expect(r.maxTurn).toBeLessThan(1.5);
    expect(r.maxFov).toBeLessThan(0.5);
  });
  it("tactical", () => {
    const r = run("tactical", null);
    expect(r.maxMove).toBeLessThan(0.5);
    expect(r.maxTurn).toBeLessThan(1.5);
  });
  it("replay following the scorer, through the goal and the dead ball", () => {
    const r = run("replay", "H08");
    expect(r.maxMove).toBeLessThan(0.6);
    expect(r.maxTurn).toBeLessThan(4);
  });
  it("replay following the ball", () => {
    const r = run("replay", null);
    expect(r.maxMove).toBeLessThan(0.8);
    expect(r.maxTurn).toBeLessThan(6);
  });
});
