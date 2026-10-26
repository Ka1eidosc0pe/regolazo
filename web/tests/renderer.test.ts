import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import { ANIMS, CONTACT_S, locomotionWeights, oneShotTime } from "../src/athletes/animMap";
import { angleDelta, facingToYaw, threeToWorld, worldToThree } from "../src/coords";
import { buildScriptedClip, CONTACT_S as CLIP_CONTACT_S } from "../src/dev/scriptedClip";
import { interpolate, StateBuffer } from "../src/source/interpolate";
import type { MatchMeta, SimState } from "../src/types";
import { PLAYER_ANIM } from "../src/types/generated/enums";

const repo = (p: string) => fileURLToPath(new URL(`../../${p}`, import.meta.url));
const readJson = (p: string): unknown => JSON.parse(readFileSync(repo(p), "utf8"));
const meta = readJson("shared/fixtures/MatchMeta.json") as MatchMeta;
const manifest = readJson("web/public/assets/athlete_clips.json") as { clips: Record<string, { duration_s: number; contact_s?: number; kind: string }> };

describe("coordinates", () => {
  it("maps world (z up) to three (y up) and back", () => {
    expect(worldToThree(1, 2, 3)).toEqual([1, 3, -2]);
    expect(threeToWorld(...worldToThree(4, -5, 6))).toEqual([4, -5, 6]);
  });
  it("turns facing into a yaw that points the character's +z along the facing", () => {
    for (const f of [0, 0.7, Math.PI / 2, -2]) {
      const yaw = facingToYaw(f);
      // local +z rotated by yaw about +y, mapped back to world
      const [wx, wy] = threeToWorld(Math.sin(yaw), 0, Math.cos(yaw));
      expect(wx).toBeCloseTo(Math.cos(f), 6);
      expect(wy).toBeCloseTo(Math.sin(f), 6);
    }
  });
  it("takes the shortest angle", () => {
    expect(angleDelta(3.0, -3.0)).toBeCloseTo(2 * Math.PI - 6, 6);
    expect(angleDelta(0.1, 0.3)).toBeCloseTo(0.2, 6);
  });
});

describe("scripted dev clip", () => {
  const clip = buildScriptedClip(meta);
  const ajv = new Ajv({ allErrors: true, strict: false });
  ajv.addSchema(readJson("shared/schemas/json/contracts.schema.json") as object, "contracts");
  const validate = ajv.getSchema("contracts#/$defs/SimState")!;

  it("is 25 Hz, ~27 s, deterministic", () => {
    expect(clip.frames.length).toBeGreaterThan(25 * 20);
    const dts = clip.frames.slice(1).map((f, i) => f.t - clip.frames[i]!.t);
    for (const d of dts) expect(d).toBeCloseTo(0.04, 6);
    expect(JSON.stringify(buildScriptedClip(meta))).toEqual(JSON.stringify(clip));
  });
  it("every frame validates against the SimState schema", () => {
    for (const f of clip.frames) expect(validate(f), JSON.stringify(validate.errors)).toBe(true);
  });
  it("contains runs, passes, a lofted cross, a shot and a goal", () => {
    const anims = new Set(clip.frames.flatMap((f) => f.players.map((p) => p.anim)));
    for (const a of ["sprint", "pass_short", "pass_long", "cross", "header_jumping", "shoot", "gk_dive_left", "celebrate"]) {
      expect(anims.has(a as SimState["players"][number]["anim"]), a).toBe(true);
    }
    expect(Math.max(...clip.frames.map((f) => f.ball.pos[2]))).toBeGreaterThan(3);
    expect(clip.frames[clip.frames.length - 1]!.score.home).toBe(1);
  });
  it("uses the renderer's strike convention", () => {
    expect(CLIP_CONTACT_S).toBe(CONTACT_S);
  });
  it("ball carriers face where they carry the ball, never backpedal with it", () => {
    for (const f of clip.frames) {
      const owner = f.possession.player;
      if (!owner) continue;
      const p = f.players.find((q) => q.id === owner)!;
      const speed = Math.hypot(p.vel[0], p.vel[1]);
      if (speed < 1.5) continue;
      const where = `${owner} at t=${(f.t - clip.frames[0]!.t).toFixed(2)}s`;
      expect(["jog_back", "strafe_left", "strafe_right"], where).not.toContain(p.anim);
      expect(Math.abs(angleDelta(p.facing, Math.atan2(p.vel[1], p.vel[0]))), where).toBeLessThan(Math.PI / 2);
      // during the first touch the ball is still arriving from the passer, so it may be level with or behind the run
      if (p.anim.startsWith("receive_")) continue;
      const ahead = (f.ball.pos[0] - p.pos[0]) * p.vel[0] + (f.ball.pos[1] - p.pos[1]) * p.vel[1];
      expect(ahead, `ball behind ${where}`).toBeGreaterThan(0);
    }
  });
  it("receivers only gain the ball when they can reach it, and the ball never teleports", () => {
    const t0 = clip.frames[0]!.t;
    for (let i = 1; i < clip.frames.length; i++) {
      const a = clip.frames[i - 1]!;
      const b = clip.frames[i]!;
      const owner = b.possession.player;
      if (owner && owner !== a.possession.player) {
        const p = a.players.find((q) => q.id === owner)!;
        const gap = Math.hypot(a.ball.pos[0] - p.pos[0], a.ball.pos[1] - p.pos[1]);
        expect(gap, `${owner} gains the ball at t=${(b.t - t0).toFixed(2)}s from ${gap.toFixed(2)} m`).toBeLessThanOrEqual(1.05);
      }
      const moved = Math.hypot(b.ball.pos[0] - a.ball.pos[0], b.ball.pos[1] - a.ball.pos[1], b.ball.pos[2] - a.ball.pos[2]);
      const speed = Math.max(Math.hypot(...a.ball.vel), Math.hypot(...b.ball.vel));
      expect(moved, `ball jumps ${moved.toFixed(2)} m at t=${(b.t - t0).toFixed(2)}s`).toBeLessThanOrEqual(speed * 0.04 + 0.3);
    }
  });
});

describe("interpolation", () => {
  const clip = buildScriptedClip(meta);
  it("returns the tick states at the tick times and blends in between", () => {
    const a = clip.frames[100]!;
    const b = clip.frames[101]!;
    const at0 = interpolate(a, b, 0);
    expect(at0.ball.pos[0]).toBeCloseTo(a.ball.pos[0], 6);
    const mid = interpolate(a, b, 0.5);
    const pa = a.players[3]!;
    const pb = b.players[3]!;
    expect(mid.players[3]!.x).toBeCloseTo((pa.pos[0] + pb.pos[0]) / 2, 6);
    expect(mid.t).toBeCloseTo((a.t + b.t) / 2, 6);
  });
  it("samples a buffer by time", () => {
    const buf = new StateBuffer();
    buf.setAll(clip.frames);
    const s = buf.sample(clip.frames[50]!.t + 0.01)!;
    expect(s.tick).toBe(clip.frames[50]!.tick);
    expect(buf.sample(-5)!.tick).toBe(clip.frames[0]!.tick);
  });
});

describe("animation map", () => {
  it("covers every PlayerAnim with clips that exist in the processed GLB", () => {
    for (const a of PLAYER_ANIM) {
      const spec = ANIMS[a];
      expect(spec, a).toBeDefined();
      for (const c of spec.clips) expect(manifest.clips[c], `${a} -> ${c}`).toBeDefined();
    }
  });
  it("strikes reach their contact frame at CONTACT_S", () => {
    for (const a of ["pass_short", "pass_long", "shoot", "header_jumping"] as const) {
      const clip = manifest.clips[ANIMS[a].clips[0]!]!;
      expect(oneShotTime(a, clip, CONTACT_S)).toBeCloseTo(clip.contact_s!, 6);
    }
  });
  it("locomotion weights sum to one and follow speed", () => {
    for (const s of [0, 1, 2.2, 3.5, 5, 6.5, 9]) {
      const w = locomotionWeights(s);
      expect(w.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 6);
    }
    expect(locomotionWeights(9)[3]).toBe(1);
    expect(locomotionWeights(0)[0]).toBe(1);
  });
});
