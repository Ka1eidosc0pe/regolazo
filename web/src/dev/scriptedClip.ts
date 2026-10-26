// A believable scripted 25 Hz clip for renderer development, used until the simulator's first recording exists.
// Home (Harbour City) builds from the back, switches play to the right wing, crosses to the far post; the header
// is parried by the away keeper and the loose ball is driven into the net.
// Deterministic: the same meta and options always give the same frames.
import type { Clip, MatchMeta, PlayerAnim, PlayerMeta, PlayerState, SimState } from "../types.ts";
import { angleDelta, clamp, HALF_L, HALF_W, mulberry32 } from "../coords.ts";

/** Strike anims reach ball contact this long after anim_t = 0 (renderer convention, see athletes/animMap.ts). */
export const CONTACT_S = 0.4;

type Vec = [number, number];

interface Kick {
  t: number;
  by: string;
  anim: PlayerAnim;
  /** Ball target (world) and arrival time. */
  target: [number, number, number];
  arrive: number;
  kind: "ground" | "lofted" | "shot";
  receiver?: string;
  receiveAnim?: PlayerAnim;
  spin?: [number, number, number];
  mirror?: boolean;
}

interface Waypoint {
  t: number;
  p: Vec;
}

const HZ = 25;
const ROLL_DECEL = 1.4;
const G = 9.81;

// 4-3-3 base positions in the attacking frame (attacking +x, +y on the attacker's left).
const BASE: Record<string, Vec> = {
  GK: [-49, 0],
  RB: [-27, -24],
  RCB: [-33, -8],
  CB: [-34, 0],
  LCB: [-33, 8],
  LB: [-27, 24],
  DM: [-19, 0],
  RCM: [-9, -13],
  LCM: [-9, 13],
  AM: [-2, 0],
  RW: [6, -26],
  LW: [6, 26],
  RM: [-2, -24],
  LM: [-2, 24],
  ST: [10, 0],
  CF: [10, 0],
  RWB: [-20, -28],
  LWB: [-20, 28],
};

function lerpWaypoints(w: Waypoint[], t: number): Vec | null {
  if (w.length === 0 || t < w[0]!.t || t > w[w.length - 1]!.t) return null;
  for (let i = 0; i < w.length - 1; i++) {
    const a = w[i]!;
    const b = w[i + 1]!;
    if (t >= a.t && t <= b.t) {
      const s = (t - a.t) / Math.max(1e-6, b.t - a.t);
      return [a.p[0] + (b.p[0] - a.p[0]) * s, a.p[1] + (b.p[1] - a.p[1]) * s];
    }
  }
  return w[w.length - 1]!.p;
}

export interface ScriptOptions {
  t0?: number;
  duration?: number;
}

export function buildScriptedClip(meta: MatchMeta, opts: ScriptOptions = {}): Clip {
  const t0 = opts.t0 ?? 1520.0;
  const duration = opts.duration ?? 27;
  const rng = mulberry32(meta.seed);
  const players: PlayerMeta[] = [
    ...meta.home.players.filter((p) => p.starter),
    ...meta.away.players.filter((p) => p.starter),
  ];
  const byPos = (team: "home" | "away", pos: string): string => {
    const p = players.find((q) => q.team === team && q.position === pos);
    if (!p) throw new Error(`no ${team} ${pos}`);
    return p.id;
  };
  const H = (pos: string) => byPos("home", pos);
  const A = (pos: string) => byPos("away", pos);

  // ------------------------------------------------------------------ the script (seconds from clip start)
  const kicks: Kick[] = [
    { t: 1.2, by: H("LCB"), anim: "pass_short", target: [-14, 4, 0.11], arrive: 2.4, kind: "ground", receiver: H("DM"), receiveAnim: "receive_ground" },
    { t: 4.0, by: H("DM"), anim: "pass_short", target: [-1, 15, 0.11], arrive: 5.3, kind: "ground", receiver: H("LCM"), receiveAnim: "receive_ground" },
    {
      t: 8.2, by: H("LCM"), anim: "pass_long", target: [27, -25, 0.9], arrive: 10.3, kind: "lofted", receiver: H("RW"),
      receiveAnim: "receive_aerial", spin: [0, -18, 0],
    },
    {
      t: 15.0, by: H("RW"), anim: "cross", target: [46.5, 2.5, 2.25], arrive: 16.25, kind: "lofted", receiver: H("ST"),
      spin: [0, -12, 6],
    },
    { t: 16.25, by: H("ST"), anim: "header_jumping", target: [52.0, -2.0, 0.9], arrive: 16.75, kind: "shot" },
    // keeper parry (the keeper "kicks" the ball away with a dive)
    { t: 16.75, by: A("GK"), anim: "gk_dive_left", target: [44.5, -9.5, 0.11], arrive: 17.75, kind: "lofted", receiver: H("RCM"), receiveAnim: "receive_ground" },
    { t: 18.3, by: H("RCM"), anim: "shoot", target: [53.2, 2.9, 1.05], arrive: 18.62, kind: "shot", spin: [0, 0, 9] },
  ];
  const GOAL_T = 18.62;

  // Scripted runs (world frame; home attacks +x).
  const runs: Record<string, Waypoint[]> = {
    [H("LCB")]: [{ t: 0, p: [-22, 9] }, { t: 1.2, p: [-21, 9] }, { t: 6, p: [-16, 9] }],
    [H("DM")]: [{ t: 0, p: [-12, 0] }, { t: 2.4, p: [-14, 4] }, { t: 4.0, p: [-10, 5] }, { t: 8, p: [-4, 2] }],
    [H("LCM")]: [{ t: 0, p: [-2, 18] }, { t: 5.3, p: [-1, 15] }, { t: 8.2, p: [9, 14] }, { t: 12, p: [16, 10] }],
    [H("RW")]: [
      { t: 0, p: [12, -27] }, { t: 7, p: [18, -27] }, { t: 10.3, p: [27, -25] }, { t: 11.2, p: [29, -25] },
      { t: 15.0, p: [41, -23] }, { t: 17, p: [43, -20] },
    ],
    [H("ST")]: [
      { t: 0, p: [16, 2] }, { t: 9, p: [22, 0] }, { t: 13, p: [34, -2] }, { t: 15.0, p: [41, -1] },
      { t: 16.25, p: [46.5, 2.3] }, { t: 18, p: [48, 2] },
    ],
    [H("RCM")]: [
      { t: 0, p: [0, -12] }, { t: 10, p: [14, -14] }, { t: 15, p: [32, -12] }, { t: 17.75, p: [44.5, -9.5] },
      { t: 18.3, p: [45.3, -9.0] }, { t: 19.5, p: [46, -10] }, { t: 23, p: [44, -32.5] }, { t: duration, p: [44, -32.5] },
    ],
    [H("LW")]: [{ t: 0, p: [14, 26] }, { t: 13, p: [33, 18] }, { t: 16, p: [42, 10] }, { t: 18.5, p: [44, 8] }],
    [A("GK")]: [{ t: 0, p: [48, 0] }, { t: 14, p: [50, -1.0] }, { t: 16.25, p: [51.0, -0.5] }, { t: 16.75, p: [51.0, -1.2] }, { t: 18.3, p: [51.6, -3.6] }],
  };
  // One-shot animation schedule: [player, anim, start, end, mirror]
  const oneShots: [string, PlayerAnim, number, number, boolean][] = [];
  for (const k of kicks) {
    oneShots.push([k.by, k.anim, k.t - CONTACT_S, k.t + (k.anim.startsWith("gk_dive") ? 2.0 : 0.75), k.mirror ?? false]);
    if (k.receiver && k.receiveAnim) oneShots.push([k.receiver, k.receiveAnim, k.arrive - 0.15, k.arrive + 0.55, false]);
  }
  oneShots.push([A("GK"), "get_up", 18.8, 21.5, false]);
  oneShots.push([A("LCB"), "slide_tackle", 17.9, 19.2, false]);
  for (const p of players) {
    if (p.team === "home" && p.position !== "GK") oneShots.push([p.id, "celebrate", GOAL_T + 2.6 + rng() * 1.5, duration + 1, false]);
    if (p.team === "away" && rng() < 0.6 && p.position !== "GK") oneShots.push([p.id, "disappointed", GOAL_T + 1.2 + rng(), duration + 1, false]);
  }

  // ------------------------------------------------------------------ simulation state
  interface Body {
    meta: PlayerMeta;
    pos: Vec;
    vel: Vec;
    facing: number;
    anim: PlayerAnim;
    animStart: number;
    mirror: boolean;
    variant: number;
  }
  const sign = (team: "home" | "away") => (team === "home" ? 1 : -1);
  const baseOf = (p: PlayerMeta): Vec => {
    const b = BASE[p.position] ?? [0, 0];
    return [b[0] * sign(p.team), b[1] * sign(p.team)];
  };
  const bodies: Body[] = players.map((p) => {
    const b = baseOf(p);
    const start: Vec = runs[p.id]?.[0]?.p ?? [b[0] * 0.6 + (p.team === "home" ? 2 : 6), b[1]];
    return {
      meta: p,
      pos: [...start],
      vel: [0, 0],
      facing: p.team === "home" ? 0 : Math.PI,
      anim: p.position === "GK" ? "gk_idle" : "idle",
      animStart: 0,
      mirror: false,
      variant: Math.floor(rng() * 3),
    };
  });
  const body = (id: string) => bodies.find((b) => b.meta.id === id)!;

  let ball: [number, number, number] = [-21.4, 9.2, 0.11];
  let ballVel: [number, number, number] = [0, 0, 0];
  let spin: [number, number, number] = [0, 0, 0];
  let owner: string | null = H("LCB");
  let flight: { k: Kick; p0: [number, number, number]; v0: [number, number, number] } | null = null;
  let afterGoal = false;
  let lastTouch: string | null = owner;
  let possessionId = 141;
  let possTeam: "home" | "away" | null = "home";
  let ref: { pos: Vec; vel: Vec; facing: number } = { pos: [-8, 14], vel: [0, 0], facing: -Math.PI / 2 };

  const frames: SimState[] = [];
  const nTicks = Math.round(duration * HZ);
  const tick0 = Math.round(t0 * HZ);
  for (let i = 0; i <= nTicks; i++) {
    const s = i / HZ;
    const dt = 1 / HZ;
    let keyframe = false;

    // --- kicks
    for (const k of kicks) {
      if (Math.abs(s - k.t) < dt / 2) {
        const from: [number, number, number] = [...ball];
        const T = k.arrive - k.t;
        const dx = k.target[0] - from[0];
        const dy = k.target[1] - from[1];
        const d = Math.hypot(dx, dy);
        let v0: [number, number, number];
        if (k.kind === "ground") {
          const v = (d + 0.5 * ROLL_DECEL * T * T) / T;
          v0 = [(dx / d) * v, (dy / d) * v, 0];
        } else {
          v0 = [dx / T, dy / T, (k.target[2] - from[2] + 0.5 * G * T * T) / T];
        }
        flight = { k, p0: from, v0 };
        owner = null;
        lastTouch = k.by;
        spin = k.spin ?? [0, 0, 0];
        keyframe = true;
        const team = body(k.by).meta.team;
        if (team !== possTeam) {
          possTeam = team;
          possessionId++;
        }
      }
    }
    // --- ball
    if (flight) {
      const ft = s - flight.k.t;
      const { p0, v0, k } = flight;
      if (k.kind === "ground") {
        const sp = Math.hypot(v0[0], v0[1]);
        const tt = Math.min(ft, sp / ROLL_DECEL);
        const dist = sp * tt - 0.5 * ROLL_DECEL * tt * tt;
        const cur = Math.max(0, sp - ROLL_DECEL * tt);
        ball = [p0[0] + (v0[0] / sp) * dist, p0[1] + (v0[1] / sp) * dist, 0.11];
        ballVel = [(v0[0] / sp) * cur, (v0[1] / sp) * cur, 0];
      } else {
        ball = [p0[0] + v0[0] * ft, p0[1] + v0[1] * ft, Math.max(0.11, p0[2] + v0[2] * ft - 0.5 * G * ft * ft)];
        ballVel = [v0[0], v0[1], v0[2] - G * ft];
      }
      if (k.kind === "shot" && k.by === H("RCM") && ball[0] > HALF_L) {
        // in the net: decelerate hard against the back netting and drop
        afterGoal = true;
      }
      if (ft >= k.arrive - k.t - 1e-6) {
        if (k.receiver) {
          owner = k.receiver;
          lastTouch = k.receiver;
          keyframe = true;
          flight = null;
          ballVel = [0, 0, 0];
          spin = [0, 0, 0];
        } else if (k.kind === "shot" && k.by !== H("RCM")) {
          // header reaches the keeper; the parry kick takes over
        } else {
          flight = null;
        }
      }
    }
    if (afterGoal && !flight) {
      // ball settles in the goal
      ball = [Math.min(54.2, ball[0] + ballVel[0] * dt), ball[1] + ballVel[1] * dt, Math.max(0.11, ball[2] + ballVel[2] * dt)];
      ballVel = [ballVel[0] * 0.8, ballVel[1] * 0.8, ball[2] > 0.12 ? ballVel[2] - G * dt : -ballVel[2] * 0.3];
      if (ball[0] >= 54.2) ballVel[0] = -Math.abs(ballVel[0]) * 0.2;
    }
    if (afterGoal && flight && ball[0] > 54.0) {
      ballVel = [2, ballVel[1] * 0.3, ballVel[2]];
      ball = [54.0, ball[1], ball[2]];
      flight = null;
    }

    // --- players
    for (const b of bodies) {
      const p = b.meta;
      const scripted = lerpWaypoints(runs[p.id] ?? [], s);
      let target: Vec;
      if (scripted) {
        target = scripted;
      } else if (afterGoal) {
        const scorer = body(H("RCM")).pos;
        target = p.team === "home" && p.position !== "GK" ? [scorer[0] - 4 + rng() * 0.2, scorer[1] + 3] : [baseOf(p)[0] * 0.5, baseOf(p)[1] * 0.6];
      } else {
        const bb = baseOf(p);
        if (p.position === "GK") {
          const gx = p.team === "home" ? -49.5 : 49.5;
          target = [gx, clamp(ball[1] * 0.12, -3, 3)];
        } else if (p.team === "home") {
          target = [clamp(bb[0] * 0.62 + ball[0] * 0.55 + 10, -45, 47), bb[1] * 0.95 + ball[1] * 0.12];
        } else {
          // away defends towards +x: compact block shifted to the ball
          target = [clamp(bb[0] * 0.42 + ball[0] * 0.5 + 6, -40, 50), bb[1] * 0.62 + ball[1] * 0.32];
        }
      }
      if (owner === p.id) target = [target[0], target[1]];
      const maxV = p.physical.max_speed_mps * 0.92;
      const desired: Vec = [(target[0] - b.pos[0]) * 1.1, (target[1] - b.pos[1]) * 1.1];
      const dm = Math.hypot(desired[0], desired[1]);
      if (dm > maxV) {
        desired[0] *= maxV / dm;
        desired[1] *= maxV / dm;
      }
      const acc = p.physical.max_accel_mps2 * 0.8;
      const dvx = desired[0] - b.vel[0];
      const dvy = desired[1] - b.vel[1];
      const dvm = Math.hypot(dvx, dvy);
      const k = dvm > acc * dt ? (acc * dt) / dvm : 1;
      b.vel = [b.vel[0] + dvx * k, b.vel[1] + dvy * k];
      b.pos = [clamp(b.pos[0] + b.vel[0] * dt, -HALF_L - 2, HALF_L + 2), clamp(b.pos[1] + b.vel[1] * dt, -HALF_W - 2, HALF_W + 2)];

      // facing: towards motion when moving fast, otherwise towards the ball
      const sp = Math.hypot(b.vel[0], b.vel[1]);
      const toBall = Math.atan2(ball[1] - b.pos[1], ball[0] - b.pos[0]);
      const moveDir = Math.atan2(b.vel[1], b.vel[0]);
      let want = sp > 2.2 ? moveDir : toBall;
      if (sp > 0.8 && sp < 4.5 && Math.abs(angleDelta(moveDir, toBall)) > 1.9) want = toBall; // backpedal
      if (owner === p.id && sp < 1.5) {
        // shaping to pass: face the next kick target
        const next = kicks.find((kk) => kk.by === p.id && kk.t >= s);
        if (next) want = Math.atan2(next.target[1] - b.pos[1], next.target[0] - b.pos[0]);
      }
      const nextKick = kicks.find((kk) => kk.by === p.id && kk.t >= s && kk.t - s < 0.6);
      if (nextKick) want = Math.atan2(nextKick.target[1] - b.pos[1], nextKick.target[0] - b.pos[0]);
      b.facing += clamp(angleDelta(b.facing, want), -7 * dt, 7 * dt);

      // animation
      let anim: PlayerAnim;
      let mirror = false;
      const os = oneShots.find(([id, , st, en]) => id === p.id && s >= st && s < en);
      if (os) {
        anim = os[1];
        mirror = os[4];
      } else if (p.position === "GK" && sp < 2.5) {
        anim = "gk_idle";
      } else {
        const rel = angleDelta(b.facing, moveDir);
        if (sp < 0.5) anim = "idle";
        else if (Math.abs(rel) > 2.2) anim = "jog_back";
        else if (Math.abs(rel) > 0.9 && sp < 4.5) anim = rel > 0 ? "strafe_left" : "strafe_right";
        else if (owner === p.id && sp > 1.2) anim = "dribble";
        else if (sp < 3.6) anim = "jog";
        else if (sp < 6.2) anim = "run";
        else anim = "sprint";
      }
      if (anim !== b.anim || (os && os[2] !== b.animStart && Math.abs(s - os[2]) < dt / 2)) {
        b.anim = anim;
        b.animStart = os ? os[2] : s;
      }
      b.mirror = mirror;
    }

    // ball at the owner's feet (dribbling touches)
    if (owner) {
      const o = body(owner);
      const sp = Math.hypot(o.vel[0], o.vel[1]);
      const lead = 0.45 + Math.min(1.2, sp * 0.12) * (0.5 + 0.5 * Math.sin(s * 7.5));
      const nb: [number, number, number] = [o.pos[0] + Math.cos(o.facing) * lead, o.pos[1] + Math.sin(o.facing) * lead, 0.11];
      ballVel = [(nb[0] - ball[0]) * HZ, (nb[1] - ball[1]) * HZ, 0];
      ball = nb;
    }

    // referee: diagonal system, trailing the ball
    {
      const tgt: Vec = [clamp(ball[0] - 8, -40, 40), clamp(ball[1] * 0.5 + 10, -25, 25)];
      const dv: Vec = [(tgt[0] - ref.pos[0]) * 0.8, (tgt[1] - ref.pos[1]) * 0.8];
      const m = Math.hypot(dv[0], dv[1]);
      if (m > 6) {
        dv[0] *= 6 / m;
        dv[1] *= 6 / m;
      }
      ref.vel = [ref.vel[0] + (dv[0] - ref.vel[0]) * 0.1, ref.vel[1] + (dv[1] - ref.vel[1]) * 0.1];
      ref.pos = [ref.pos[0] + ref.vel[0] * dt, ref.pos[1] + ref.vel[1] * dt];
      const sp = Math.hypot(ref.vel[0], ref.vel[1]);
      const want = sp > 2.5 ? Math.atan2(ref.vel[1], ref.vel[0]) : Math.atan2(ball[1] - ref.pos[1], ball[0] - ref.pos[0]);
      ref = { ...ref, facing: ref.facing + clamp(angleDelta(ref.facing, want), -5 * dt, 5 * dt) };
    }

    const t = +(t0 + s).toFixed(3);
    const goalScored = s >= GOAL_T;
    const ps: PlayerState[] = bodies.map((b) => ({
      id: b.meta.id,
      pos: [round(b.pos[0]), round(b.pos[1])],
      vel: [round(b.vel[0]), round(b.vel[1])],
      facing: round(normAngle(b.facing)),
      z: 0,
      anim: b.anim,
      anim_t: round(Math.max(0, s - b.animStart)),
      anim_mirror: b.mirror,
      anim_variant: b.variant,
      stamina: 0.9,
      on_pitch: true,
    }));
    const phase: SimState["phase"] = goalScored
      ? "dead_ball"
      : ball[0] > 17.5
        ? "final_third"
        : ball[0] > -17.5
          ? "progression"
          : "build_up";
    frames.push({
      schema_version: meta.schema_version,
      tick: tick0 + i,
      t,
      period: 1,
      status: "first_half",
      ball: {
        pos: [round(ball[0]), round(ball[1]), round(ball[2])],
        vel: [round(ballVel[0]), round(ballVel[1]), round(ballVel[2])],
        spin: [round(spin[0]), round(spin[1]), round(spin[2])],
        in_play: !goalScored,
        last_touch: lastTouch,
      },
      players: ps,
      referee: { pos: [round(ref.pos[0]), round(ref.pos[1])], vel: [round(ref.vel[0]), round(ref.vel[1])], facing: round(normAngle(ref.facing)) },
      possession: { team: possTeam, player: owner, possession_id: possessionId },
      phase,
      restart: goalScored ? "kick_off" : null,
      score: { home: goalScored ? 1 : 0, away: 0 },
      attacking_positive_x: meta.home_attacks_positive_x_first_half ? "home" : "away",
      keyframe,
      snapshot_ref: keyframe ? `snap:${meta.match_id}:${tick0 + i}` : null,
    });
  }
  return { meta, frames };
}

function round(v: number): number {
  return Math.round(v * 1000) / 1000;
}

function normAngle(a: number): number {
  let r = a % (2 * Math.PI);
  if (r > Math.PI) r -= 2 * Math.PI;
  if (r <= -Math.PI) r += 2 * Math.PI;
  return r;
}
