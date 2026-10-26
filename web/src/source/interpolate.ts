// Interpolation between two 25 Hz SimState ticks into a render state.
import { angleDelta, lerp } from "../coords.ts";
import type { PlayerAnim, SimState } from "../types.ts";

export interface RenderPlayer {
  id: string;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  speed: number;
  facing: number;
  anim: PlayerAnim;
  animT: number;
  mirror: boolean;
  variant: number;
  onPitch: boolean;
}

export interface RenderBall {
  pos: [number, number, number];
  vel: [number, number, number];
  spin: [number, number, number];
  inPlay: boolean;
}

export interface RenderState {
  t: number;
  /** Tick of the earlier of the two interpolated states. */
  tick: number;
  base: SimState;
  ball: RenderBall;
  players: RenderPlayer[];
  referee: { x: number; y: number; vx: number; vy: number; facing: number } | null;
}

const G = 9.81;

function hermite(p0: number, v0: number, p1: number, v1: number, dt: number, s: number): number {
  const s2 = s * s;
  const s3 = s2 * s;
  return (
    (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * dt * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * dt * v1
  );
}

/** Interpolate states `a` (earlier) and `b` at fraction `alpha` in [0, 1]. `b` may equal `a`. */
export function interpolate(a: SimState, b: SimState, alpha: number): RenderState {
  const dt = Math.max(1e-6, b.t - a.t);
  const s = a === b ? 0 : Math.min(1, Math.max(0, alpha));
  const t = a.t + (b.t - a.t) * s;

  // Ball: cubic Hermite on position + velocity for smooth flight; plain lerp across bounces and kicks,
  // where the velocity changes abruptly between ticks.
  const [ax, ay, az] = a.ball.pos;
  const [bx, by, bz] = b.ball.pos;
  const av = a.ball.vel;
  const bv = b.ball.vel;
  const expectedDvz = az > 0.12 || bz > 0.12 ? -G * dt : 0;
  const jump = Math.hypot(bv[0] - av[0], bv[1] - av[1], bv[2] - av[2] - expectedDvz);
  const smooth = jump < 2.0;
  const pos: [number, number, number] = smooth
    ? [
        hermite(ax, av[0], bx, bv[0], dt, s),
        hermite(ay, av[1], by, bv[1], dt, s),
        Math.max(0, hermite(az, av[2], bz, bv[2], dt, s)),
      ]
    : [lerp(ax, bx, s), lerp(ay, by, s), lerp(az, bz, s)];
  const vel: [number, number, number] = [lerp(av[0], bv[0], s), lerp(av[1], bv[1], s), lerp(av[2], bv[2], s)];

  const byId = new Map(b.players.map((p) => [p.id, p]));
  const players: RenderPlayer[] = a.players.map((pa) => {
    const pb = byId.get(pa.id) ?? pa;
    const vx = lerp(pa.vel[0], pb.vel[0], s);
    const vy = lerp(pa.vel[1], pb.vel[1], s);
    const sameAnim = pa.anim === pb.anim && pb.anim_t >= pa.anim_t;
    return {
      id: pa.id,
      x: lerp(pa.pos[0], pb.pos[0], s),
      y: lerp(pa.pos[1], pb.pos[1], s),
      z: lerp(pa.z, pb.z, s),
      vx,
      vy,
      speed: Math.hypot(vx, vy),
      facing: pa.facing + angleDelta(pa.facing, pb.facing) * s,
      anim: pa.anim,
      animT: sameAnim ? lerp(pa.anim_t, pb.anim_t, s) : pa.anim_t + (t - a.t),
      mirror: pa.anim_mirror,
      variant: pa.anim_variant,
      onPitch: pa.on_pitch,
    };
  });

  let referee: RenderState["referee"] = null;
  if (a.referee) {
    const rb = b.referee ?? a.referee;
    referee = {
      x: lerp(a.referee.pos[0], rb.pos[0], s),
      y: lerp(a.referee.pos[1], rb.pos[1], s),
      vx: lerp(a.referee.vel[0], rb.vel[0], s),
      vy: lerp(a.referee.vel[1], rb.vel[1], s),
      facing: a.referee.facing + angleDelta(a.referee.facing, rb.facing) * s,
    };
  }

  return {
    t,
    tick: a.tick,
    base: a,
    ball: { pos, vel, spin: [...a.ball.spin], inPlay: a.ball.in_play },
    players,
    referee,
  };
}

/** Time-ordered buffer of SimStates with interpolated sampling. */
export class StateBuffer {
  private states: SimState[] = [];

  get length(): number {
    return this.states.length;
  }

  get first(): SimState | undefined {
    return this.states[0];
  }

  get last(): SimState | undefined {
    return this.states[this.states.length - 1];
  }

  at(i: number): SimState | undefined {
    return this.states[i];
  }

  clear(): void {
    this.states = [];
  }

  push(s: SimState): void {
    const last = this.last;
    if (!last || s.t > last.t) {
      this.states.push(s);
      return;
    }
    if (s.t === last.t) {
      this.states[this.states.length - 1] = s;
      return;
    }
    // out of order (rare): insert in place
    const i = this.indexBefore(s.t);
    if (this.states[i]?.t === s.t) this.states[i] = s;
    else this.states.splice(i + 1, 0, s);
  }

  setAll(states: SimState[]): void {
    this.states = [...states].sort((p, q) => p.t - q.t);
  }

  /** Drop states older than `t` (keeping one before it for interpolation). */
  trimBefore(t: number): void {
    const i = this.indexBefore(t);
    if (i > 0) this.states.splice(0, i);
  }

  /** Index of the last state with state.t <= t (or 0). */
  indexBefore(t: number): number {
    let lo = 0;
    let hi = this.states.length - 1;
    if (hi < 0) return 0;
    if (t < this.states[0]!.t) return 0;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.states[mid]!.t <= t) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  sample(t: number): RenderState | null {
    const n = this.states.length;
    if (n === 0) return null;
    const i = this.indexBefore(t);
    const a = this.states[i]!;
    const b = this.states[Math.min(i + 1, n - 1)]!;
    if (a === b || t <= a.t) return interpolate(a, a, 0);
    return interpolate(a, b, (t - a.t) / (b.t - a.t));
  }
}
