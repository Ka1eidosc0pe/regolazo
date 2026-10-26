// World frame (pitch, z up; shared/schemas/common.py) <-> Three.js (y up): (x, y, z) -> (x, z, -y).

export const PITCH_LENGTH = 105;
export const PITCH_WIDTH = 68;
export const HALF_L = PITCH_LENGTH / 2;
export const HALF_W = PITCH_WIDTH / 2;
export const GOAL_WIDTH = 7.32;
export const GOAL_HEIGHT = 2.44;
export const BALL_RADIUS = 0.11;

export type V3 = [number, number, number];

/** World (x, y, z) -> Three.js (x, y, z). */
export function worldToThree(x: number, y: number, z: number): V3 {
  return [x, z, -y];
}

/** Three.js (x, y, z) -> world (x, y, z). */
export function threeToWorld(x: number, y: number, z: number): V3 {
  return [x, -z, y];
}

/**
 * Three.js rotation about +y for a body facing `facing` (radians from world +x, counter-clockwise).
 * Characters are authored facing +z (Three.js local).
 */
export function facingToYaw(facing: number): number {
  return facing + Math.PI / 2;
}

/** Shortest signed angle from a to b, in (-pi, pi]. */
export function angleDelta(a: number, b: number): number {
  let d = (b - a) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d <= -Math.PI) d += 2 * Math.PI;
  return d;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** Deterministic 32-bit hash of a string (FNV-1a). */
export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Small deterministic PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
