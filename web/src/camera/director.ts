// Camera language (docs/art_direction.md section 7): broadcast (gantry pan/tilt/zoom with lead and danger zoom),
// tactical (high, top-down-ish), replay/cinematic (low long-lens orbit with DOF). Eased transitions between modes.
// Everything is driven by explicit dt, so a fixed frame sequence gives identical cameras (recording mode).
import * as THREE from "three";
import { clamp, HALF_L } from "../coords.ts";
import type { RenderState } from "../source/interpolate.ts";

export type CameraMode = "broadcast" | "tactical" | "replay" | "free";

/** Main camera gantry: world (0, -58, 21) -> Three.js (0, 21, 58). */
export const GANTRY = new THREE.Vector3(0, 21, 58);

interface Pose {
  pos: THREE.Vector3;
  look: THREE.Vector3;
  fov: number;
}

/** Critically damped smoothing (Unity SmoothDamp), stable for any dt. */
class Damp {
  value: number;
  vel = 0;
  constructor(v: number) {
    this.value = v;
  }
  step(target: number, smoothTime: number, dt: number, maxSpeed = Infinity): number {
    const st = Math.max(1e-4, smoothTime);
    const omega = 2 / st;
    const x = omega * dt;
    const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    let change = this.value - target;
    const maxChange = maxSpeed * st;
    change = clamp(change, -maxChange, maxChange);
    const tgt = this.value - change;
    const temp = (this.vel + omega * change) * dt;
    this.vel = (this.vel - omega * temp) * exp;
    let out = tgt + (change + temp) * exp;
    if (target - this.value > 0 === out > target) {
      out = target;
      this.vel = (out - target) / Math.max(dt, 1e-6);
    }
    this.value = out;
    return out;
  }
  set(v: number): void {
    this.value = v;
    this.vel = 0;
  }
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export class CameraDirector {
  readonly camera: THREE.PerspectiveCamera;
  mode: CameraMode = "broadcast";
  private aimX = new Damp(0);
  private aimY = new Damp(0);
  private width = new Damp(40);
  private tacX = new Damp(0);
  private orbit = 0;
  // replay smoothing: subject weight (player vs ball), look-at centre and camera position
  private repW = new Damp(0.7);
  private repX = new Damp(0);
  private repY = new Damp(0);
  private repPX = new Damp(0);
  private repPZ = new Damp(0);
  private repPrimed = false;
  private transition: { from: Pose; t: number; dur: number } | null = null;
  private free: Pose = { pos: new THREE.Vector3(0, 3, 20), look: new THREE.Vector3(), fov: 30 };
  replayTarget: string | null = null;
  /** Focus distance for depth of field (replay). */
  focusDistance = 30;
  private primed = false;

  constructor(aspect: number) {
    this.camera = new THREE.PerspectiveCamera(30, aspect, 0.3, 4000);
    this.camera.position.copy(GANTRY);
  }

  setAspect(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  setMode(mode: CameraMode, transitionS = 1.3): void {
    if (mode === this.mode) return;
    if (mode === "replay") this.repPrimed = false;
    this.transition = transitionS > 0 ? { from: this.currentPose(), t: 0, dur: transitionS } : null;
    this.mode = mode;
  }

  /** A fixed pose (art previews, close-ups). Switches to "free". */
  setFreePose(pos: THREE.Vector3, look: THREE.Vector3, fov: number, transitionS = 0): void {
    this.free = { pos: pos.clone(), look: look.clone(), fov };
    this.setMode("free", transitionS);
    if (transitionS === 0) this.transition = null;
  }

  private currentPose(): Pose {
    const look = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion).multiplyScalar(30).add(this.camera.position);
    return { pos: this.camera.position.clone(), look, fov: this.camera.fov };
  }

  /** Forget all smoothing state (seek, rewind, recording start). */
  reset(): void {
    this.primed = false;
    this.repPrimed = false;
    this.orbit = 0;
    this.transition = null;
  }

  /** Snap the broadcast smoothing to the current state (after seeks / at recording start). */
  prime(state: RenderState): void {
    const { tx, ty, w } = this.broadcastTargets(state);
    this.aimX.set(tx);
    this.aimY.set(ty);
    this.width.set(w);
    this.tacX.set(clamp(state.ball.pos[0] * 0.55, -20, 20));
    this.primed = true;
  }

  private broadcastTargets(state: RenderState): { tx: number; ty: number; w: number } {
    const [bx, by, bz] = state.ball.pos;
    const [vx, vy] = state.ball.vel;
    const lead = 0.55;
    // drift towards the side the team in possession attacks
    const team = state.base.possession.team;
    const attackSign = team ? (state.base.attacking_positive_x === team ? 1 : -1) : 0;
    let tx = bx + clamp(vx * lead, -12, 12) + attackSign * 4;
    let ty = by + clamp(vy * lead, -8, 8);
    tx = clamp(tx, -HALF_L + 6, HALF_L - 6);
    ty = clamp(ty * 0.85, -22, 22);
    // zoom: spread of nearby players, tighter in the final third, wider on long balls
    let spread = 0;
    for (const p of state.players) {
      const d = Math.hypot(p.x - bx, p.y - by);
      if (d < 22) spread = Math.max(spread, d);
    }
    const speed = Math.hypot(vx, vy);
    let w = 22 + spread * 0.7 + Math.max(0, speed - 12) * 0.8 + Math.min(bz, 8) * 1.0;
    const danger = clamp((Math.abs(bx) - 30) / 18, 0, 1);
    w -= danger * 7;
    return { tx, ty, w: clamp(w, 24, 50) };
  }

  private broadcastPose(state: RenderState, dt: number): Pose {
    if (!this.primed) this.prime(state);
    const { tx, ty, w } = this.broadcastTargets(state);
    const ax = this.aimX.step(tx, 0.75, dt, 40);
    const ay = this.aimY.step(ty, 0.9, dt, 25);
    const width = this.width.step(w, 1.2, dt, 30);
    const look = new THREE.Vector3(ax, 0, -ay);
    const dist = GANTRY.distanceTo(look);
    const hfov = 2 * Math.atan(width / 2 / dist);
    const vfov = (2 * Math.atan(Math.tan(hfov / 2) / this.camera.aspect) * 180) / Math.PI;
    return { pos: GANTRY.clone(), look, fov: vfov };
  }

  private tacticalPose(state: RenderState, dt: number): Pose {
    if (!this.primed) this.prime(state);
    const x = this.tacX.step(clamp(state.ball.pos[0] * 0.55, -20, 20), 1.4, dt);
    return { pos: new THREE.Vector3(x, 74, 52), look: new THREE.Vector3(x, 0, -2), fov: 44 };
  }

  private replayPose(state: RenderState, dt: number): Pose {
    this.orbit += dt * 0.12;
    const tgtPlayer = this.replayTarget ? state.players.find((p) => p.id === this.replayTarget) : undefined;
    const [bx, by] = state.ball.pos;
    const wTarget = state.ball.inPlay ? 0.7 : 1.0;
    const sx = tgtPlayer ? (tx: number) => tgtPlayer.x * tx + bx * (1 - tx) : () => bx;
    const sy = tgtPlayer ? (tx: number) => tgtPlayer.y * tx + by * (1 - tx) : () => by;
    if (!this.repPrimed) {
      this.repW.set(wTarget);
      this.repX.set(sx(wTarget));
      this.repY.set(sy(wTarget));
    }
    // everything is damped: the subject weight changes smoothly (e.g. when the ball goes dead) and so do the
    // look-at centre and the camera position, so the shot never jumps between frames
    const w = this.repW.step(wTarget, 0.6, dt);
    const cx = this.repX.step(sx(w), 0.35, dt, 30);
    const cy = this.repY.step(sy(w), 0.35, dt, 30);
    const look = new THREE.Vector3(cx, 1.1, -cy);
    const r = 11;
    // stay on the pitch side of the subject, looking outwards towards the stands
    const base = Math.atan2(-cy * 0.6 - 30, -cx * 0.8);
    const a = base + Math.sin(this.orbit) * 0.6;
    const px = cx + Math.cos(a) * r;
    const pz = -(cy + Math.sin(a) * r);
    if (!this.repPrimed) {
      this.repPX.set(px);
      this.repPZ.set(pz);
      this.repPrimed = true;
    }
    const pos = new THREE.Vector3(this.repPX.step(px, 0.5, dt, 25), 2.2, this.repPZ.step(pz, 0.5, dt, 25));
    this.focusDistance = pos.distanceTo(look);
    return { pos, look, fov: 17 };
  }

  update(state: RenderState | null, dt: number): void {
    let pose: Pose;
    if (this.mode === "free" || !state) pose = this.free;
    else if (this.mode === "tactical") pose = this.tacticalPose(state, dt);
    else if (this.mode === "replay") pose = this.replayPose(state, dt);
    else pose = this.broadcastPose(state, dt);
    if (this.mode === "free") this.focusDistance = pose.pos.distanceTo(pose.look);

    if (this.transition) {
      this.transition.t += dt;
      const k = ease(clamp(this.transition.t / this.transition.dur, 0, 1));
      const f = this.transition.from;
      pose = {
        pos: f.pos.clone().lerp(pose.pos, k),
        look: f.look.clone().lerp(pose.look, k),
        fov: f.fov + (pose.fov - f.fov) * k,
      };
      if (this.transition.t >= this.transition.dur) this.transition = null;
    }
    this.camera.position.copy(pose.pos);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(pose.look);
    if (Math.abs(this.camera.fov - pose.fov) > 1e-6) {
      this.camera.fov = pose.fov;
      this.camera.updateProjectionMatrix();
    }
    this.camera.updateMatrixWorld(true);
  }

  /** Point the shadow cameras at: the current look-at point on the ground. */
  groundFocus(): THREE.Vector3 {
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const p = this.camera.position;
    const t = dir.y < -1e-3 ? -p.y / dir.y : 60;
    return p.clone().add(dir.multiplyScalar(Math.min(t, 140)));
  }
}
