// Rewind-only world layers: the pitch-control glow on the grass, translucent cyan "reality" ghosts, and the
// preview arc of the chosen ball. All are hidden outside a Rewind, so the approved pitch look is untouched.
import * as THREE from "three";
import { worldToThree } from "../coords.ts";
import type { RenderState } from "../source/interpolate.ts";
import type { PitchControlGrid } from "../types/generated/contracts.ts";

const REWIND_CYAN = new THREE.Color("#38e1ff");
const OWN = new THREE.Color("#38e1ff");
const OPP = new THREE.Color("#ff7a1a");

export class PitchControlLayer {
  readonly mesh: THREE.Mesh;
  private readonly mat: THREE.MeshBasicMaterial;
  private tex: THREE.DataTexture | null = null;
  private target = 0;
  private opacity = 0;

  constructor() {
    this.mat = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, toneMapped: false, opacity: 0 });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(105, 68), this.mat);
    this.mesh.rotation.x = -Math.PI / 2;
    this.mesh.position.y = 0.025;
    this.mesh.renderOrder = 2;
    this.mesh.visible = false;
    this.mesh.name = "rewind-pitch-control";
  }

  /** Show the grid from the point of view of `team` (cyan where it controls the ground, flare where the opponent does). */
  set(grid: PitchControlGrid, team: "home" | "away"): void {
    const { nx, ny } = grid;
    const data = new Uint8Array(nx * ny * 4);
    const c = new THREE.Color();
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const raw = grid.values[iy * nx + ix] ?? 0.5;
        const p = grid.team === team ? raw : 1 - raw;
        const d = (p - 0.5) * 2;
        c.copy(d >= 0 ? OWN : OPP);
        const i = (iy * nx + ix) * 4;
        data[i] = Math.round(c.r * 255);
        data[i + 1] = Math.round(c.g * 255);
        data[i + 2] = Math.round(c.b * 255);
        data[i + 3] = Math.round(Math.min(1, Math.abs(d) ** 1.4) * 120);
      }
    }
    this.tex?.dispose();
    const tex = new THREE.DataTexture(data, nx, ny, THREE.RGBAFormat);
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    this.tex = tex;
    this.mat.map = tex;
    this.mat.needsUpdate = true;
    const w = nx * grid.dx;
    const h = ny * grid.dy;
    this.mesh.scale.set(w / 105, h / 68, 1);
    const cx = grid.x0 - grid.dx / 2 + w / 2;
    const cy = grid.y0 - grid.dy / 2 + h / 2;
    const [tx, , tz] = worldToThree(cx, cy, 0);
    this.mesh.position.x = tx;
    this.mesh.position.z = tz;
  }

  show(on: boolean): void {
    this.target = on ? 0.75 : 0;
    if (on) this.mesh.visible = true;
  }

  update(dt: number): void {
    this.opacity += (this.target - this.opacity) * Math.min(1, dt * 4);
    this.mat.opacity = this.opacity;
    if (this.target === 0 && this.opacity < 0.01) this.mesh.visible = false;
  }
}

/** Reality as translucent cyan figures (and ball) while the alternate reality plays in full colour. */
export class RealityGhosts {
  readonly group = new THREE.Group();
  private readonly bodies = new Map<string, THREE.Mesh>();
  private readonly ball: THREE.Mesh;
  private readonly mat: THREE.MeshBasicMaterial;
  private readonly geo = new THREE.CapsuleGeometry(0.32, 1.15, 4, 10);

  constructor() {
    this.mat = new THREE.MeshBasicMaterial({
      color: REWIND_CYAN,
      transparent: true,
      opacity: 0.32,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    });
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(0.2, 16, 12), this.mat);
    this.group.add(this.ball);
    this.group.visible = false;
    this.group.name = "rewind-ghosts";
  }

  show(on: boolean): void {
    this.group.visible = on;
  }

  update(s: RenderState | null): void {
    if (!s || !this.group.visible) return;
    const seen = new Set<string>();
    for (const p of s.players) {
      if (!p.onPitch) continue;
      let m = this.bodies.get(p.id);
      if (!m) {
        m = new THREE.Mesh(this.geo, this.mat);
        this.bodies.set(p.id, m);
        this.group.add(m);
      }
      const [x, , z] = worldToThree(p.x, p.y, 0);
      m.position.set(x, 0.92, z);
      m.visible = true;
      seen.add(p.id);
    }
    for (const [id, m] of this.bodies) if (!seen.has(id)) m.visible = false;
    const [bx, by, bz] = worldToThree(...s.ball.pos);
    this.ball.position.set(bx, Math.max(0.2, by), bz);
  }
}

/** The noise-free ball path of the chosen option, as a glowing tube. */
export class PreviewArc {
  readonly group = new THREE.Group();
  private readonly mat = new THREE.MeshBasicMaterial({ color: new THREE.Color("#d7ff3a"), transparent: true, opacity: 0.9, toneMapped: false, depthWrite: false });
  private mesh: THREE.Mesh | null = null;

  constructor() {
    this.group.name = "rewind-preview-arc";
  }

  set(points: [number, number, number][] | null): void {
    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh = null;
    }
    if (!points || points.length < 2) return;
    const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...worldToThree(p[0], p[1], Math.max(0.12, p[2])))));
    this.mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.min(120, points.length * 3), 0.2, 8, false), this.mat);
    this.group.add(this.mesh);
  }
}

/** A straight fallback arc (ball -> target) when the engine gives no preview samples. */
export function straightArc(from: [number, number, number], to: [number, number, number], lift: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let i = 0; i <= 24; i++) {
    const s = i / 24;
    out.push([from[0] + (to[0] - from[0]) * s, from[1] + (to[1] - from[1]) * s, from[2] + (to[2] - from[2]) * s + lift * 4 * s * (1 - s)]);
  }
  return out;
}
