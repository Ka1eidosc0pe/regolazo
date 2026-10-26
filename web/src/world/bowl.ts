// Geometry helpers for the stadium bowl: a profile (outward offset d, height h) swept along a rounded rectangle.
import * as THREE from "three";

export interface PathPoint {
  /** Point on the stand-front path in the world (pitch) frame. */
  x: number;
  y: number;
  /** Outward unit normal (world frame). */
  nx: number;
  ny: number;
  /** Arc length from the start. */
  s: number;
}

/** Rounded rectangle with half extents (a, b) and corner radius r, sampled about every `ds` metres, closed. */
export function roundedRectPath(a: number, b: number, r: number, ds: number): PathPoint[] {
  const pts: PathPoint[] = [];
  let s = 0;
  const push = (x: number, y: number, nx: number, ny: number) => {
    const last = pts[pts.length - 1];
    if (last) s += Math.hypot(x - last.x, y - last.y);
    pts.push({ x, y, nx, ny, s });
  };
  const straight = (x0: number, y0: number, x1: number, y1: number, nx: number, ny: number) => {
    const n = Math.max(1, Math.round(Math.hypot(x1 - x0, y1 - y0) / ds));
    for (let i = 0; i < n; i++) push(x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * i) / n, nx, ny);
  };
  const arc = (cx: number, cy: number, a0: number) => {
    const n = Math.max(2, Math.round((Math.PI / 2) * r / ds));
    for (let i = 0; i < n; i++) {
      const ang = a0 + ((Math.PI / 2) * i) / n;
      push(cx + Math.cos(ang) * r, cy + Math.sin(ang) * r, Math.cos(ang), Math.sin(ang));
    }
  };
  // counter-clockwise starting at the bottom side (y = -b), moving +x
  straight(-a + r, -b, a - r, -b, 0, -1);
  arc(a - r, -b + r, -Math.PI / 2);
  straight(a, -b + r, a, b - r, 1, 0);
  arc(a - r, b - r, 0);
  straight(a - r, b, -a + r, b, 0, 1);
  arc(-a + r, b - r, Math.PI / 2);
  straight(-a, b - r, -a, -b + r, -1, 0);
  arc(-a + r, -b + r, Math.PI);
  return pts;
}

export interface ProfileSeg {
  d0: number;
  h0: number;
  d1: number;
  h1: number;
  color: THREE.Color;
}

/** Sweep profile segments around a closed path into one flat-shaded geometry with vertex colours. */
export function sweep(path: PathPoint[], segs: ProfileSeg[], colorFn?: (seg: ProfileSeg, p: PathPoint, i: number) => THREE.Color): THREE.BufferGeometry {
  const n = path.length;
  const pos: number[] = [];
  const nor: number[] = [];
  const colr: number[] = [];
  const idx: number[] = [];
  let v = 0;
  for (const seg of segs) {
    for (let i = 0; i < n; i++) {
      const p = path[i]!;
      const q = path[(i + 1) % n]!;
      const a0 = [p.x + p.nx * seg.d0, seg.h0, -(p.y + p.ny * seg.d0)];
      const a1 = [p.x + p.nx * seg.d1, seg.h1, -(p.y + p.ny * seg.d1)];
      const b0 = [q.x + q.nx * seg.d0, seg.h0, -(q.y + q.ny * seg.d0)];
      const b1 = [q.x + q.nx * seg.d1, seg.h1, -(q.y + q.ny * seg.d1)];
      // normal: profile direction x path tangent, oriented towards the pitch / up
      const t = new THREE.Vector3(b0[0]! - a0[0]!, 0, b0[2]! - a0[2]!).normalize();
      const u = new THREE.Vector3(a1[0]! - a0[0]!, a1[1]! - a0[1]!, a1[2]! - a0[2]!).normalize();
      const nn = new THREE.Vector3().crossVectors(u, t).normalize();
      const inward = new THREE.Vector3(-p.nx, 0, p.ny);
      if (nn.y < -0.01 || (Math.abs(nn.y) <= 0.01 && nn.dot(inward) < 0)) nn.negate();
      const c = colorFn ? colorFn(seg, p, i) : seg.color;
      for (const w of [a0, a1, b0, b1]) {
        pos.push(w[0]!, w[1]!, w[2]!);
        nor.push(nn.x, nn.y, nn.z);
        colr.push(c.r, c.g, c.b);
      }
      idx.push(v, v + 2, v + 1, v + 1, v + 2, v + 3);
      v += 4;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(colr, 3));
  g.setIndex(idx);
  return g;
}
