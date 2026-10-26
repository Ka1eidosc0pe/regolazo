// Sculpted hair silhouettes, built around the fitted head ellipsoid in bind-pose space.
import * as THREE from "three";

export const HAIR_STYLES = ["crop", "buzz", "hightop", "bun", "quiff", "tied", "volume", "shaved"] as const;
export type HairStyle = (typeof HAIR_STYLES)[number];

export const HAIR_COLORS = ["#16110e", "#2b1d14", "#4a3020", "#7a5634", "#b48a52", "#5a2a1a"];

export interface HeadFit {
  center: THREE.Vector3;
  radii: THREE.Vector3;
}

/** Fit the head as an ellipsoid from body-mesh vertices above the neck (bind pose, metres). */
export function fitHead(geometry: THREE.BufferGeometry, minY = 1.6): HeadFit {
  const pos = geometry.getAttribute("position") as THREE.BufferAttribute;
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    if (v.y > minY && Math.abs(v.x) < 0.15) box.expandByPoint(v);
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  // the fit region starts at the jaw; the cranium centre sits higher than the box centre
  center.y += size.y * 0.08;
  return { center, radii: new THREE.Vector3(size.x / 2, size.y * 0.46, size.z / 2) };
}

function hash3(x: number, y: number, z: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
  return s - Math.floor(s);
}

function noise3(x: number, y: number, z: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const xf = x - xi;
  const yf = y - yi;
  const zf = z - zi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const w = zf * zf * (3 - 2 * zf);
  const l = (a: number, b: number, t: number) => a + (b - a) * t;
  const h = (i: number, j: number, k: number) => hash3(xi + i, yi + j, zi + k);
  return l(
    l(l(h(0, 0, 0), h(1, 0, 0), u), l(h(0, 1, 0), h(1, 1, 0), u), v),
    l(l(h(0, 0, 1), h(1, 0, 1), u), l(h(0, 1, 1), h(1, 1, 1), u), v),
    w,
  );
}

/** Hair geometry in bind-pose space for a style; null for "shaved". */
export function hairGeometry(style: HairStyle, head: HeadFit, seed: number): THREE.BufferGeometry | null {
  if (style === "shaved") return null;
  const g = new THREE.SphereGeometry(1, 40, 28);
  const pos = g.getAttribute("position") as THREE.BufferAttribute;
  const d = new THREE.Vector3();
  const { center: c, radii: r } = head;
  for (let i = 0; i < pos.count; i++) {
    d.fromBufferAttribute(pos, i).normalize();
    // hairline: high on the forehead, low at the nape, around the ears on the sides
    const frontness = d.z; // +z = face
    const hairline = frontness > 0 ? 0.18 + frontness * 0.22 : -0.05 + frontness * 0.42;
    const above = THREE.MathUtils.smoothstep(d.y - hairline, -0.04, 0.06);
    let thick = 0.06;
    switch (style) {
      case "buzz":
        thick = 0.03;
        break;
      case "crop":
        thick = 0.07 + 0.05 * Math.max(0, d.y);
        break;
      case "hightop":
        thick = 0.08 + 0.55 * Math.pow(Math.max(0, d.y), 3);
        break;
      case "volume":
        thick = 0.22 + 0.12 * Math.max(0, d.y);
        break;
      case "quiff": {
        const q = Math.exp(-((d.x * d.x) / 0.12 + (d.y - 0.6) ** 2 / 0.12 + (d.z - 0.75) ** 2 / 0.2));
        thick = 0.06 + 0.32 * q;
        break;
      }
      case "bun":
      case "tied":
        thick = 0.05;
        break;
    }
    // sculpted clumps
    thick *= 0.85 + 0.3 * noise3(d.x * 5 + seed, d.y * 5, d.z * 5);
    // continuous: inside the skull below the hairline, rising to the hair surface above it
    const k = 0.9 + (0.1 + thick) * above;
    pos.setXYZ(i, c.x + d.x * r.x * k, c.y + d.y * r.y * k, c.z + d.z * r.z * k);
  }
  g.computeVertexNormals();
  const parts: THREE.BufferGeometry[] = [g];
  if (style === "bun") {
    const b = new THREE.SphereGeometry(r.x * 0.42, 16, 12);
    b.translate(c.x, c.y + r.y * 0.72, c.z - r.z * 0.62);
    parts.push(b);
  }
  if (style === "tied") {
    const t = new THREE.CapsuleGeometry(r.x * 0.2, r.y * 0.9, 6, 12);
    t.rotateX(0.5);
    t.translate(c.x, c.y - r.y * 0.25, c.z - r.z * 1.15);
    parts.push(t);
  }
  if (parts.length === 1) return g;
  return mergeSimple(parts);
}

function mergeSimple(geoms: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const ng = geoms.map((x) => (x.index ? x.toNonIndexed() : x));
  const count = ng.reduce((a, x) => a + x.getAttribute("position").count, 0);
  const pos = new Float32Array(count * 3);
  const nor = new Float32Array(count * 3);
  let o = 0;
  for (const x of ng) {
    pos.set((x.getAttribute("position") as THREE.BufferAttribute).array as Float32Array, o * 3);
    nor.set((x.getAttribute("normal") as THREE.BufferAttribute).array as Float32Array, o * 3);
    o += x.getAttribute("position").count;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  out.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  return out;
}

export function hairMaterial(color: string): THREE.MeshPhysicalMaterial {
  const c = new THREE.Color(color);
  return new THREE.MeshPhysicalMaterial({
    color: c,
    roughness: 0.55,
    sheen: 1,
    sheenRoughness: 0.4,
    sheenColor: c.clone().lerp(new THREE.Color(1, 1, 1), 0.35),
    clearcoat: 0.25,
    clearcoatRoughness: 0.4,
    envMapIntensity: 0.6,
  });
}
