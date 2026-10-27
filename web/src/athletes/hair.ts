// Sculpted hair silhouettes on the cranium ellipsoid (bind-pose space, from athlete_clips.json `skull_m`).
// Each style is a shell over the skull whose thickness follows the style; the shell meets the scalp tangentially
// at a shaped hairline (forehead, temples, sideburns, nape), so it reads as hair rather than a helmet. The
// material adds combed grooves (or curl bumps) and fades into the skin tone where the style is shaved short.
import * as THREE from "three";

export const HAIR_STYLES = ["crop", "fade", "curls", "bun", "ponytail", "shaved", "quiff"] as const;
export type HairStyle = (typeof HAIR_STYLES)[number];

export const HAIR_COLORS = ["#16110e", "#2b1d14", "#4a3020", "#7a5634", "#b48a52", "#5a2a1a"];

export interface HeadFit {
  center: THREE.Vector3;
  radii: THREE.Vector3;
}

/** Fallback when the manifest has no skull: fit an ellipsoid to body vertices above the neck. */
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

const ss = THREE.MathUtils.smoothstep;

/**
 * Hairline as a height on the unit sphere of the skull (d.y above which hair grows), by direction:
 * high on the forehead, dipping at the temples into sideburns, low at the nape.
 */
function hairline(d: THREE.Vector3, style: HairStyle): number {
  const front = Math.max(0, d.z);
  const back = Math.max(0, -d.z);
  const side = Math.abs(d.x);
  let h = 0.32 * front * front - 0.48 * back - 0.02 + 0.18 * front * side;
  // sideburn notch in front of the ears
  h -= 0.22 * Math.exp(-(((d.z - 0.25) / 0.18) ** 2)) * ss(side, 0.6, 0.85);
  if (style === "fade" || style === "shaved") h += 0.04;
  return h;
}

/** Thickness (fraction of the skull radius) and short-side "fade" (0 = full colour, 1 = skin) per direction. */
function profile(style: HairStyle, d: THREE.Vector3, seed: number): { t: number; fade: number } {
  const top = Math.max(0, d.y);
  const n = noise3(d.x * 4 + seed, d.y * 4, d.z * 4);
  switch (style) {
    case "crop":
      return { t: 0.045 + 0.05 * top + 0.012 * n, fade: 0 };
    case "quiff": {
      const q = Math.exp(-((d.x * d.x) / 0.1 + (d.y - 0.55) ** 2 / 0.1 + (d.z - 0.7) ** 2 / 0.16));
      return { t: 0.04 + 0.04 * top + 0.22 * q + 0.01 * n, fade: ss(-d.y, -0.25, 0.1) * 0.55 };
    }
    case "fade":
      // a short top over clipper-short sides that fade into the skin
      return { t: 0.012 + 0.05 * ss(top, 0.35, 0.8), fade: 1 - ss(top, 0.2, 0.65) };
    case "curls": {
      // tight curls: many small rounded bumps on a full volume
      const c = Math.abs(Math.sin(d.x * 21 + seed) * Math.sin(d.y * 21) * Math.sin(d.z * 21 + seed * 0.5));
      return { t: 0.1 + 0.07 * top + 0.05 * Math.pow(c, 0.6) + 0.02 * n, fade: 0 };
    }
    case "bun":
    case "ponytail":
      return { t: 0.035 + 0.02 * top, fade: 0 };
    case "shaved":
      return { t: 0.006, fade: 0.62 };
  }
}

/** Hair geometry in bind-pose space (positions, normals, aFade). */
export function hairGeometry(style: HairStyle, head: HeadFit, seed: number): THREE.BufferGeometry {
  const g = new THREE.SphereGeometry(1, 48, 34);
  const pos = g.getAttribute("position") as THREE.BufferAttribute;
  const fade = new Float32Array(pos.count);
  const d = new THREE.Vector3();
  const { center: c, radii: r } = head;
  for (let i = 0; i < pos.count; i++) {
    d.fromBufferAttribute(pos, i).normalize();
    const hl = hairline(d, style);
    // 0 below the hairline, rising smoothly over a band above it: the shell meets the scalp tangentially
    const grow = ss(d.y - hl, -0.03, 0.1);
    const p = profile(style, d, seed);
    // just inside the scalp where there is no hair (hidden), the style's thickness where there is
    // (and pulled well inside under the jaw, where the neck is narrower than the cranium)
    const inside = 0.985 - 0.45 * ss(-d.y, 0.25, 0.7) * (1 - grow);
    const k = inside + (0.015 + p.t) * grow;
    pos.setXYZ(i, c.x + d.x * r.x * k, c.y + d.y * r.y * k, c.z + d.z * r.z * k);
    fade[i] = Math.max(p.fade, 1 - ss(grow, 0.0, 0.6) * 0.85);
  }
  g.setAttribute("aFade", new THREE.BufferAttribute(fade, 1));
  g.computeVertexNormals();
  const parts: THREE.BufferGeometry[] = [g];
  const extra = (geo: THREE.BufferGeometry) => {
    geo.setAttribute("aFade", new THREE.BufferAttribute(new Float32Array(geo.getAttribute("position").count), 1));
    parts.push(geo);
  };
  if (style === "bun") {
    const b = new THREE.SphereGeometry(r.x * 0.42, 20, 14);
    b.scale(1, 0.85, 1);
    b.translate(c.x, c.y + r.y * 0.62, c.z - r.z * 0.78);
    extra(b);
  }
  if (style === "ponytail") {
    const tie = new THREE.SphereGeometry(r.x * 0.2, 12, 10);
    tie.translate(c.x, c.y + r.y * 0.25, c.z - r.z * 1.02);
    extra(tie);
    const tail = new THREE.CapsuleGeometry(r.x * 0.2, r.y * 1.1, 6, 12);
    // taper the tail towards its tip
    const tp = tail.getAttribute("position") as THREE.BufferAttribute;
    for (let i = 0; i < tp.count; i++) {
      const y = tp.getY(i);
      const s = 1 - 0.55 * ss(-y, -r.y * 0.5, r.y * 0.7);
      tp.setXYZ(i, tp.getX(i) * s, y, tp.getZ(i) * s);
    }
    tail.computeVertexNormals();
    tail.rotateX(0.35);
    tail.translate(c.x, c.y - r.y * 0.45, c.z - r.z * 1.18);
    extra(tail);
  }
  return parts.length === 1 ? g : mergeSimple(parts);
}

function mergeSimple(geoms: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const ng = geoms.map((x) => (x.index ? x.toNonIndexed() : x));
  const count = ng.reduce((a, x) => a + x.getAttribute("position").count, 0);
  const pos = new Float32Array(count * 3);
  const nor = new Float32Array(count * 3);
  const fad = new Float32Array(count);
  let o = 0;
  for (const x of ng) {
    pos.set((x.getAttribute("position") as THREE.BufferAttribute).array as Float32Array, o * 3);
    nor.set((x.getAttribute("normal") as THREE.BufferAttribute).array as Float32Array, o * 3);
    fad.set((x.getAttribute("aFade") as THREE.BufferAttribute).array as Float32Array, o);
    o += x.getAttribute("position").count;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  out.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  out.setAttribute("aFade", new THREE.BufferAttribute(fad, 1));
  return out;
}

/** Hair material: combed grooves (curls: bumps), sheen, and a fade into the skin tone on short sides. */
export function hairMaterial(color: string, skin: string, style: HairStyle): THREE.MeshPhysicalMaterial {
  const c = new THREE.Color(color);
  const m = new THREE.MeshPhysicalMaterial({
    color: c,
    roughness: 0.5,
    sheen: 1,
    sheenRoughness: 0.35,
    sheenColor: c.clone().lerp(new THREE.Color(1, 1, 1), 0.3),
    clearcoat: 0.15,
    clearcoatRoughness: 0.45,
    envMapIntensity: 0.6,
  });
  const uniforms = {
    uSkin: { value: new THREE.Color(skin) },
    uCurl: { value: style === "curls" ? 1 : 0 },
  };
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float aFade;\nvarying float vFade;\nvarying vec3 vHairP;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\n  vFade = aFade;\n  vHairP = position;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform vec3 uSkin;
uniform float uCurl;
varying float vFade;
varying vec3 vHairP;
vec3 hairPerturb(vec3 surfPos, vec3 surfNorm, float h, float faceDir) {
  vec3 sx = dFdx(surfPos); vec3 sy = dFdy(surfPos);
  vec3 r1 = cross(sy, surfNorm); vec3 r2 = cross(surfNorm, sx);
  float det = dot(sx, r1) * faceDir;
  vec3 grad = sign(det) * (dFdx(h) * r1 + dFdy(h) * r2);
  return normalize(abs(det) * surfNorm - grad);
}`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  // short sides fade into the skin tone, with a little stubble darkening
  diffuseColor.rgb = mix(diffuseColor.rgb, uSkin * 0.78, clamp(vFade, 0.0, 1.0));
  // combed strands flow front to back; curls are small bumps
  float across = vHairP.x * 900.0 + sin(vHairP.z * 60.0) * 2.0;
  float strands = sin(across) * 0.5 + 0.5;
  float curls = sin(vHairP.x * 520.0) * sin(vHairP.y * 520.0) * sin(vHairP.z * 520.0);
  float hairH = mix(strands * 0.00025, curls * 0.0006, uCurl) * (1.0 - vFade);
  float fwH = length(fwidth(vHairP)) * 900.0;
  hairH *= 1.0 - smoothstep(0.6, 2.0, fwH);
  diffuseColor.rgb *= mix(0.9 + 0.1 * strands, 1.0, uCurl);`,
      )
      .replace("#include <normal_fragment_maps>", "#include <normal_fragment_maps>\n  normal = hairPerturb(-vViewPosition, normal, hairH, faceDirection);");
  };
  m.customProgramCacheKey = () => "regolazo-hair-v2";
  return m;
}
