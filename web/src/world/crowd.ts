// Instanced crowd: ~30k seated figures on the tier treads, animated in the vertex shader (sway, bob, rise and
// arms up when excited), plus a sparse layer of twinkling phone lights.
import * as THREE from "three";
import { mulberry32 } from "../coords.ts";
import type { PathPoint } from "./bowl.ts";
import { LOWER, UPPER, type Tier } from "./stadium.ts";

export interface Crowd {
  group: THREE.Group;
  count: number;
  /** excitement 0..1 (goals raise it), time in seconds. */
  update(t: number, excitement: number): void;
}

/** A camera-agnostic "card" per spectator; the silhouette (head, torso, lap, raised arms) is drawn in the shader. */
function cardGeometry(): THREE.BufferGeometry {
  // tall enough for a standing spectator; seated ones use the lower part
  const g = new THREE.PlaneGeometry(0.66, 1.85);
  g.translate(0, 0.925, 0.05);
  return g;
}

const PALETTE_HOME = ["#0b3d91", "#0b3d91", "#4cc9f0", "#f2f4f8", "#13254a", "#0b3d91", "#1d2738", "#3a4458"];
const PALETTE_AWAY = ["#c8102e", "#c8102e", "#1a1a1a", "#f5b700", "#5a0a14", "#c8102e"];
const PALETTE_NEUTRAL = ["#2c3340", "#3a4250", "#59606c", "#1d2230", "#6b5a4a", "#40506a", "#7a7f88", "#5b3a2e", "#2f4a3a", "#8a8170", "#3b2f4f", "#6d6f73"];

export function createCrowd(path: PathPoint[]): Crowd {
  const rng = mulberry32(1234);
  const seats: { x: number; y: number; h: number; nx: number; ny: number; away: boolean; row: number; rows: number; lower: boolean }[] = [];
  const spacing = 0.56;
  const place = (tier: Tier, occupancy: number) => {
    for (let r = 0; r < tier.rows; r++) {
      const d = tier.d0 + r * tier.tread + 0.42;
      const h = tier.h0 + r * tier.riser;
      // walk the offset path at constant spacing
      let acc = 0;
      for (let i = 0; i < path.length; i++) {
        const p = path[i]!;
        const q = path[(i + 1) % path.length]!;
        const ax = p.x + p.nx * d;
        const ay = p.y + p.ny * d;
        const bx = q.x + q.nx * d;
        const by = q.y + q.ny * d;
        const len = Math.hypot(bx - ax, by - ay);
        let s = acc;
        while (s < len) {
          const f = s / len;
          const x = ax + (bx - ax) * f;
          const y = ay + (by - ay) * f;
          // aisles every ~14 m and the tunnel gaps
          const aisle = Math.abs(((p.s + s) % 14) - 7) < 0.5;
          if (!aisle && rng() < occupancy) {
            const away = x > 40 && y < -8;
            seats.push({ x, y, h, nx: p.nx, ny: p.ny, away, row: r, rows: tier.rows, lower: tier === LOWER });
          }
          s += spacing;
        }
        acc = s - len;
      }
    }
  };
  place(LOWER, 0.9);
  place(UPPER, 0.86);

  const geo = cardGeometry();
  const count = seats.length;
  const phase = new Float32Array(count);
  // 0 seated, 1 seated broad, 2 standing (away end and behind the goals stand; never in the back rows of the lower
  // tier, where heads would cover the LED fascia)
  const kind = new Float32Array(count);
  const mat = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide, alphaToCoverage: true });
  const uniforms = { uTime: { value: 0 }, uExcite: { value: 0 } };
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
uniform float uTime;
uniform float uExcite;
attribute float aPhase;
attribute float aKind;
varying vec2 vCard;
varying float vHype;
varying float vPhase;
varying float vDim;
varying float vKind;
varying float vFar;`,
      )
      .replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>
  float ph = aPhase * 6.2831;
  float hype = clamp(uExcite * (0.6 + 0.6 * fract(aPhase * 7.13)), 0.0, 1.0);
  float bounce = max(0.0, sin(uTime * 7.5 + ph)) * 0.22 * hype;
  vCard = vec2(position.x, position.y);
  vHype = hype * (0.75 + 0.25 * sin(uTime * 6.0 + ph));
  vPhase = aPhase;
  vKind = aKind;
  transformed.x += sin(uTime * 1.3 + ph) * 0.03;
  // seated fans rise when excited; standing ones bounce
  transformed.y += (aKind > 1.5 ? 0.0 : hype * 0.3) + bounce;
  vec4 crowdWorld = modelMatrix * instanceMatrix * vec4(0.0, 0.9, 0.0, 1.0);
  vFar = smoothstep(55.0, 150.0, distance(crowdWorld.xyz, cameraPosition));`,
      )
      .replace(
        "#include <color_vertex>",
        `#include <color_vertex>
  float wy = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).y;
  vDim = 0.42 * mix(1.0, 0.55, smoothstep(4.0, 30.0, wy));`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec2 vCard;
varying float vHype;
varying float vPhase;
varying float vDim;
varying float vKind;
varying float vFar;
float sdBox(vec2 p, vec2 b, float r) { vec2 d = abs(p) - b + r; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r; }
float sdSeg(vec2 p, vec2 a, vec2 b, float r) { vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - r; }`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  vec2 q = vCard;
  bool standing = vKind > 1.5;
  float broad = vKind > 0.5 && vKind < 1.5 ? 1.0 : 0.0;
  float vary = fract(vPhase * 31.7);
  // seated: head, torso, lap; standing: head, torso, legs. Width varies per fan.
  float wid = 0.17 + 0.04 * broad + 0.025 * vary;
  float lift = standing ? 0.6 : 0.0;
  float head = length(q - vec2(0.0, 1.0 + lift)) - (0.1 + 0.012 * vary);
  float torso = sdBox(q - vec2(0.0, 0.64 + lift), vec2(wid, 0.24), 0.08);
  float lap = standing ? min(sdSeg(q, vec2(0.085, 0.05), vec2(0.08, 0.9), 0.07), sdSeg(q, vec2(-0.085, 0.05), vec2(-0.08, 0.9), 0.07))
                       : sdBox(q - vec2(0.0, 0.38), vec2(wid - 0.01, 0.08), 0.04);
  float up = vHype;
  vec2 sh = vec2(wid - 0.01, 0.8 + lift);
  vec2 hand = vec2(wid + 0.06 + 0.05 * up, 0.8 + lift + 0.42 * up);
  float arms = min(sdSeg(q, sh, hand, 0.045), sdSeg(q, sh * vec2(-1.0, 1.0), hand * vec2(-1.0, 1.0), 0.045));
  arms = mix(1.0, arms, step(0.15, up));
  // some fans wear a cap, some a scarf in the club colour
  float cap = fract(vPhase * 53.1) < 0.12 ? sdBox(q - vec2(0.0, 1.07 + lift), vec2(0.11, 0.035), 0.03) : 1.0;
  float body = min(min(torso, lap), arms);
  float d = min(min(head, body), cap);
  float aa = max(fwidth(d), 1e-4);
  float mask = 1.0 - smoothstep(-aa, aa, d);
  if (mask < 0.5) discard;
  diffuseColor.a = mask;
  // skin tones for heads (and raised forearms), cylindrical shading for a rounded read
  float sk = fract(vPhase * 13.7);
  vec3 skin = mix(vec3(0.85, 0.62, 0.48), vec3(0.28, 0.17, 0.11), sk);
  if (head < 0.0 && cap > 0.0) diffuseColor.rgb = skin * 0.8;
  float scarf = fract(vPhase * 71.3) < 0.18 ? 1.0 : 0.0;
  if (scarf > 0.5 && abs(q.y - (0.86 + lift)) < 0.045 && torso < 0.0) diffuseColor.rgb = mix(vec3(0.95), diffuseColor.rgb * 1.6, step(0.5, fract(q.x * 18.0)));
  if (standing && lap < 0.0 && torso > 0.0) diffuseColor.rgb = vec3(0.07, 0.075, 0.09); // trousers
  float prof = 1.0 - pow(clamp(abs(q.x) / 0.22, 0.0, 1.0), 2.0) * 0.45;
  float top = 0.85 + 0.25 * smoothstep(0.3, 1.1, q.y);
  diffuseColor.rgb *= prof * top * vDim;
  // distance fade: far stands read as a soft texture, not shimmering detail
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.05, 0.06, 0.085) * (0.8 + 0.4 * vary), vFar * 0.45);`,
      );
  };
  mat.customProgramCacheKey = () => "regolazo-crowd-v3";
  const mesh = new THREE.InstancedMesh(geo, mat, count);
  const m4 = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const color = new THREE.Color();
  const up = new THREE.Vector3(0, 1, 0);
  seats.forEach((s, i) => {
    const yaw = Math.atan2(-s.nx, s.ny) + (rng() - 0.5) * 0.12;
    q.setFromAxisAngle(up, yaw);
    const behindGoal = Math.abs(s.x) > 52;
    const standChance = s.away ? 0.6 : behindGoal ? 0.3 : 0.08;
    const backRows = s.lower && s.row >= s.rows - 4;
    kind[i] = !backRows && rng() < standChance ? 2 : rng() < 0.3 ? 1 : 0;
    const sc = 0.86 + rng() * 0.26;
    m4.compose(new THREE.Vector3(s.x, s.h, -s.y), q, new THREE.Vector3(sc, sc, sc));
    mesh.setMatrixAt(i, m4);
    const pal = s.away ? PALETTE_AWAY : rng() < 0.62 ? PALETTE_HOME : PALETTE_NEUTRAL;
    color.set(pal[Math.floor(rng() * pal.length)]!);
    color.multiplyScalar(0.75 + rng() * 0.35);
    mesh.setColorAt(i, color);
    phase[i] = rng();
  });
  geo.setAttribute("aPhase", new THREE.InstancedBufferAttribute(phase, 1));
  geo.setAttribute("aKind", new THREE.InstancedBufferAttribute(kind, 1));
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.frustumCulled = false;
  mesh.name = "crowd";

  // phone lights
  const nPhones = Math.floor(count * 0.012);
  const pPos = new Float32Array(nPhones * 3);
  const pPh = new Float32Array(nPhones);
  for (let i = 0; i < nPhones; i++) {
    const s = seats[Math.floor(rng() * count)]!;
    pPos.set([s.x - s.nx * 0.2, s.h + 1.15, -(s.y - s.ny * 0.2)], i * 3);
    pPh[i] = rng();
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
  pg.setAttribute("aPhase", new THREE.BufferAttribute(pPh, 1));
  const pm = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uTime: uniforms.uTime, uScale: { value: 900 } },
    vertexShader: /* glsl */ `
      uniform float uTime; uniform float uScale; attribute float aPhase; varying float vI;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float tw = fract(uTime * (0.08 + 0.1 * aPhase) + aPhase * 13.0);
        vI = smoothstep(0.0, 0.05, tw) * (1.0 - smoothstep(0.1, 0.35, tw));
        gl_PointSize = clamp(uScale * 0.06 / -mv.z, 1.5, 6.0);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      varying float vI;
      void main() {
        vec2 c = gl_PointCoord - 0.5; float d = dot(c, c);
        float a = exp(-d * 18.0) * vI;
        gl_FragColor = vec4(vec3(1.0, 0.97, 0.9) * a * 6.0, 1.0);
      }`,
  });
  const phones = new THREE.Points(pg, pm);
  phones.frustumCulled = false;

  const group = new THREE.Group();
  group.add(mesh, phones);
  return {
    group,
    count,
    update(t, excitement) {
      uniforms.uTime.value = t;
      uniforms.uExcite.value = excitement;
    },
  };
}
