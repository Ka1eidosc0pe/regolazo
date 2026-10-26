// Lantern Park: two-tier bowl, hospitality glass, LED fascia, cantilevered roof with a floodlight ring, haze cones.
import * as THREE from "three";
import { mulberry32 } from "../coords.ts";
import { roundedRectPath, sweep, type PathPoint, type ProfileSeg } from "./bowl.ts";

export const STAND_A = 61.5;
export const STAND_B = 43.5;
export const STAND_R = 12;

export interface Tier {
  rows: number;
  d0: number;
  h0: number;
  tread: number;
  riser: number;
}

export const LOWER: Tier = { rows: 22, d0: 0.3, h0: 1.3, tread: 0.8, riser: 0.42 };
export const UPPER: Tier = { rows: 24, d0: 19.4, h0: 16.6, tread: 0.8, riser: 0.56 };
export const ROOF = { dIn: 5.5, hIn: 35.2, dOut: 41, hOut: 34.0 };

export interface Stadium {
  group: THREE.Group;
  path: PathPoint[];
  /** Floodlight head positions (Three.js) and aim points, for lights and haze. */
  lamps: { pos: THREE.Vector3; aim: THREE.Vector3 }[];
  update(t: number): void;
}

const C = (hex: string) => new THREE.Color(hex);

function tierSegs(t: Tier, seat: THREE.Color, riser: THREE.Color): ProfileSeg[] {
  const segs: ProfileSeg[] = [];
  for (let i = 0; i < t.rows; i++) {
    const d = t.d0 + i * t.tread;
    const h = t.h0 + i * t.riser;
    segs.push({ d0: d, h0: h, d1: d + t.tread, h1: h, color: seat });
    segs.push({ d0: d + t.tread, h0: h, d1: d + t.tread, h1: h + t.riser, color: riser });
  }
  return segs;
}

/** Windows of the hospitality glass and the LED fascia, as an emissive pattern on a strip mesh. */
function stripMesh(path: PathPoint[], d: number, h0: number, h1: number, mat: THREE.Material): THREE.Mesh {
  const n = path.length;
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  const total = path[n - 1]!.s + Math.hypot(path[0]!.x - path[n - 1]!.x, path[0]!.y - path[n - 1]!.y);
  for (let i = 0; i <= n; i++) {
    const p = path[i % n]!;
    const s = i === n ? total : p.s;
    pos.push(p.x + p.nx * d, h0, -(p.y + p.ny * d), p.x + p.nx * d, h1, -(p.y + p.ny * d));
    uv.push(total - s, 0, total - s, 1);
    if (i < n) {
      const v = i * 2;
      idx.push(v, v + 2, v + 1, v + 1, v + 2, v + 3);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, mat);
  return m;
}

function glassMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    fog: true,
    uniforms: { ...THREE.UniformsLib.fog },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      #include <fog_pars_vertex>
      void main() {
        vUv = uv;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */ `
      varying vec2 vUv;
      #include <fog_pars_fragment>
      float h1(float x) { return fract(sin(x * 91.7) * 43758.5); }
      void main() {
        float pane = floor(vUv.x / 2.4);
        float fx = fract(vUv.x / 2.4);
        float mull = step(0.06, fx) * step(fx, 0.94) * step(0.08, vUv.y) * step(vUv.y, 0.9);
        float lit = step(0.45, h1(pane)) * (0.55 + 0.45 * smoothstep(0.2, 0.8, vUv.y));
        vec3 warm = mix(vec3(1.0, 0.62, 0.32), vec3(1.0, 0.82, 0.6), h1(pane + 3.1));
        vec3 col = mix(vec3(0.015, 0.02, 0.03), warm * (0.12 + 0.3 * h1(pane + 7.0)) * lit, mull);
        // silhouettes of people in some panes
        float person = step(0.55, h1(pane + 11.0)) * step(abs(fx - 0.3 - 0.4 * h1(pane + 5.0)), 0.05) * step(vUv.y, 0.55);
        col *= 1.0 - 0.85 * person * mull;
        gl_FragColor = vec4(col, 1.0);
        #include <fog_fragment>
      }`,
  });
}

export function fasciaTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 2048;
  c.height = 64;
  const ctx = c.getContext("2d")!;
  const items: [string, string, string][] = [
    ["REGOLAZO", "#05070d", "#d7ff3a"],
    ["NOVA LUME", "#0a1a3a", "#ffffff"],
    ["HARBOUR CITY MARINERS", "#0b3d91", "#4cc9f0"],
    ["ORBIT FIZZ", "#ff7a1a", "#1a0a00"],
    ["LANTERN PARK", "#05070d", "#f4f7ff"],
    ["MERIDIAN", "#123c2e", "#c9ffe8"],
  ];
  const w = c.width / items.length;
  items.forEach(([label, bg, fg], i) => {
    ctx.fillStyle = bg;
    ctx.fillRect(i * w, 0, w, 64);
    ctx.fillStyle = fg;
    ctx.font = `800 40px "Barlow Condensed", "Arial Narrow", Arial, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, i * w + w / 2, 34);
  });
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = THREE.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}

function hazeCone(from: THREE.Vector3, to: THREE.Vector3, radius: number): THREE.Mesh {
  const len = from.distanceTo(to);
  const g = new THREE.CylinderGeometry(0.6, radius, len, 24, 1, true);
  g.translate(0, -len / 2, 0);
  const m = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    uniforms: { uStrength: { value: 0.0035 } },
    vertexShader: /* glsl */ `
      varying float vAlong;
      varying vec3 vN;
      varying vec3 vV;
      void main() {
        vAlong = -position.y / ${len.toFixed(3)};
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vN = normalize(normalMatrix * normal);
        vV = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform float uStrength;
      varying float vAlong;
      varying vec3 vN;
      varying vec3 vV;
      void main() {
        float edge = pow(abs(dot(normalize(vN), normalize(vV))), 2.5);
        float fall = (1.0 - vAlong) * smoothstep(0.0, 0.08, vAlong);
        gl_FragColor = vec4(vec3(0.82, 0.88, 1.0) * uStrength * edge * fall, 1.0);
      }`,
  });
  const mesh = new THREE.Mesh(g, m);
  mesh.position.copy(from);
  const dir = to.clone().sub(from).normalize();
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir);
  mesh.renderOrder = 10;
  return mesh;
}

export function createStadium(): Stadium {
  const group = new THREE.Group();
  group.name = "stadium";
  const path = roundedRectPath(STAND_A, STAND_B, STAND_R, 1.6);
  const rng = mulberry32(7);

  // ---- bowl structure
  const seat = C("#1a2130");
  const riser = C("#0b0f17");
  const concrete = C("#2a3038");
  const segs: ProfileSeg[] = [
    { d0: -0.2, h0: 0, d1: -0.2, h1: 1.3, color: C("#0e1a1c") },
    { d0: -0.2, h0: 1.3, d1: 0.3, h1: 1.3, color: concrete },
    ...tierSegs(LOWER, seat, riser),
    // concourse lip under the LED fascia
    { d0: LOWER.d0 + LOWER.rows * LOWER.tread, h0: LOWER.h0 + LOWER.rows * LOWER.riser, d1: 18.2, h1: 10.6, color: concrete },
    { d0: 18.2, h0: 13.8, d1: 19.3, h1: 15.6, color: C("#141922") },
    { d0: 19.3, h0: 15.6, d1: 19.3, h1: 16.6, color: C("#20262f") },
    { d0: 19.3, h0: 16.6, d1: UPPER.d0, h1: UPPER.h0, color: concrete },
    ...tierSegs(UPPER, seat, riser),
    { d0: UPPER.d0 + UPPER.rows * UPPER.tread, h0: UPPER.h0 + UPPER.rows * UPPER.riser, d1: 39.0, h1: 33.2, color: C("#151a22") },
  ];
  const awaySeat = C("#3a1016");
  const bowlGeo = sweep(path, segs, (seg, p) => {
    // away end (behind the -x goal... away fans sit in the +x/-y corner) gets darker red seats
    if (seg.color === seat && p.x > 40 && p.y < -8) return awaySeat;
    return seg.color;
  });
  const bowl = new THREE.Mesh(
    bowlGeo,
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.05, side: THREE.DoubleSide, envMapIntensity: 0.3 }),
  );
  bowl.receiveShadow = false;
  group.add(bowl);

  // ---- LED fascia between tiers, hospitality glass
  const fasciaTex = fasciaTexture();
  fasciaTex.repeat.set(1 / 60, 1);
  const fascia = stripMesh(
    path,
    18.2,
    10.6,
    11.9,
    new THREE.MeshBasicMaterial({ map: fasciaTex, color: new THREE.Color(1.6, 1.6, 1.6), side: THREE.DoubleSide }),
  );
  fascia.name = "fascia";
  group.add(fascia);
  const glass = stripMesh(path, 18.6, 11.9, 13.8, glassMaterial());
  group.add(glass);

  // ---- roof (underside + inner fascia) and floodlight ring
  const roofSegs: ProfileSeg[] = [
    { d0: ROOF.dIn, h0: ROOF.hIn, d1: ROOF.dOut, h1: ROOF.hOut, color: C("#05070b") },
    { d0: ROOF.dIn, h0: ROOF.hIn - 1.4, d1: ROOF.dIn, h1: ROOF.hIn + 0.4, color: C("#1b212b") },
    { d0: ROOF.dIn, h0: ROOF.hIn + 0.4, d1: ROOF.dOut, h1: ROOF.hOut + 1.2, color: C("#10141b") },
    { d0: 39.0, h0: 33.2, d1: ROOF.dOut, h1: ROOF.hOut, color: C("#0d1118") },
  ];
  const roof = new THREE.Mesh(
    sweep(path, roofSegs),
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, metalness: 0.0, side: THREE.DoubleSide, envMapIntensity: 0.15 }),
  );
  group.add(roof);

  // roof truss ribs (radial) for structure under the roof
  const ribGeo = new THREE.BoxGeometry(0.35, 0.9, ROOF.dOut - ROOF.dIn);
  const ribs = new THREE.InstancedMesh(ribGeo, new THREE.MeshStandardMaterial({ color: 0x1a1f28, roughness: 0.6, metalness: 0.4 }), Math.ceil(path.length / 4));
  const m4 = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  let ri = 0;
  for (let i = 0; i < path.length; i += 4) {
    const p = path[i]!;
    const dm = (ROOF.dIn + ROOF.dOut) / 2;
    const pos = new THREE.Vector3(p.x + p.nx * dm, (ROOF.hIn + ROOF.hOut) / 2 - 0.7, -(p.y + p.ny * dm));
    const yaw = Math.atan2(p.nx, -p.ny);
    q.setFromEuler(new THREE.Euler(Math.atan2(ROOF.hIn - ROOF.hOut, ROOF.dOut - ROOF.dIn), yaw, 0, "YXZ"));
    m4.compose(pos, q, new THREE.Vector3(1, 1, 1));
    ribs.setMatrixAt(ri++, m4);
  }
  ribs.count = ri;
  group.add(ribs);

  // floodlight heads
  const lamps: Stadium["lamps"] = [];
  const lampGeo = new THREE.PlaneGeometry(1.25, 0.62);
  const lampMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(6.5, 6.8, 7.5), side: THREE.DoubleSide, fog: false });
  const housingMat = new THREE.MeshStandardMaterial({ color: 0x1c2230, roughness: 0.5, metalness: 0.6 });
  const housingGeo = new THREE.BoxGeometry(1.45, 0.8, 0.35);
  const lampSpots: { p: PathPoint; k: number }[] = [];
  // clusters of lamp heads with dark gaps between them
  for (let i = 0; i < path.length; i++) {
    if (i % 6 === 5) continue;
    const p = path[i]!;
    for (let k = 0; k < 2; k++) lampSpots.push({ p, k });
  }
  const heads = new THREE.InstancedMesh(lampGeo, lampMat, lampSpots.length);
  const housings = new THREE.InstancedMesh(housingGeo, housingMat, lampSpots.length);
  lampSpots.forEach(({ p, k }, i) => {
    const d = ROOF.dIn - 0.3;
    const pos = new THREE.Vector3(p.x + p.nx * d + p.ny * k * 0.8, ROOF.hIn - 1.9 - k * 0.0, -(p.y + p.ny * d) + p.nx * k * 0.8);
    // aim: towards a point on the pitch in front of this stand
    const aim = new THREE.Vector3(p.x * 0.45, 0, -p.y * 0.45);
    const look = new THREE.Matrix4().lookAt(pos, aim, new THREE.Vector3(0, 1, 0));
    q.setFromRotationMatrix(look);
    m4.compose(pos, q, new THREE.Vector3(1, 1, 1));
    heads.setMatrixAt(i, m4);
    const back = new THREE.Vector3(0, 0, 0.2).applyQuaternion(q).add(pos);
    m4.compose(back, q, new THREE.Vector3(1, 1, 1));
    housings.setMatrixAt(i, m4);
    if (i % 16 === 0) lamps.push({ pos, aim });
  });
  heads.name = "floodlights";
  group.add(heads, housings);

  // haze cones from a subset of lamps
  const hazeGroup = new THREE.Group();
  hazeGroup.name = "haze";
  lamps.forEach((l, i) => {
    if (i % 4 !== 1) return;
    const to = l.aim.clone().lerp(l.pos, 0.05);
    to.y = 0;
    hazeGroup.add(hazeCone(l.pos, to, 12 + rng() * 6));
  });
  group.add(hazeGroup);

  const update = (t: number) => {
    fasciaTex.offset.x = (t * 0.004) % 1;
  };
  return { group, path, lamps, update };
}
