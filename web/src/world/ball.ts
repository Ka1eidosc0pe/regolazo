// The match ball: original panel design, spin from the simulated angular velocity, contact shadow, strike trail.
import * as THREE from "three";
import { BALL_RADIUS } from "../coords.ts";

function ballTexture(): THREE.CanvasTexture {
  const W = 1024;
  const H = 512;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#f4f5f2";
  ctx.fillRect(0, 0, W, H);
  // three swept bands (equirectangular): navy, ice, flare pinstripe
  const band = (phase: number, amp: number, width: number, color: string) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    for (let x = 0; x <= W; x += 4) {
      const y = H / 2 + Math.sin((x / W) * Math.PI * 2 * 2 + phase) * amp;
      if (x === 0) ctx.moveTo(x, y - width);
      else ctx.lineTo(x, y - width);
    }
    for (let x = W; x >= 0; x -= 4) {
      const y = H / 2 + Math.sin((x / W) * Math.PI * 2 * 2 + phase) * amp;
      ctx.lineTo(x, y + width);
    }
    ctx.closePath();
    ctx.fill();
  };
  band(0, 150, 34, "#13254a");
  band(0, 150, 10, "#4cc9f0");
  band(Math.PI, 150, 22, "#13254a");
  band(Math.PI, 150, 5, "#ff7a1a");
  // panel seams
  ctx.strokeStyle = "rgba(20,24,30,0.35)";
  ctx.lineWidth = 3;
  for (let i = 0; i < 6; i++) {
    ctx.beginPath();
    ctx.moveTo((i * W) / 6, 0);
    ctx.bezierCurveTo((i * W) / 6 + 60, H * 0.35, (i * W) / 6 - 60, H * 0.65, (i * W) / 6, H);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(0, H * 0.18);
  ctx.lineTo(W, H * 0.18);
  ctx.moveTo(0, H * 0.82);
  ctx.lineTo(W, H * 0.82);
  ctx.stroke();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function blobTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, "rgba(0,0,0,0.85)");
  g.addColorStop(0.45, "rgba(0,0,0,0.5)");
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

let BLOB: THREE.CanvasTexture | null = null;
export function blobShadow(size: number, opacity: number): THREE.Mesh {
  BLOB ??= blobTexture();
  const m = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshBasicMaterial({ map: BLOB, transparent: true, depthWrite: false, opacity, color: 0x000000 }),
  );
  m.rotation.x = -Math.PI / 2;
  m.renderOrder = 2;
  return m;
}

const TRAIL_N = 28;
const TRAIL_LIFE = 0.24;

export interface Ball {
  mesh: THREE.Mesh;
  shadow: THREE.Mesh;
  trail: THREE.Mesh;
  group: THREE.Group;
  update(pos: [number, number, number], vel: [number, number, number], spin: [number, number, number], t: number, dt: number, camera: THREE.Camera): void;
  reset(): void;
}

export function createBall(): Ball {
  const group = new THREE.Group();
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(BALL_RADIUS, 48, 32),
    new THREE.MeshPhysicalMaterial({ map: ballTexture(), roughness: 0.38, clearcoat: 0.8, clearcoatRoughness: 0.2, envMapIntensity: 0.9 }),
  );
  mesh.castShadow = true;
  mesh.name = "ball";
  const shadow = blobShadow(0.5, 0.75);
  shadow.name = "ball-shadow";

  // trail ribbon
  const tg = new THREE.BufferGeometry();
  const tPos = new Float32Array(TRAIL_N * 2 * 3);
  const tA = new Float32Array(TRAIL_N * 2);
  tg.setAttribute("position", new THREE.BufferAttribute(tPos, 3));
  tg.setAttribute("aAlpha", new THREE.BufferAttribute(tA, 1));
  const idx: number[] = [];
  for (let i = 0; i < TRAIL_N - 1; i++) {
    const v = i * 2;
    idx.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
  }
  tg.setIndex(idx);
  const trail = new THREE.Mesh(
    tg,
    new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      vertexShader: /* glsl */ `
        attribute float aAlpha; varying float vA;
        void main() { vA = aAlpha; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */ `
        varying float vA;
        void main() { gl_FragColor = vec4(mix(vec3(0.35, 0.85, 1.0), vec3(1.0), vA) * vA * 0.6, 1.0); }`,
    }),
  );
  trail.frustumCulled = false;
  trail.renderOrder = 20;
  group.add(mesh, shadow, trail);

  const hist: { t: number; p: THREE.Vector3; s: number }[] = [];
  const q = new THREE.Quaternion();
  const axis = new THREE.Vector3();
  const tmp = new THREE.Vector3();
  const camPos = new THREE.Vector3();
  let lastSpinT = -1;

  return {
    mesh,
    shadow,
    trail,
    group,
    reset() {
      hist.length = 0;
      lastSpinT = -1;
      mesh.quaternion.identity();
    },
    update(pos, vel, spin, t, dt, camera) {
      mesh.position.set(pos[0], pos[2], -pos[1]);
      // integrate rotation: simulated spin if present, else rolling without slipping on the ground
      let w = new THREE.Vector3(spin[0], spin[2], -spin[1]);
      if (w.lengthSq() < 1e-4) {
        const v = new THREE.Vector3(vel[0], 0, -vel[1]);
        // rolling: omega = up x v / r
        w = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), v).divideScalar(BALL_RADIUS);
        if (pos[2] > 0.25) w.multiplyScalar(0.3);
      }
      // rotate by the match-time step: frozen while paused, independent of the display frame rate
      const dtm = lastSpinT < 0 ? 0 : Math.min(0.2, Math.max(0, t - lastSpinT));
      void dt;
      const ang = w.length() * dtm;
      if (ang > 0) {
        axis.copy(w).normalize();
        q.setFromAxisAngle(axis, ang);
        mesh.quaternion.premultiply(q);
      }
      lastSpinT = t;
      // contact shadow
      const hgt = Math.max(0, pos[2] - BALL_RADIUS);
      shadow.position.set(pos[0], 0.012, -pos[1]);
      const s = 1 + hgt * 0.6;
      shadow.scale.set(s, s, s);
      (shadow.material as THREE.MeshBasicMaterial).opacity = 0.75 / (1 + hgt * 1.4);
      // trail
      const speed = Math.hypot(vel[0], vel[1], vel[2]);
      const last = hist[hist.length - 1];
      if (!last || t > last.t) hist.push({ t, p: mesh.position.clone(), s: speed });
      else if (t < last.t) hist.length = 0;
      while (hist.length > 0 && t - hist[0]!.t > TRAIL_LIFE) hist.shift();
      while (hist.length > TRAIL_N) hist.shift();
      camera.getWorldPosition(camPos);
      const n = hist.length;
      for (let i = 0; i < TRAIL_N; i++) {
        const h = hist[Math.max(0, n - TRAIL_N + i)] ?? { t, p: mesh.position, s: 0 };
        const next = hist[Math.min(n - 1, Math.max(0, n - TRAIL_N + i + 1))] ?? h;
        tmp.copy(next.p).sub(h.p);
        const view = camPos.clone().sub(h.p);
        const side = new THREE.Vector3().crossVectors(tmp, view).normalize();
        if (!Number.isFinite(side.x)) side.set(0, 0, 0);
        const age = (t - h.t) / TRAIL_LIFE;
        const k = i / (TRAIL_N - 1);
        // strike trail from ~12 m/s, full by ~20 m/s
        const fast = THREE.MathUtils.smoothstep(h.s, 11, 20);
        const alpha = n < 3 ? 0 : Math.max(0, 1 - age) * k * fast;
        const width = BALL_RADIUS * (0.15 + 0.55 * k);
        side.multiplyScalar(width);
        tPos.set([h.p.x + side.x, h.p.y + side.y, h.p.z + side.z, h.p.x - side.x, h.p.y - side.y, h.p.z - side.z], i * 6);
        tA[i * 2] = alpha;
        tA[i * 2 + 1] = alpha;
      }
      tg.attributes.position!.needsUpdate = true;
      tg.attributes.aAlpha!.needsUpdate = true;
    },
  };
}
