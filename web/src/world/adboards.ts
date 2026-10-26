// LED advertising boards around the pitch with fictional sponsors, scrolling and glowing.
import * as THREE from "three";
import { BOARD_X, BOARD_Y } from "./pitch.ts";

export const BOARD_SPONSORS: [string, string, string][] = [
  ["NOVA LUME", "#0a1a3a", "#ffffff"],
  ["TIDEWATER", "#0b3d91", "#e8f6ff"],
  ["KESTREL AIR", "#c8102e", "#ffffff"],
  ["ORBIT FIZZ", "#ff7a1a", "#1a0a00"],
  ["HALCYON MOTORS", "#101418", "#d7ff3a"],
  ["BLUEFIN", "#0a6b8a", "#ffffff"],
  ["MERIDIAN", "#123c2e", "#c9ffe8"],
  ["PIXEL HARVEST", "#3b1d6e", "#ffe14d"],
  ["REGOLAZO", "#05070d", "#d7ff3a"],
];

function boardTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 4096;
  c.height = 96;
  const ctx = c.getContext("2d")!;
  const w = c.width / BOARD_SPONSORS.length;
  BOARD_SPONSORS.forEach(([label, bg, fg], i) => {
    const x = i * w;
    const grad = ctx.createLinearGradient(x, 0, x, 96);
    grad.addColorStop(0, bg);
    grad.addColorStop(1, new THREE.Color(bg).multiplyScalar(0.6).getStyle());
    ctx.fillStyle = grad;
    ctx.fillRect(x, 0, w, 96);
    ctx.fillStyle = fg;
    ctx.font = `800 62px "Barlow Condensed", "Arial Narrow", Arial, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.save();
    ctx.translate(x + w / 2, 50);
    ctx.transform(1, 0, -0.15, 1, 0, 0);
    ctx.fillText(label, 0, 0);
    ctx.restore();
    // a simple original mark next to the word
    ctx.beginPath();
    ctx.arc(x + 34, 48, 14, 0, Math.PI * 2);
    ctx.fill();
  });
  // LED pixel grid
  ctx.fillStyle = "rgba(0,0,0,0.22)";
  for (let y = 0; y < 96; y += 4) ctx.fillRect(0, y, c.width, 1);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = THREE.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}

export interface AdBoards {
  group: THREE.Group;
  update(t: number): void;
}

export function createAdBoards(): AdBoards {
  const group = new THREE.Group();
  group.name = "adboards";
  const tex = boardTexture();
  const H = 0.95;
  const faceMat = new THREE.MeshBasicMaterial({ map: tex, color: new THREE.Color(1.35, 1.35, 1.35) });
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x0c0f14, roughness: 0.6, metalness: 0.4 });
  const runs: { cx: number; cy: number; len: number; yaw: number }[] = [
    { cx: 0, cy: -BOARD_Y, len: 104, yaw: Math.PI },
    { cx: 0, cy: BOARD_Y, len: 112, yaw: 0 },
    { cx: BOARD_X, cy: 0, len: 62, yaw: -Math.PI / 2 },
    { cx: -BOARD_X, cy: 0, len: 62, yaw: Math.PI / 2 },
  ];
  const textures: THREE.Texture[] = [];
  for (const r of runs) {
    const t = tex.clone();
    t.needsUpdate = true;
    t.repeat.set(r.len / 36, 1);
    textures.push(t);
    const face = new THREE.Mesh(new THREE.PlaneGeometry(r.len, H), faceMat.clone());
    (face.material as THREE.MeshBasicMaterial).map = t;
    face.position.set(0, H / 2 + 0.05, 0.08);
    const body = new THREE.Mesh(new THREE.BoxGeometry(r.len, H + 0.1, 0.15), frameMat);
    body.position.set(0, (H + 0.1) / 2, 0);
    body.castShadow = true;
    const g = new THREE.Group();
    g.add(body, face);
    // boards face the pitch: local +z towards the centre
    g.position.set(r.cx, 0, -r.cy);
    g.rotation.y = r.yaw;
    group.add(g);
  }
  return {
    group,
    update(t) {
      // slow scroll with a page flip every 12 s, deterministic in t
      const page = Math.floor(t / 12);
      const within = (t % 12) / 12;
      textures.forEach((tx, i) => {
        tx.offset.x = (page * 0.37 + i * 0.21 + within * 0.03) % 1;
      });
    },
  };
}
