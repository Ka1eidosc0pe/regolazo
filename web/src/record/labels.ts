// Recording-mode ground truth (shared/schemas/labels.py): camera calibration in the OpenCV convention, per-object
// boxes from projected skinned geometry, visibility from an ID-buffer pass, ball and ball-shadow pixels.
import * as THREE from "three";
import type { App, PlayerRig } from "../app.ts";
import { threeToWorld, worldToThree } from "../coords.ts";
import type { CameraCalibration, FrameLabels, ObjectLabel } from "../types.ts";

const LAYER_ID = 2;
const LAYER_SOLO = 3;
/** ID pass resolution relative to the frame. */
const ID_SCALE = 0.5;

/** World (z up) -> Three.js (y up) as a 4x4 matrix. */
const W2T = new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
/** Three.js camera (looks down -z, y up) -> OpenCV camera (looks down +z, y down). */
const FLIP = new THREE.Matrix4().makeScale(1, -1, -1);

/** K (row-major 3x3) and [R | t] (row-major 3x4) mapping world points to pixels: p ~ K [R | t] X. */
export function calibration(camera: THREE.PerspectiveCamera, width: number, height: number): CameraCalibration {
  camera.updateMatrixWorld(true);
  const fy = height / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const fx = fy; // square pixels, aspect = width / height
  const ext = new THREE.Matrix4().multiplyMatrices(FLIP, camera.matrixWorldInverse).multiply(W2T);
  const e = ext.elements; // column-major
  const extrinsics = [e[0], e[4], e[8], e[12], e[1], e[5], e[9], e[13], e[2], e[6], e[10], e[14]].map((v) => +v!.toFixed(9));
  const p = camera.getWorldPosition(new THREE.Vector3());
  return {
    intrinsics: [fx, 0, width / 2, 0, fy, height / 2, 0, 0, 1].map((v) => +v.toFixed(6)) as CameraCalibration["intrinsics"],
    extrinsics: extrinsics as CameraCalibration["extrinsics"],
    fov_y_deg: +camera.fov.toFixed(6),
    position: threeToWorld(p.x, p.y, p.z).map((v) => +v.toFixed(6)) as [number, number, number],
  };
}

/** Project a world-frame point with a calibration (the same maths the Spotter uses). */
export function projectWorld(cal: CameraCalibration, X: [number, number, number]): [number, number, number] {
  const r = cal.extrinsics;
  const k = cal.intrinsics;
  const xc = r[0] * X[0] + r[1] * X[1] + r[2] * X[2] + r[3];
  const yc = r[4] * X[0] + r[5] * X[1] + r[6] * X[2] + r[7];
  const zc = r[8] * X[0] + r[9] * X[1] + r[10] * X[2] + r[11];
  return [(k[0] * xc + k[1] * yc) / zc + k[2], (k[4] * yc) / zc + k[5], zc];
}

interface Target {
  label: Omit<ObjectLabel, "box" | "visible_fraction" | "center_px">;
  objects: THREE.Object3D[];
  points: () => THREE.Vector3[];
  id: number;
}

export interface LabelDebug {
  /** Tight bounds of the object's own silhouette from the solo ID pass (full-frame pixels). */
  silhouette: Record<string, [number, number, number, number] | null>;
  visiblePx: Record<string, number>;
  soloPx: Record<string, number>;
  /** `world` projected with the Three.js camera (independent of the calibration matrices). */
  threeCenter: Record<string, [number, number]>;
}

export class Labeler {
  private readonly app: App;
  private readonly samples = new Map<THREE.Mesh, number[]>();
  private idRT: THREE.WebGLRenderTarget | null = null;
  private readonly idMats = new Map<number, THREE.MeshBasicMaterial>();
  private readonly occluderMat = new THREE.MeshBasicMaterial({ color: 0x000000 });

  constructor(app: App) {
    this.app = app;
  }

  private sampleIndices(mesh: THREE.Mesh, stride: number): number[] {
    let s = this.samples.get(mesh);
    if (!s) {
      const n = mesh.geometry.getAttribute("position").count;
      s = [];
      for (let i = 0; i < n; i += stride) s.push(i);
      this.samples.set(mesh, s);
    }
    return s;
  }

  private rigPoints(rig: PlayerRig): THREE.Vector3[] {
    const out: THREE.Vector3[] = [];
    for (const m of rig.athlete.meshes) {
      for (const i of this.sampleIndices(m, 3)) {
        const v = new THREE.Vector3();
        m.getVertexPosition(i, v);
        out.push(v.applyMatrix4(m.matrixWorld));
      }
    }
    const hair = rig.athlete.hair;
    if (hair) {
      const pos = hair.geometry.getAttribute("position");
      for (let i = 0; i < pos.count; i += 4) out.push(new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(hair.matrixWorld));
    }
    return out;
  }

  private targets(): Target[] {
    const app = this.app;
    const state = app.state;
    if (!state) return [];
    const out: Target[] = [];
    let id = 1;
    const byId = new Map(state.players.map((p) => [p.id, p]));
    for (const [pid, rig] of app.rigs) {
      const p = byId.get(pid);
      if (!p || !p.onPitch || !rig.athlete.root.visible || !rig.meta) continue;
      const team = rig.team!;
      out.push({
        label: {
          cls: `${rig.isGK ? "goalkeeper" : "player"}_${team}` as ObjectLabel["cls"],
          track_id: pid,
          team,
          player: pid,
          world: [r3(p.x), r3(p.y), r3(p.z)],
        },
        objects: [...rig.athlete.meshes, ...(rig.athlete.hair ? [rig.athlete.hair] : [])],
        points: () => this.rigPoints(rig),
        id: id++,
      });
    }
    if (state.referee && app.referee) {
      const rig = app.referee;
      out.push({
        label: { cls: "referee", track_id: "referee", team: null, player: null, world: [r3(state.referee.x), r3(state.referee.y), 0] },
        objects: [...rig.athlete.meshes, ...(rig.athlete.hair ? [rig.athlete.hair] : [])],
        points: () => this.rigPoints(rig),
        id: id++,
      });
    }
    const ball = app.ball.mesh;
    const [bx, by, bz] = state.ball.pos;
    out.push({
      label: { cls: "ball", track_id: "ball", team: null, player: null, world: [r3(bx), r3(by), r3(bz)] },
      objects: [ball],
      points: () => {
        const c = ball.getWorldPosition(new THREE.Vector3());
        const r = 0.11;
        const pts: THREE.Vector3[] = [];
        for (let i = 0; i < 64; i++) {
          // Fibonacci sphere: an even sample of the silhouette from any view direction
          const y = 1 - (2 * (i + 0.5)) / 64;
          const rad = Math.sqrt(1 - y * y);
          const a = i * 2.399963;
          pts.push(new THREE.Vector3(c.x + Math.cos(a) * rad * r, c.y + y * r, c.z + Math.sin(a) * rad * r));
        }
        return pts;
      },
      id,
    });
    return out;
  }

  private idMat(id: number): THREE.MeshBasicMaterial {
    let m = this.idMats.get(id);
    if (!m) {
      m = new THREE.MeshBasicMaterial({ fog: false, side: THREE.DoubleSide });
      m.color.setRGB((id * 10) / 255, 0, 0, THREE.LinearSRGBColorSpace);
      this.idMats.set(id, m);
    }
    return m;
  }

  /** Labels for the frame that was just rendered. */
  label(frame: number, image: string, withDebug = false): { labels: FrameLabels; debug?: LabelDebug } {
    const app = this.app;
    const { width: W, height: H } = app.size;
    const camera = app.director.camera;
    const cal = calibration(camera, W, H);
    const state = app.state!;
    app.scene.updateMatrixWorld(true);
    const targets = this.targets();
    const pv = new THREE.Vector3();

    // ---- projected boxes
    interface Box {
      raw: [number, number, number, number];
      clip: [number, number, number, number];
      inImage: number;
    }
    const boxes = new Map<number, Box>();
    for (const t of targets) {
      let x0 = Infinity;
      let y0 = Infinity;
      let x1 = -Infinity;
      let y1 = -Infinity;
      for (const p of t.points()) {
        pv.copy(p).applyMatrix4(camera.matrixWorldInverse);
        if (pv.z > -0.05) continue; // behind the camera
        pv.copy(p).project(camera);
        const px = ((pv.x + 1) / 2) * W;
        const py = ((1 - pv.y) / 2) * H;
        x0 = Math.min(x0, px);
        y0 = Math.min(y0, py);
        x1 = Math.max(x1, px);
        y1 = Math.max(y1, py);
      }
      if (!Number.isFinite(x0)) continue;
      const clip: [number, number, number, number] = [clampTo(x0, W), clampTo(y0, H), clampTo(x1, W), clampTo(y1, H)];
      const rawArea = Math.max(1e-6, (x1 - x0) * (y1 - y0));
      const clipArea = (clip[2] - clip[0]) * (clip[3] - clip[1]);
      if (clipArea <= 0) continue;
      boxes.set(t.id, { raw: [x0, y0, x1, y1], clip, inImage: Math.min(1, clipArea / rawArea) });
    }

    // ---- ID passes: everything together (occlusion), then each object alone
    const { visible, solo, silhouette } = this.idPasses(targets.filter((t) => boxes.has(t.id)), boxes, W, H);

    const objects: ObjectLabel[] = [];
    for (const t of targets) {
      const b = boxes.get(t.id);
      if (!b) continue;
      const vis = visible.get(t.id) ?? 0;
      const sol = solo.get(t.id) ?? 0;
      const occl = sol > 0 ? Math.min(1, vis / sol) : 1;
      const [cx, cy] = projectWorld(cal, t.label.world);
      objects.push({
        ...t.label,
        box: b.clip.map((v) => +v.toFixed(2)) as ObjectLabel["box"],
        visible_fraction: +(b.inImage * occl).toFixed(4),
        center_px: [+cx.toFixed(2), +cy.toFixed(2)],
      });
    }
    const [bx, by] = state.ball.pos;
    const sh = projectWorld(cal, [bx, by, 0]);
    const shadowIn = sh[2] > 0 && sh[0] >= 0 && sh[0] <= W && sh[1] >= 0 && sh[1] <= H;
    const mode = app.director.mode;
    const labels: FrameLabels = {
      schema_version: state.base.schema_version,
      match_id: app.meta.match_id,
      frame,
      t: +state.t.toFixed(4),
      tick: state.tick,
      image,
      width: W,
      height: H,
      camera_mode: mode === "free" ? "replay" : mode,
      camera: cal,
      objects,
      ball_shadow_px: shadowIn ? [+sh[0].toFixed(2), +sh[1].toFixed(2)] : null,
    };
    if (!withDebug) return { labels };
    const dbg: LabelDebug = { silhouette: {}, visiblePx: {}, soloPx: {}, threeCenter: {} };
    for (const t of targets) {
      dbg.silhouette[t.label.track_id] = silhouette.get(t.id) ?? null;
      dbg.visiblePx[t.label.track_id] = visible.get(t.id) ?? 0;
      dbg.soloPx[t.label.track_id] = solo.get(t.id) ?? 0;
      const [wx, wy, wz] = t.label.world;
      const v = new THREE.Vector3(...worldToThree(wx, wy, wz)).project(camera);
      dbg.threeCenter[t.label.track_id] = [((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H];
    }
    return { labels, debug: dbg };
  }

  private idPasses(
    targets: Target[],
    boxes: Map<number, { clip: [number, number, number, number] }>,
    W: number,
    H: number,
  ): { visible: Map<number, number>; solo: Map<number, number>; silhouette: Map<number, [number, number, number, number]> } {
    const app = this.app;
    const renderer = app.renderer;
    const camera = app.director.camera;
    const w = Math.round(W * ID_SCALE);
    const h = Math.round(H * ID_SCALE);
    if (!this.idRT || this.idRT.width !== w || this.idRT.height !== h) {
      this.idRT?.dispose();
      this.idRT = new THREE.WebGLRenderTarget(w, h, { depthBuffer: true, samples: 0 });
    }
    const rt = this.idRT;
    // swap materials, tag layers
    const saved = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>();
    for (const t of targets) {
      for (const o of t.objects) {
        const m = o as THREE.Mesh;
        saved.set(m, m.material);
        m.material = this.idMat(t.id);
        m.layers.enable(LAYER_ID);
      }
    }
    const occluders: THREE.Mesh[] = [];
    for (const g of app.goals) {
      for (const m of g.frame) {
        saved.set(m, m.material);
        m.material = this.occluderMat;
        m.layers.enable(LAYER_ID);
        occluders.push(m);
      }
    }
    const prevLayers = camera.layers.mask;
    const prevBg = app.scene.background;
    const prevFog = app.scene.fog;
    const prevAuto = renderer.shadowMap.autoUpdate;
    const prevTarget = renderer.getRenderTarget();
    const prevClear = renderer.getClearColor(new THREE.Color());
    const prevAlpha = renderer.getClearAlpha();
    app.scene.background = null;
    app.scene.fog = null;
    renderer.shadowMap.autoUpdate = false;
    renderer.setClearColor(0x000000, 1);
    renderer.setRenderTarget(rt);

    const visible = new Map<number, number>();
    const solo = new Map<number, number>();
    const silhouette = new Map<number, [number, number, number, number]>();
    try {
      camera.layers.set(LAYER_ID);
      renderer.setScissorTest(false);
      renderer.clear();
      renderer.render(app.scene, camera);
      const buf = new Uint8Array(w * h * 4);
      renderer.readRenderTargetPixels(rt, 0, 0, w, h, buf);
      for (let i = 0; i < w * h; i++) {
        const id = Math.round(buf[i * 4]! / 10);
        if (id > 0) visible.set(id, (visible.get(id) ?? 0) + 1);
      }
      // solo passes, scissored to each box
      camera.layers.set(LAYER_SOLO);
      renderer.setScissorTest(true);
      for (const t of targets) {
        const b = boxes.get(t.id)!.clip;
        // scissor / viewport origin is bottom-left in GL
        const sx = Math.max(0, Math.floor(b[0] * ID_SCALE) - 1);
        const sy = Math.max(0, Math.floor(h - b[3] * ID_SCALE) - 1);
        const sw = Math.min(w - sx, Math.ceil((b[2] - b[0]) * ID_SCALE) + 3);
        const shh = Math.min(h - sy, Math.ceil((b[3] - b[1]) * ID_SCALE) + 3);
        if (sw <= 0 || shh <= 0) continue;
        for (const o of t.objects) o.layers.enable(LAYER_SOLO);
        renderer.setScissor(sx, sy, sw, shh);
        renderer.clear();
        renderer.render(app.scene, camera);
        for (const o of t.objects) o.layers.disable(LAYER_SOLO);
        const sb = new Uint8Array(sw * shh * 4);
        renderer.readRenderTargetPixels(rt, sx, sy, sw, shh, sb);
        let n = 0;
        let x0 = Infinity;
        let y0 = Infinity;
        let x1 = -Infinity;
        let y1 = -Infinity;
        for (let yy = 0; yy < shh; yy++) {
          for (let xx = 0; xx < sw; xx++) {
            if (Math.round(sb[(yy * sw + xx) * 4]! / 10) === t.id) {
              n++;
              const fx = (sx + xx) / ID_SCALE;
              const fy = (h - (sy + yy) - 1) / ID_SCALE;
              x0 = Math.min(x0, fx);
              x1 = Math.max(x1, fx + 1 / ID_SCALE);
              y0 = Math.min(y0, fy);
              y1 = Math.max(y1, fy + 1 / ID_SCALE);
            }
          }
        }
        solo.set(t.id, n);
        if (n > 0) silhouette.set(t.id, [x0, y0, x1, y1]);
      }
    } finally {
      renderer.setScissorTest(false);
      renderer.setRenderTarget(prevTarget);
      renderer.setClearColor(prevClear, prevAlpha);
      renderer.shadowMap.autoUpdate = prevAuto;
      app.scene.background = prevBg;
      app.scene.fog = prevFog;
      camera.layers.mask = prevLayers;
      for (const [m, mat] of saved) {
        m.material = mat;
        m.layers.disable(LAYER_ID);
        m.layers.disable(LAYER_SOLO);
      }
    }
    return { visible, solo, silhouette };
  }
}

function r3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

function clampTo(v: number, max: number): number {
  return Math.min(max, Math.max(0, v));
}
