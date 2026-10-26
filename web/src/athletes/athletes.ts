// Athlete assets (GLB from scripts/assets/convert_mixamo.mjs), per-player rigs and the animation state machine.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import * as SkeletonUtils from "three/examples/jsm/utils/SkeletonUtils.js";
import { facingToYaw } from "../coords.ts";
import type { RenderPlayer } from "../source/interpolate.ts";
import type { PlayerAnim } from "../types.ts";
import {
  ANIMS,
  clipFor,
  FADE_S,
  locomotionCadence,
  locomotionWeights,
  LOCO_CLIPS,
  oneShotTime,
  type ClipInfo,
} from "./animMap.ts";
import { fitHead, hairGeometry, hairMaterial, type HairStyle, type HeadFit } from "./hair.ts";
import { createKitMaterial, kitUniforms, type KitLook } from "./kitMaterial.ts";

const YBOT_HEIGHT_M = 1.805;

interface Manifest {
  clips: Record<string, ClipInfo>;
}

export class AthleteFactory {
  readonly clips: Map<string, THREE.AnimationClip>;
  readonly info: Record<string, ClipInfo>;
  readonly head: HeadFit;
  private readonly template: THREE.Object3D;
  private readonly hairCache = new Map<string, THREE.BufferGeometry | null>();

  private constructor(template: THREE.Object3D, clips: THREE.AnimationClip[], manifest: Manifest) {
    this.template = template;
    this.clips = new Map(clips.map((c) => [c.name, c]));
    this.info = manifest.clips;
    let body: THREE.SkinnedMesh | null = null;
    template.traverse((o) => {
      if ((o as THREE.SkinnedMesh).isSkinnedMesh && o.name === "Alpha_Surface") body = o as THREE.SkinnedMesh;
    });
    if (!body) throw new Error("athlete.glb: no Alpha_Surface mesh");
    this.head = fitHead((body as THREE.SkinnedMesh).geometry);
  }

  static async load(base = "/assets/"): Promise<AthleteFactory> {
    const [gltf, manifest] = await Promise.all([
      new GLTFLoader().loadAsync(`${base}athlete.glb`),
      fetch(`${base}athlete_clips.json`).then((r) => r.json() as Promise<Manifest>),
    ]);
    return new AthleteFactory(gltf.scene, gltf.animations, manifest);
  }

  hair(style: HairStyle, seed: number): THREE.BufferGeometry | null {
    const key = `${style}:${seed % 4}`;
    if (!this.hairCache.has(key)) this.hairCache.set(key, hairGeometry(style, this.head, seed % 4));
    return this.hairCache.get(key) ?? null;
  }

  create(opts: { id: string; look: KitLook; heightCm: number; hair: HairStyle; hairColor: string; seed: number }): Athlete {
    const model = SkeletonUtils.clone(this.template);
    const material = createKitMaterial(opts.look);
    const meshes: THREE.SkinnedMesh[] = [];
    model.traverse((o) => {
      const m = o as THREE.SkinnedMesh;
      if (m.isSkinnedMesh) {
        m.material = material;
        m.castShadow = true;
        m.receiveShadow = true;
        m.frustumCulled = false;
        meshes.push(m);
      }
    });
    let hair: THREE.Mesh | null = null;
    const hg = this.hair(opts.hair, opts.seed);
    const body = meshes.find((m) => m.name === "Alpha_Surface") ?? meshes[0]!;
    const headIdx = body.skeleton.bones.findIndex((b) => b.name === "mixamorigHead");
    if (hg && headIdx >= 0) {
      const headBone = body.skeleton.bones[headIdx]!;
      const local = hg.clone().applyMatrix4(body.skeleton.boneInverses[headIdx]!);
      hair = new THREE.Mesh(local, hairMaterial(opts.hairColor));
      hair.castShadow = true;
      hair.receiveShadow = true;
      hair.frustumCulled = false;
      headBone.add(hair);
    }
    return new Athlete(opts.id, model, meshes, hair, material, this, opts.heightCm / 100 / YBOT_HEIGHT_M);
  }
}

interface Layer {
  anim: PlayerAnim;
  /** Match time at which the state began (t - anim_t). */
  start: number;
  variant: number;
  mirror: boolean;
  /** Match time at which this layer stopped being current (for fade-out). */
  end: number;
}

export class Athlete {
  readonly id: string;
  readonly root = new THREE.Group();
  readonly model: THREE.Object3D;
  readonly meshes: THREE.SkinnedMesh[];
  readonly hair: THREE.Mesh | null;
  readonly material: THREE.MeshPhysicalMaterial;
  readonly heightScale: number;
  private readonly factory: AthleteFactory;
  private readonly mixer: THREE.AnimationMixer;
  private readonly actions = new Map<string, THREE.AnimationAction>();
  private readonly active = new Set<string>();
  private phase = 0;
  private idleT = 0;
  private cur: Layer | null = null;
  private prev: Layer | null = null;

  constructor(
    id: string,
    model: THREE.Object3D,
    meshes: THREE.SkinnedMesh[],
    hair: THREE.Mesh | null,
    material: THREE.MeshPhysicalMaterial,
    factory: AthleteFactory,
    heightScale: number,
  ) {
    this.id = id;
    this.model = model;
    this.meshes = meshes;
    this.hair = hair;
    this.material = material;
    this.factory = factory;
    this.heightScale = heightScale;
    this.mixer = new THREE.AnimationMixer(model);
    this.root.add(model);
    this.root.name = `athlete-${id}`;
    this.phase = (hashId(id) % 1000) / 1000;
    this.idleT = (hashId(id) % 5000) / 1000;
  }

  private action(name: string): THREE.AnimationAction | null {
    let a = this.actions.get(name);
    if (!a) {
      const clip = this.factory.clips.get(name);
      if (!clip) return null;
      a = this.mixer.clipAction(clip);
      a.setLoop(THREE.LoopRepeat, Infinity);
      this.actions.set(name, a);
    }
    return a;
  }

  /** Clip weights for one layer. */
  private layerPose(l: Layer, p: RenderPlayer, t: number, out: Map<string, { time: number; weight: number }>, w: number): void {
    if (w <= 0) return;
    const spec = ANIMS[l.anim] ?? ANIMS.idle;
    const info = this.factory.info;
    const add = (name: string, time: number, weight: number) => {
      if (weight <= 0) return;
      const e = out.get(name);
      if (!e || e.weight < weight) out.set(name, { time, weight: (e?.weight ?? 0) + weight });
      else e.weight += weight;
    };
    const loopTime = (name: string) => {
      const c = info[name];
      if (!c) return 0;
      const ph = (this.phase + (c.phase0_s ?? 0) / c.duration_s) % 1;
      return ph * c.duration_s;
    };
    if (spec.kind === "locomotion") {
      const lw = locomotionWeights(p.speed);
      LOCO_CLIPS.forEach((name, i) => {
        const wi = lw[i] ?? 0;
        if (wi === 0) return;
        const c = info[name];
        const time = i === 0 ? this.idleT % (c?.duration_s ?? 1) : loopTime(name);
        add(name, time, wi * w);
      });
    } else if (spec.kind === "loop") {
      const name = clipFor(l.anim, l.variant);
      const c = info[name];
      const time = c?.natural_speed_mps ? loopTime(name) : (t - l.start) % (c?.duration_s ?? 1);
      add(name, time, w);
    } else {
      const name = clipFor(l.anim, l.variant);
      const c = info[name];
      if (c) add(name, oneShotTime(l.anim, c, Math.max(0, t - l.start)), w);
    }
  }

  update(p: RenderPlayer, t: number, dt: number): void {
    // state change detection (also a restart of the same anim)
    const start = t - p.animT;
    if (!this.cur || this.cur.anim !== p.anim || Math.abs(this.cur.start - start) > 0.12 || this.cur.variant !== p.variant) {
      const isRestart = this.cur && this.cur.anim === p.anim && Math.abs(this.cur.start - start) > 0.12;
      const sameLoco = this.cur && ANIMS[this.cur.anim]?.kind === "locomotion" && ANIMS[p.anim]?.kind === "locomotion";
      if (this.cur && !sameLoco) this.prev = { ...this.cur, end: t };
      if (!this.cur || !sameLoco || isRestart) this.cur = { anim: p.anim, start, variant: p.variant, mirror: p.mirror, end: Infinity };
      else this.cur.anim = p.anim;
    }
    if (this.cur) this.cur.mirror = p.mirror;

    // shared locomotion phase advances with speed (deterministic for a fixed dt sequence)
    const spec = ANIMS[p.anim] ?? ANIMS.idle;
    let cadence: number;
    if (spec.kind === "loop") {
      const c = this.factory.info[clipFor(p.anim, p.variant)];
      const nat = c?.natural_speed_mps ?? 0;
      cadence = c ? (nat > 0 ? Math.min(1.35, Math.max(0.7, p.speed / nat)) : 1) / c.duration_s : 1;
    } else {
      cadence = locomotionCadence(p.speed, this.factory.info);
    }
    this.phase = (this.phase + cadence * dt) % 1;
    this.idleT += dt;

    let wCur = 1;
    if (this.prev) {
      const since = t - this.prev.end;
      if (since >= FADE_S || since < 0) this.prev = null;
      else wCur = since / FADE_S;
    }
    const pose = new Map<string, { time: number; weight: number }>();
    if (this.cur) this.layerPose(this.cur, p, t, pose, wCur);
    if (this.prev) this.layerPose(this.prev, p, t, pose, 1 - wCur);

    for (const name of this.active) {
      if (!pose.has(name)) {
        this.actions.get(name)?.stop();
        this.active.delete(name);
      }
    }
    for (const [name, { time, weight }] of pose) {
      const a = this.action(name);
      if (!a) continue;
      if (!this.active.has(name)) {
        a.play();
        this.active.add(name);
      }
      a.enabled = true;
      a.setEffectiveWeight(weight);
      a.time = time;
    }
    this.mixer.update(0);

    const ownsHeight = (ANIMS[p.anim] ?? ANIMS.idle).ownsHeight ?? false;
    this.root.position.set(p.x, ownsHeight ? 0 : p.z, -p.y);
    this.root.rotation.set(0, facingToYaw(p.facing), 0);
    const mirror = this.cur?.mirror ?? false;
    this.root.scale.set(this.heightScale * (mirror ? -1 : 1), this.heightScale, this.heightScale);
    kitUniforms(this.material).uMirror!.value = mirror ? -1 : 1;
    this.root.visible = p.onPitch;
  }

  /** Static pose for art previews: a clip at a time, no state machine. */
  posePreview(clip: string, time: number): void {
    for (const name of this.active) this.actions.get(name)?.stop();
    this.active.clear();
    const a = this.action(clip);
    if (!a) return;
    a.play();
    a.setEffectiveWeight(1);
    a.time = time;
    this.active.add(clip);
    this.mixer.update(0);
  }
}

function hashId(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
