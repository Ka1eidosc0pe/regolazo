// Athlete assets (GLB from scripts/assets/convert_mixamo.mjs), per-player rigs and the animation state machine.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import * as SkeletonUtils from "three/examples/jsm/utils/SkeletonUtils.js";
import { angleDelta, facingToYaw, smoothstep } from "../coords.ts";
import type { RenderPlayer } from "../source/interpolate.ts";
import type { PlayerAnim } from "../types.ts";
import {
  ANIMS,
  clipFor,
  FADE_S,
  locomotionCadence,
  locomotionWeights,
  LOCO_CLIPS,
  loopRate,
  oneShotTime,
  type ClipInfo,
} from "./animMap.ts";
import { fitHead, hairGeometry, hairMaterial, type HairStyle, type HeadFit } from "./hair.ts";
import { createKitMaterial, kitUniforms, type KitLook } from "./kitMaterial.ts";

const YBOT_HEIGHT_M = 1.805;

interface Manifest {
  clips: Record<string, ClipInfo>;
}

const LOWER_BODY = /mixamorig(Hips|LeftUpLeg|LeftLeg|LeftFoot|LeftToe|RightUpLeg|RightLeg|RightFoot|RightToe)/;

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
    // aPlate = 1 on vertices driven by the rib-cage bone (Spine1), 2 on the shoulder yoke (Spine2): rigid plates,
    // so the number / crest / sponsor (ribs) and the name (yoke) never straddle the torso segments
    template.traverse((o) => {
      const m = o as THREE.SkinnedMesh;
      if (m.isSkinnedMesh) markPlate(m);
    });
  }

  static async load(base = "/assets/"): Promise<AthleteFactory> {
    const [gltf, manifest] = await Promise.all([
      new GLTFLoader().loadAsync(`${base}athlete.glb`),
      fetch(`${base}athlete_clips.json`).then((r) => r.json() as Promise<Manifest>),
    ]);
    return new AthleteFactory(gltf.scene, gltf.animations, manifest);
  }

  /** A clip by name, or a copy with only the upper-body (`name#upper`) or lower-body + hips (`name#lower`) tracks. */
  clip(name: string): THREE.AnimationClip | undefined {
    const known = this.clips.get(name);
    if (known) return known;
    const [base, part] = name.split("#");
    const src = base ? this.clips.get(base) : undefined;
    if (!src || (part !== "upper" && part !== "lower")) return undefined;
    const keep = (track: string) => (part === "lower" ? LOWER_BODY.test(track) : !LOWER_BODY.test(track));
    const c = new THREE.AnimationClip(name, src.duration, src.tracks.filter((t) => keep(t.name)));
    this.clips.set(name, c);
    return c;
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

function markPlate(m: THREE.SkinnedMesh): void {
  const g = m.geometry;
  const si = g.getAttribute("skinIndex");
  const sw = g.getAttribute("skinWeight");
  const ribs = m.skeleton.bones.findIndex((b) => b.name === "mixamorigSpine1");
  const yoke = m.skeleton.bones.findIndex((b) => b.name === "mixamorigSpine2");
  const out = new Float32Array(si.count);
  for (let i = 0; i < si.count; i++) {
    let best = 0;
    let bi = -1;
    for (let k = 0; k < 4; k++) {
      const w = sw.getComponent(i, k);
      if (w > best) {
        best = w;
        bi = si.getComponent(i, k);
      }
    }
    out[i] = bi === ribs ? 1 : bi === yoke ? 2 : 0;
  }
  g.setAttribute("aPlate", new THREE.BufferAttribute(out, 1));
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

type Pose = Map<string, { time: number; weight: number }>;

/** Upper-body gestures play full body below SPLIT_FROM m/s; above SPLIT_TO the legs come from locomotion. */
const SPLIT_FROM = 1.0;
const SPLIT_TO = 1.8;

export class Athlete {
  readonly id: string;
  readonly root = new THREE.Group();
  readonly model: THREE.Object3D;
  readonly meshes: THREE.SkinnedMesh[];
  readonly hair: THREE.Mesh | null;
  readonly material: THREE.MeshPhysicalMaterial;
  readonly heightScale: number;
  private readonly factory: AthleteFactory;
  private mixer: THREE.AnimationMixer;
  private readonly actions = new Map<string, THREE.AnimationAction>();
  private readonly active = new Set<string>();
  private readonly seedPhase: number;
  private readonly seedIdle: number;
  private phase = 0;
  private idleT = 0;
  /** Match time of the last update. Phases advance with match time, so a paused match is a frozen pose. */
  private lastT: number | null = null;
  private cur: Layer | null = null;
  private prev: Layer | null = null;
  /** Bind pose of every bone, restored before each evaluation (see apply). */
  private readonly rest: { bone: THREE.Bone; q: THREE.Quaternion; p: THREE.Vector3 }[] = [];

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
    model.traverse((o) => {
      if ((o as THREE.Bone).isBone) this.rest.push({ bone: o as THREE.Bone, q: o.quaternion.clone(), p: o.position.clone() });
    });
    this.root.add(model);
    this.root.name = `athlete-${id}`;
    this.seedPhase = (hashId(id) % 1000) / 1000;
    this.seedIdle = (hashId(id) % 5000) / 1000;
    this.resetDynamics();
  }

  /** Forget all history (phases, blends): the next update depends only on its inputs. */
  resetDynamics(): void {
    // drop every action and binding too: their loop counters and saved values are history as well
    this.mixer.stopAllAction();
    this.mixer.uncacheRoot(this.model);
    this.mixer = new THREE.AnimationMixer(this.model);
    this.actions.clear();
    this.active.clear();
    for (const r of this.rest) {
      r.bone.quaternion.copy(r.q);
      r.bone.position.copy(r.p);
    }
    this.phase = this.seedPhase;
    this.idleT = this.seedIdle;
    this.lastT = null;
    this.cur = null;
    this.prev = null;
  }

  /** Diagnostics for tests: the shared locomotion phase and idle clock. */
  get dynamics(): { phase: number; idleT: number } {
    return { phase: this.phase, idleT: this.idleT };
  }

  private action(name: string): THREE.AnimationAction | null {
    let a = this.actions.get(name);
    if (!a) {
      const clip = this.factory.clip(name);
      if (!clip) return null;
      a = this.mixer.clipAction(clip);
      a.setLoop(THREE.LoopRepeat, Infinity);
      this.actions.set(name, a);
    }
    return a;
  }

  private loopTime(name: string): number {
    const c = this.factory.info[name];
    if (!c) return 0;
    return ((this.phase + (c.phase0_s ?? 0) / c.duration_s) % 1) * c.duration_s;
  }

  private idleTime(name: string): number {
    const c = this.factory.info[name];
    return c ? this.idleT % c.duration_s : 0;
  }

  /**
   * Locomotion clips for the player's motion: the forward idle/jog/run/sprint blend or, when `directional`, a
   * backpedal / strafe clip if the player moves away from the facing (keepers' footwork).
   */
  private motionClips(p: RenderPlayer, directional: boolean): [string, number][] {
    const rel = angleDelta(p.facing, Math.atan2(p.vy, p.vx));
    if (directional && p.speed > 0.3 && Math.abs(rel) > 0.8) {
      const name = Math.abs(rel) > 2.3 ? "Running_Backward" : rel > 0 ? "Jog_Strafe_Left" : "Jog_Strafe_Right";
      const s = Math.min(1, p.speed / 0.8);
      return s < 1 ? [[name, s], ["Breathing_Idle", 1 - s]] : [[name, 1]];
    }
    const out: [string, number][] = [];
    const lw = locomotionWeights(p.speed);
    LOCO_CLIPS.forEach((name, i) => {
      const wi = lw[i] ?? 0;
      if (wi > 0) out.push([name, wi]);
    });
    return out;
  }

  private cadenceOf(clips: [string, number][], speed: number): number {
    let num = 0;
    let den = 0;
    for (const [name, w] of clips) {
      const c = this.factory.info[name];
      if (!c?.natural_speed_mps) continue;
      num += w * (loopRate(speed, c.natural_speed_mps) / c.duration_s);
      den += w;
    }
    return den > 0 ? num / den : 0;
  }

  private addMotion(pose: Pose, clips: [string, number][], w: number, suffix = ""): void {
    for (const [name, cw] of clips) {
      const c = this.factory.info[name];
      add(pose, name + suffix, c?.natural_speed_mps ? this.loopTime(name) : this.idleTime(name), cw * w);
    }
  }

  /** Clip weights for one layer. */
  private layerPose(l: Layer, p: RenderPlayer, t: number, pose: Pose, w: number): void {
    if (w <= 0) return;
    const spec = ANIMS[l.anim] ?? ANIMS.idle;
    const info = this.factory.info;
    if (spec.kind === "locomotion") {
      this.addMotion(pose, this.motionClips(p, false), w);
    } else if (l.anim === "gk_idle") {
      // the keeper's set stance when still, real footwork (shuffle, backpedal, run) when moving
      const s = smoothstep(0.6, 1.4, p.speed);
      add(pose, "Goalkeeper_Idle", this.idleTime("Goalkeeper_Idle"), (1 - s) * w);
      if (s > 0) this.addMotion(pose, this.motionClips(p, true), s * w);
    } else if (spec.kind === "loop") {
      const name = clipFor(l.anim, l.variant);
      const c = info[name];
      if (c?.natural_speed_mps) {
        // fade to idle as the speed drops, so slow strafes and backpedals never skate
        const s = Math.min(1, p.speed / 0.8);
        add(pose, name, this.loopTime(name), s * w);
        add(pose, "Breathing_Idle", this.idleTime("Breathing_Idle"), (1 - s) * w);
      } else {
        add(pose, name, (t - l.start) % (c?.duration_s ?? 1), w);
      }
    } else {
      const name = clipFor(l.anim, l.variant);
      const c = info[name];
      if (!c) return;
      const time = oneShotTime(l.anim, c, Math.max(0, t - l.start));
      const split = spec.upperBody ? smoothstep(SPLIT_FROM, SPLIT_TO, p.speed) : 0;
      add(pose, name, time, (1 - split) * w);
      if (split > 0) {
        add(pose, `${name}#upper`, time, split * w);
        this.addMotion(pose, this.motionClips(p, false), split * w, "#lower");
      }
    }
  }

  /** Cycles per second of the shared phase for what the legs are doing now. */
  private cadence(p: RenderPlayer): number {
    const spec = ANIMS[p.anim] ?? ANIMS.idle;
    if (p.anim === "gk_idle") return this.cadenceOf(this.motionClips(p, true), p.speed);
    if (spec.kind === "loop") {
      const c = this.factory.info[clipFor(p.anim, p.variant)];
      return c ? loopRate(p.speed, c.natural_speed_mps) / c.duration_s : 1;
    }
    return locomotionCadence(p.speed, this.factory.info);
  }

  update(p: RenderPlayer, t: number): void {
    // match-time step: zero while paused; a jump (seek, rewind) resets the dynamics
    let dtm = this.lastT === null ? 0 : t - this.lastT;
    if (dtm < 0 || dtm > 0.5) {
      this.resetDynamics();
      dtm = 0;
    }
    this.lastT = t;

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

    this.phase = (this.phase + this.cadence(p) * dtm) % 1;
    this.idleT += dtm;

    let wCur = 1;
    if (this.prev) {
      const since = t - this.prev.end;
      if (since >= FADE_S || since < 0) this.prev = null;
      else wCur = since / FADE_S;
    }
    const pose: Pose = new Map();
    if (this.cur) this.layerPose(this.cur, p, t, pose, wCur);
    if (this.prev) this.layerPose(this.prev, p, t, pose, 1 - wCur);
    this.apply(pose);

    const ownsHeight = (ANIMS[p.anim] ?? ANIMS.idle).ownsHeight ?? false;
    this.root.position.set(p.x, ownsHeight ? 0 : p.z, -p.y);
    this.root.rotation.set(0, facingToYaw(p.facing), 0);
    const mirror = this.cur?.mirror ?? false;
    this.root.scale.set(this.heightScale * (mirror ? -1 : 1), this.heightScale, this.heightScale);
    kitUniforms(this.material).uMirror!.value = mirror ? -1 : 1;
    this.root.visible = p.onPitch;
  }

  private apply(pose: Pose): void {
    // Start every evaluation from the bind pose. The mixer saves a binding's "original" value when an action
    // starts and blends any missing weight with it; without this the saved value is whatever pose the bone had
    // then, so the result would depend on history (and recordings would not be reproducible).
    for (const r of this.rest) {
      r.bone.quaternion.copy(r.q);
      r.bone.position.copy(r.p);
    }
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
  }

  /** Static pose for art previews and tests: one clip at a time, no state machine. */
  posePreview(clip: string, time: number): void {
    this.apply(new Map([[clip, { time, weight: 1 }]]));
  }
}

function add(pose: Pose, name: string, time: number, weight: number): void {
  if (weight <= 1e-4) return;
  const e = pose.get(name);
  if (!e) {
    pose.set(name, { time, weight });
    return;
  }
  if (weight > e.weight) e.time = time;
  e.weight += weight;
}

function hashId(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
