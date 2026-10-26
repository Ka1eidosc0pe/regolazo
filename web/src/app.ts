// The renderer app: builds the world, owns the frame source, the athletes, the camera director and post.
import * as THREE from "three";
import { Athlete, AthleteFactory } from "./athletes/athletes.ts";
import { backTexture, blankTexture, frontTexture, SPONSORS } from "./athletes/kitTextures.ts";
import { kitFor, lookFor, REFEREE_KIT } from "./athletes/roster.ts";
import { CameraDirector, type CameraMode } from "./camera/director.ts";
import { Post } from "./post/post.ts";
import type { FrameSource } from "./source/FrameSource.ts";
import type { RenderState } from "./source/interpolate.ts";
import type { MatchMeta, PlayerMeta } from "./types.ts";
import { createAdBoards, type AdBoards } from "./world/adboards.ts";
import { blobShadow, createBall, type Ball } from "./world/ball.ts";
import { createCrowd, type Crowd } from "./world/crowd.ts";
import { createGoal, type Goal } from "./world/goals.ts";
import { buildEnvironment, createLighting, createSky, FOG_COLOR, type Lighting } from "./world/lighting.ts";
import { createPitch } from "./world/pitch.ts";
import { createStadium, type Stadium } from "./world/stadium.ts";

export interface AppOptions {
  width?: number;
  height?: number;
  quality?: "high" | "low";
  /** Overrides: MSAA samples of the HDR target and shadow map size. */
  msaa?: number;
  shadowMap?: number;
  record?: boolean;
  assetBase?: string;
}

export interface PlayerRig {
  meta: PlayerMeta | null;
  athlete: Athlete;
  blob: THREE.Mesh;
  team: "home" | "away" | null;
  isGK: boolean;
}

export class App {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly director: CameraDirector;
  readonly post: Post;
  readonly source: FrameSource;
  meta!: MatchMeta;
  readonly rigs = new Map<string, PlayerRig>();
  referee: PlayerRig | null = null;
  ball!: Ball;
  goals: Goal[] = [];
  private stadium!: Stadium;
  private crowd!: Crowd;
  private boards!: AdBoards;
  private lighting!: Lighting;
  private frameNo = 0;
  private lastT = 0;
  private goalT = -Infinity;
  private goalScore = -1;
  state: RenderState | null = null;
  private running = false;
  private width: number;
  private height: number;
  private readonly fixedSize: boolean;
  private readonly opts: AppOptions;
  private readonly container: HTMLElement;
  fps = 0;

  constructor(container: HTMLElement, source: FrameSource, opts: AppOptions = {}) {
    this.opts = opts;
    this.source = source;
    this.fixedSize = opts.width !== undefined && opts.height !== undefined;
    this.container = container;
    // fall back to the window if the container has not been laid out yet
    this.width = opts.width ?? (container.clientWidth || window.innerWidth || 1280);
    this.height = opts.height ?? (container.clientHeight || window.innerHeight || 720);
    this.renderer = new THREE.WebGLRenderer({
      antialias: false,
      powerPreference: "high-performance",
      preserveDrawingBuffer: opts.record ?? false,
      stencil: false,
    });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(this.width, this.height, !this.fixedSize);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.92;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.domElement.id = "stage";
    container.appendChild(this.renderer.domElement);
    this.director = new CameraDirector(this.width / this.height);
    // live view: 2x MSAA keeps 1080p above 60 fps on the M3 Pro; fixed-size renders (recording, screenshots) use 4x
    const msaa = opts.msaa ?? (opts.quality === "low" ? 0 : this.fixedSize || opts.record ? 4 : 2);
    this.post = new Post(this.renderer, this.scene, this.director.camera, this.width, this.height, msaa);
    if (!this.fixedSize) {
      new ResizeObserver(() => this.ensureSize()).observe(container);
    }
  }

  resize(w: number, h: number): void {
    this.width = w;
    this.height = h;
    this.renderer.setSize(w, h, !this.fixedSize);
    this.post.setSize(w, h);
    this.director.setAspect(w / h);
  }

  /** Match the drawing buffer and post targets to the container (never zero-sized). */
  ensureSize(): void {
    if (this.fixedSize) return;
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    if (w > 0 && h > 0 && (w !== this.width || h !== this.height)) this.resize(w, h);
  }

  get size(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }

  async init(): Promise<void> {
    const [meta, factory] = await Promise.all([this.source.connect(), AthleteFactory.load(this.opts.assetBase ?? "/assets/")]);
    this.meta = meta;
    const scene = this.scene;
    scene.fog = new THREE.FogExp2(FOG_COLOR, 0.0042);
    scene.background = new THREE.Color("#03050a");
    scene.environment = buildEnvironment(this.renderer);
    scene.environmentIntensity = 1.0;
    scene.add(createSky());
    this.lighting = createLighting(scene, this.opts.shadowMap ?? (this.opts.quality === "low" ? 1024 : 2048));
    scene.add(createPitch());
    this.stadium = createStadium();
    scene.add(this.stadium.group);
    this.crowd = createCrowd(this.stadium.path);
    scene.add(this.crowd.group);
    this.boards = createAdBoards();
    scene.add(this.boards.group);
    this.goals = [createGoal(1), createGoal(-1)];
    for (const g of this.goals) scene.add(g.group);
    this.ball = createBall();
    scene.add(this.ball.group);
    this.addCornerFlags();

    // athletes
    for (const side of ["home", "away"] as const) {
      const team = side === "home" ? meta.home : meta.away;
      const front = frontTexture(team, team.kit, SPONSORS[side] ?? "");
      const gkFront = frontTexture(team, team.gk_kit, SPONSORS[side] ?? "");
      for (const p of team.players) {
        const kit = kitFor(team, p);
        const look = lookFor(p.id, p.last_name);
        const isGK = p.position === "GK";
        const athlete = factory.create({
          id: p.id,
          heightCm: p.height_cm,
          hair: look.hair,
          hairColor: look.hairColor,
          seed: look.seed,
          look: {
            kit,
            shorts: kit.secondary,
            socks: kit.primary,
            skin: look.skin,
            boot: look.boot,
            glove: isGK ? "#f2f4f8" : look.skin,
            goalkeeper: isGK,
            back: backTexture(p.display_name, p.shirt_number, kit),
            front: isGK ? gkFront : front,
            hasNumber: true,
            hasFront: true,
          },
        });
        this.addRig(p.id, { meta: p, athlete, team: side, isGK });
      }
    }
    const refLook = lookFor("REF", "referee");
    const ref = factory.create({
      id: "referee",
      heightCm: 181,
      hair: "buzz",
      hairColor: "#2b1d14",
      seed: refLook.seed,
      look: {
        kit: REFEREE_KIT,
        shorts: REFEREE_KIT.primary,
        socks: "#0c0d10",
        skin: refLook.skin,
        boot: "#101114",
        glove: refLook.skin,
        goalkeeper: false,
        back: blankTexture(),
        front: blankTexture(),
        hasNumber: false,
        hasFront: false,
      },
    });
    this.referee = { meta: null, athlete: ref, blob: this.makeBlob(), team: null, isGK: false };
    scene.add(ref.root);

    const c = this.source.clock();
    this.lastT = c.t;
    this.renderer.compile(scene, this.director.camera);
  }

  private makeBlob(): THREE.Mesh {
    const b = blobShadow(1.3, 0.55);
    this.scene.add(b);
    return b;
  }

  private addRig(id: string, r: Omit<PlayerRig, "blob">): void {
    const rig: PlayerRig = { ...r, blob: this.makeBlob() };
    this.rigs.set(id, rig);
    this.scene.add(r.athlete.root);
  }

  private addCornerFlags(): void {
    const poleMat = new THREE.MeshStandardMaterial({ color: 0xf2f2f0, roughness: 0.4 });
    const flagMat = new THREE.MeshStandardMaterial({ color: new THREE.Color("#d7ff3a"), roughness: 0.6, side: THREE.DoubleSide, emissive: new THREE.Color("#2a3300") });
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1.5, 8), poleMat);
        pole.position.set(sx * 52.5, 0.75, -sy * 34);
        pole.castShadow = true;
        const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.4, 0.3), flagMat);
        flag.position.set(sx * 52.5 + 0.2 * -sx, 1.35, -sy * 34);
        this.scene.add(pole, flag);
      }
    }
  }

  setCamera(mode: CameraMode, transitionS = 1.3): void {
    this.director.setMode(mode, transitionS);
  }

  /** Advance the world to match time t with frame step dt and update every object (no rendering). */
  step(t: number, dt: number): void {
    const state = this.source.sample(t);
    this.state = state;
    if (!state) return;
    for (const p of state.players) {
      const rig = this.rigs.get(p.id);
      if (!rig) continue;
      rig.athlete.update(p, t, dt);
      rig.blob.position.set(p.x, 0.011, -p.y);
      rig.blob.visible = p.onPitch;
    }
    if (state.referee && this.referee) {
      const r = state.referee;
      const sp = Math.hypot(r.vx, r.vy);
      this.referee.athlete.update(
        { id: "referee", x: r.x, y: r.y, z: 0, vx: r.vx, vy: r.vy, speed: sp, facing: r.facing, anim: "jog", animT: 0, mirror: false, variant: 0, onPitch: true },
        t,
        dt,
      );
      this.referee.blob.position.set(r.x, 0.011, -r.y);
    }
    this.ball.update(state.ball.pos, state.ball.vel, state.ball.spin, t, dt, this.director.camera);
    for (const g of this.goals) g.update(state.ball.pos, dt);
    // crowd excitement: rises after a goal, decays over ~10 s, plus a little tension near the boxes
    const total = state.base.score.home + state.base.score.away;
    if (total !== this.goalScore) {
      if (this.goalScore >= 0 || total > 0) this.goalT = this.findGoalTime(total);
      this.goalScore = total;
    }
    const since = t - this.goalT;
    const goalHype = since >= 0 ? Math.exp(-since / 9) : 0;
    const tension = Math.max(0, (Math.abs(state.ball.pos[0]) - 35) / 17) * 0.25;
    this.crowd.update(t, Math.min(1, goalHype + tension));
    this.boards.update(t);
    this.stadium.update(t);
    this.director.update(state, dt);
    const focus = this.director.groundFocus();
    this.lighting.focus(focus, this.director.mode === "tactical" ? 70 : 42);
    this.post.setDof(this.director.mode === "replay", this.director.focusDistance, 0.0006);
  }

  private findGoalTime(total: number): number {
    const buf = this.source.buffer;
    let lo = 0;
    let hi = buf.length - 1;
    const sum = (i: number) => {
      const s = buf.at(i)!;
      return s.score.home + s.score.away;
    };
    if (hi < 0 || sum(hi) < total) return -Infinity;
    if (sum(0) >= total) return buf.at(0)!.t - 1000;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sum(mid) >= total) hi = mid;
      else lo = mid + 1;
    }
    return buf.at(lo)!.t;
  }

  render(): void {
    this.ensureSize();
    this.post.render(this.frameNo++);
  }

  /** Jump to t: prime the camera and settle the dynamics over `warm` seconds of fixed steps, then render. */
  jumpTo(t: number, warm = 2.5, fps = 25): void {
    this.source.seek(t - warm);
    this.ball.reset();
    const t0 = Math.max(this.source.clock().t, t - warm);
    const s0 = this.source.sample(t0);
    if (s0) this.director.prime(s0);
    const n = Math.max(1, Math.round((t - t0) * fps));
    for (let i = 1; i <= n; i++) this.step(t0 + ((t - t0) * i) / n, 1 / fps);
    this.source.seek(t);
    this.lastT = t;
    this.render();
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    let prev = performance.now();
    let acc = 0;
    let frames = 0;
    const loop = (now: number) => {
      if (!this.running) return;
      const dt = Math.min(0.1, (now - prev) / 1000);
      prev = now;
      const t = this.source.advance(dt);
      if (t < this.lastT - 0.5) this.ball.reset();
      this.lastT = t;
      this.step(t, dt);
      this.render();
      frames++;
      acc += dt;
      if (acc >= 0.5) {
        this.fps = frames / acc;
        frames = 0;
        acc = 0;
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
  }
}
