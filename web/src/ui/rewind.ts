// Rewind & Rewrite (PLAN.md Step 9, sprint scope): freeze on a moment, swing to the tactical camera, glow the pitch
// control on the grass, let the fan pick a teammate (or a goal spot) and a technique, commit to a rollout that plays
// in full colour with reality as cyan ghosts, then the verdict, the "1,000 realities" distribution and back to live.
import type { App } from "../app.ts";
import { ClipSource } from "../source/ClipSource.ts";
import { StateBuffer } from "../source/interpolate.ts";
import type {
  ActionIntent,
  DecisionMoment,
  MatchEvent,
  OptionEvaluation,
  ResolvedAction,
  RolloutDistribution,
  RolloutResult,
  SimState,
} from "../types/generated/contracts.ts";
import { PitchControlLayer, PreviewArc, RealityGhosts, straightArc } from "../world/rewindFx.ts";
import type { Api, RolloutBody } from "./api.ts";
import type { Hud } from "./hud.ts";
import {
  dedupeOptions,
  distributionBars,
  eventCard,
  FRAMING_TEXT,
  GOAL_X,
  openTeammate,
  OUTCOME_TEXT,
  pct,
  realOption,
  realOutcome,
  verdictFraming,
  type Outcome,
} from "./logic.ts";
import type { Narration } from "./narration.ts";
import { el, projectWorld } from "./project.ts";
import type { SwitchSource } from "./switchSource.ts";

export type RewindPhase = "idle" | "loading" | "choose" | "committing" | "playing" | "verdict" | "realities";

const SEED = 2026;
const HORIZON_S = 10;
const GHOST_STRIDE_S = 0.2;
const DESAT_FREEZE = 0.62;

export class Rewind {
  phase: RewindPhase = "idle";
  private readonly app: App;
  private readonly source: SwitchSource;
  private readonly api: Api;
  private readonly hud: Hud;
  private readonly narration: Narration;
  private readonly layer: HTMLDivElement;
  private readonly rings: HTMLDivElement;
  private readonly panel: HTMLDivElement;
  private readonly card: HTMLDivElement;
  private readonly banner: HTMLDivElement;
  private readonly pc = new PitchControlLayer();
  private readonly ghosts = new RealityGhosts();
  private readonly arc = new PreviewArc();
  private moment: DecisionMoment | null = null;
  private frozen: SimState | null = null;
  private selected: OptionEvaluation | null = null;
  private preview: ResolvedAction | null = null;
  private technique: string | null = null;
  private analyst = false;
  private desat = 0;
  private desatTarget = 0;
  private result: RolloutResult | null = null;
  private rollout: ClipSource | null = null;
  private reality = new StateBuffer();
  private rolloutEvents: MatchEvent[] = [];
  private nextEvent = 0;
  private realOutcomeOverride: Outcome | null = null;
  private resolveVerdict: (() => void) | null = null;
  verdictShown: Promise<void> = Promise.resolve();
  private body: RolloutBody | null = null;

  constructor(overlay: HTMLElement, app: App, source: SwitchSource, api: Api, hud: Hud, narration: Narration) {
    this.app = app;
    this.source = source;
    this.api = api;
    this.hud = hud;
    this.narration = narration;
    this.layer = el("div", "rewind-layer");
    this.rings = el("div", "rw-rings");
    this.panel = el("div", "rw-panel");
    this.panel.dataset.testid = "rewind-panel";
    this.card = el("div", "rw-card");
    this.card.dataset.testid = "verdict";
    this.banner = el("div", "rw-banner");
    this.banner.dataset.testid = "rewind-banner";
    this.layer.append(this.rings, this.banner, this.panel, this.card);
    overlay.appendChild(this.layer);
    app.scene.add(this.pc.mesh, this.ghosts.group, this.arc.group);
  }

  private setPhase(p: RewindPhase): void {
    this.phase = p;
    document.body.dataset.rewind = p;
  }

  /** Fan-initiated rewind on any on-ball event (a keyframe at every touch). */
  async openAt(e: MatchEvent): Promise<void> {
    if (!e.player || !e.team) return;
    const ref = `snap:${this.app.meta.match_id}:${e.tick ?? 0}`;
    const mo = await this.api.options(ref);
    const q = (e.qualifiers ?? {}) as Record<string, unknown>;
    const recipient = typeof q.recipient === "string" ? q.recipient : null;
    const kind: ActionIntent["kind"] = e.type === "shot" ? "shot" : "pass_player";
    const realIntent: ActionIntent = {
      kind,
      player: e.player,
      target_player: kind === "pass_player" ? recipient : null,
      target: e.end_location ?? null,
      path: [],
      curve: 0,
      power: 0.5,
      pass_technique_override: null,
      shot_technique_override: null,
      source: "ai",
    };
    const options = mo.options.map((o) => ({
      ...o,
      is_real: kind === "shot" ? o.intent.kind === "shot" : o.intent.kind === "pass_player" && o.intent.target_player === recipient,
    }));
    this.realOutcomeOverride =
      e.type === "shot" ? (q.outcome === "goal" ? "goal" : q.outcome === "saved" ? "shot_saved" : "shot_off_target") : q.outcome === "complete" ? "possession_retained" : "turnover";
    const m: DecisionMoment = {
      schema_version: "1.0.0",
      id: `fan_${e.tick}`,
      tick: e.tick ?? 0,
      t: e.t,
      period: e.period ?? 1,
      snapshot_ref: ref,
      team: e.team,
      ball_carrier: e.player,
      moment_type: e.type === "shot" ? "shot" : "other",
      origin: "fan",
      real_event_id: e.id,
      scores: null,
      real_action: {
        intent: realIntent,
        action_type: e.type === "shot" ? "shot" : "pass",
        body_part: null,
        pass_technique: null,
        pass_type: null,
        pass_height: null,
        shot_technique: null,
        intended_end: e.end_location ?? e.location ?? [0, 0, 0],
        launch: null,
        ellipse: null,
        preview_arc: [],
        curve_applied: 0,
        override_rejected: false,
        weak_foot: false,
        execution_quality: 1,
        available_pass_techniques: [],
        available_shot_techniques: [],
        max_curve: 0,
      },
      consequence: { kind: "none", xt_swing: 0, horizon_s: 10, event_ids: [] },
      options,
      flagged: false,
      hook: `${this.hud.name(e.player)} had ${options.filter((o) => o.viable).length} options here. What would you have done?`,
    };
    await this.open(m, mo.pitch_control ?? null, true);
  }

  async open(m: DecisionMoment, pitchControl: import("../types/generated/contracts.ts").PitchControlGrid | null = null, fan = false): Promise<void> {
    if (this.phase !== "idle") this.teardown();
    if (!fan) this.realOutcomeOverride = null;
    this.moment = m;
    this.setPhase("loading");
    this.verdictShown = new Promise((res) => (this.resolveVerdict = res));
    this.source.live.pause();
    this.hud.tagsEnabled = false;
    this.hud.clearTags();
    this.hud.hideCard();
    this.banner.replaceChildren(el("span", "rw-tag", "Rewind"), el("span", "rw-hook", "Freezing the moment…"));
    this.banner.classList.add("show");
    const [state, mo] = await Promise.all([
      this.api.state(m.t + 0.001),
      pitchControl ? Promise.resolve(null) : this.api.options(m.snapshot_ref).catch(() => null),
    ]);
    this.frozen = state;
    const freeze = new ClipSource({ meta: this.app.meta, frames: [state] }, { loop: false });
    await freeze.connect();
    freeze.pause();
    this.source.use(freeze);
    this.app.ball.reset();
    this.app.setCamera("tactical", 1.6);
    this.desatTarget = DESAT_FREEZE;
    const grid = pitchControl ?? mo?.pitch_control ?? null;
    if (grid) {
      this.pc.set(grid, m.team);
      this.pc.show(true);
    }
    if (m.options.length === 0 && mo) m.options = mo.options;
    this.banner.replaceChildren(el("span", "rw-tag", "Rewind & Rewrite"), el("span", "rw-hook", m.hook ?? "What would you have done?"));
    this.narration.say("Rewind", m.hook ?? "What would you have done?");
    this.selected = null;
    this.preview = null;
    this.technique = null;
    this.arc.set(null);
    this.buildRings();
    this.renderPanel();
    this.setPhase("choose");
  }

  private carrierXY(): [number, number, number] {
    const s = this.frozen;
    const p = s?.players.find((q) => q.id === this.moment?.ball_carrier);
    if (s) return [s.ball.pos[0], s.ball.pos[1], s.ball.pos[2]];
    return p ? [p.pos[0], p.pos[1], 0.11] : [0, 0, 0.11];
  }

  private goalX(): number {
    const s = this.frozen;
    const team = this.moment?.team ?? "home";
    return s && s.attacking_positive_x === team ? GOAL_X : -GOAL_X;
  }

  private shotOption(): OptionEvaluation | null {
    return this.moment?.options.find((o) => o.intent.kind === "shot") ?? null;
  }

  private buildRings(): void {
    this.rings.replaceChildren();
    const m = this.moment;
    if (!m) return;
    const carrier = el("div", "rw-ring carrier");
    carrier.dataset.player = m.ball_carrier;
    carrier.appendChild(el("span", "rw-ring-label", this.hud.name(m.ball_carrier)));
    this.rings.appendChild(carrier);
    for (const o of m.options) {
      if (o.intent.kind !== "pass_player" || !o.intent.target_player) continue;
      const b = el("button", "rw-ring mate");
      b.dataset.option = o.option_id;
      b.dataset.player = o.intent.target_player;
      b.dataset.testid = `mate-${o.intent.target_player}`;
      const lab = el("span", "rw-ring-label", this.hud.name(o.intent.target_player));
      const p = el("span", "rw-ring-p", pct(o.p_success));
      b.append(lab, p);
      b.addEventListener("click", () => void this.select(o));
      this.rings.appendChild(b);
    }
    const goal = el("button", "rw-ring goal");
    goal.dataset.testid = "goal-spot";
    goal.append(el("span", "rw-ring-label", "Shoot"), el("span", "rw-ring-p", this.shotOption() ? pct(this.shotOption()!.p_success) : ""));
    goal.addEventListener("click", () => void this.selectShot());
    this.rings.appendChild(goal);
    this.rings.classList.toggle("numbers", this.analyst);
  }

  private async selectShot(): Promise<void> {
    const m = this.moment;
    if (!m) return;
    const existing = this.shotOption();
    if (existing) return this.select(existing);
    const keeper = this.frozen?.players.find((p) => Math.abs(p.pos[0] - this.goalX()) < 6);
    const y = keeper && keeper.pos[1] > 0 ? -2.6 : 2.6;
    const intent: ActionIntent = {
      kind: "shot",
      player: m.ball_carrier,
      target_player: null,
      target: [this.goalX(), y, 0.5],
      path: [],
      curve: 0,
      power: 0.75,
      pass_technique_override: null,
      shot_technique_override: null,
      source: "fan",
    };
    const o: OptionEvaluation = {
      option_id: "fan_shot",
      intent,
      resolved: null,
      label: "Shot",
      label_key: "option.shot",
      label_args: {},
      p_success: 0,
      p_success_baseline: null,
      value_if_success: 0,
      xg: null,
      xt_gain: null,
      decision_value: 0,
      viable: true,
      is_real: false,
    };
    return this.select(o);
  }

  async select(o: OptionEvaluation): Promise<void> {
    const m = this.moment;
    if (!m || (this.phase !== "choose" && this.phase !== "loading")) return;
    this.selected = o;
    this.technique = null;
    for (const r of this.rings.querySelectorAll<HTMLElement>(".rw-ring")) r.classList.toggle("selected", r.dataset.option === o.option_id || (o.intent.kind === "shot" && r.classList.contains("goal")));
    this.renderPanel();
    const intent: ActionIntent = { ...o.intent, source: "fan" };
    this.preview = await this.api.preview(m.snapshot_ref, intent).catch(() => o.resolved ?? null);
    const arc = this.preview?.preview_arc ?? [];
    if (arc.length >= 2) this.arc.set(arc as [number, number, number][]);
    else {
      const end = this.preview?.intended_end ?? o.intent.target ?? this.targetXY(o);
      this.arc.set(straightArc(this.carrierXY(), end as [number, number, number], o.intent.kind === "shot" ? 1 : 0.6));
    }
    this.renderPanel();
  }

  private targetXY(o: OptionEvaluation): [number, number, number] {
    const p = this.frozen?.players.find((q) => q.id === o.intent.target_player);
    return p ? [p.pos[0], p.pos[1], 0] : this.carrierXY();
  }

  async pickOptionId(id: string): Promise<void> {
    const o = this.moment?.options.find((x) => x.option_id === id);
    if (o) await this.select(o);
  }

  async pickOpenTeammate(): Promise<string | null> {
    if (!this.moment) return null;
    const o = openTeammate(this.moment);
    if (o) await this.select(o);
    return o?.intent.target_player ?? null;
  }

  setAnalyst(on: boolean): void {
    this.analyst = on;
    this.rings.classList.toggle("numbers", on);
    this.renderPanel();
  }

  private renderPanel(): void {
    const m = this.moment;
    this.panel.replaceChildren();
    if (!m || this.phase === "playing" || this.phase === "verdict" || this.phase === "realities") {
      this.panel.classList.remove("show");
      return;
    }
    this.panel.classList.add("show");
    const head = el("div", "rw-head");
    head.append(el("span", "rw-kicker", m.origin === "fan" ? "Your rewind" : "Flagged moment"), el("span", "rw-title", `${this.hud.name(m.ball_carrier)} on the ball`));
    this.panel.appendChild(head);
    const list = el("div", "rw-options");
    const ranked = dedupeOptions(m.options).filter((o) => o.viable || o === this.selected).sort((a, b) => b.decision_value - a.decision_value).slice(0, 7);
    if (this.selected && !ranked.includes(this.selected)) ranked.unshift(this.selected);
    for (const o of ranked) {
      const b = el("button", "rw-opt");
      b.dataset.option = o.option_id;
      b.classList.toggle("selected", o === this.selected);
      b.append(el("span", "rw-opt-label", this.label(o)));
      if (this.analyst) b.append(el("span", "rw-opt-p", pct(o.p_success)));
      b.addEventListener("click", () => void this.select(o));
      list.appendChild(b);
    }
    this.panel.appendChild(list);
    if (this.selected && this.preview) {
      const techs: string[] = this.selected.intent.kind === "shot" ? this.preview.available_shot_techniques : this.preview.available_pass_techniques;
      const def = this.selected.intent.kind === "shot" ? this.preview.shot_technique : this.preview.pass_technique;
      if (techs.length > 0) {
        const tw = el("div", "rw-techs");
        tw.appendChild(el("span", "rw-sub", "Technique"));
        for (const t of techs) {
          const c = el("button", "rw-tech", t.replace(/_/g, " "));
          c.classList.toggle("on", (this.technique ?? def) === t);
          c.addEventListener("click", () => {
            this.technique = t === def ? null : t;
            this.renderPanel();
          });
          tw.appendChild(c);
        }
        this.panel.appendChild(tw);
      }
    }
    const foot = el("div", "rw-foot");
    const toggle = el("button", "rw-analyst", this.analyst ? "Hide numbers" : "Show numbers first");
    toggle.dataset.testid = "analyst-toggle";
    toggle.addEventListener("click", () => this.setAnalyst(!this.analyst));
    const commit = el("button", "rw-commit", this.phase === "committing" ? "Simulating…" : "Commit");
    commit.dataset.testid = "commit";
    commit.disabled = !this.selected || this.phase === "committing";
    commit.addEventListener("click", () => void this.commit());
    const cancel = el("button", "rw-cancel", "Back to live");
    cancel.addEventListener("click", () => this.backToLive());
    foot.append(toggle, cancel, commit);
    this.panel.appendChild(foot);
  }

  private label(o: OptionEvaluation): string {
    const who = o.intent.target_player ? this.hud.name(o.intent.target_player) : "";
    switch (o.label_key) {
      case "option.through_ball":
        return `Through ball · ${who || o.label.replace(/^Through ball (for )?/, "")}`;
      case "option.cutback":
        return `Cutback to ${who}`;
      case "option.cross":
        return `Cross to ${who}`;
      case "option.shot":
        return "Shoot";
      case "option.carry":
        return "Carry forward";
      default:
        return who ? `Pass to ${who}` : o.label;
    }
  }

  async commit(): Promise<void> {
    const m = this.moment;
    const o = this.selected;
    if (!m || !o || this.phase !== "choose") return;
    this.setPhase("committing");
    this.renderPanel();
    const intent: ActionIntent = { ...o.intent, source: "fan" };
    if (this.technique) {
      if (o.intent.kind === "shot") intent.shot_technique_override = this.technique as ActionIntent["shot_technique_override"];
      else intent.pass_technique_override = this.technique as ActionIntent["pass_technique_override"];
    }
    this.body = { snapshot_ref: m.snapshot_ref, intent, horizon_s: HORIZON_S, seed: SEED };
    const n = Math.round(HORIZON_S / GHOST_STRIDE_S);
    const [result, real] = await Promise.all([
      this.api.rollout(this.body),
      Promise.all(Array.from({ length: n + 1 }, (_, k) => this.api.state(m.t + k * GHOST_STRIDE_S + 0.001).catch(() => null))),
    ]);
    this.result = result;
    this.reality = new StateBuffer();
    this.reality.setAll(real.filter((s): s is SimState => s !== null && s.t >= m.t - 0.01));
    const clip = new ClipSource({ meta: this.app.meta, frames: result.frames }, { loop: false });
    await clip.connect();
    this.rollout = clip;
    this.rolloutEvents = [...result.events].sort((a, b) => a.t - b.t);
    this.nextEvent = 0;
    this.source.use(clip);
    clip.play();
    this.app.ball.reset();
    this.arc.set(null);
    this.rings.replaceChildren();
    this.pc.show(false);
    this.desatTarget = 0;
    this.ghosts.show(true);
    this.app.setCamera("broadcast", 1.2);
    this.hud.tagsEnabled = true;
    this.banner.replaceChildren(el("span", "rw-tag alt", "Alternate reality"), el("span", "rw-hook", `${this.label(o)} · reality in cyan`));
    this.setPhase("playing");
    this.renderPanel();
  }

  update(dt: number): void {
    this.desat += (this.desatTarget - this.desat) * Math.min(1, dt * 3);
    const u = (this.app.post.grade.uniforms as Record<string, { value: number }>).uDesat;
    if (u) u.value = this.desat;
    this.pc.update(dt);
    if (this.phase === "choose" || this.phase === "committing" || this.phase === "loading") this.placeRings();
    if (this.phase === "playing" && this.rollout && this.result) {
      const t = this.rollout.clock().t;
      this.ghosts.update(this.reality.sample(t));
      while (this.nextEvent < this.rolloutEvents.length && this.rolloutEvents[this.nextEvent]!.t <= t) {
        const card = eventCard(this.rolloutEvents[this.nextEvent]!);
        if (card) this.hud.showCard(card);
        this.nextEvent++;
      }
      const end = Math.min(this.rollout.endT, this.result.outcome.terminal_t + 2.2);
      if (t >= end - 0.01) {
        this.rollout.pause();
        this.showVerdict();
      }
    } else if (this.phase === "verdict" || this.phase === "realities") {
      if (this.rollout) this.ghosts.update(this.reality.sample(this.rollout.clock().t));
    }
  }

  private placeRings(): void {
    const s = this.frozen;
    if (!s) return;
    for (const r of this.rings.querySelectorAll<HTMLElement>(".rw-ring")) {
      let x: number;
      let y: number;
      let z = 0;
      if (r.classList.contains("goal")) {
        x = this.goalX();
        y = 0;
        z = 1.2;
      } else {
        const p = s.players.find((q) => q.id === r.dataset.player);
        if (!p) continue;
        [x, y] = p.pos;
      }
      const sp = projectWorld(this.app, x, y, z);
      r.style.transform = `translate(${sp.x.toFixed(1)}px, ${sp.y.toFixed(1)}px) translate(-50%, -50%)`;
    }
  }

  private showVerdict(): void {
    const m = this.moment;
    const r = this.result;
    if (!m || !r || this.phase !== "playing") return;
    this.setPhase("verdict");
    this.hud.hideCard();
    const real = realOption(m);
    const framing = verdictFraming(r.evaluation, real, r.outcome);
    const realOut = this.realOutcomeOverride ?? realOutcome(m);
    this.card.replaceChildren();
    this.card.dataset.framing = framing;
    const head = el("div", "vd-head");
    head.append(el("span", "vd-kicker", "Verdict"), el("span", `vd-framing ${framing}`, FRAMING_TEXT[framing]));
    const col = (who: string, label: string, p: number | null, base: number | null, outcome: string, cls: string, isShot: boolean) => {
      const c = el("div", `vd-col ${cls}`);
      c.append(el("span", "vd-who", who), el("span", "vd-label", label));
      const stat = el("div", "vd-stat");
      stat.append(el("span", "vd-p", p === null ? "–" : pct(p)), el("span", "vd-p-k", isShot ? "xG" : "chance it works"));
      c.appendChild(stat);
      if (base !== null) c.appendChild(el("span", "vd-base", `Average player: ${pct(base)}`));
      c.appendChild(el("span", "vd-out", outcome));
      return c;
    };
    const userLabel = this.label(this.selected ?? r.evaluation);
    const tech = r.resolved.pass_technique ?? r.resolved.shot_technique;
    const grid = el("div", "vd-grid");
    grid.append(
      col("Your call", tech ? `${userLabel} · ${tech.replace(/_/g, " ")}` : userLabel, r.evaluation.p_success, r.evaluation.p_success_baseline ?? null, OUTCOME_TEXT[r.outcome.label], "you", r.resolved.action_type === "shot"),
      col("Reality", real ? this.label(real) : (m.real_action.action_type ?? "").replace(/_/g, " "), real?.p_success ?? null, real?.p_success_baseline ?? null, OUTCOME_TEXT[realOut], "real", (real?.intent.kind ?? m.real_action.intent.kind) === "shot"),
    );
    const actions = el("div", "vd-actions");
    const runBtn = el("button", "vd-run", "Run 1,000 realities");
    runBtn.dataset.testid = "run-realities";
    runBtn.addEventListener("click", () => void this.runRealities());
    const replay = el("button", "vd-btn", "Replay");
    replay.addEventListener("click", () => this.replay());
    const back = el("button", "vd-btn", "Back to live");
    back.dataset.testid = "back-to-live";
    back.addEventListener("click", () => this.backToLive());
    actions.append(runBtn, replay, back);
    const dist = el("div", "vd-dist");
    dist.dataset.testid = "distribution";
    this.card.append(head, grid, dist, actions);
    this.card.classList.add("show");
    this.banner.classList.remove("show");
    this.narration.say("Verdict", `${FRAMING_TEXT[framing]}. ${userLabel}: ${OUTCOME_TEXT[r.outcome.label].toLowerCase()}, ${pct(r.evaluation.p_success)} for this player. Reality: ${OUTCOME_TEXT[realOut].toLowerCase()}.`);
    this.resolveVerdict?.();
  }

  private replay(): void {
    if (!this.rollout || this.phase !== "verdict") return;
    this.card.classList.remove("show");
    this.rollout.seek(this.rollout.startT);
    this.rollout.play();
    this.nextEvent = 0;
    this.app.ball.reset();
    this.banner.classList.add("show");
    this.setPhase("playing");
  }

  async runRealities(): Promise<void> {
    if (!this.body || (this.phase !== "verdict" && this.phase !== "realities")) return;
    this.setPhase("realities");
    const dist = this.card.querySelector<HTMLDivElement>(".vd-dist")!;
    const runBtn = this.card.querySelector<HTMLButtonElement>(".vd-run");
    if (runBtn) runBtn.disabled = true;
    dist.replaceChildren();
    const prog = el("div", "vd-progress");
    const fill = el("div", "vd-progress-fill");
    prog.appendChild(fill);
    const counter = el("span", "vd-counter", "Simulating 0 realities");
    dist.append(counter, prog);
    dist.classList.add("show");
    const started = performance.now();
    const expectMs = 6500;
    const tick = window.setInterval(() => {
      const f = 1 - Math.exp(-(performance.now() - started) / expectMs * 1.6);
      fill.style.width = `${(f * 100).toFixed(1)}%`;
      counter.textContent = `Simulating ${Math.round(f * 1000).toLocaleString("en")} realities`;
    }, 60);
    let d: RolloutDistribution;
    try {
      d = await this.api.distribution({ ...this.body, n: 1000 });
    } finally {
      window.clearInterval(tick);
    }
    this.renderDistribution(dist, d);
  }

  private renderDistribution(dist: HTMLDivElement, d: RolloutDistribution): void {
    dist.replaceChildren();
    const stats = el("div", "vd-dstats");
    const stat = (k: string, v: string, sub?: string) => {
      const s = el("div", "vd-dstat");
      s.append(el("span", "vd-dv", v), el("span", "vd-dk", k));
      if (sub) s.append(el("span", "vd-dsub", sub));
      stats.appendChild(s);
    };
    const shot = d.resolved.action_type === "shot";
    stat(shot ? "Scored" : "Action worked", pct(d.p_primary_success), `95% CI ${pct(d.p_primary_success_ci95.lo)}–${pct(d.p_primary_success_ci95.hi)}`);
    stat("Led to a goal", `${(d.p_goal * 100).toFixed(1)}%`);
    stat("Model said", pct(d.model_p_success));
    stat("Realities", d.n.toLocaleString("en"), `${(d.elapsed_ms / 1000).toFixed(1)} s`);
    const body = el("div", "vd-dbody");
    const bars = el("div", "vd-bars");
    for (const b of distributionBars(d)) {
      const row = el("div", `vd-bar ${b.label}`);
      const track = el("div", "vd-bar-track");
      const fillEl = el("div", "vd-bar-fill");
      fillEl.style.width = `${Math.max(1.5, b.fraction * 100).toFixed(1)}%`;
      track.appendChild(fillEl);
      row.append(el("span", "vd-bar-k", b.text), track, el("span", "vd-bar-v", `${b.count}`));
      bars.appendChild(row);
    }
    const xs = d.samples.map((q) => q.end_xy[0]);
    const ys = d.samples.map((q) => q.end_xy[1]);
    const spread = xs.length ? Math.max(...xs) - Math.min(...xs) + Math.max(...ys) - Math.min(...ys) : 0;
    body.append(bars);
    if (spread > 4) body.append(this.samplePlot(d));
    else body.classList.add("bars-only");
    dist.append(stats, body);
    dist.dataset.n = String(d.n);
    dist.dataset.pSuccess = String(d.p_primary_success);
  }

  /** Where the ball ended in each sampled reality: on a mini pitch (passes) or the goal mouth (shots). */
  private samplePlot(d: RolloutDistribution): SVGSVGElement {
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.classList.add("vd-plot");
    const shot = d.resolved.action_type === "shot";
    const flip = this.goalX() < 0 ? -1 : 1;
    if (shot) {
      svg.setAttribute("viewBox", "-6 -0.6 12 4");
      const frame = document.createElementNS(NS, "path");
      frame.setAttribute("d", "M -3.66 3 L -3.66 0.56 L 3.66 0.56 L 3.66 3");
      frame.setAttribute("class", "vd-frame");
      svg.appendChild(frame);
    } else {
      svg.setAttribute("viewBox", "-54 -36 108 72");
      const p = document.createElementNS(NS, "rect");
      p.setAttribute("x", "-52.5");
      p.setAttribute("y", "-34");
      p.setAttribute("width", "105");
      p.setAttribute("height", "68");
      p.setAttribute("class", "vd-pitch");
      const half = document.createElementNS(NS, "line");
      half.setAttribute("x1", "0");
      half.setAttribute("x2", "0");
      half.setAttribute("y1", "-34");
      half.setAttribute("y2", "34");
      half.setAttribute("class", "vd-pitch");
      svg.append(p, half);
    }
    for (const s of d.samples) {
      const c = document.createElementNS(NS, "circle");
      const [a, b] = s.end_xy;
      if (shot) {
        // goal-mouth (y, z), seen from the pitch: the shooter's left (+y when attacking +x) is screen left
        c.setAttribute("cx", (-a * flip).toFixed(2));
        c.setAttribute("cy", (3 - Math.min(3.5, b)).toFixed(2));
        c.setAttribute("r", "0.12");
      } else {
        // attacking to the right on the mini pitch
        c.setAttribute("cx", (a * flip).toFixed(2));
        c.setAttribute("cy", (-b * flip).toFixed(2));
        c.setAttribute("r", "1.1");
      }
      const onTarget = s.label === "shot_saved" || s.label === "woodwork";
      c.setAttribute("class", `vd-dot ${s.label === "goal" ? "goal" : s.primary_success || (shot && onTarget) ? "ok" : "bad"}`);
      svg.appendChild(c);
    }
    return svg;
  }

  private teardown(): void {
    this.card.classList.remove("show");
    this.panel.classList.remove("show");
    this.banner.classList.remove("show");
    this.rings.replaceChildren();
    this.arc.set(null);
    this.pc.show(false);
    this.ghosts.show(false);
    this.desatTarget = 0;
    this.rollout = null;
    this.result = null;
  }

  backToLive(): void {
    const m = this.moment;
    this.teardown();
    this.source.backToLive();
    this.hud.hideCard();
    this.app.ball.reset();
    this.app.setCamera("broadcast", 1.0);
    this.hud.tagsEnabled = true;
    if (m) this.source.live.seek(Math.max(0, m.t - 4));
    this.source.live.play();
    this.moment = null;
    this.setPhase("idle");
  }
}
