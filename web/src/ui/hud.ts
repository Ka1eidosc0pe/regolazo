// Broadcast HUD: scorebug and clock, the playback timeline with Rewind markers, name tags with speed badges,
// and pass / shot cards. DOM over the canvas, driven every frame from the rendered state.
import type { App } from "../app.ts";
import { crestCanvas } from "../brand/crest.ts";
import type { MatchEvent, DecisionMoment, PlayerMeta, TeamMeta } from "../types/generated/contracts.ts";
import type { MatchMeta } from "../types.ts";
import { eventCard, formatClock, kmh, markerFraction, pct, speedBadge, type PassCard, type ShotCard } from "./logic.ts";
import { el, projectWorld } from "./project.ts";

export interface HudCallbacks {
  onMarker(m: DecisionMoment): void;
  onRewindHere(): void;
  onSeek(t: number): void;
  onTogglePlay(): void;
}

const CARD_HOLD_S = 4.5;

export class Hud {
  readonly root: HTMLDivElement;
  private readonly app: App;
  private readonly meta: MatchMeta;
  private readonly players = new Map<string, PlayerMeta>();
  private readonly scoreHome: HTMLSpanElement;
  private readonly scoreAway: HTMLSpanElement;
  private readonly clockEl: HTMLSpanElement;
  private readonly periodEl: HTMLSpanElement;
  private readonly bar: HTMLDivElement;
  private readonly progress: HTMLDivElement;
  private readonly markers: HTMLDivElement;
  private readonly playBtn: HTMLButtonElement;
  readonly rewindBtn: HTMLButtonElement;
  private readonly tagLayer: HTMLDivElement;
  private readonly tags = new Map<string, HTMLDivElement>();
  private readonly cardEl: HTMLDivElement;
  private cardUntil = -1;
  private readonly top = new Map<string, number>();
  private secondHalfStart: number | null = null;
  durationS = 0;
  /** Live tags and cards are hidden while the Rewind UI owns the screen. */
  tagsEnabled = true;
  private moments: DecisionMoment[] = [];
  private lastCardT = -1;
  private lastPeriod = 0;

  constructor(overlay: HTMLElement, app: App, meta: MatchMeta, cb: HudCallbacks) {
    this.app = app;
    this.meta = meta;
    for (const p of [...meta.home.players, ...meta.away.players]) this.players.set(p.id, p);
    this.root = el("div", "hud");
    overlay.appendChild(this.root);

    // scorebug
    const bug = el("div", "scorebug");
    bug.dataset.testid = "scorebug";
    const team = (t: TeamMeta) => {
      const box = el("div", "sb-team");
      box.style.setProperty("--club", t.kit.primary);
      const crest = crestCanvas(t.crest, 64);
      crest.className = "sb-crest";
      box.append(crest, el("span", "sb-code", t.short_name.slice(0, 3).toUpperCase()));
      return box;
    };
    this.scoreHome = el("span", "sb-score", "0");
    this.scoreAway = el("span", "sb-score", "0");
    const score = el("div", "sb-scores");
    score.append(this.scoreHome, el("span", "sb-dash", "-"), this.scoreAway);
    this.clockEl = el("span", "sb-clock", "00:00");
    this.periodEl = el("span", "sb-period", "1H");
    const clock = el("div", "sb-clockbox");
    clock.append(this.clockEl, this.periodEl);
    bug.append(team(meta.home), score, team(meta.away), clock);
    this.root.appendChild(bug);

    // tags + cards
    this.tagLayer = el("div", "tag-layer");
    this.root.appendChild(this.tagLayer);
    this.cardEl = el("div", "stat-card");
    this.cardEl.dataset.testid = "stat-card";
    this.root.appendChild(this.cardEl);

    // timeline
    const tl = el("div", "timeline");
    tl.dataset.testid = "timeline";
    this.playBtn = el("button", "tl-play", "Pause");
    this.playBtn.dataset.testid = "play";
    this.playBtn.addEventListener("click", () => cb.onTogglePlay());
    this.bar = el("div", "tl-bar");
    this.progress = el("div", "tl-progress");
    this.markers = el("div", "tl-markers");
    this.bar.append(this.progress, this.markers);
    this.bar.addEventListener("click", (e) => {
      if ((e.target as HTMLElement).closest(".tl-marker")) return;
      const r = this.bar.getBoundingClientRect();
      if (this.durationS > 0) cb.onSeek(((e.clientX - r.left) / r.width) * this.durationS);
    });
    this.rewindBtn = el("button", "tl-rewind", "Rewind this moment");
    this.rewindBtn.dataset.testid = "rewind-here";
    this.rewindBtn.addEventListener("click", () => cb.onRewindHere());
    tl.append(this.playBtn, this.bar, this.rewindBtn);
    this.root.appendChild(tl);
    this.cb = cb;
  }

  private readonly cb: HudCallbacks;

  setMoments(moments: DecisionMoment[]): void {
    this.moments = moments;
    this.renderMarkers();
  }

  setSecondHalfStart(t: number | null): void {
    this.secondHalfStart = t;
  }

  private renderMarkers(): void {
    this.markers.replaceChildren();
    for (const m of this.moments) {
      const b = el("button", "tl-marker");
      b.dataset.testid = `marker-${m.id}`;
      b.dataset.moment = m.id;
      b.style.left = `${markerFraction(m.t, this.durationS) * 100}%`;
      b.title = m.hook ?? m.moment_type;
      const tip = el("span", "tl-tip", m.hook ?? m.moment_type.replace(/_/g, " "));
      b.appendChild(tip);
      b.addEventListener("click", (e) => {
        e.stopPropagation();
        this.cb.onMarker(m);
      });
      this.markers.appendChild(b);
    }
  }

  setDuration(d: number): void {
    if (d > 0 && Math.abs(d - this.durationS) > 0.5) {
      this.durationS = d;
      this.renderMarkers();
    }
  }

  name(id: string | null | undefined): string {
    if (!id) return "";
    return this.players.get(id)?.display_name ?? id;
  }

  /** Per-frame update from the rendered state. `liveT` drives the timeline (the live clock, also during a Rewind). */
  update(liveT: number, playing: boolean, live: boolean): void {
    const s = this.app.state;
    if (s) {
      this.scoreHome.textContent = String(s.base.score.home);
      this.scoreAway.textContent = String(s.base.score.away);
      if (s.base.period >= 2 && this.lastPeriod === 1 && this.secondHalfStart === null && live) this.secondHalfStart = s.t;
      this.lastPeriod = s.base.period;
      this.clockEl.textContent = formatClock(s.t, s.base.period, this.secondHalfStart);
      this.periodEl.textContent = s.base.period >= 2 ? "2H" : "1H";
    }
    this.progress.style.width = `${markerFraction(liveT, this.durationS) * 100}%`;
    this.playBtn.textContent = playing ? "Pause" : "Play";
    this.playBtn.classList.toggle("paused", !playing);
    this.rewindBtn.classList.toggle("show", live && !playing);
    this.updateTags();
    if (s && s.t > this.cardUntil + 30) this.cardUntil = -1;
    if (s && this.cardUntil >= 0 && (s.t > this.cardUntil || s.t < this.lastCardT - 1)) this.hideCard();
  }

  private updateTags(): void {
    const s = this.app.state;
    const want = new Map<string, string>();
    if (s && this.tagsEnabled) {
      const carrier = s.base.possession.player;
      // speed badges: the two fastest players near the ball, plus the carrier
      const bx = s.ball.pos[0];
      const by = s.ball.pos[1];
      const near = s.players
        .filter((p) => p.onPitch && Math.hypot(p.x - bx, p.y - by) < 30)
        .sort((a, b) => b.speed - a.speed);
      let badges = 0;
      for (const p of s.players) {
        const prev = this.top.get(p.id) ?? 0;
        if (p.speed <= 12 && p.speed > prev) this.top.set(p.id, p.speed);
      }
      for (const p of near) {
        const badge = speedBadge(p.speed, this.top.get(p.id) ?? 0);
        if (!badge || badges >= 2) continue;
        badges++;
        const label = badge.kind === "top_speed" ? `Top speed ${badge.kmh.toFixed(1)}` : `${badge.kmh.toFixed(1)} km/h`;
        want.set(p.id, `${badge.kind}|${label}`);
      }
      if (carrier && !want.has(carrier)) {
        const p = s.players.find((q) => q.id === carrier);
        want.set(carrier, p && p.speed > 5.5 ? `run|${kmh(p.speed).toFixed(1)} km/h` : "none|");
      }
    }
    for (const [id, tag] of this.tags) {
      if (!want.has(id)) {
        tag.remove();
        this.tags.delete(id);
      }
    }
    if (!s) return;
    for (const [id, spec] of want) {
      const p = s.players.find((q) => q.id === id);
      if (!p) continue;
      let tag = this.tags.get(id);
      const meta = this.players.get(id);
      if (!tag) {
        tag = el("div", "name-tag");
        tag.dataset.testid = "name-tag";
        tag.dataset.player = id;
        const team = meta?.team === "away" ? this.meta.away : this.meta.home;
        tag.style.setProperty("--club", team.kit.primary);
        const nm = el("span", "nt-name");
        nm.append(el("b", "nt-num", String(meta?.shirt_number ?? "")), document.createTextNode(meta?.display_name ?? id));
        tag.append(nm, el("span", "nt-badge"));
        this.tagLayer.appendChild(tag);
        this.tags.set(id, tag);
      }
      const [kind, label] = spec.split("|") as [string, string];
      const badge = tag.querySelector<HTMLSpanElement>(".nt-badge")!;
      badge.textContent = label;
      badge.className = `nt-badge ${kind}`;
      tag.classList.toggle("carrier", id === s.base.possession.player);
      const sp = projectWorld(this.app, p.x, p.y, 2.25);
      tag.style.transform = `translate(${sp.x.toFixed(1)}px, ${sp.y.toFixed(1)}px) translate(-50%, -100%)`;
      tag.style.opacity = sp.visible ? "1" : "0";
    }
  }

  /** Show a pass or shot card for a live event. */
  onEvent(e: MatchEvent): void {
    if (!this.tagsEnabled) return;
    const card = eventCard(e);
    if (card) this.showCard(card);
  }

  showCard(card: PassCard | ShotCard): void {
    this.cardEl.replaceChildren();
    const team = this.players.get(card.kind === "pass" ? card.passer : card.shooter)?.team === "away" ? this.meta.away : this.meta.home;
    this.cardEl.style.setProperty("--club", team.kit.primary);
    const head = el("div", "sc-head");
    const rows = el("div", "sc-rows");
    const row = (k: string, v: string) => {
      const r = el("div", "sc-row");
      r.append(el("span", "sc-k", k), el("span", "sc-v", v));
      rows.appendChild(r);
    };
    if (card.kind === "pass") {
      head.append(el("span", "sc-kind", "Pass"), el("span", "sc-who", `${this.name(card.passer)} → ${this.name(card.recipient)}`));
      row("Distance", `${card.distanceM.toFixed(0)} m`);
      if (card.xpass !== null) row("Difficulty", `${card.difficulty} · ${pct(card.xpass)} xPass`);
      if (card.technique) row("Technique", card.technique.replace(/_/g, " "));
    } else {
      head.append(el("span", "sc-kind shot", "Shot"), el("span", "sc-who", this.name(card.shooter)));
      if (card.xg !== null) row("xG", card.xg.toFixed(2));
      if (card.speedKmh !== null) row("Shot speed", `${card.speedKmh.toFixed(0)} km/h`);
      row("Result", card.outcome.replace(/_/g, " "));
    }
    this.cardEl.append(head, rows);
    this.cardEl.dataset.kind = card.kind;
    this.cardEl.classList.add("show");
    this.cardUntil = card.t + CARD_HOLD_S;
    this.lastCardT = card.t;
  }

  hideCard(): void {
    this.cardEl.classList.remove("show");
    this.cardUntil = -1;
  }

  clearTags(): void {
    for (const t of this.tags.values()) t.remove();
    this.tags.clear();
  }
}
