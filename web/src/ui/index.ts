// The live-match UI: HUD, Rewind & Rewrite, narration and accuracy panels, and the `window.regolazo.demo` hooks.
import "./ui.css";
import type { App } from "../app.ts";
import type { DecisionMoment, MatchEvent, Narrative } from "../types/generated/contracts.ts";
import type { ServerMessage } from "../types.ts";
import { Api } from "./api.ts";
import { Hud } from "./hud.ts";
import { lastOnBall } from "./logic.ts";
import { Narration } from "./narration.ts";
import { Rewind } from "./rewind.ts";
import type { SwitchSource } from "./switchSource.ts";

export interface LiveUi {
  hud: Hud;
  rewind: Rewind;
  narration: Narration;
  moments: DecisionMoment[];
}

export async function mountUi(app: App, source: SwitchSource, api: Api): Promise<LiveUi> {
  const overlay = document.querySelector<HTMLDivElement>("#overlay-root")!;
  document.body.dataset.ui = "live";
  const recent: MatchEvent[] = [];
  let rewind: Rewind | null = null;
  const hud = new Hud(overlay, app, app.meta, {
    onMarker: (m) => void rewind?.open(m),
    onRewindHere: () => {
      const t = source.live.clock().t;
      const local = lastOnBall(recent, t, 60);
      // the stream only carries events from where playback started; ask the server for the last minute otherwise
      const found = local
        ? Promise.resolve(local)
        : fetch(`${api.base}/api/events?from=${Math.max(0, t - 60).toFixed(2)}&to=${(t + 0.05).toFixed(2)}`)
            .then((r) => (r.ok ? (r.json() as Promise<MatchEvent[]>) : []))
            .then((evs) => lastOnBall(evs, t, 60));
      void found.then((e) => {
        if (e) void rewind?.openAt(e);
        else narration.say("Rewind", "No touch of the ball in the last minute to rewind.");
      });
    },
    onSeek: (t) => {
      if (!source.isLive) return;
      source.live.seek(t);
      hud.hideCard();
    },
    onTogglePlay: () => {
      if (!source.isLive) return;
      if (source.live.clock().playing) source.live.pause();
      else source.live.play();
    },
  });
  const narration = new Narration(overlay, source, api);
  rewind = new Rewind(overlay, app, source, api, hud, narration);

  source.onMessage((msg: ServerMessage) => {
    if (msg.type === "events") {
      for (const e of msg.events) {
        if (e.rollout_id) continue;
        if (e.type === "pass" || e.type === "shot") recent.push(e);
        if (source.isLive) hud.onEvent(e);
        if (e.type === "period_start" && e.period === 2) hud.setSecondHalfStart(e.t);
      }
      if (recent.length > 400) recent.splice(0, recent.length - 400);
    } else if (msg.type === "clock") {
      hud.setDuration(msg.clock.duration_s);
    } else if (msg.type === "hello") {
      hud.setDuration(msg.clock.duration_s);
    }
  });

  const moments = await api.moments().catch(() => [] as DecisionMoment[]);
  hud.setMoments(moments);
  // second-half kick-off for the broadcast clock (the stream only sends events as playback passes them)
  void fetch(`${api.base}/api/events?from=2600&to=3600`)
    .then((r) => (r.ok ? (r.json() as Promise<MatchEvent[]>) : []))
    .then((evs) => {
      const k = evs.find((e) => e.type === "period_start" && e.period === 2);
      if (k) hud.setSecondHalfStart(k.t);
    })
    .catch(() => undefined);

  app.frameHooks.push((dt) => {
    const c = source.live.clock();
    if (c.durationS) hud.setDuration(c.durationS);
    hud.update(c.t, c.playing, source.isLive);
    rewind?.update(dt);
    narration.update();
  });

  const ui: LiveUi = { hud, rewind, narration, moments };
  installDemo(ui, source);
  return ui;
}

export interface DemoHooks {
  moments(): DecisionMoment[];
  seek(t: number): void;
  play(): void;
  pause(): void;
  openMoment(id: string): Promise<void>;
  pickOpenTeammate(): Promise<string | null>;
  pickOption(optionId: string): Promise<void>;
  setAnalyst(on: boolean): void;
  commit(): Promise<void>;
  /** Resolves when the rollout finished playing and the verdict is on screen. */
  verdict(): Promise<void>;
  runRealities(): Promise<void>;
  backToLive(): void;
  phase(): string;
  /** Feed a Narrative as if it came over the stream (tests, demos without the narration agent). */
  narrate(n: Narrative): void;
  profile(): { mode: string; language: string } | null;
}

declare global {
  interface Window {
    regolazoDemo?: DemoHooks;
  }
}

function installDemo(ui: LiveUi, source: SwitchSource): void {
  const hooks: DemoHooks = {
    moments: () => ui.moments,
    seek: (t) => source.live.seek(t),
    play: () => source.live.play(),
    pause: () => source.live.pause(),
    openMoment: async (id) => {
      const m = ui.moments.find((x) => x.id === id);
      if (!m) throw new Error(`no moment ${id}`);
      await ui.rewind.open(m);
    },
    pickOpenTeammate: () => ui.rewind.pickOpenTeammate(),
    pickOption: (id) => ui.rewind.pickOptionId(id),
    setAnalyst: (on) => ui.rewind.setAnalyst(on),
    commit: () => ui.rewind.commit(),
    verdict: () => ui.rewind.verdictShown,
    runRealities: () => ui.rewind.runRealities(),
    backToLive: () => ui.rewind.backToLive(),
    phase: () => ui.rewind.phase,
    narrate: (n) => ui.narration.show(n),
    profile: () => ui.narration.currentProfile(),
  };
  window.regolazoDemo = hooks;
}
