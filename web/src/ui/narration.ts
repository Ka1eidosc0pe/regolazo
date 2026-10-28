// Narration captions (NarrativeMsg from the stream, with optional TTS audio), the mode / language pickers that send
// ProfileCmd, and the Spotter accuracy panel (shown only when /api/files/spotter/accuracy.json exists).
import type { AccuracyPanelPayload, Narrative, ViewerProfile } from "../types/generated/contracts.ts";
import type { ClientMessage, ServerMessage } from "../types.ts";
import type { Api } from "./api.ts";
import { el } from "./project.ts";
import type { SwitchSource } from "./switchSource.ts";

const LANGS = ["en", "es", "fr"] as const;
const MODES = ["casual", "analyst"] as const;

export function captionHoldS(text: string): number {
  return Math.min(9, Math.max(3.5, text.split(/\s+/).length * 0.38));
}

export class Narration {
  private readonly caption: HTMLDivElement;
  private readonly captionText: HTMLDivElement;
  private readonly captionKind: HTMLSpanElement;
  private profile: ViewerProfile | null = null;
  private readonly source: SwitchSource;
  private readonly api: Api;
  private hideAt = 0;
  private wallNow = 0;
  private audio: HTMLAudioElement | null = null;
  voice = true;
  readonly received: Narrative[] = [];
  private readonly picker: HTMLDivElement;
  private readonly accuracyBtn: HTMLButtonElement;
  private readonly accuracyPanel: HTMLDivElement;

  constructor(overlay: HTMLElement, source: SwitchSource, api: Api) {
    this.source = source;
    this.api = api;
    this.caption = el("div", "caption");
    this.caption.dataset.testid = "caption";
    this.captionKind = el("span", "cap-kind");
    this.captionText = el("div", "cap-text");
    this.caption.append(this.captionKind, this.captionText);
    overlay.appendChild(this.caption);

    this.picker = el("div", "profile-picker");
    this.picker.dataset.testid = "profile-picker";
    const group = <T extends string>(values: readonly T[], label: (v: T) => string, key: "mode" | "language") => {
      const g = el("div", "pp-group");
      for (const v of values) {
        const b = el("button", "pp-btn", label(v));
        b.dataset.value = v;
        b.dataset.key = key;
        b.addEventListener("click", () => this.setProfile({ [key]: v } as Partial<ViewerProfile>));
        g.appendChild(b);
      }
      return g;
    };
    this.picker.append(
      group(MODES, (m) => (m === "casual" ? "Casual" : "Analyst"), "mode"),
      group(LANGS, (l) => l.toUpperCase(), "language"),
    );
    this.accuracyBtn = el("button", "pp-btn acc-btn", "Spotter");
    this.accuracyBtn.style.display = "none";
    this.accuracyBtn.addEventListener("click", () => this.accuracyPanel.classList.toggle("show"));
    this.picker.appendChild(this.accuracyBtn);
    overlay.appendChild(this.picker);
    this.accuracyPanel = el("div", "accuracy-panel");
    this.accuracyPanel.dataset.testid = "accuracy-panel";
    overlay.appendChild(this.accuracyPanel);

    source.onMessage((msg: ServerMessage) => {
      if (msg.type === "hello") {
        this.profile = msg.profile;
        this.syncPicker();
      } else if (msg.type === "narrative") {
        this.show(msg.narrative);
      }
    });
    if (source.hello) {
      this.profile = source.hello.profile;
      this.syncPicker();
    }
    void api.accuracy().then((p) => p && this.renderAccuracy(p));
  }

  setProfile(patch: Partial<ViewerProfile>): void {
    if (!this.profile) return;
    this.profile = { ...this.profile, ...patch };
    // WsSource keeps its sender private; ProfileCmd is a plain protocol message, so send it through the same socket.
    const live = this.source.live as unknown as { send?: (m: ClientMessage) => void };
    live.send?.({ type: "profile", profile: this.profile });
    this.syncPicker();
  }

  currentProfile(): { mode: string; language: string } | null {
    return this.profile ? { mode: this.profile.mode, language: this.profile.language } : null;
  }

  get mode(): ViewerProfile["mode"] {
    return this.profile?.mode ?? "casual";
  }

  private syncPicker(): void {
    for (const b of this.picker.querySelectorAll<HTMLButtonElement>(".pp-btn[data-key]")) {
      const key = b.dataset.key as "mode" | "language";
      b.classList.toggle("on", this.profile?.[key] === b.dataset.value);
    }
  }

  show(n: Narrative): void {
    this.received.push(n);
    if (this.profile && n.language !== this.profile.language) return;
    this.captionKind.textContent = n.kind.replace(/_/g, " ");
    this.captionText.textContent = n.text;
    this.caption.classList.add("show");
    this.hideAt = this.wallNow + captionHoldS(n.text);
    if (n.audio_url && this.voice) {
      this.audio?.pause();
      this.audio = new Audio(this.api.fileUrl(n.audio_url));
      void this.audio.play().catch(() => undefined);
    }
  }

  /** Local caption (e.g. the Rewind hook) when no narrative arrives from the server. */
  say(kind: string, text: string): void {
    this.captionKind.textContent = kind;
    this.captionText.textContent = text;
    this.caption.classList.add("show");
    this.hideAt = this.wallNow + captionHoldS(text);
  }

  update(): void {
    this.wallNow = performance.now() / 1000;
    if (this.caption.classList.contains("show") && this.wallNow > this.hideAt) this.caption.classList.remove("show");
  }

  private renderAccuracy(p: AccuracyPanelPayload): void {
    this.accuracyBtn.style.display = "";
    this.accuracyPanel.replaceChildren(el("div", "acc-title", p.window === "live_window" ? "Spotter accuracy · live" : "Spotter accuracy · match so far"));
    const table = el("table", "acc-table");
    const head = el("tr");
    for (const h of ["Metric", "P", "R", "F1 / MAE", "n"]) head.appendChild(el("th", undefined, h));
    table.appendChild(head);
    const f = (v: number | null | undefined) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);
    for (const r of p.rows.slice(0, 14)) {
      const tr = el("tr");
      const name = r.qualifier ?? r.event_type ?? "metric";
      tr.append(
        el("td", undefined, name.replace(/_/g, " ")),
        el("td", undefined, r.metric === "mae" ? "" : f(r.precision)),
        el("td", undefined, r.metric === "mae" ? "" : f(r.recall)),
        el("td", undefined, r.metric === "mae" ? `${(r.mae ?? 0).toFixed(2)} ${r.unit ?? ""}` : f(r.f1)),
        el("td", undefined, String(r.support)),
      );
      table.appendChild(tr);
    }
    this.accuracyPanel.appendChild(table);
  }
}
