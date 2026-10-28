// REST client for the live engine (server/app.py). The base URL is the server's HTTP origin.
import type {
  ActionIntent,
  AccuracyPanelPayload,
  DecisionMoment,
  ModelOutputs,
  ResolvedAction,
  RolloutDistribution,
  RolloutResult,
  SimState,
} from "../types/generated/contracts.ts";

export interface RolloutBody {
  snapshot_ref: string;
  intent: ActionIntent;
  horizon_s?: number;
  n?: number;
  seed?: number;
  return_frames?: boolean;
  frame_stride?: number;
}

export class Api {
  readonly base: string;

  constructor(base: string) {
    this.base = base.replace(/\/$/, "");
  }

  /** HTTP origin for a WebSocket URL (ws://host:8000/ws -> http://host:8000). */
  static fromWs(wsUrl: string): Api {
    const u = new URL(wsUrl, location.href);
    u.protocol = u.protocol === "wss:" ? "https:" : "http:";
    return new Api(u.origin);
  }

  private async get<T>(path: string): Promise<T> {
    const res = await fetch(this.base + path);
    if (!res.ok) throw new Error(`GET ${path}: HTTP ${res.status}`);
    return (await res.json()) as T;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(this.base + path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`POST ${path}: HTTP ${res.status} ${await res.text()}`);
    return (await res.json()) as T;
  }

  moments(): Promise<DecisionMoment[]> {
    return this.get("/api/moments");
  }

  state(t: number): Promise<SimState> {
    return this.get(`/api/state?t=${t.toFixed(3)}`);
  }

  options(snapshotRef: string): Promise<ModelOutputs> {
    return this.get(`/api/options?snapshot_ref=${encodeURIComponent(snapshotRef)}`);
  }

  preview(snapshotRef: string, intent: ActionIntent): Promise<ResolvedAction> {
    return this.post("/api/preview", { snapshot_ref: snapshotRef, intent });
  }

  rollout(body: RolloutBody): Promise<RolloutResult> {
    return this.post("/api/rollout", { n: 1, return_frames: true, ...body });
  }

  distribution(body: RolloutBody): Promise<RolloutDistribution> {
    return this.post("/api/distribution", { n: 1000, return_frames: false, ...body });
  }

  /** Spotter accuracy (another component writes it); null when absent. */
  async accuracy(): Promise<AccuracyPanelPayload | null> {
    try {
      const res = await fetch(this.base + "/api/files/spotter/accuracy.json");
      if (!res.ok) return null;
      return (await res.json()) as AccuracyPanelPayload;
    } catch {
      return null;
    }
  }

  fileUrl(path: string): string {
    if (/^https?:\/\//.test(path)) return path;
    if (path.startsWith("/")) return this.base + path;
    return `${this.base}/api/files/${path}`;
  }
}
