// Live source: a WebSocket client for the shared/schemas/stream.py protocol.
// The first server message is `hello` (MatchMeta + clock); `frame` messages carry full SimStates at the
// playback rate; `clock` follows every play / pause / seek / speed change.
import type { ClientMessage, MatchMeta, ServerMessage } from "../types.ts";
import type { FrameSource, PlaybackClock } from "./FrameSource.ts";
import { StateBuffer, type RenderState } from "./interpolate.ts";

/** Render this far behind the newest frame so there is always a next tick to interpolate towards. */
const RENDER_DELAY_S = 0.12;
const KEEP_HISTORY_S = 20;

export class WsSource implements FrameSource {
  readonly kind = "ws" as const;
  readonly buffer = new StateBuffer();
  private ws: WebSocket | null = null;
  private listeners = new Set<(msg: ServerMessage) => void>();
  private renderT = 0;
  private serverClock: PlaybackClock = { t: 0, playing: false, speed: 1, durationS: 0 };
  private latestT = -1;
  /** Time of a seek we sent and whose `clock` answer has not arrived yet. */
  private pendingSeek: number | null = null;
  private readonly url: string;

  constructor(url: string) {
    this.url = url;
  }

  connect(): Promise<MatchMeta> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      this.ws = ws;
      let gotHello = false;
      ws.onerror = () => {
        if (!gotHello) reject(new Error(`WebSocket ${this.url} failed`));
      };
      ws.onclose = () => {
        if (!gotHello) reject(new Error(`WebSocket ${this.url} closed before hello`));
      };
      ws.onmessage = (ev: MessageEvent<string>) => {
        let msg: ServerMessage;
        try {
          msg = JSON.parse(ev.data) as ServerMessage;
        } catch {
          return;
        }
        this.handle(msg);
        if (msg.type === "hello" && !gotHello) {
          gotHello = true;
          resolve(msg.meta);
        }
      };
    });
  }

  private handle(msg: ServerMessage): void {
    switch (msg.type) {
      case "hello":
      case "clock": {
        const c = msg.clock;
        // A clock message follows every play / pause / speed change and every seek. It is a seek when it answers
        // the seek we sent, or when the server's time is far from what we are showing (a seek from elsewhere);
        // pause and speed changes keep the buffer.
        const answered = this.pendingSeek !== null && Math.abs(c.t - this.pendingSeek) < 0.25;
        const showing = this.latestT >= 0 ? this.latestT : this.renderT;
        const jumped = msg.type === "clock" && Math.abs(c.t - showing) > 1.5;
        this.serverClock = { t: c.t, playing: c.playing, speed: c.speed, durationS: c.duration_s };
        if (msg.type === "hello" || answered || jumped) {
          this.buffer.clear();
          this.latestT = -1;
          this.renderT = c.t;
          this.pendingSeek = null;
        }
        break;
      }
      case "frame": {
        this.buffer.push(msg.state);
        if (this.latestT < 0) this.renderT = msg.state.t - RENDER_DELAY_S;
        this.latestT = Math.max(this.latestT, msg.state.t);
        this.buffer.trimBefore(this.latestT - KEEP_HISTORY_S);
        break;
      }
      default:
        break;
    }
    for (const l of this.listeners) l(msg);
  }

  advance(dtReal: number): number {
    if (this.latestT < 0 || !this.serverClock.playing) return this.renderT; // paused: hold the frame
    this.renderT += dtReal * this.serverClock.speed;
    const target = this.latestT - RENDER_DELAY_S * Math.max(1, this.serverClock.speed);
    const err = target - this.renderT;
    if (Math.abs(err) > 1.5) this.renderT = target;
    else this.renderT += err * Math.min(1, dtReal * 3);
    return this.renderT;
  }

  sample(t: number): RenderState | null {
    return this.buffer.sample(t);
  }

  clock(): PlaybackClock {
    return { ...this.serverClock, t: this.renderT };
  }

  private send(msg: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  play(): void {
    this.send({ type: "play" });
  }

  pause(): void {
    this.send({ type: "pause" });
  }

  seek(t: number): void {
    this.pendingSeek = t;
    this.send({ type: "seek", t });
  }

  setSpeed(speed: number): void {
    this.send({ type: "speed", speed });
  }

  onMessage(cb: (msg: ServerMessage) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  dispose(): void {
    this.ws?.close();
    this.ws = null;
    this.listeners.clear();
  }
}
