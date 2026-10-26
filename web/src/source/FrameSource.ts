// Where SimState frames come from: the live server (WebSocket) or a local clip (development, recording).
import type { MatchMeta, ServerMessage } from "../types.ts";
import type { RenderState, StateBuffer } from "./interpolate.ts";

export interface PlaybackClock {
  t: number;
  playing: boolean;
  speed: number;
  durationS: number;
}

export interface FrameSource {
  readonly kind: "clip" | "ws";
  readonly buffer: StateBuffer;
  /** Connects or loads, resolving with the match meta. */
  connect(): Promise<MatchMeta>;
  /** Advances the playback clock by `dtReal` wall-clock seconds; returns the match time to render. */
  advance(dtReal: number): number;
  /** Interpolated render state at match time `t`. */
  sample(t: number): RenderState | null;
  clock(): PlaybackClock;
  play(): void;
  pause(): void;
  seek(t: number): void;
  setSpeed(speed: number): void;
  /** Subscribe to every server message (events, cues, narrative, moments) for later UI layers. */
  onMessage(cb: (msg: ServerMessage) => void): () => void;
  dispose(): void;
}
