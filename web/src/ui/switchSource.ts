// A FrameSource that delegates to a swappable inner source: the live stream, a frozen Rewind frame or a rollout.
// Messages (events, narrative, moments) always come from the live source.
import type { MatchMeta, ServerMessage } from "../types.ts";
import type { FrameSource, PlaybackClock } from "../source/FrameSource.ts";
import type { RenderState, StateBuffer } from "../source/interpolate.ts";

export class SwitchSource implements FrameSource {
  readonly live: FrameSource;
  private inner: FrameSource;
  /** The live stream's `hello` (session profile), kept for UI layers mounted after connect. */
  hello: Extract<ServerMessage, { type: "hello" }> | null = null;

  constructor(live: FrameSource) {
    this.live = live;
    this.inner = live;
    live.onMessage((m) => {
      if (m.type === "hello") this.hello = m;
    });
  }

  get kind(): "clip" | "ws" {
    return this.inner.kind;
  }

  get buffer(): StateBuffer {
    return this.inner.buffer;
  }

  get active(): FrameSource {
    return this.inner;
  }

  get isLive(): boolean {
    return this.inner === this.live;
  }

  use(src: FrameSource): void {
    if (this.inner !== this.live && this.inner !== src) this.inner.dispose();
    this.inner = src;
  }

  backToLive(): void {
    this.use(this.live);
  }

  connect(): Promise<MatchMeta> {
    return this.live.connect();
  }

  advance(dtReal: number): number {
    return this.inner.advance(dtReal);
  }

  sample(t: number): RenderState | null {
    return this.inner.sample(t);
  }

  clock(): PlaybackClock {
    return this.inner.clock();
  }

  play(): void {
    this.inner.play();
  }

  pause(): void {
    this.inner.pause();
  }

  seek(t: number): void {
    this.inner.seek(t);
  }

  setSpeed(speed: number): void {
    this.inner.setSpeed(speed);
  }

  onMessage(cb: (msg: ServerMessage) => void): () => void {
    return this.live.onMessage(cb);
  }

  dispose(): void {
    this.inner.dispose();
    if (this.inner !== this.live) this.live.dispose();
  }
}
