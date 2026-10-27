// Local clip playback: a JSON file `{ meta, frames }` (see scripts/make-dev-clip.ts) played with a local clock.
import type { Clip, MatchMeta } from "../types.ts";
import type { FrameSource, PlaybackClock } from "./FrameSource.ts";
import { StateBuffer, type RenderState } from "./interpolate.ts";

export class ClipSource implements FrameSource {
  readonly kind = "clip" as const;
  readonly buffer = new StateBuffer();
  private t = 0;
  private playing = true;
  private speed = 1;
  private t0 = 0;
  private t1 = 0;
  private clip: Clip | null = null;
  loop: boolean;
  private readonly input: string | Clip;

  constructor(input: string | Clip, opts: { loop?: boolean } = {}) {
    this.input = input;
    this.loop = opts.loop ?? true;
  }

  async connect(): Promise<MatchMeta> {
    const clip = typeof this.input === "string" ? await fetchClip(this.input) : this.input;
    this.clip = clip;
    this.buffer.setAll(clip.frames);
    this.t0 = this.buffer.first?.t ?? 0;
    this.t1 = this.buffer.last?.t ?? 0;
    this.t = this.t0;
    return clip.meta;
  }

  get frames(): Clip["frames"] {
    return this.clip?.frames ?? [];
  }

  get startT(): number {
    return this.t0;
  }

  get endT(): number {
    return this.t1;
  }

  advance(dtReal: number): number {
    if (this.playing) {
      this.t += dtReal * this.speed;
      if (this.t > this.t1) this.t = this.loop ? this.t0 + ((this.t - this.t0) % Math.max(1e-6, this.t1 - this.t0)) : this.t1;
    }
    return this.t;
  }

  sample(t: number): RenderState | null {
    return this.buffer.sample(t);
  }

  clock(): PlaybackClock {
    return { t: this.t, playing: this.playing, speed: this.speed, durationS: this.t1 - this.t0 };
  }

  play(): void {
    this.playing = true;
  }

  pause(): void {
    this.playing = false;
  }

  seek(t: number): void {
    this.t = Math.min(this.t1, Math.max(this.t0, t));
  }

  setSpeed(speed: number): void {
    this.speed = speed;
  }

  onMessage(): () => void {
    return () => undefined;
  }

  dispose(): void {
    this.buffer.clear();
  }
}

export async function fetchClip(url: string): Promise<Clip> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`clip ${url}: HTTP ${res.status}`);
  return (await res.json()) as Clip;
}
