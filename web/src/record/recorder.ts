// Deterministic frame-by-frame recording: fixed time step, the broadcast camera settled before the first frame,
// image read straight from the canvas after each render, labels from the same scene state.
import type { App } from "../app.ts";
import type { CameraMode } from "../camera/director.ts";
import type { FrameLabels } from "../types.ts";
import { Labeler, type LabelDebug } from "./labels.ts";

export interface RecordOptions {
  fps?: number;
  start?: number;
  end?: number;
  camera?: CameraMode;
  warm?: number;
  format?: "jpeg" | "png";
  quality?: number;
}

export interface RecordedFrame {
  labels: FrameLabels;
  image: string;
  debug?: LabelDebug;
}

export class Recorder {
  private readonly app: App;
  private readonly labeler: Labeler;
  private opts: Required<RecordOptions> | null = null;
  private next = 0;

  constructor(app: App) {
    this.app = app;
    this.labeler = new Labeler(app);
  }

  /** Prepare a recording; returns the number of frames. */
  start(o: RecordOptions = {}): { frames: number; start: number; end: number; fps: number } {
    const app = this.app;
    const buf = app.source.buffer;
    const t0 = buf.first?.t ?? 0;
    const t1 = buf.last?.t ?? 0;
    const fps = o.fps ?? 25;
    const start = Math.max(t0, o.start ?? t0);
    const end = Math.min(t1, o.end ?? t1);
    this.opts = { fps, start, end, camera: o.camera ?? "broadcast", warm: o.warm ?? 2.0, format: o.format ?? "jpeg", quality: o.quality ?? 0.92 };
    app.stop();
    app.source.pause();
    app.director.replayTarget = null;
    app.setCamera(this.opts.camera, 0);
    // settle the camera on the opening state (the warm-up never runs before the clip start)
    const warmStart = Math.max(t0, start - this.opts.warm);
    app.jumpTo(start, start - warmStart, fps);
    this.next = 0;
    return { frames: Math.floor((end - start) * fps + 1e-6) + 1, start, end, fps };
  }

  /** Render frame i (must be called in order 0, 1, 2, ...). */
  frame(i: number, withDebug = false): RecordedFrame {
    const o = this.opts;
    if (!o) throw new Error("Recorder.start() first");
    if (i !== this.next) throw new Error(`frames must be recorded in order: expected ${this.next}, got ${i}`);
    const app = this.app;
    const t = o.start + i / o.fps;
    if (i > 0) app.step(t, 1 / o.fps);
    app.render();
    const canvas = app.renderer.domElement;
    const ext = o.format === "png" ? "png" : "jpg";
    const image = canvas.toDataURL(o.format === "png" ? "image/png" : "image/jpeg", o.quality);
    const name = `frames/${String(i).padStart(6, "0")}.${ext}`;
    const { labels, debug } = this.labeler.label(i, name, withDebug);
    this.next = i + 1;
    return { labels, image, debug };
  }
}
