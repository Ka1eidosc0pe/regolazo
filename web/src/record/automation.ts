// `window.regolazo`: the automation surface used by Playwright (reference screenshots, visual tests, recording).
import type * as THREE_NS from "three";
import type { App } from "../app.ts";
import type { CameraMode } from "../camera/director.ts";
import { worldToThree, type V3 } from "../coords.ts";
import { Recorder } from "./recorder.ts";

export interface Automation {
  ready: Promise<void>;
  app: App;
  /** Deterministic recording (frames + FrameLabels); see scripts/record.ts. */
  recorder: Recorder;
  /** Settle dynamics and render the frame at match time t (paused afterwards). */
  jumpTo(t: number, warm?: number): void;
  setCamera(mode: CameraMode, replayTarget?: string | null): void;
  /** Free camera in world coordinates (pitch frame, z up). */
  setFreePose(pos: V3, look: V3, fov: number): void;
  setClean(clean: boolean): void;
  /** Performance experiments: hide named objects, toggle shadows / bloom. */
  debug(flags: { hide?: string[]; shadows?: boolean; bloom?: boolean }): void;
  info(): { t: number; start: number; end: number; fps: number; mode: CameraMode; width: number; height: number; renderer: string };
}

declare global {
  interface Window {
    regolazo?: Automation;
  }
}

export function installAutomation(app: App, ready: Promise<void>, THREE: typeof THREE_NS): void {
  const api: Automation = {
    ready,
    app,
    recorder: new Recorder(app),
    jumpTo(t, warm = 2.5) {
      app.stop();
      app.source.pause();
      app.jumpTo(t, warm);
    },
    setCamera(mode, replayTarget = null) {
      app.director.replayTarget = replayTarget;
      app.setCamera(mode, 0);
    },
    setFreePose(pos, look, fov) {
      app.director.setFreePose(new THREE.Vector3(...worldToThree(...pos)), new THREE.Vector3(...worldToThree(...look)), fov, 0);
    },
    debug(flags) {
      for (const name of flags.hide ?? []) {
        const o = app.scene.getObjectByName(name);
        if (o) o.visible = false;
      }
      if (flags.shadows !== undefined) app.renderer.shadowMap.enabled = flags.shadows;
      if (flags.bloom !== undefined) app.post.bloom.enabled = flags.bloom;
    },
    setClean(clean) {
      document.body.dataset.clean = clean ? "1" : "0";
    },
    info() {
      const gl = app.renderer.getContext();
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      const buf = app.source.buffer;
      return {
        t: app.source.clock().t,
        start: buf.first?.t ?? 0,
        end: buf.last?.t ?? 0,
        fps: app.fps,
        mode: app.director.mode,
        width: app.size.width,
        height: app.size.height,
        renderer: ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "unknown",
      };
    },
  };
  window.regolazo = api;
}
