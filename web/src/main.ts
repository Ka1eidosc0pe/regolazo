// Entry point. URL parameters:
//   ?clip=/dev/clip_demo.json   local clip (default)        ?ws=ws://host:8000/ws   live server instead
//   ?cam=broadcast|tactical|replay                          ?t=<match seconds>      start (or freeze) time
//   ?paused=1  freeze                                       ?quality=low            1k shadows, no MSAA
//   ?w=1920&h=1080  fixed render size (recording / screenshots)
import "./style.css";
import * as THREE from "three";
import { App } from "./app.ts";
import type { CameraMode } from "./camera/director.ts";
import { installAutomation } from "./record/automation.ts";
import { ClipSource } from "./source/ClipSource.ts";
import type { FrameSource } from "./source/FrameSource.ts";
import { WsSource } from "./source/WsSource.ts";

const params = new URLSearchParams(location.search);
const root = document.querySelector<HTMLDivElement>("#app")!;
root.innerHTML = `
  <div id="stage-root"></div>
  <div id="overlay-root" aria-live="polite"></div>
  <div class="brand-bug"><span data-testid="title">ReGolazo</span></div>`;

const stageRoot = root.querySelector<HTMLDivElement>("#stage-root")!;
const wsUrl = params.get("ws");
const source: FrameSource = wsUrl ? new WsSource(wsUrl) : new ClipSource(params.get("clip") ?? "/dev/clip_demo.json");
const w = params.get("w");
const h = params.get("h");
const app = new App(stageRoot, source, {
  width: w ? Number(w) : undefined,
  height: h ? Number(h) : undefined,
  quality: params.get("quality") === "low" ? "low" : "high",
  msaa: params.get("msaa") ? Number(params.get("msaa")) : undefined,
  shadowMap: params.get("shadow") ? Number(params.get("shadow")) : undefined,
  record: params.get("record") === "1",
});

const ready = app.init().then(() => {
  const cam = params.get("cam") as CameraMode | null;
  if (cam) app.setCamera(cam, 0);
  const t = params.get("t");
  if (t) source.seek(Number(t));
  if (params.get("paused") === "1") source.pause();
  if (params.get("record") !== "1") {
    app.start();
    installKeys();
  }
  document.body.dataset.ready = "1";
});
installAutomation(app, ready, THREE);
ready.catch((err: unknown) => {
  console.error(err);
  root.insertAdjacentHTML("beforeend", `<pre class="fatal">${String(err)}</pre>`);
});

/** Live-view keys: 1 broadcast, 2 tactical, 3 replay (follows the ball carrier), space play/pause, arrows seek. */
function installKeys(): void {
  window.addEventListener("keydown", (e) => {
    const c = source.clock();
    switch (e.key) {
      case "1":
        app.setCamera("broadcast");
        break;
      case "2":
        app.setCamera("tactical");
        break;
      case "3":
        app.director.replayTarget = app.state?.base.possession.player ?? app.state?.base.ball.last_touch ?? null;
        app.setCamera("replay");
        break;
      case " ":
        if (c.playing) source.pause();
        else source.play();
        e.preventDefault();
        break;
      case "ArrowLeft":
        source.seek(c.t - 5);
        break;
      case "ArrowRight":
        source.seek(c.t + 5);
        break;
      default:
        break;
    }
  });
}
