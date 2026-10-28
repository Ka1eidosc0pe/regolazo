// World (pitch frame, z up) -> CSS pixels over the canvas, through the director's camera.
import * as THREE from "three";
import type { App } from "../app.ts";
import { worldToThree } from "../coords.ts";

const v = new THREE.Vector3();

export interface ScreenPoint {
  x: number;
  y: number;
  visible: boolean;
}

export function projectWorld(app: App, x: number, y: number, z: number): ScreenPoint {
  const [tx, ty, tz] = worldToThree(x, y, z);
  v.set(tx, ty, tz).project(app.director.camera);
  const r = app.renderer.domElement.getBoundingClientRect();
  return {
    x: r.left + ((v.x + 1) / 2) * r.width,
    y: r.top + ((1 - v.y) / 2) * r.height,
    visible: v.z > -1 && v.z < 1 && Math.abs(v.x) < 1.15 && Math.abs(v.y) < 1.15,
  };
}

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
