// Checks on the processed athlete GLB itself (evaluated in Node): strike contact frames, dive lying frames,
// root motion stripped.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { GLTFLoader, type GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { beforeAll, describe, expect, it } from "vitest";
import { ANIMS, CONTACT_S, oneShotTime, type ClipInfo } from "../src/athletes/animMap";

const file = (p: string) => fileURLToPath(new URL(`../public/assets/${p}`, import.meta.url));
const manifest = JSON.parse(readFileSync(file("athlete_clips.json"), "utf8")) as { clips: Record<string, ClipInfo & { contact_part?: string }> };

let gltf: GLTF;
beforeAll(async () => {
  const b = readFileSync(file("athlete.glb"));
  gltf = await new GLTFLoader().parseAsync(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer, "");
});

function sampler(clipName: string) {
  const scene = gltf.scene;
  const mixer = new THREE.AnimationMixer(scene);
  const clip = gltf.animations.find((c) => c.name === clipName)!;
  const action = mixer.clipAction(clip).play();
  return {
    at(t: number, bone: string): THREE.Vector3 {
      action.time = t;
      mixer.update(0);
      scene.updateMatrixWorld(true);
      return scene.getObjectByName(bone)!.getWorldPosition(new THREE.Vector3());
    },
    stop() {
      action.stop();
      mixer.uncacheRoot(scene);
    },
  };
}

describe("strike contact frames", () => {
  for (const anim of ["pass_short", "pass_long", "shoot"] as const) {
    it(`${anim}: at contact_s the kicking toe is at ball height and sweeping forward`, () => {
      const name = ANIMS[anim].clips[0]!;
      const info = manifest.clips[name]!;
      const toe = info.contact_part === "left" ? "mixamorigLeftToeBase" : "mixamorigRightToeBase";
      const s = sampler(name);
      const c = info.contact_s!;
      const p = s.at(c, toe);
      const before = s.at(c - 1 / 30, toe);
      // the ball (radius 0.11) sits on the grass: the toe at contact is within its height band
      expect(p.y, `${name} toe height at ${c}`).toBeLessThan(0.2);
      // and moving forward (+z, the character's facing) at speed
      expect((p.z - before.z) * 30, `${name} forward toe speed`).toBeGreaterThan(2);
      // contact is the lowest point of the forward swing (toe moving forward fast) in the 0.25 s around it
      let low = Infinity;
      for (let t = c - 0.25; t <= c + 0.25; t += 1 / 60) {
        const a = s.at(t - 1 / 60, toe);
        const b = s.at(t, toe);
        if ((b.z - a.z) * 60 > 2) low = Math.min(low, b.y);
      }
      expect(p.y - low).toBeLessThan(0.03);
      s.stop();
      // and the renderer lands that frame at anim_t = CONTACT_S
      expect(oneShotTime(anim, info, CONTACT_S)).toBeCloseTo(c, 6);
    });
  }
});

describe("dives", () => {
  for (const anim of ["gk_dive_left", "gk_dive_right"] as const) {
    it(`${anim} holds on the lying frame instead of playing the clip's recovery`, () => {
      const name = ANIMS[anim].clips[0]!;
      const info = manifest.clips[name]!;
      expect(info.lie_s).toBeGreaterThan(0.5);
      const s = sampler(name);
      const lying = s.at(info.lie_s!, "mixamorigHips").y;
      expect(lying).toBeLessThan(0.45);
      // long after the dive the renderer still shows the lying frame
      expect(oneShotTime(anim, info, 10)).toBeCloseTo(info.lie_s!, 6);
      // whereas the raw clip stands back up by its end
      expect(s.at(info.duration_s - 0.05, "mixamorigHips").y).toBeGreaterThan(lying + 0.3);
      s.stop();
    });
  }
});
