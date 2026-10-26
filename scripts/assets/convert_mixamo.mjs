#!/usr/bin/env node
// Convert the raw Mixamo FBX downloads (gitignored, docs/assets_mixamo.md) into one optimized in-app GLB.
//
//   node scripts/assets/convert_mixamo.mjs [--src <dir with FBX files>] [--out web/public/assets]
//
// What it does:
// - loads the Y Bot character and every animation FBX with Three.js FBXLoader (Node, no browser needed),
// - re-binds the body mesh to the real bone hierarchy (FBXLoader creates duplicate leaf bones), drops UVs and
//   welds vertices (the FBX is a triangle soup), and bakes centimetres -> metres into bones, geometry and clips,
// - keeps only rotation tracks plus the hip translation, and strips horizontal (XZ) hip root motion: in-place
//   loops are linearly detrended (sway is kept), one-shots with real displacement are pinned to their first frame.
//   Vertical (Y) hip motion is kept for dives, jumps and slides. The simulator owns player positions.
// - measures per clip: duration, extracted root displacement, natural ground speed of locomotion loops (from the
//   planted foot), and the ball-contact time of kicks/headers (peak foot/head speed),
// - writes `athlete.glb` (mesh + skeleton + all clips) and `athlete_clips.json` (the measurements) to --out.
//
// Mixamo's terms allow shipping these processed assets inside the app; the raw FBX files are never committed.

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../..");
const threeDir = join(repo, "web/node_modules/three/");
const THREE = await import(join(threeDir, "build/three.module.js"));
const { FBXLoader } = await import(join(threeDir, "examples/jsm/loaders/FBXLoader.js"));
const { GLTFExporter } = await import(join(threeDir, "examples/jsm/exporters/GLTFExporter.js"));
const { mergeVertices } = await import(join(threeDir, "examples/jsm/utils/BufferGeometryUtils.js"));

const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const SRC = resolve(arg("--src", join(repo, "../../../assets/third_party/mixamo")));
const SRC_FALLBACK = "/Users/adityaselvam/Desktop/Aage/Hackathons/PL_MSFT_InsideTheGame_2026/assets/third_party/mixamo";
const OUT = resolve(arg("--out", join(repo, "web/public/assets")));
const CM = 0.01;

// GLTFExporter needs FileReader (browser API); a minimal Node polyfill is enough for binary output.
globalThis.FileReader = class {
  readAsArrayBuffer(blob) {
    blob.arrayBuffer().then((r) => {
      this.result = r;
      this.onloadend?.();
    });
  }
  readAsDataURL(blob) {
    blob.arrayBuffer().then((r) => {
      this.result = `data:${blob.type};base64,${Buffer.from(r).toString("base64")}`;
      this.onloadend?.();
    });
  }
};

/** Clip file (without .fbx) -> how it is used. kind: loop (locomotion), oneshot, kick (contact = foot), head. */
const CLIPS = {
  Breathing_Idle: "loop",
  Jog_Forward: "loop",
  Run_forward: "loop",
  Sprint: "loop",
  Running_Backward: "loop",
  Jog_Strafe_Left: "loop",
  Jog_Strafe_Right: "loop",
  Running_Turn_180: "oneshot",
  Dribble: "loop",
  Goalkeeper_Idle: "loop",
  Soccer_Pass: "kick",
  long_pass: "kick",
  Strike_Foward: "kick",
  Goalkeeper_Drop_Kick: "kick",
  standing_Header: "head",
  jumping_Header: "head",
  Receive_ground_pass: "oneshot",
  Receive_aerial_pass: "oneshot",
  Throw_In: "oneshot",
  Soccer_Tackle: "kick",
  Slide_Tackle: "oneshot",
  Soccer_Trip: "oneshot",
  Getting_Up: "oneshot",
  Goalkeeper_Diving_Save_left: "oneshot",
  Goalkeeper_Diving_Save_Right: "oneshot",
  Goalkeeper_Body_Block_left: "oneshot",
  Goalkeeper_Body_Block_right: "oneshot",
  Goalkeeper_Catch_highshot: "oneshot",
  Goalkeeper_Catch_lowshot: "oneshot",
  Goalkeeper_Overhand_Throw: "oneshot",
  Goalkeeper_Scoop: "oneshot",
  celebration_Cheering: "oneshot",
  celebration_Fist_Pump: "oneshot",
  celebration_Victory: "oneshot",
  Disappointed: "oneshot",
};

const srcDir = (() => {
  try {
    readdirSync(SRC);
    return SRC;
  } catch {
    return SRC_FALLBACK;
  }
})();
const load = (file) => {
  const b = readFileSync(join(srcDir, file));
  return new FBXLoader().parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), "");
};

// ---------------------------------------------------------------- character
const character = load("character_ybot.fbx");
const meshes = [];
character.traverse((o) => {
  if (o.isSkinnedMesh) meshes.push(o);
});
const realHips = character.children.find((c) => c.isBone);
const realByName = new Map();
realHips.traverse((b) => {
  if (b.isBone && !realByName.has(b.name)) realByName.set(b.name, b);
});

for (const mesh of meshes) {
  const bones = mesh.skeleton.bones.map((b) => {
    const real = realByName.get(b.name);
    if (!real) throw new Error(`no real bone for ${b.name}`);
    return real;
  });
  const inverses = mesh.skeleton.boneInverses.map((m) => {
    const c = m.clone();
    c.elements[12] *= CM;
    c.elements[13] *= CM;
    c.elements[14] *= CM;
    return c;
  });
  let g = mesh.geometry;
  g.deleteAttribute("uv");
  g.scale(CM, CM, CM);
  g = mergeVertices(g, 1e-4);
  g.computeBoundingBox();
  g.computeBoundingSphere();
  mesh.geometry = g;
  mesh.bind(new THREE.Skeleton(bones, inverses), new THREE.Matrix4());
  mesh.material = new THREE.MeshStandardMaterial({ name: mesh.name, color: 0xcccccc, roughness: 0.6 });
}
// Remove FBXLoader's duplicate leaf bones and bake cm -> m into the hierarchy.
const dupes = [];
realHips.traverse((b) => {
  if (b.isBone && realByName.get(b.name) !== b) dupes.push(b);
});
for (const d of dupes) d.removeFromParent();
realHips.traverse((b) => b.position.multiplyScalar(CM));
character.name = "athlete";
character.updateMatrixWorld(true);

// ---------------------------------------------------------------- clips
const boneNames = new Set(realByName.keys());
const clips = [];
const report = {};
const mixer = new THREE.AnimationMixer(character);
const v = new THREE.Vector3();

function sampleBones(clip, names, fps = 60) {
  const action = mixer.clipAction(clip);
  action.reset().play();
  const n = Math.max(2, Math.round(clip.duration * fps) + 1);
  const out = names.map(() => []);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * clip.duration;
    mixer.setTime(t);
    character.updateMatrixWorld(true);
    names.forEach((nm, k) => out[k].push(realByName.get(nm).getWorldPosition(v).clone()));
  }
  action.stop();
  mixer.uncacheAction(clip);
  return { n, dt: clip.duration / (n - 1), out };
}

for (const [file, kind] of Object.entries(CLIPS)) {
  const src = load(`${file}.fbx`).animations[0];
  if (!src) throw new Error(`${file}: no animation`);
  const tracks = [];
  let root = { dx: 0, dz: 0, dy_max: 0 };
  for (const t of src.tracks) {
    const [node, prop] = t.name.split(".");
    if (!boneNames.has(node)) continue;
    if (prop === "quaternion") {
      tracks.push(t.clone());
    } else if (prop === "position" && node === "mixamorigHips") {
      const tr = t.clone();
      const vals = tr.values;
      const n = vals.length / 3;
      for (let i = 0; i < vals.length; i++) vals[i] *= CM;
      const x0 = vals[0];
      const z0 = vals[2];
      const dx = vals[(n - 1) * 3] - x0;
      const dz = vals[(n - 1) * 3 + 2] - z0;
      const y0 = vals[1];
      let dyMax = 0;
      for (let i = 0; i < n; i++) dyMax = Math.max(dyMax, Math.abs(vals[i * 3 + 1] - y0));
      root = { dx: +dx.toFixed(3), dz: +dz.toFixed(3), dy_max: +dyMax.toFixed(3) };
      const drift = Math.hypot(dx, dz);
      for (let i = 0; i < n; i++) {
        const f = n > 1 ? i / (n - 1) : 0;
        if (kind === "loop") {
          // remove the linear drift, keep the natural sway
          vals[i * 3] -= dx * f;
          vals[i * 3 + 2] -= dz * f;
        } else if (drift > 0.05) {
          vals[i * 3] = x0;
          vals[i * 3 + 2] = z0;
        }
      }
      tracks.push(tr);
    }
  }
  const clip = new THREE.AnimationClip(file, src.duration, tracks);
  clip.optimize();
  clips.push(clip);

  const info = { kind, duration_s: +clip.duration.toFixed(4), root_motion_m: root };
  const feet = ["mixamorigLeftToeBase", "mixamorigRightToeBase", "mixamorigHead"];
  const { n, dt, out } = sampleBones(clip, feet);
  if (kind === "loop" && file !== "Breathing_Idle" && file !== "Goalkeeper_Idle") {
    // natural ground speed: horizontal speed of a foot while it is planted (lowest 3 cm of its height range)
    const speeds = [];
    for (const path of out.slice(0, 2)) {
      const minY = Math.min(...path.map((p) => p.y));
      for (let i = 1; i < n; i++) {
        if (path[i].y < minY + 0.03 && path[i - 1].y < minY + 0.03) {
          speeds.push(Math.hypot(path[i].x - path[i - 1].x, path[i].z - path[i - 1].z) / dt);
        }
      }
    }
    speeds.sort((a, b) => a - b);
    const median = speeds.length ? speeds[Math.floor(speeds.length / 2)] : 0;
    // Sprint carries real root motion: its natural speed is the root displacement per loop
    // phase reference: the moment the left foot is furthest forward (lets the renderer sync loop phases)
    const left = out[0];
    let best = -Infinity;
    let at = 0;
    for (let i = 0; i < n; i++) {
      const fwd = file === "Running_Backward" ? -left[i].z : file === "Jog_Strafe_Left" ? left[i].x : file === "Jog_Strafe_Right" ? -left[i].x : left[i].z;
      if (fwd > best) {
        best = fwd;
        at = i * dt;
      }
    }
    info.phase0_s = +at.toFixed(3);
    info.natural_speed_mps = +(Math.hypot(root.dx, root.dz) > 0.5 ? Math.hypot(root.dx, root.dz) / clip.duration : median).toFixed(3);
  }
  if (kind === "kick" || kind === "head") {
    const paths = kind === "head" ? [out[2]] : out.slice(0, 2);
    let best = 0;
    let bestT = 0;
    let foot = "";
    paths.forEach((path, k) => {
      // ignore the last frames: some clips jump back to the start pose at the loop seam
      for (let i = 1; i < n - 4; i++) {
        const s = path[i].distanceTo(path[i - 1]) / dt;
        if (s > best) {
          best = s;
          bestT = i * dt;
          foot = kind === "head" ? "head" : k === 0 ? "left" : "right";
        }
      }
    });
    info.contact_s = +bestT.toFixed(3);
    info.contact_part = foot;
    info.contact_speed_mps = +best.toFixed(2);
  }
  report[file] = info;
}

// ---------------------------------------------------------------- export
const scene = new THREE.Scene();
scene.add(character);
const glb = await new GLTFExporter().parseAsync(scene, { binary: true, animations: clips, onlyVisible: false });
mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, "athlete.glb"), Buffer.from(glb));
const bounds = new THREE.Box3().setFromObject(character);
meshes[0].skeleton.pose();
character.updateMatrixWorld(true);
const landmarks = {};
for (const nm of ["mixamorigHips", "mixamorigSpine2", "mixamorigNeck", "mixamorigHead", "mixamorigHeadTop_End", "mixamorigLeftShoulder", "mixamorigLeftArm", "mixamorigLeftForeArm", "mixamorigLeftHand", "mixamorigLeftUpLeg", "mixamorigLeftLeg", "mixamorigLeftFoot", "mixamorigLeftToeBase"]) {
  const b = realByName.get(nm);
  if (b) landmarks[nm.replace("mixamorig", "")] = b.getWorldPosition(v).toArray().map((x) => +x.toFixed(3));
}
const manifest = {
  bind_landmarks_m: landmarks,
  source: "Mixamo Y Bot + 35 Mixamo clips, processed by scripts/assets/convert_mixamo.mjs",
  units: "metres, y-up, character faces +z",
  height_m: +(bounds.max.y - bounds.min.y).toFixed(3),
  bones: realByName.size,
  vertices: meshes.map((m) => ({ name: m.name, count: m.geometry.attributes.position.count })),
  clips: report,
};
writeFileSync(join(OUT, "athlete_clips.json"), JSON.stringify(manifest, null, 2) + "\n");
console.log(`wrote ${join(OUT, "athlete.glb")} (${(glb.byteLength / 1e6).toFixed(2)} MB), ${clips.length} clips`);
console.log(JSON.stringify(manifest.vertices), "height", manifest.height_m);
for (const [k, i] of Object.entries(report)) console.log(k.padEnd(30), JSON.stringify(i));
