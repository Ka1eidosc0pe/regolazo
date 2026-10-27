// A smooth, sculpted figurine body for the Mixamo skeleton (replaces the segmented Y Bot surface + joints).
// The body is a signed distance field built from tapered capsules and ellipsoids placed on the bind-pose bones
// (T-pose, metres), smoothly unioned, polygonised with marching cubes, given SDF-gradient normals, and skinned
// to the bones by distance to each bone's shapes. No joints, no seams, a faceless egg head.

/** @typedef {[number, number, number]} V */

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Round cone (tapered capsule) from a (radius ra) to b (radius rb). */
function roundCone(a, b, ra, rb) {
  const ba = sub(b, a);
  const l2 = dot(ba, ba);
  return (p) => {
    const pa = sub(p, a);
    const h = clamp(dot(pa, ba) / l2, 0, 1);
    const q = sub(pa, [ba[0] * h, ba[1] * h, ba[2] * h]);
    return len(q) - (ra + (rb - ra) * h);
  };
}

/** Ellipsoid (approximate distance, good near the surface). */
function ellipsoid(c, r) {
  return (p) => {
    const q = sub(p, c);
    const k0 = Math.hypot(q[0] / r[0], q[1] / r[1], q[2] / r[2]);
    const k1 = Math.hypot(q[0] / (r[0] * r[0]), q[1] / (r[1] * r[1]), q[2] / (r[2] * r[2]));
    return k1 > 0 ? (k0 * (k0 - 1)) / k1 : -Math.min(r[0], r[1], r[2]);
  };
}

/** Rounded box centred at c with half extents h and corner radius r. */
function roundBox(c, h, r) {
  return (p) => {
    const q = [Math.abs(p[0] - c[0]) - h[0] + r, Math.abs(p[1] - c[1]) - h[1] + r, Math.abs(p[2] - c[2]) - h[2] + r];
    const o = [Math.max(q[0], 0), Math.max(q[1], 0), Math.max(q[2], 0)];
    return len(o) + Math.min(Math.max(q[0], q[1], q[2]), 0) - r;
  };
}

function smin(a, b, k) {
  const h = clamp(0.5 + (0.5 * (b - a)) / k, 0, 1);
  return b + (a - b) * h - k * h * (1 - h);
}

/** The cranium ellipsoid (also exported to the manifest so hair fits it exactly). */
export function skull(head) {
  return { center: add(head, [0, 0.104, 0.022]), radii: [0.087, 0.113, 0.1] };
}

/**
 * Shapes on the bind-pose skeleton. `bone(name)` returns the bone's world position (metres).
 * Each shape: { bone, sdf, k } (k = smooth-union radius with the rest of the body).
 */
export function figurineShapes(bone) {
  const P = (n) => bone(`mixamorig${n}`);
  const shapes = [];
  const S = (b, sdf, k = 0.03) => shapes.push({ bone: `mixamorig${b}`, sdf, k });
  // torso: pelvis, abdomen, rib cage, chest (a V taper from broad shoulders to the waist)
  S("Hips", ellipsoid([0, 0.955, -0.012], [0.148, 0.112, 0.108]));
  S("Hips", ellipsoid([0, 0.915, -0.07], [0.135, 0.085, 0.075]), 0.04); // glutes
  S("Spine", ellipsoid([0, 1.115, 0.0], [0.128, 0.115, 0.098]), 0.05);
  S("Spine1", ellipsoid([0, 1.265, -0.012], [0.152, 0.135, 0.11]), 0.05);
  S("Spine2", ellipsoid([0, 1.385, -0.02], [0.18, 0.108, 0.115]), 0.05);
  // neck and head (a smooth egg with a soft jaw, no face)
  const neckA = add(P("Neck"), [0, -0.03, 0.0]);
  S("Neck", roundCone(neckA, add(P("Head"), [0, 0.02, 0.0]), 0.056, 0.05), 0.03);
  S("Head", ellipsoid(skull(P("Head")).center, skull(P("Head")).radii), 0.03);
  S("Head", ellipsoid(add(P("Head"), [0, 0.045, 0.04]), [0.062, 0.058, 0.072]), 0.045);
  for (const side of ["Left", "Right"]) {
    const sx = side === "Left" ? 1 : -1;
    const arm = P(`${side}Arm`);
    const elbow = P(`${side}ForeArm`);
    const wrist = P(`${side}Hand`);
    const hip = P(`${side}UpLeg`);
    const knee = P(`${side}Leg`);
    const ankle = P(`${side}Foot`);
    const toe = P(`${side}ToeBase`);
    // shoulders: trapezius slope and a rounded deltoid
    S(`${side}Shoulder`, roundCone([sx * 0.045, 1.47, -0.04], add(arm, [-sx * 0.01, 0.012, 0.005]), 0.05, 0.058), 0.035);
    S(`${side}Arm`, ellipsoid(add(arm, [sx * 0.02, 0.0, 0.005]), [0.08, 0.07, 0.072]), 0.03);
    // arms
    S(`${side}Arm`, roundCone(arm, elbow, 0.056, 0.043), 0.03);
    S(`${side}Arm`, ellipsoid(add(arm, [sx * 0.12, -0.004, 0.0]), [0.085, 0.052, 0.054]), 0.03); // biceps / triceps
    S(`${side}ForeArm`, roundCone(elbow, wrist, 0.045, 0.032), 0.02);
    S(`${side}ForeArm`, ellipsoid(add(elbow, [sx * 0.07, 0, 0.0]), [0.075, 0.047, 0.049]), 0.03); // forearm muscle
    // mitten hand, palm down in the T-pose
    S(`${side}Hand`, ellipsoid(add(wrist, [sx * 0.075, -0.006, 0.0]), [0.078, 0.026, 0.047]), 0.02);
    S(`${side}Hand`, roundCone(add(wrist, [sx * 0.03, -0.01, 0.04]), add(wrist, [sx * 0.07, -0.018, 0.07]), 0.016, 0.013), 0.015); // thumb
    // legs: strong thighs, calves with a muscle belly
    S(`${side}UpLeg`, roundCone(add(hip, [sx * 0.008, 0.02, 0.0]), knee, 0.092, 0.057), 0.045);
    S(`${side}UpLeg`, ellipsoid(add(hip, [sx * 0.012, -0.17, 0.02]), [0.08, 0.155, 0.08]), 0.04); // quads
    S(`${side}Leg`, roundCone(knee, ankle, 0.05, 0.032), 0.025);
    S(`${side}Leg`, ellipsoid(add(knee, [0, -0.14, -0.024]), [0.056, 0.115, 0.058]), 0.03); // calf
    // boot: heel to ball on the foot bone, toe cap on the toe bone
    S(`${side}Foot`, roundBox([ankle[0], 0.042, 0.03], [0.04, 0.042, 0.095], 0.032), 0.03);
    S(`${side}ToeBase`, roundBox([toe[0], 0.03, toe[2] + 0.03], [0.038, 0.03, 0.055], 0.028), 0.02);
  }
  return shapes;
}

/** Evaluate the body SDF and every shape's own distance at p. */
function evalShapes(shapes, p, out) {
  let d = Infinity;
  for (let i = 0; i < shapes.length; i++) {
    const s = shapes[i];
    const di = s.sdf(p);
    if (out) out[i] = di;
    d = d === Infinity ? di : smin(d, di, s.k);
  }
  return d;
}

/**
 * Polygonise the body. Returns { positions, normals, indices, skinIndex, skinWeight } for a skeleton whose bone
 * names are `boneNames` (index = position in that array).
 */
export function buildFigurine({ shapes, boneNames, step = 0.009, edgeTable, triTable }) {
  const min = [-0.98, -0.02, -0.24];
  const max = [0.98, 1.86, 0.3];
  const nx = Math.ceil((max[0] - min[0]) / step) + 1;
  const ny = Math.ceil((max[1] - min[1]) / step) + 1;
  const nz = Math.ceil((max[2] - min[2]) / step) + 1;
  const field = new Float32Array(nx * ny * nz);
  const idx = (x, y, z) => x + nx * (y + ny * z);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        field[idx(x, y, z)] = evalShapes(shapes, [min[0] + x * step, min[1] + y * step, min[2] + z * step]);
      }
    }
  }
  // marching cubes (Paul Bourke corner/edge convention, as in three's MarchingCubes tables)
  const corner = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const positions = [];
  const indices = [];
  const vertexOf = new Map();
  const vals = new Float32Array(8);
  const edgeVert = new Int32Array(12);
  for (let z = 0; z < nz - 1; z++) {
    for (let y = 0; y < ny - 1; y++) {
      for (let x = 0; x < nx - 1; x++) {
        let cube = 0;
        for (let c = 0; c < 8; c++) {
          const [cx, cy, cz] = corner[c];
          vals[c] = field[idx(x + cx, y + cy, z + cz)];
          if (vals[c] < 0) cube |= 1 << c;
        }
        const em = edgeTable[cube];
        if (!em) continue;
        for (let e = 0; e < 12; e++) {
          if (!(em & (1 << e))) continue;
          const [a, b] = edges[e];
          const A = corner[a];
          const B = corner[b];
          const ga = [x + A[0], y + A[1], z + A[2]];
          const gb = [x + B[0], y + B[1], z + B[2]];
          const ka = idx(ga[0], ga[1], ga[2]);
          const kb = idx(gb[0], gb[1], gb[2]);
          const key = ka < kb ? ka * 3 + (gb[0] - ga[0] ? 0 : gb[1] - ga[1] ? 1 : 2) : kb * 3 + (gb[0] - ga[0] ? 0 : gb[1] - ga[1] ? 1 : 2);
          let v = vertexOf.get(key);
          if (v === undefined) {
            const t = vals[a] / (vals[a] - vals[b]);
            v = positions.length / 3;
            positions.push(
              min[0] + (ga[0] + (gb[0] - ga[0]) * t) * step,
              min[1] + (ga[1] + (gb[1] - ga[1]) * t) * step,
              min[2] + (ga[2] + (gb[2] - ga[2]) * t) * step,
            );
            vertexOf.set(key, v);
          }
          edgeVert[e] = v;
        }
        for (let i = 0; triTable[cube * 16 + i] !== -1; i += 3) {
          indices.push(edgeVert[triTable[cube * 16 + i]], edgeVert[triTable[cube * 16 + i + 1]], edgeVert[triTable[cube * 16 + i + 2]]);
        }
      }
    }
  }
  const nv = positions.length / 3;
  // normals from the SDF gradient (smooth regardless of the grid)
  const normals = new Float32Array(nv * 3);
  const e = step * 0.5;
  for (let i = 0; i < nv; i++) {
    const p = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
    const g = [
      evalShapes(shapes, [p[0] + e, p[1], p[2]]) - evalShapes(shapes, [p[0] - e, p[1], p[2]]),
      evalShapes(shapes, [p[0], p[1] + e, p[2]]) - evalShapes(shapes, [p[0], p[1] - e, p[2]]),
      evalShapes(shapes, [p[0], p[1], p[2] + e]) - evalShapes(shapes, [p[0], p[1], p[2] - e]),
    ];
    const l = len(g) || 1;
    normals.set([g[0] / l, g[1] / l, g[2] / l], i * 3);
  }
  // consistent outward winding
  for (let t = 0; t < indices.length; t += 3) {
    const [a, b, c] = [indices[t], indices[t + 1], indices[t + 2]];
    const pa = [positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]];
    const pb = [positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]];
    const pc = [positions[c * 3], positions[c * 3 + 1], positions[c * 3 + 2]];
    const u = sub(pb, pa);
    const w = sub(pc, pa);
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    if (dot(n, [normals[a * 3], normals[a * 3 + 1], normals[a * 3 + 2]]) < 0) {
      indices[t + 1] = c;
      indices[t + 2] = b;
    }
  }
  // skin weights: softmin over each bone's shapes
  const boneIndex = new Map(boneNames.map((n, i) => [n, i]));
  const skinIndex = new Uint16Array(nv * 4);
  const skinWeight = new Float32Array(nv * 4);
  const ds = new Float32Array(shapes.length);
  const tau = 0.014;
  for (let i = 0; i < nv; i++) {
    evalShapes(shapes, [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]], ds);
    let dmin = Infinity;
    for (let k = 0; k < ds.length; k++) dmin = Math.min(dmin, ds[k]);
    const acc = new Map();
    for (let k = 0; k < ds.length; k++) {
      const w = Math.exp(-(ds[k] - dmin) / tau);
      if (w < 1e-3) continue;
      const b = boneIndex.get(shapes[k].bone);
      if (b === undefined) throw new Error(`no bone ${shapes[k].bone}`);
      acc.set(b, (acc.get(b) ?? 0) + w);
    }
    const top = [...acc.entries()].sort((p, q) => q[1] - p[1]).slice(0, 4);
    const sum = top.reduce((s, [, w]) => s + w, 0);
    top.forEach(([b, w], k) => {
      skinIndex[i * 4 + k] = b;
      skinWeight[i * 4 + k] = w / sum;
    });
  }
  return { positions: new Float32Array(positions), normals, indices, skinIndex, skinWeight, grid: [nx, ny, nz] };
}
