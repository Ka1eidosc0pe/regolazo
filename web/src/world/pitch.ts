// The pitch and its surround: mowing stripes with view-dependent grain, shader-drawn markings (anti-aliased SDF),
// goalmouth wear, wet sheen (low roughness + environment reflections) and grass micro-bump.
import * as THREE from "three";
import { HALF_L, HALF_W } from "../coords.ts";

export const BOARD_X = HALF_L + 6.0;
export const BOARD_Y = HALF_W + 5.0;

const PARS = /* glsl */ `
varying vec3 vWorldPos;
uniform float uWet;
float pHash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float pNoise(vec2 x) {
  vec2 i = floor(x); vec2 f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(pHash(i), pHash(i + vec2(1, 0)), f.x), mix(pHash(i + vec2(0, 1)), pHash(i + vec2(1, 1)), f.x), f.y);
}
float pFbm(vec2 p) { float v = 0.0; float a = 0.5; for (int i = 0; i < 4; i++) { v += a * pNoise(p); p *= 2.03; a *= 0.5; } return v; }
float segD(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h); }
float boxOutline(vec2 p, vec2 h) { vec2 d = abs(p) - h; return abs(length(max(d, 0.0)) + min(max(d.x, d.y), 0.0)); }

float markings(vec2 p) {
  float d = boxOutline(p, vec2(${HALF_L.toFixed(2)}, ${HALF_W.toFixed(2)}));
  d = min(d, segD(p, vec2(0.0, -${HALF_W.toFixed(2)}), vec2(0.0, ${HALF_W.toFixed(2)})));
  d = min(d, abs(length(p) - 9.15));
  d = min(d, max(length(p) - 0.15, 0.0));
  for (int k = 0; k < 2; k++) {
    float s = k == 0 ? 1.0 : -1.0;
    float gx = s * ${HALF_L.toFixed(2)};
    // penalty area 16.5 x 40.32
    d = min(d, segD(p, vec2(gx, -20.16), vec2(gx - s * 16.5, -20.16)));
    d = min(d, segD(p, vec2(gx, 20.16), vec2(gx - s * 16.5, 20.16)));
    d = min(d, segD(p, vec2(gx - s * 16.5, -20.16), vec2(gx - s * 16.5, 20.16)));
    // goal area 5.5 x 18.32
    d = min(d, segD(p, vec2(gx, -9.16), vec2(gx - s * 5.5, -9.16)));
    d = min(d, segD(p, vec2(gx, 9.16), vec2(gx - s * 5.5, 9.16)));
    d = min(d, segD(p, vec2(gx - s * 5.5, -9.16), vec2(gx - s * 5.5, 9.16)));
    // penalty spot and arc
    vec2 spot = vec2(gx - s * 11.0, 0.0);
    d = min(d, max(length(p - spot) - 0.12, 0.0));
    if (s * p.x < ${(HALF_L - 16.5).toFixed(2)}) d = min(d, abs(length(p - spot) - 9.15));
    // corner arcs
    for (int j = 0; j < 2; j++) {
      float sy = j == 0 ? 1.0 : -1.0;
      vec2 c = vec2(gx, sy * ${HALF_W.toFixed(2)});
      vec2 q = p - c;
      if (q.x * s <= 0.0 && q.y * sy <= 0.0) d = min(d, abs(length(q) - 1.0));
    }
  }
  return d;
}

// out: albedo, roughness, bump height
void pitchSurface(vec3 wp, vec3 viewDir, out vec3 albedo, out float rough, out float h) {
  vec2 p = vec2(wp.x, -wp.z);
  float inBoards = step(abs(p.x), ${BOARD_X.toFixed(2)} + 0.4) * step(abs(p.y), ${BOARD_Y.toFixed(2)} + 0.4);
  // grass
  float stripe = floor((p.x + ${HALF_L.toFixed(2)}) / 5.25);
  float par = mod(stripe, 2.0) * 2.0 - 1.0;
  // mowing grain: stripes are mown towards / away from the main camera, so their brightness depends on view
  float grain = par * (0.55 + 0.45 * clamp(-viewDir.z * 1.3, -1.0, 1.0));
  vec3 g1 = vec3(0.085, 0.215, 0.055);
  vec3 g2 = vec3(0.060, 0.165, 0.040);
  vec3 grass = mix(g2, g1, 0.5 + 0.5 * grain);
  float n1 = pFbm(p * 0.35);
  float n2 = pFbm(p * 2.2 + 7.0);
  grass *= 0.86 + 0.22 * n1;
  grass *= 0.94 + 0.12 * n2;
  // wear: goalmouths, penalty spots, centre circle, touchline where linesmen run
  float wear = 0.0;
  for (int k = 0; k < 2; k++) {
    float s = k == 0 ? 1.0 : -1.0;
    vec2 q = (p - vec2(s * (${HALF_L.toFixed(2)} - 3.5), 0.0)) / vec2(4.5, 4.0);
    wear = max(wear, (1.0 - smoothstep(0.3, 1.1, length(q))) * (0.55 + 0.45 * pNoise(p * 1.7)));
    wear = max(wear, (1.0 - smoothstep(0.0, 1.2, length(p - vec2(s * (${HALF_L.toFixed(2)} - 11.0), 0.0)))) * 0.6);
  }
  wear = max(wear, (1.0 - smoothstep(0.0, 2.5, length(p))) * 0.35);
  grass = mix(grass, vec3(0.20, 0.20, 0.10), wear * 0.55);
  // fine blades (faded with distance to avoid shimmer)
  float fw = length(fwidth(p));
  float blades = pNoise(p * 55.0);
  float bladeFade = 1.0 - smoothstep(0.02, 0.09, fw);
  grass *= 1.0 + (blades - 0.5) * 0.25 * bladeFade;
  h = (blades - 0.5) * 0.0016 * bladeFade + (n2 - 0.5) * 0.003;
  rough = mix(0.62, 0.36, uWet) + 0.14 * n1 + wear * 0.2;
  // markings
  float d = markings(p);
  float aa = max(fwidth(d), 0.004);
  float line = 1.0 - smoothstep(0.06 - aa, 0.06 + aa, d);
  line *= step(abs(p.x), ${HALF_L.toFixed(2)} + 0.2) * step(abs(p.y), ${HALF_W.toFixed(2)} + 0.2);
  vec3 paint = vec3(0.80, 0.82, 0.78) * (0.92 + 0.08 * n2);
  albedo = mix(grass, paint, line * 0.95);
  rough = mix(rough, 0.75, line);
  h *= 1.0 - line;
  // synthetic track outside the boards
  vec3 track = vec3(0.022, 0.05, 0.046) * (0.85 + 0.3 * n2);
  albedo = mix(track, albedo, inBoards);
  rough = mix(0.82, rough, inBoards);
}

vec3 pitchPerturb(vec3 surfPos, vec3 surfNorm, float h, float faceDir) {
  vec3 sx = dFdx(surfPos);
  vec3 sy = dFdy(surfPos);
  vec3 r1 = cross(sy, surfNorm);
  vec3 r2 = cross(surfNorm, sx);
  float det = dot(sx, r1) * faceDir;
  vec3 grad = sign(det) * (dFdx(h) * r1 + dFdy(h) * r2);
  return normalize(abs(det) * surfNorm - grad);
}
`;

export function createPitch(): THREE.Mesh {
  const geo = new THREE.PlaneGeometry(150, 110, 1, 1);
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0, envMapIntensity: 1.15 });
  const uniforms = { uWet: { value: 1.0 } };
  mat.userData.uniforms = uniforms;
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = "varying vec3 vWorldPos;\n" + shader.vertexShader.replace(
      "#include <begin_vertex>",
      "#include <begin_vertex>\n  vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;",
    );
    shader.fragmentShader = PARS + shader.fragmentShader
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  vec3 pAlbedo; float pRough; float pH;
  pitchSurface(vWorldPos, normalize(vWorldPos - cameraPosition), pAlbedo, pRough, pH);
  diffuseColor.rgb = pAlbedo;`,
      )
      .replace("#include <roughnessmap_fragment>", "#include <roughnessmap_fragment>\n  roughnessFactor = pRough;")
      .replace(
        "#include <normal_fragment_maps>",
        "#include <normal_fragment_maps>\n  normal = pitchPerturb(-vViewPosition, normal, pH, faceDirection);",
      );
  };
  mat.customProgramCacheKey = () => "regolazo-pitch-v1";
  const mesh = new THREE.Mesh(geo, mat);
  mesh.receiveShadow = true;
  mesh.name = "pitch";
  return mesh;
}
