// The athlete shader: one MeshPhysicalMaterial program for every player. Each pixel decides, in bind-pose space
// (metres, Y Bot T-pose, +z forward, +x the character's left), whether it is skin, shirt, sleeve, collar, shorts,
// sock, boot or glove, then applies kit colours, the pattern, decals (name, number, crest, sponsor), fabric sheen
// and a procedural knit bump. See docs/art_direction.md sections 3-4.
import * as THREE from "three";
import type { KitColors } from "../types.ts";

export const PATTERN_ID: Record<KitColors["pattern"], number> = {
  plain: 0,
  stripes: 1,
  hoops: 2,
  sash: 3,
  halves: 4,
  chevron: 5,
};

export interface KitLook {
  kit: KitColors;
  /** Shorts colour override (defaults to kit.secondary). */
  shorts?: string;
  socks?: string;
  skin: string;
  boot: string;
  glove: string;
  goalkeeper: boolean;
  back: THREE.Texture;
  front: THREE.Texture;
  hasNumber: boolean;
  hasFront: boolean;
}

const VERT_PARS = /* glsl */ `
attribute float aPlate;
varying vec3 vBind;
varying vec3 vBindN;
varying float vPlate;
`;

const FRAG_PARS = /* glsl */ `
varying vec3 vBind;
varying vec3 vBindN;
varying float vPlate;
uniform vec3 uShirt;
uniform vec3 uSecondary;
uniform vec3 uShorts;
uniform vec3 uSocks;
uniform vec3 uAccent;
uniform vec3 uNumberCol;
uniform vec3 uSkin;
uniform vec3 uBoot;
uniform vec3 uGlove;
uniform float uPattern;
uniform float uGK;
uniform float uMirror;
uniform float uHasNumber;
uniform float uHasFront;
uniform float uGhost;
uniform sampler2D uBackTex;
uniform sampler2D uFrontTex;

float kHash(vec3 p) { p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3)); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float kNoise(vec3 x) {
  vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(kHash(i), kHash(i + vec3(1,0,0)), f.x), mix(kHash(i + vec3(0,1,0)), kHash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(kHash(i + vec3(0,0,1)), kHash(i + vec3(1,0,1)), f.x), mix(kHash(i + vec3(0,1,1)), kHash(i + vec3(1,1,1)), f.x), f.y), f.z);
}

// region ids: 0 skin, 1 shirt, 2 sleeve, 3 trim (accent), 4 shorts, 5 sock, 6 boot, 7 glove, 8 collar (secondary), 9 sole
int kitRegion(vec3 p, vec3 n) {
  float ax = abs(p.x);
  float y = p.y;
  bool arm = ax > 0.205 && y > 1.30;
  if (arm) {
    if (ax > 0.715) return uGK > 0.5 ? 7 : 0;
    float sleeveEnd = uGK > 0.5 ? 0.705 : 0.355;
    if (ax < sleeveEnd - 0.022) return 2;
    if (ax < sleeveEnd) return 3;
    return 0;
  }
  if (y > 1.535) return 0;
  if (y > 1.492) return 8;
  if (y > 0.985) return 1;
  if (y > 0.965) return 3;
  if (y > 0.625) return 4;
  if (y > 0.605) return 3;
  if (y > 0.47) return 0;
  if (y > 0.43) return 3;
  if (y > 0.118) return 5;
  if (y > 0.022) return 6;
  return 9;
}

vec4 decal(sampler2D t, vec2 uv) {
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return vec4(0.0);
  return texture2D(t, vec2(uv.x, 1.0 - uv.y));
}

// out: albedo, roughness, sheen strength, clearcoat, bump height (metres)
void kitSurface(out vec3 albedo, out float rough, out float sheenK, out float cc, out float bumpH) {
  vec3 p = vBind;
  vec3 n = normalize(vBindN);
  int r = kitRegion(p, n);
  float dx = p.x * uMirror; // decals read correctly when the body is mirrored
  bool front = n.z > 0.25;
  bool back = n.z < -0.25;
  sheenK = 0.0; cc = 0.0; bumpH = 0.0; rough = 0.62;
  float knit = 0.0;
  if (r == 0) {
    albedo = uSkin; rough = 0.42; cc = 0.35;
    // subtle sculpt variation so skin is not flat plastic
    albedo *= 0.94 + 0.06 * kNoise(p * 40.0);
  } else if (r == 6) {
    albedo = uBoot; rough = 0.2; cc = 1.0;
    if (p.y < 0.07 && n.y < 0.5 && abs(n.x) > 0.6) albedo = mix(albedo, uAccent, 0.7);
  } else if (r == 9) {
    albedo = vec3(0.04); rough = 0.5;
  } else if (r == 7) {
    albedo = uGlove; rough = 0.55; sheenK = 0.4;
    if (abs(p.x) > 0.84) albedo = mix(albedo, uAccent, 0.85);
    knit = 1.0;
  } else {
    sheenK = 1.0; knit = 1.0;
    if (r == 1 || r == 2) albedo = uShirt;
    else if (r == 3) albedo = uAccent;
    else if (r == 8) albedo = uSecondary;
    else if (r == 4) albedo = uShorts;
    else albedo = uSocks;

    if (r == 1) {
      float ax = abs(p.x);
      if (uPattern < 0.5) {
        // plain: a thin raglan piping from the collar to the armpit in the accent colour
        float rag = abs(p.y - (1.50 - (ax - 0.06) * 0.95));
        albedo = mix(albedo, uAccent, step(rag, 0.009) * step(0.06, ax) * step(ax, 0.2));
      } else if (uPattern < 1.5) {
        if (fract(p.x / 0.075) < 0.5) albedo = uSecondary;
      } else if (uPattern < 2.5) {
        if (fract(p.y / 0.1) < 0.5) albedo = uSecondary;
      } else if (uPattern < 3.5) {
        if (front && abs((p.y - 1.22) - p.x * 1.1) < 0.06) albedo = uSecondary;
      } else if (uPattern < 4.5) {
        if (p.x > 0.0) albedo = uSecondary;
      } else if (front) {
        // chevron across the chest, piped in the accent colour
        float v = p.y - (1.19 + ax * 0.62);
        if (v > 0.0 && v < 0.085) albedo = uSecondary;
        if (abs(v) < 0.007 || abs(v - 0.085) < 0.007) albedo = uAccent;
      }
      // decals sit on the rib-cage plate only (one rigid piece), so they never break across torso segments
      bool plate = vPlate > 0.5 && vPlate < 1.5;
      bool yoke = vPlate > 1.5;
      if (back && (plate || yoke) && uHasNumber > 0.5) {
        // number on the rib plate, name on the shoulder yoke (texture: name in the top 20 %, number below)
        float u = 0.5 - dx / 0.28;
        vec2 uv = plate ? vec2(u, 0.22 + 0.78 * (1.36 - p.y) / 0.17) : vec2(u, 0.2 * (1.452 - p.y) / 0.062);
        if (plate && uv.y < 0.22) uv.y = -1.0;
        if (yoke && uv.y > 0.2) uv.y = -1.0;
        vec4 d = decal(uBackTex, uv);
        albedo = mix(albedo, d.rgb, d.a);
        rough = mix(rough, 0.45, d.a); sheenK *= 1.0 - 0.7 * d.a; knit *= 1.0 - d.a;
      }
      if (front && plate && uHasFront > 0.5) {
        // crest on the left chest (character's left = +x)
        vec2 cuv = vec2((dx - 0.045) / 0.065, (1.395 - p.y) / 0.065);
        vec4 c = decal(uFrontTex, vec2(cuv.x * 0.25, cuv.y));
        if (cuv.x >= 0.0 && cuv.x <= 1.0) albedo = mix(albedo, c.rgb, c.a);
        // sponsor across the chest
        vec2 suv = vec2((dx + 0.12) / 0.24, (1.3 - p.y) / 0.08);
        if (suv.x >= 0.0 && suv.x <= 1.0) {
          vec4 s = decal(uFrontTex, vec2(0.25 + suv.x * 0.75, suv.y));
          albedo = mix(albedo, s.rgb, s.a);
          knit *= 1.0 - s.a;
        }
      }
    }
    if (r == 4) {
      // side stripe on the shorts and a small number on the front of the left leg
      float side = smoothstep(0.9, 0.96, abs(n.x));
      albedo = mix(albedo, uAccent, side * 0.85);
      if (front && uHasNumber > 0.5) {
        vec2 uv = vec2((dx - 0.055) / 0.075, (0.77 - p.y) / 0.075);
        if (uv.x >= 0.0 && uv.x <= 1.0) {
          vec4 d = decal(uBackTex, vec2(uv.x, 0.22 + uv.y * 0.78));
          albedo = mix(albedo, uSecondary == uShorts ? uShirt : d.rgb, d.a);
        }
      }
    }
    if (r == 5 && p.y > 0.37 && p.y < 0.40) albedo = uSecondary;
    // fabric folds + knit
    float folds = kNoise(vec3(p.x * 18.0, p.y * 7.0, p.z * 18.0));
    albedo *= 0.93 + 0.07 * folds;
    rough = mix(0.62, 0.78, folds);
    bumpH = (folds - 0.5) * 0.004;
  }
  if (knit > 0.0) {
    float k = sin(p.x * 1400.0 + p.z * 900.0) * sin(p.y * 1400.0);
    float fw = length(fwidth(p)) * 1400.0;
    bumpH += k * 0.00018 * knit * (1.0 - smoothstep(0.5, 1.5, fw));
  }
}

vec3 kitPerturb(vec3 surfPos, vec3 surfNorm, float h, float faceDir) {
  vec3 sx = dFdx(surfPos);
  vec3 sy = dFdy(surfPos);
  vec3 r1 = cross(sy, surfNorm);
  vec3 r2 = cross(surfNorm, sx);
  float det = dot(sx, r1) * faceDir;
  vec3 grad = sign(det) * (dFdx(h) * r1 + dFdy(h) * r2);
  return normalize(abs(det) * surfNorm - grad);
}
`;

let shared: { vertexShader: string; fragmentShader: string } | null = null;

function patch(shader: THREE.WebGLProgramParametersWithUniforms): void {
  shader.vertexShader = VERT_PARS + shader.vertexShader.replace(
    "#include <begin_vertex>",
    "#include <begin_vertex>\n  vBind = position;\n  vBindN = normal;\n  vPlate = aPlate;",
  );
  shader.fragmentShader = FRAG_PARS + shader.fragmentShader
    .replace(
      "#include <color_fragment>",
      `#include <color_fragment>
  vec3 kAlbedo; float kRough; float kSheen; float kCC; float kBump;
  kitSurface(kAlbedo, kRough, kSheen, kCC, kBump);
  diffuseColor.rgb = kAlbedo;`,
    )
    .replace("#include <roughnessmap_fragment>", "#include <roughnessmap_fragment>\n  roughnessFactor = kRough;")
    .replace(
      "#include <normal_fragment_maps>",
      "#include <normal_fragment_maps>\n  normal = kitPerturb(-vViewPosition, normal, kBump, faceDirection);",
    )
    .replace(
      "#include <lights_physical_fragment>",
      `#include <lights_physical_fragment>
  material.sheenColor = mix(kAlbedo, vec3(1.0), 0.35) * kSheen * 0.9;
  material.clearcoat = kCC;
  material.clearcoatRoughness = 0.3;`,
    );
  shared = { vertexShader: shader.vertexShader, fragmentShader: shader.fragmentShader };
}

const col = (hex: string) => new THREE.Color(hex);

export function createKitMaterial(look: KitLook): THREE.MeshPhysicalMaterial {
  const m = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.65,
    metalness: 0,
    sheen: 1,
    sheenRoughness: 0.55,
    sheenColor: new THREE.Color(1, 1, 1),
    clearcoat: 1,
    clearcoatRoughness: 0.3,
    envMapIntensity: 0.55,
  });
  const k = look.kit;
  const uniforms = {
    uShirt: { value: col(k.primary) },
    uSecondary: { value: col(k.secondary) },
    uShorts: { value: col(look.shorts ?? k.secondary) },
    uSocks: { value: col(look.socks ?? k.primary) },
    uAccent: { value: col(k.accent) },
    uNumberCol: { value: col(k.number) },
    uSkin: { value: col(look.skin) },
    uBoot: { value: col(look.boot) },
    uGlove: { value: col(look.glove) },
    uPattern: { value: PATTERN_ID[k.pattern] },
    uGK: { value: look.goalkeeper ? 1 : 0 },
    uMirror: { value: 1 },
    uHasNumber: { value: look.hasNumber ? 1 : 0 },
    uHasFront: { value: look.hasFront ? 1 : 0 },
    uGhost: { value: 0 },
    uBackTex: { value: look.back },
    uFrontTex: { value: look.front },
  };
  m.userData.kitUniforms = uniforms;
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    patch(shader);
  };
  m.customProgramCacheKey = () => "regolazo-kit-v1";
  return m;
}

export function kitUniforms(m: THREE.Material): Record<string, { value: unknown }> {
  return m.userData.kitUniforms as Record<string, { value: unknown }>;
}

/** For debugging: the patched shader source once compiled. */
export function kitShaderSource(): typeof shared {
  return shared;
}
