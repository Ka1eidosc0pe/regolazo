// Post-processing: HDR render (MSAA) -> optional depth of field -> bloom -> ACES tone mapping + sRGB -> grade
// (vignette, cool-shadow / warm-highlight split, gentle contrast, light grain, faint edge chromatic aberration).
import * as THREE from "three";
import { BokehPass } from "three/examples/jsm/postprocessing/BokehPass.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

const GradeShader = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uFrame: { value: 0 },
    uVignette: { value: 0.32 },
    uGrain: { value: 0.018 },
    uAberration: { value: 0.0012 },
    uDesat: { value: 0 },
  },
  vertexShader: /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse; uniform float uFrame; uniform float uVignette; uniform float uGrain; uniform float uAberration; uniform float uDesat;
    varying vec2 vUv;
    float h(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)) + uFrame * 0.618) * 43758.5453); }
    void main(){
      vec2 c = vUv - 0.5;
      float r2 = dot(c, c);
      vec2 off = c * uAberration * r2 * 4.0;
      vec3 col;
      col.r = texture2D(tDiffuse, vUv + off).r;
      col.g = texture2D(tDiffuse, vUv).g;
      col.b = texture2D(tDiffuse, vUv - off).b;
      // split tone: cool shadows, slightly warm highlights
      float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
      col += vec3(-0.012, 0.0, 0.022) * (1.0 - smoothstep(0.0, 0.45, l));
      col += vec3(0.012, 0.006, -0.01) * smoothstep(0.55, 1.0, l);
      // gentle S-curve
      col = mix(col, col * col * (3.0 - 2.0 * col), 0.18);
      col = mix(col, vec3(l), uDesat);
      // vignette
      col *= 1.0 - uVignette * smoothstep(0.08, 0.62, r2 * 1.6);
      // grain (luma-weighted)
      col += (h(vUv * 1000.0) - 0.5) * uGrain * (0.6 + 0.4 * (1.0 - l));
      gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
    }`,
};

export class Post {
  readonly composer: EffectComposer;
  readonly bloom: UnrealBloomPass;
  readonly bokeh: BokehPass;
  readonly grade: ShaderPass;
  private readonly renderPass: RenderPass;

  constructor(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, w: number, h: number, samples = 4) {
    const rt = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, samples });
    this.composer = new EffectComposer(renderer, rt);
    this.composer.setPixelRatio(1);
    this.renderPass = new RenderPass(scene, camera);
    this.bokeh = new BokehPass(scene, camera, { focus: 30, aperture: 0.0008, maxblur: 0.006 });
    this.bokeh.enabled = false;
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.42, 0.42, 1.0);
    this.grade = new ShaderPass(GradeShader);
    this.composer.addPass(this.renderPass);
    this.composer.addPass(this.bokeh);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.composer.addPass(this.grade);
  }

  setSize(w: number, h: number): void {
    this.composer.setSize(w, h);
  }

  setDof(enabled: boolean, focus = 30, aperture = 0.0008): void {
    this.bokeh.enabled = enabled;
    const u = this.bokeh.uniforms as Record<string, THREE.IUniform>;
    u.focus!.value = focus;
    u.aperture!.value = aperture;
  }

  render(frame: number, dt?: number): void {
    (this.grade.uniforms as Record<string, THREE.IUniform>).uFrame!.value = frame % 997;
    this.composer.render(dt);
  }
}
