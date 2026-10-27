// Post-processing: HDR render (MSAA, with a depth texture) -> optional depth of field -> bloom (half res) ->
// ACES tone mapping + sRGB -> grade (vignette, cool-shadow / warm-highlight split, gentle contrast, light grain,
// faint edge chromatic aberration).
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { FullScreenQuad, Pass } from "three/examples/jsm/postprocessing/Pass.js";
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

const DOF_FRAGMENT = /* glsl */ `
  uniform sampler2D tDiffuse;
  uniform sampler2D tDepth;
  uniform float uNear;
  uniform float uFar;
  uniform float uFocus;
  uniform float uAperture;
  uniform float uMaxBlur;
  uniform vec2 uTexel;
  varying vec2 vUv;
  float linDepth(vec2 uv) {
    float z = texture2D(tDepth, uv).x;
    // perspective depth -> view distance
    return (uNear * uFar) / ((uFar - uNear) * z - uFar) * -1.0;
  }
  float coc(float d) { return clamp(abs(d - uFocus) / max(d, 1e-3) * uAperture * uMaxBlur * 4.0, 0.0, uMaxBlur); }
  void main() {
    float d0 = linDepth(vUv);
    float c0 = coc(d0);
    vec4 base = texture2D(tDiffuse, vUv);
    if (c0 < 0.5) { gl_FragColor = base; return; }
    vec3 acc = base.rgb;
    float wsum = 1.0;
    for (int i = 0; i < 24; i++) {
      float fi = float(i) + 0.5;
      float r = sqrt(fi / 24.0) * c0;
      float a = fi * 2.39996;
      vec2 uv = vUv + vec2(cos(a), sin(a)) * r * uTexel;
      float ds = linDepth(uv);
      // a sharp foreground does not bleed into the blurred background behind it
      float w = ds < d0 - 0.5 ? clamp(coc(ds) / max(c0, 1e-3), 0.0, 1.0) : 1.0;
      acc += texture2D(tDiffuse, uv).rgb * w;
      wsum += w;
    }
    gl_FragColor = vec4(acc / wsum, base.a);
  }`;

/**
 * Depth of field from the main render's own depth texture (BokehPass renders the whole scene a second time):
 * circle of confusion from view distance vs the focus distance, gathered with a 24-tap golden-angle disc.
 */
class DofPass extends Pass {
  readonly uniforms = {
    tDiffuse: { value: null as THREE.Texture | null },
    tDepth: { value: null as THREE.Texture | null },
    uNear: { value: 0.3 },
    uFar: { value: 4000 },
    uFocus: { value: 30 },
    uAperture: { value: 0.6 },
    uMaxBlur: { value: 9 },
    uTexel: { value: new THREE.Vector2(1 / 1920, 1 / 1080) },
  };
  private readonly quad: FullScreenQuad;
  private readonly camera: THREE.PerspectiveCamera;

  constructor(camera: THREE.PerspectiveCamera) {
    super();
    this.camera = camera;
    this.quad = new FullScreenQuad(
      new THREE.ShaderMaterial({
        uniforms: this.uniforms,
        vertexShader: GradeShader.vertexShader,
        fragmentShader: DOF_FRAGMENT,
      }),
    );
  }

  setSize(w: number, h: number): void {
    this.uniforms.uTexel.value.set(1 / w, 1 / h);
  }

  render(renderer: THREE.WebGLRenderer, writeBuffer: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget): void {
    this.uniforms.tDiffuse.value = readBuffer.texture;
    this.uniforms.tDepth.value = readBuffer.depthTexture;
    this.uniforms.uNear.value = this.camera.near;
    this.uniforms.uFar.value = this.camera.far;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    this.quad.render(renderer);
  }
}

export class Post {
  readonly composer: EffectComposer;
  readonly bloom: UnrealBloomPass;
  readonly dof: DofPass;
  readonly grade: ShaderPass;
  private readonly renderPass: RenderPass;

  constructor(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, w: number, h: number, samples = 4) {
    const depthTexture = new THREE.DepthTexture(w, h);
    depthTexture.type = THREE.UnsignedIntType;
    const rt = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, samples, depthTexture });
    this.composer = new EffectComposer(renderer, rt);
    this.composer.setPixelRatio(1);
    this.renderPass = new RenderPass(scene, camera);
    this.dof = new DofPass(camera);
    this.dof.enabled = false;
    // bloom runs at half resolution: a quarter of the fill cost, and the glow is soft anyway
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w / 2, h / 2), 0.42, 0.42, 1.0);
    this.grade = new ShaderPass(GradeShader);
    this.composer.addPass(this.renderPass);
    this.composer.addPass(this.dof);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.composer.addPass(this.grade);
    this.setSize(w, h);
  }

  setSize(w: number, h: number): void {
    this.composer.setSize(w, h);
    this.bloom.setSize(Math.round(w / 2), Math.round(h / 2));
    this.dof.setSize(w, h);
  }

  /**
   * Start from the same ping-pong buffer every time: the number of swaps per frame depends on which passes are
   * enabled (DOF), and the two buffers are not bit-identical, so the parity would leak history into frames.
   */
  resetBuffers(): void {
    if (this.composer.readBuffer !== this.composer.renderTarget1) this.composer.swapBuffers();
  }

  /** Depth of field (replays): focus distance in metres; aperture scales the blur. */
  setDof(enabled: boolean, focus = 30, aperture = 0.6): void {
    this.dof.enabled = enabled;
    this.dof.uniforms.uFocus.value = focus;
    this.dof.uniforms.uAperture.value = aperture;
  }

  render(frame: number, dt?: number): void {
    (this.grade.uniforms as Record<string, THREE.IUniform>).uFrame!.value = frame % 997;
    this.composer.render(dt);
  }
}
