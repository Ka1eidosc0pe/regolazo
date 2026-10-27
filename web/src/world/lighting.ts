// Night lighting: four corner floodlight banks (shadow-casting directional lights), a dim hemisphere fill,
// an environment map built from a dark dome with a ring of lamp heads, the sky dome and fog.
import * as THREE from "three";

export const FOG_COLOR = new THREE.Color("#0f1622");

export interface Lighting {
  group: THREE.Group;
  lights: THREE.DirectionalLight[];
  /** Fit the shadow cameras around a focus point (Three.js coords) with a half extent in metres. */
  focus(center: THREE.Vector3, halfExtent: number): void;
}

export function createLighting(scene: THREE.Scene, mapSize: number): Lighting {
  const group = new THREE.Group();
  group.name = "lighting";
  const hemi = new THREE.HemisphereLight(new THREE.Color("#4a5a7a"), new THREE.Color("#18301a"), 0.22);
  group.add(hemi);
  const lights: THREE.DirectionalLight[] = [];
  const corners: [number, number, number][] = [
    [-62, 46, 48],
    [62, 46, 48],
    [-62, 46, -48],
    [62, 46, -48],
  ];
  corners.forEach((c, i) => {
    const l = new THREE.DirectionalLight(new THREE.Color("#eef3ff"), i < 2 ? 1.05 : 0.85);
    l.position.set(...c);
    // the two near-side banks cast shadows (each player gets a two-way shadow, thrown away from the main
    // camera); the far banks only light. Four shadow maps cost ~20 % of the frame (p95 over budget).
    l.castShadow = i < 2;
    l.shadow.mapSize.set(mapSize, mapSize);
    l.shadow.bias = -0.0004;
    l.shadow.normalBias = 0.03;
    l.shadow.radius = 2.5;
    const cam = l.shadow.camera;
    cam.near = 10;
    cam.far = 220;
    cam.left = -45;
    cam.right = 45;
    cam.top = 45;
    cam.bottom = -45;
    group.add(l, l.target);
    lights.push(l);
  });
  scene.add(group);
  const offsets = lights.map((l) => l.position.clone().sub(l.target.position));

  return {
    group,
    lights,
    focus(center, half) {
      lights.forEach((l, i) => {
        // fixed per-light offsets (no accumulated floating-point drift: results depend only on the focus)
        const offset = offsets[i]!;
        // snap the focus to whole shadow-map texels in the light's view plane, so shadow edges do not crawl
        // while the camera pans
        const texel = (2 * half) / l.shadow.mapSize.x;
        const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(offset, new THREE.Vector3(), new THREE.Vector3(0, 1, 0)));
        const local = center.clone().applyQuaternion(q.clone().invert());
        local.x = Math.round(local.x / texel) * texel;
        local.y = Math.round(local.y / texel) * texel;
        const snapped = local.applyQuaternion(q);
        l.target.position.copy(snapped);
        l.position.copy(snapped).add(offset);
        const cam = l.shadow.camera;
        cam.left = -half;
        cam.right = half;
        cam.top = half;
        cam.bottom = -half;
        cam.updateProjectionMatrix();
      });
    },
  };
}

/** Builds the environment map with the app's renderer (call once after the renderer exists). */
export function buildEnvironment(renderer: THREE.WebGLRenderer): THREE.Texture {
  const env = new THREE.Scene();
  const dome = new THREE.Mesh(
    new THREE.SphereGeometry(100, 32, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      vertexShader: "varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0);} ",
      fragmentShader: /* glsl */ `varying vec3 vP; void main(){
        float y = normalize(vP).y;
        vec3 sky = mix(vec3(0.03,0.035,0.055), vec3(0.006,0.008,0.016), smoothstep(0.0,0.8,y));
        vec3 c = y > 0.0 ? sky : mix(vec3(0.03,0.06,0.03), vec3(0.05,0.1,0.045), smoothstep(0.0,-0.6,y));
        c += vec3(0.03,0.032,0.04) * (1.0 - smoothstep(0.0, 0.3, abs(y - 0.15)));
        gl_FragColor = vec4(c, 1.0); }`,
    }),
  );
  env.add(dome);
  const lampMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(30, 31, 34) });
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    const lamp = new THREE.Mesh(new THREE.BoxGeometry(6, 1.4, 1.4), lampMat);
    lamp.position.set(Math.cos(a) * 62, 36, Math.sin(a) * 46);
    lamp.lookAt(0, 0, 0);
    env.add(lamp);
  }
  // warm hospitality band
  const band = new THREE.Mesh(
    new THREE.CylinderGeometry(70, 70, 2, 48, 1, true),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(1.2, 0.7, 0.35), side: THREE.BackSide }),
  );
  band.position.y = 12;
  env.add(band);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const rt = pmrem.fromScene(env, 0.02);
  pmrem.dispose();
  return rt.texture;
}

export function createSky(): THREE.Mesh {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(1500, 48, 24),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      vertexShader: "varying vec3 vP; void main(){ vP = position; vec4 p = projectionMatrix*modelViewMatrix*vec4(position,1.0); gl_Position = p.xyww; }",
      fragmentShader: /* glsl */ `varying vec3 vP;
        float h(vec3 p){ return fract(sin(dot(p, vec3(12.9898,78.233,45.164))) * 43758.5453); }
        float n3(vec3 x){ vec3 i=floor(x); vec3 f=fract(x); f=f*f*(3.0-2.0*f);
          return mix(mix(mix(h(i),h(i+vec3(1,0,0)),f.x),mix(h(i+vec3(0,1,0)),h(i+vec3(1,1,0)),f.x),f.y),
                     mix(mix(h(i+vec3(0,0,1)),h(i+vec3(1,0,1)),f.x),mix(h(i+vec3(0,1,1)),h(i+vec3(1,1,1)),f.x),f.y),f.z); }
        void main(){
          vec3 d = normalize(vP);
          float y = d.y;
          vec3 zen = vec3(0.004, 0.006, 0.014);
          vec3 hor = vec3(0.05, 0.04, 0.055);
          vec3 c = mix(hor, zen, smoothstep(-0.05, 0.45, y));
          // clouds lit from below by the stadium and the city
          float cl = n3(d * 5.0) * 0.6 + n3(d * 11.0) * 0.4;
          c += vec3(0.05, 0.045, 0.06) * smoothstep(0.45, 0.85, cl) * (1.0 - smoothstep(0.05, 0.6, y));
          // a few stars
          float st = step(0.9985, h(floor(d * 420.0)));
          c += vec3(0.5) * st * smoothstep(0.25, 0.6, y);
          gl_FragColor = vec4(c, 1.0);
        }`,
    }),
  );
  m.renderOrder = -10;
  m.frustumCulled = false;
  m.name = "sky";
  return m;
}
