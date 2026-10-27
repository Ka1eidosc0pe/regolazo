// Goals: round white posts and crossbar, dark back frame, a fine shader net that bulges where the ball hits it.
import * as THREE from "three";
import { GOAL_HEIGHT, GOAL_WIDTH, HALF_L } from "../coords.ts";

const DEPTH_BOTTOM = 2.0;
const DEPTH_TOP = 1.0;
const POST_R = 0.06;

export interface Goal {
  group: THREE.Group;
  /** Sign of the goal line (+1 at x = +52.5). */
  side: number;
  /** Solid posts/bar meshes (label occluders). */
  frame: THREE.Mesh[];
  /** ball position in world coords; dt seconds. */
  update(ball: [number, number, number], dt: number): void;
  reset(): void;
}

function netMaterial(uniforms: Record<string, THREE.IUniform>): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    uniforms: { ...THREE.UniformsLib.fog, ...uniforms },
    fog: true,
    vertexShader: /* glsl */ `
      uniform vec3 uImpact;
      uniform float uStrength;
      uniform vec3 uPush;
      varying vec3 vLocal;
      varying vec3 vWorld;
      #include <fog_pars_vertex>
      void main() {
        vec3 p = position;
        vLocal = p;
        vec4 w = modelMatrix * vec4(p, 1.0);
        float d = length(w.xyz - uImpact);
        float k = exp(-d * d / 0.9) * uStrength;
        w.xyz += uPush * k;
        vWorld = w.xyz;
        vec4 mvPosition = viewMatrix * w;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */ `
      varying vec3 vLocal;
      varying vec3 vWorld;
      #include <fog_pars_fragment>
      float gridLine(vec2 c, float cell, float w) {
        vec2 g = abs(fract(c / cell - 0.5) - 0.5) * cell;
        vec2 fw = fwidth(c) * 0.75;
        vec2 l = 1.0 - smoothstep(vec2(w) - fw, vec2(w) + fw, g);
        // fade the mesh into a soft veil when cells get smaller than a few pixels
        float far = smoothstep(0.02, 0.06, max(fw.x, fw.y));
        return mix(max(l.x, l.y), 0.32, far);
      }
      void main() {
        // hexagon-ish diamond mesh in world coords on whichever plane this fragment lies on
        vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
        vec2 c = abs(n.x) > 0.6 ? vWorld.zy : (abs(n.y) > 0.6 ? vWorld.xz : vWorld.xy);
        c = vec2(c.x + c.y, c.x - c.y) * 0.7071;
        float a = gridLine(c, 0.12, 0.007);
        gl_FragColor = vec4(vec3(0.93, 0.94, 0.96), a * 0.9);
        #include <fog_fragment>
      }`,
  });
}

export function createGoal(side: 1 | -1): Goal {
  const group = new THREE.Group();
  group.name = side > 0 ? "goal-east" : "goal-west";
  const x0 = side * HALF_L;
  const hw = GOAL_WIDTH / 2 + POST_R; // inner faces of the posts are 7.32 apart
  const postMat = new THREE.MeshPhysicalMaterial({ color: 0xf4f6f8, roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.15, envMapIntensity: 1.0 });
  const backMat = new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: 0.5, metalness: 0.6 });
  const frame: THREE.Mesh[] = [];
  const cyl = (len: number, r: number, mat: THREE.Material) => {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 20), mat);
    m.castShadow = true;
    return m;
  };
  // posts (world y = +/-hw -> three z = -/+hw)
  for (const s of [-1, 1]) {
    const post = cyl(GOAL_HEIGHT + POST_R, POST_R, postMat);
    post.position.set(x0, (GOAL_HEIGHT + POST_R) / 2, s * hw);
    group.add(post);
    frame.push(post);
  }
  const bar = cyl(hw * 2 + POST_R * 2, POST_R, postMat);
  bar.rotation.x = Math.PI / 2;
  bar.position.set(x0, GOAL_HEIGHT + POST_R, 0);
  group.add(bar);
  frame.push(bar);
  // back frame: top-back bar, back uprights, ground bars, roof stays
  const xb = x0 + side * DEPTH_BOTTOM;
  const xt = x0 + side * DEPTH_TOP;
  const thin = 0.025;
  const add = (a: THREE.Vector3, b: THREE.Vector3) => {
    const len = a.distanceTo(b);
    const m = cyl(len, thin, backMat);
    m.position.copy(a).add(b).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
    group.add(m);
  };
  const H = GOAL_HEIGHT + POST_R;
  for (const s of [-1, 1]) {
    add(new THREE.Vector3(x0, H, s * hw), new THREE.Vector3(xt, H, s * hw));
    add(new THREE.Vector3(xt, H, s * hw), new THREE.Vector3(xb, 0.02, s * hw));
    add(new THREE.Vector3(x0, 0.02, s * hw), new THREE.Vector3(xb, 0.02, s * hw));
  }
  add(new THREE.Vector3(xt, H, -hw), new THREE.Vector3(xt, H, hw));
  add(new THREE.Vector3(xb, 0.02, -hw), new THREE.Vector3(xb, 0.02, hw));

  // net surfaces (segmented so the bulge is smooth)
  const uniforms = {
    uImpact: { value: new THREE.Vector3(0, -100, 0) },
    uStrength: { value: 0 },
    uPush: { value: new THREE.Vector3(side, 0.15, 0) },
  };
  const nm = netMaterial(uniforms);
  const quad = (p00: THREE.Vector3, p10: THREE.Vector3, p01: THREE.Vector3, p11: THREE.Vector3, nu: number, nv: number) => {
    const g = new THREE.PlaneGeometry(1, 1, nu, nv);
    const pos = g.getAttribute("position") as THREE.BufferAttribute;
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      const u = pos.getX(i) + 0.5;
      const w = pos.getY(i) + 0.5;
      const a = p00.clone().lerp(p10, u);
      const b = p01.clone().lerp(p11, u);
      v.copy(a).lerp(b, w);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, nm);
    m.renderOrder = 5;
    group.add(m);
  };
  // back
  quad(new THREE.Vector3(xb, 0, -hw), new THREE.Vector3(xb, 0, hw), new THREE.Vector3(xt, H, -hw), new THREE.Vector3(xt, H, hw), 28, 12);
  // roof
  quad(new THREE.Vector3(x0, H, -hw), new THREE.Vector3(x0, H, hw), new THREE.Vector3(xt, H, -hw), new THREE.Vector3(xt, H, hw), 28, 6);
  // sides
  for (const s of [-1, 1]) {
    quad(new THREE.Vector3(x0, 0, s * hw), new THREE.Vector3(xb, 0, s * hw), new THREE.Vector3(x0, H, s * hw), new THREE.Vector3(xt, H, s * hw), 10, 12);
  }

  let strength = 0;
  const impact = new THREE.Vector3();
  return {
    group,
    side,
    frame,
    reset() {
      strength = 0;
      uniforms.uStrength.value = 0;
    },
    update(ball, dt) {
      const [bx, by, bz] = ball;
      const inside = side * bx > HALF_L && Math.abs(by) < hw && bz < H;
      if (inside) {
        const depth = side * bx - HALF_L;
        const target = Math.max(0, depth - DEPTH_TOP * 0.6) * 0.6;
        if (target > strength) {
          strength = Math.min(0.45, target);
          impact.set(bx, bz, -by);
        }
      }
      strength *= Math.exp(-dt * 2.2);
      uniforms.uStrength.value = strength;
      uniforms.uImpact.value.copy(impact);
    },
  };
}
