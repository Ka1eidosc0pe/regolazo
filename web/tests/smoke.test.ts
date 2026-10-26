import { describe, expect, it } from "vitest";
import * as THREE from "three";

describe("skeleton", () => {
  it("loads three.js", () => {
    expect(new THREE.Vector3(3, 4, 0).length()).toBe(5);
  });
});
