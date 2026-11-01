import assert from "node:assert/strict";
import { test } from "node:test";
import { Quaternion as ThreeQuaternion, Vector3 } from "three";
import { mugBounds, mugFitsBasket } from "../src/simulation/mug-geometry.ts";
import { createSimulation } from "../src/simulation/mujoco.ts";
import type { Quaternion, Vec3 } from "../src/simulation/types.ts";

const sideways = { position: [0.5571981048, 0.4673075879, -0.1935505269] as Vec3,
  rotation: [0.2714421463, 0.4634677136, -0.6529302963, 0.5340401370] as Quaternion };
const basket = { position: [0.48, 0.425, -0.2] as Vec3, rotation: [0, 0, 0, 1] as Quaternion };

test("upright mug bounds retain the real bottom, segmented walls and full handle", () => {
  const bounds = mugBounds({ position: [0, 0, 0], rotation: [0, 0, 0, 1] });
  for (const [values, expected] of [[bounds.min, [-0.055, -0.063, -0.055]], [bounds.max, [0.106, 0.06, 0.055]]]) {
    values.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-12));
  }
});

test("sideways geometry rests at the measured basket floor without relaxing floor tolerance or clearance", () => {
  assert.equal(mugFitsBasket(sideways, basket), true);
  assert.ok(Math.abs(mugBounds(sideways).min[1] - 0.412) < 0.00001);
  assert.equal(mugFitsBasket({ ...sideways, position: [sideways.position[0], sideways.position[1] - 0.005, sideways.position[2]] }, basket), false);
  const high = { position: [0.48, 0.8, -0.2] as Vec3, rotation: [0, 0, 0, 1] as Quaternion };
  assert.equal(mugFitsBasket(high, basket), false);
  assert.equal(mugFitsBasket(high, basket, true), true);
  assert.equal(mugFitsBasket({ ...high, position: [0.57, 0.475, -0.2] }, basket, true), false, "horizontal check still contains the handle");
  const inset = { ...high, position: [0.54, 0.475, -0.2] as Vec3 };
  assert.equal(mugFitsBasket(inset, basket, true), true);
  assert.equal(mugFitsBasket(inset, basket, true, 0.015), false, "controller inset margin is preserved");
});

test("geometry bounds match the loaded MuJoCo mug shapes in world and rotated basket frames", async () => {
  const sim = await createSimulation();
  try {
    const body = sim.mugId, q = sim.model.jnt_qposadr[sim.model.body_jntadr[body]];
    const frame = { position: [1, 0.625, 2] as Vec3, rotation: new ThreeQuaternion().setFromAxisAngle(new Vector3(1, 2, 3).normalize(), 0.7).toArray() as Quaternion };
    for (const pose of [{ position: [0.48, 0.475, -0.2] as Vec3, rotation: [0, 0, 0, 1] as Quaternion }, sideways]) {
      const rotation = new ThreeQuaternion(...pose.rotation).normalize();
      sim.data.qpos.set([...pose.position, rotation.w, rotation.x, rotation.y, rotation.z], q);
      sim.mj.mj_forward(sim.model, sim.data);
      for (const reference of [undefined, basket, frame]) {
        const inverse = new ThreeQuaternion(...(reference?.rotation ?? [0, 0, 0, 1])).invert();
        const origin = new Vector3(...(reference?.position ?? [0, 0, 0]));
        const points: Vector3[] = [];
        const first = sim.model.body_geomadr[body], count = sim.model.body_geomnum[body];
        assert.equal(count, 28);
        for (let geom = first; geom < first + count; geom++) {
          const size = sim.model.geom_size.slice(geom * 3, geom * 3 + 3), matrix = sim.data.geom_xmat.slice(geom * 9, geom * 9 + 9);
          const position = sim.data.geom_xpos.slice(geom * 3, geom * 3 + 3);
          const add = (x: number, y: number, z: number) => points.push(new Vector3(
            position[0] + matrix[0] * x + matrix[1] * y + matrix[2] * z,
            position[1] + matrix[3] * x + matrix[4] * y + matrix[5] * z,
            position[2] + matrix[6] * x + matrix[7] * y + matrix[8] * z).sub(origin).applyQuaternion(inverse));
          if (sim.model.geom_type[geom] === sim.mj.mjtGeom.mjGEOM_BOX.value) {
            for (const x of [-size[0], size[0]]) for (const y of [-size[1], size[1]]) for (const z of [-size[2], size[2]]) add(x, y, z);
          } else {
            assert.equal(sim.model.geom_type[geom], sim.mj.mjtGeom.mjGEOM_CYLINDER.value);
            // Sample the real compiled cylinder's two circles independently of the helper's analytic support.
            for (let i = 0; i < 4096; i++) {
              const angle = i * Math.PI * 2 / 4096;
              for (const z of [-size[1], size[1]]) add(size[0] * Math.cos(angle), size[0] * Math.sin(angle), z);
            }
          }
        }
        const actual = mugBounds(pose, reference);
        for (const [axis, key] of ["x", "y", "z"].entries()) {
          const values = points.map(point => point[key as "x" | "y" | "z"]);
          assert.ok(Math.abs(actual.min[axis] - Math.min(...values)) < 5e-8, `minimum ${key} matches compiled geometry`);
          assert.ok(Math.abs(actual.max[axis] - Math.max(...values)) < 5e-8, `maximum ${key} matches compiled geometry`);
        }
      }
    }
  } finally { sim.dispose(); }
});
