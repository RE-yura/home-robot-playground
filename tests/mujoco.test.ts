import { TABLE_HEIGHT } from "../src/simulation/table.ts";
import assert from "node:assert/strict";
import { test } from "node:test";
import { createSimulation } from "../src/simulation/mujoco.ts";

test("seed and Reset reproduce the physical world and clear motion", async () => {
  const sim = await createSimulation();
  try {
    sim.reset(43);
    const initial = sim.observation().objects.map(o => o.position);
    sim.advance(0.1);
    assert.ok(sim.observation().objects.some(o => Math.hypot(...o.velocity) > 0.5));
    sim.reset();
    assert.deepEqual(sim.observation().objects.map(o => o.position), initial);
    assert.ok(sim.observation().objects.every(o => Math.hypot(...o.velocity) === 0));
    sim.reset(44);
    assert.notDeepEqual(sim.observation().objects.map(o => o.position), initial);
    sim.reset(43); sim.advance(2);
    const first = sim.observation().objects.map(o => o.position);
    sim.reset(); sim.advance(2);
    assert.deepEqual(sim.observation().objects.map(o => o.position), first);
  } finally { sim.dispose(); }
});

test("Reset and new seeds keep the mug directly in front and the box in the back-left slot", async () => {
  const sim = await createSimulation();
  try {
    for (const seed of [42, 43, 99]) {
      sim.reset(seed); sim.advance(2);
      const mug = sim.position(sim.mugId), cube = sim.cube(), base = sim.position(sim.baseId);
      assert.ok(Math.abs(mug[0]) < 0.05 && mug[2] > 0.42 && mug[2] < 0.46, "mug occupies the old front box slot");
      assert.ok(cube[0] < -0.5 && cube[2] < -0.24, "box occupies the old mug slot");
      assert.ok(Math.abs(mug[1] - (TABLE_HEIGHT + 0.063)) < 0.003 && Math.abs(cube[1] - (TABLE_HEIGHT + 0.065)) < 0.003, "both objects rest on the table");
      assert.ok(base[2] - mug[2] < base[2] - cube[2], "mug is closer to the chassis than the box");
      sim.setKey("KeyD", true); sim.advance(0.3); sim.reset(); sim.advance(2);
      assert.deepEqual(sim.position(sim.mugId), mug, "Reset restores the same seeded mug placement");
      assert.deepEqual(sim.cube(), cube, "Reset restores the same seeded box placement");
    }
  } finally { sim.dispose(); }
});

test("observations report measured falling object motion and robot joints", async () => {
  const sim = await createSimulation();
  try {
    const before = sim.cube();
    sim.advance(0.1);
    const observed = sim.observation();
    assert.equal(observed.objects.length, 10);
    assert.equal(new Set(observed.objects.map(o => o.type)).size, observed.objects.length, "no duplicate objects in the physical world");
    const cube = observed.objects.find(o => o.type === "cube")!;
    assert.ok(cube.velocity[1] < -0.5);
    assert.ok(cube.position[1] < before[1] - 0.04);
    assert.equal(observed.robot!.joints.length, 5);
  } finally { sim.dispose(); }
});

test("manual keyboard input drives physical base, arm, head and gripper joints", async () => {
  const sim = await createSimulation();
  try {
    const originalHead = sim.q("pan"), originalHand = sim.tcp();
    sim.setKey("KeyD", true); sim.setKey("KeyI", true); sim.setKey("ArrowLeft", true);
    sim.advance(0.5);
    assert.ok(sim.position(sim.baseId)[0] > 0.01);
    assert.ok(sim.tcp()[1] > originalHand[1] + 0.02);
    assert.ok(sim.q("pan") > originalHead + 0.1);
    sim.opening = 0; sim.advance(1.5);
    assert.ok(sim.gripperOpening() < 0.02);
    sim.opening = 1; sim.advance(1.5);
    assert.ok(sim.gripperOpening() > 0.98);
  } finally { sim.dispose(); }
});

test("the head camera's initial central ray sees the table instead of its folded arm", async () => {
  const sim = await createSimulation(), hit = new sim.mj.IntBuffer(1), normal = new sim.mj.DoubleBuffer(3);
  try {
    const k = sim.cameraId * 9, r = sim.data.site_xmat;
    const p = Array.from(sim.data.site_xpos.slice(sim.cameraId * 3, sim.cameraId * 3 + 3)) as number[];
    const distance = sim.mj.mj_ray(sim.model, sim.data, p, [-r[k + 2], -r[k + 5], -r[k + 8]],
      [1, 1, 1, 1, 1, 1], true, sim.id("BODY", "head_tilt_link"), hit, normal);
    assert.ok(distance > 0.5, "the camera is clear of nearby arm surfaces");
    assert.notEqual(sim.model.body_rootid[sim.model.geom_bodyid[hit.GetView()[0]]], sim.baseId);
  } finally { hit.delete(); normal.delete(); sim.dispose(); }
});
