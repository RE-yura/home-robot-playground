import { TABLE_HEIGHT } from "../src/simulation/table.ts";
import assert from "node:assert/strict";
import { test } from "node:test";
import { createSimulation } from "../src/simulation/mujoco.ts";

test("MuJoCo loads the HSR scene and a falling cube settles on the table", async () => {
  const sim = await createSimulation();
  assert.ok(sim, "MuJoCo simulation must initialize");
  try {
    assert.ok(sim.cube()[1] > TABLE_HEIGHT + 0.2);
    sim.advance(2);
    assert.ok(Math.abs(sim.cube()[1] - (TABLE_HEIGHT + 0.065)) < 0.005, "cube settles on table instead of falling through");
    sim.reset();
    assert.ok(sim.cube()[1] > TABLE_HEIGHT + 0.2, "reset restores initial height");
  } finally { sim.dispose(); }
});

test("finger friction lifts the front mug by 10 cm, holds 3 seconds, and opening releases it", async () => {
  const sim = await createSimulation();
  try {
    sim.startProbe("grasp"); sim.advance(11);
    assert.equal(sim.gripperContactId, "mug", "both fingers contact the front mug during the hold");
    sim.advance(6.5);
    assert.ok(sim.minHoldHeight > TABLE_HEIGHT + 0.163, "mug stays at least 10 cm above resting height during all 3 seconds");
    assert.ok(Math.abs(sim.position(sim.mugId)[1] - (TABLE_HEIGHT + 0.063)) < 0.012, "opened HSR fingers release mug onto table");
    assert.equal(sim.gripperContactId, null, "mug is no longer attached to the fingers");
    for (const type of sim.model.eq_type) assert.equal(type, sim.mj.mjtEq.mjEQ_JOINT.value, "only internal HSR joints are coupled; no object attachment");
  } finally { sim.dispose(); }
});

test("force-limited arm contacts table without sustained large penetration", async () => {
  const sim = await createSimulation();
  try {
    sim.startProbe("contact"); sim.advance(9);
    console.log({ tablePenetration: sim.maxTablePenetration, tcp: sim.tcp() });
    assert.ok(sim.maxTablePenetration > 0, "probe actually contacts the table");
    assert.ok(sim.maxTablePenetration < 0.003, "table penetration stays below 3 mm");
    assert.ok(sim.tcp()[1] > TABLE_HEIGHT - 0.04, "table prevents the requested target below the tabletop");
  } finally { sim.dispose(); }
});

test("changing robot mass scales its inertia but preserves object masses and Reset keeps the settings", async () => {
  const sim = await createSimulation();
  try {
    const originalMass = sim.model.body_subtreemass[sim.baseId];
    const originalInertia = sim.model.body_inertia[sim.baseId * 3];
    sim.setParameters(originalMass * 2, 60);
    assert.ok(Math.abs(sim.model.body_subtreemass[sim.baseId] - originalMass * 2) < 1e-8);
    assert.ok(Math.abs(sim.model.body_inertia[sim.baseId * 3] - originalInertia * 2) < 1e-8);
    assert.equal(sim.model.body_mass[sim.cubeId], 1);
    assert.equal(sim.model.body_mass[sim.id("BODY", "ball")], 0.058);
    sim.advance(1); sim.reset();
    assert.equal(sim.armTorqueNm, 60);
    assert.ok(Math.abs(sim.robotMassKg - originalMass * 2) < 1e-8);
    assert.equal(sim.peakTableForce, 0);
  } finally { sim.dispose(); }
});

test("a stronger motor limit increases measured table force while joint torques remain bounded", async () => {
  const sim = await createSimulation();
  try {
    const forces: number[] = [];
    for (const limit of [10, 100]) {
      sim.setParameters(35, limit); sim.startProbe("contact");
      for (let step = 0; step < 4500; step++) {
        sim.step();
        for (const joint of ["arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"])
          assert.ok(Math.abs(sim.data.actuator_force[sim.joints[joint].actuator]) <= limit + 1e-8);
      }
      assert.ok(Number.isFinite(sim.peakTableForce));
      if (limit === 100) assert.ok(sim.peakTableForce > 10, "a motor strong enough to carry the arm presses against the table");
      else assert.ok(sim.q("arm_flex_joint") < sim.armCommand[1] - 0.4, "a 10 Nm motor cannot support this outstretched arm against gravity");
      assert.ok(sim.maxTablePenetration < 0.005, "contact remains stable rather than tunnelling");
      forces.push(sim.meanTableForce());
    }
    assert.ok(forces[1] > forces[0] * 1.3, `stronger press should increase contact force: ${forces}`);
  } finally { sim.dispose(); }
});

test("a deeper push target increases sustained force but the real hand stays above the table", async () => {
  const sim = await createSimulation();
  try {
    sim.setParameters(35, 100); sim.contactTargetY = TABLE_HEIGHT + 0.02; sim.startProbe("contact"); sim.advance(9);
    const shallowForce = sim.meanTableForce();
    sim.contactTargetY = TABLE_HEIGHT - 0.25; sim.startProbe("contact"); sim.advance(9);
    assert.ok(sim.meanTableForce() > shallowForce * 1.3, "a deeper goal actually commands more force");
    assert.ok(sim.tcp()[1] > TABLE_HEIGHT - 0.04 && sim.maxTablePenetration < 0.005);
  } finally { sim.dispose(); }
});

test("higher robot mass slows the same force-limited base command", async () => {
  const sim = await createSimulation();
  try {
    const distances: number[] = [];
    for (const mass of [20, 80]) {
      sim.setParameters(mass, 40); const start = sim.position(sim.baseId)[0];
      sim.setKey("KeyD", true); sim.advance(0.2);
      distances.push(sim.position(sim.baseId)[0] - start);
    }
    assert.ok(distances[0] > distances[1] * 2 && distances[1] > 0, "mass must affect real dynamics, not just a label");
  } finally { sim.dispose(); }
});
