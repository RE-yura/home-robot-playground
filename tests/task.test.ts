import assert from "node:assert/strict";
import { test } from "node:test";
import { TaskEvaluator } from "../src/simulation/task.ts";
import { mugBounds } from "../src/simulation/mug-geometry.ts";
import type { Observation, Quaternion, Vec3 } from "../src/simulation/types.ts";

function observation(mug: Vec3 = [0.48, 0.475, -0.2], basket: Vec3 = [0.48, 0.425, -0.2], rotation: Quaternion = [0, 0, 0, 1]): Observation {
  return { timestamp: 0, objects: [{ id: "mug", type: "mug", position: mug, rotation: [0, 0, 0, 1], velocity: [0, 0, 0] }],
    furniture: [{ id: "basket", position: basket, rotation }], robot: { base: { position: [0, 0.1, 1], yaw: 0 }, joints: [], gripper: 1,
      endEffectorPosition: [0, 0.5, 1], endEffectorRotation: [0, 0, 0, 1], gripperContact: false, tableForce: 0, contactId: null,
      collision: { force: 0, count: 0, selfCount: 0 } } };
}

test("cup task succeeds only after a released cup fits entirely below the rim and settles for2seconds", () => {
  const task = new TaskEvaluator(), obs = observation();
  task.start(obs);
  assert.equal(task.update(obs, 1.9).status, "running");
  assert.equal(task.update(obs, 0.1).status, "success");
  assert.equal(task.result.elapsed, 2);
  assert.equal(task.update(observation([0, 0.06, 0]), 5).status, "success", "finished evaluation is stable");
});

test("a stationary physical violation takes priority over settlement on the same2ms step", () => {
  const task = new TaskEvaluator(), obs = observation();
  task.start(obs); task.update(obs,1.998);
  obs.robot!.collision!.force = 30;
  const result = task.update(obs,.002,"base-drift");
  assert.equal(result.status,"failed"); assert.equal(result.reason,"base-drift");
  assert.equal(result.elapsed,2); assert.equal(result.collisions,1);
  assert.equal(result.settledFor,0);
  assert.deepEqual(task.update(obs,2),result,"a terminal physical failure stays latched");
});

test("cup center inside the basket is insufficient when its handle protrudes or its top exceeds the rim", () => {
  for (const mug of [[0.57, 0.475, -0.2], [0.48, 0.56, -0.2]] as Vec3[]) {
    const task = new TaskEvaluator(), obs = observation(mug); task.start(obs);
    assert.equal(task.update(obs, 3).status, "running");
    assert.equal(task.result.settledFor, 0);
  }
});

test("task follows a moved yawed basket and rejects cup centers in a sideways basket", () => {
  const yaw = Math.SQRT1_2, moved = observation([1, 0.675, 2], [1, 0.625, 2], [0, yaw, 0, yaw]);
  const task = new TaskEvaluator(); task.start(moved);
  assert.equal(task.update(moved, 2).status, "success");
  const sideways = observation([0.48, 0.475, -0.2], undefined, [0, 0, yaw, yaw]);
  task.start(sideways);
  assert.equal(task.update(sideways, 3).status, "running");
});

test("gripper contact and motion interrupt the continuous settling interval", () => {
  const task = new TaskEvaluator(), obs = observation(); task.start(obs); task.update(obs, 1.9);
  obs.robot!.gripperContact = true; obs.robot!.contactId = "mug";
  assert.equal(task.update(obs, 1).status, "running");
  assert.equal(task.result.settledFor, 0);
  obs.robot!.gripperContact = false; obs.robot!.contactId = null; obs.objects[0].velocity = [0.04, 0, 0];
  assert.equal(task.update(obs, 3).status, "running");
  obs.objects[0].velocity = [0, 0, 0];
  assert.equal(task.update(obs, 1.9).status, "running");
  assert.equal(task.update(obs, 0.1).status, "success");
});

test("falling onto the floor fails once, while initial tabletop settling does not count as a drop", () => {
  const task = new TaskEvaluator(), obs = observation([0, 0.71, 0.44]); task.start(obs);
  obs.objects[0].position[1] = 0.463; obs.objects[0].velocity[1] = -0.02;
  assert.equal(task.update(obs, 1).status, "running");
  assert.equal(task.result.drops, 0);
  obs.objects[0].position[1] = 0.063;
  assert.equal(task.update(obs, 1).status, "failed");
  assert.equal(task.result.reason, "target-fell");
  assert.equal(task.result.drops, 1);
  task.update(obs, 1); assert.equal(task.result.drops, 1);
});

test("late gravity counts one drop after the grasp has been absent for 100 ms, while release over the basket does not", () => {
  const task = new TaskEvaluator(), obs = observation([0, 0.8, 0.44]);
  obs.robot!.gripperContact = true; obs.robot!.contactId = "mug"; task.start(obs);
  obs.robot!.gripperContact = false; obs.robot!.contactId = null; task.update(obs, 0.12);
  assert.equal(task.result.drops, 0, "an unheld stationary mug is not a falling mug");
  obs.objects[0].velocity = [0, -0.2, 0]; task.update(obs, 0.02);
  assert.equal(task.result.drops, 1);
  task.update(obs, 0.2); assert.equal(task.result.drops, 1);
  obs.objects[0].position = [0.48, 0.8, -0.2]; obs.robot!.gripperContact = true; obs.robot!.contactId = "mug"; task.start(obs);
  obs.robot!.gripperContact = false; obs.robot!.contactId = null; task.update(obs, 0.2);
  assert.equal(task.result.drops, 0);
});

test("a 20 ms contact telemetry gap during native seed 7 transport is not a drop and recovered contact resets the loss timer", () => {
  const task = new TaskEvaluator(), obs = observation([0.4897977785, 0.7960324889, 1.1635272377]);
  obs.objects[0].rotation = [-0.0369459278, 0.7032136681, 0.0445400596, 0.7086195866];
  obs.objects[0].velocity = [-0.0698690781, -0.0636, -0.3015502129];
  obs.robot!.gripperContact = true; obs.robot!.contactId = "mug"; task.start(obs); task.update(obs, 0.02);
  obs.robot!.gripperContact = false; obs.robot!.contactId = null; obs.objects[0].velocity[1] = -0.0923;
  task.update(obs, 0.02); assert.equal(task.result.drops, 0);
  obs.robot!.gripperContact = true; obs.robot!.contactId = "mug"; obs.objects[0].velocity[1] = -0.0685;
  task.update(obs, 0.02); assert.equal(task.result.drops, 0);
  obs.robot!.gripperContact = false; obs.robot!.contactId = null;
  task.update(obs, 0.08); assert.equal(task.result.drops, 0, "the earlier 20 ms gap must not accumulate after regrasp");
});

test("a continuously unheld cup descending outside the basket counts once only after 100 ms", () => {
  const task = new TaskEvaluator(), obs = observation([0, 0.8, 0.44]);
  obs.robot!.gripperContact = true; obs.robot!.contactId = "mug"; task.start(obs);
  obs.robot!.gripperContact = false; obs.robot!.contactId = null; obs.objects[0].velocity = [0, -0.2, 0];
  for (const dt of [0.04, 0.04]) {
    obs.objects[0].position[1] -= 0.2 * dt; task.update(obs, dt); assert.equal(task.result.drops, 0);
  }
  obs.objects[0].position[1] -= 0.004; task.update(obs, 0.02); assert.equal(task.result.drops, 1);
  obs.objects[0].position[1] -= 0.1; task.update(obs, 0.5); assert.equal(task.result.drops, 1);
});

test("reaching the floor fails and counts a drop immediately even during the 100 ms loss confirmation", () => {
  const task = new TaskEvaluator(), obs = observation([0, 0.8, 0.44]);
  obs.robot!.gripperContact = true; obs.robot!.contactId = "mug"; task.start(obs);
  obs.robot!.gripperContact = false; obs.robot!.contactId = null; obs.objects[0].position[1] = 0.063;
  assert.equal(task.update(obs, 0.02).status, "failed");
  assert.equal(task.result.reason, "target-fell"); assert.equal(task.result.drops, 1);
});

test("a mug resting on its sideways handle at floor height also fails", () => {
  const task = new TaskEvaluator(), obs = observation([0, 0.106, 0]);
  obs.objects[0].rotation = [0, 0, -Math.SQRT1_2, Math.SQRT1_2]; task.start(obs);
  assert.equal(task.result.status, "failed");
  assert.equal(task.result.reason, "target-fell");
  assert.equal(task.result.drops, 1);
});

test("collision metrics count force episodes, excluding gentle contact and sustained contact repeats", () => {
  const task = new TaskEvaluator(), obs = observation([0, 0.463, 0.44]); task.start(obs);
  obs.robot!.collision = { force: 10, count: 10, selfCount: 0 }; task.update(obs, 0.1);
  assert.equal(task.result.collisions, 0);
  obs.robot!.collision = { force: 35, count: 10, selfCount: 0 };
  for (let i = 0; i < 100; i++) task.update(obs, 0.002);
  assert.equal(task.result.collisions, 1);
  obs.robot!.collision.force = 0; task.update(obs, 0.1);
  obs.robot!.collision.force = 35; task.update(obs, 0.1);
  assert.equal(task.result.collisions, 2);
});

test("tipping and timeout fail the task, and reset clears every metric", () => {
  const task = new TaskEvaluator(), obs = observation([0, 0.463, 0.44]); task.start(obs);
  obs.robot!.base.tipped = true;
  assert.equal(task.update(obs, 0.1).reason, "robot-tipped");
  assert.deepEqual(task.reset(), { status: "ready", elapsed: 0, collisions: 0, drops: 0, settledFor: 0 });
  obs.robot!.base.tipped = false; task.start(obs);
  assert.equal(task.update(obs, 120).reason, "timeout");
  assert.throws(() => task.update(obs, Number.NaN), /finite|time/i);
});

test("measured base rotation detects tipping when the optional tilt flag is absent", () => {
  const task = new TaskEvaluator(), obs = observation([0, 0.463, 0.44]);
  obs.robot!.base.rotation = [Math.SQRT1_2, 0, 0, Math.SQRT1_2]; task.start(obs);
  assert.equal(task.result.reason, "robot-tipped");
});

test("the actual sideways mug resting on the basket floor succeeds after two released seconds", () => {
  const task = new TaskEvaluator(), obs = observation([0.5571981048, 0.4673075879, -0.1935505269]);
  obs.objects[0].rotation = [0.2714421463, 0.4634677136, -0.6529302963, 0.5340401370];
  obs.robot!.gripperContact = true; obs.robot!.contactId = "mug"; task.start(obs);
  assert.equal(task.update(obs, 3).status, "running", "being supported by the basket does not make a held mug released");
  obs.robot!.gripperContact = false; obs.robot!.contactId = null;
  assert.equal(task.update(obs, 1.9).status, "running");
  assert.equal(task.update(obs, 0.1).status, "success");
  assert.equal(task.result.drops, 0);
});

test("a sideways mug more than4mm through the basket floor still cannot succeed", () => {
  const task = new TaskEvaluator(), obs = observation([0.5571981048, 0.4623075879, -0.1935505269]);
  obs.objects[0].rotation = [0.2714421463, 0.4634677136, -0.6529302963, 0.5340401370]; task.start(obs);
  assert.equal(task.update(obs, 3).status, "running");
  assert.equal(task.result.settledFor, 0);
});

// A sideways mug moved along the basket's local z until its outline reaches `zMin` (basket at the fixture pose).
function sidewaysAt(zMin: number): Observation {
  const obs = observation([0.5571981048, 0.4673075879, -0.1935505269]);
  obs.objects[0].rotation = [0.2714421463, 0.4634677136, -0.6529302963, 0.5340401370];
  const basket = obs.furniture[0], bounds = mugBounds(obs.objects[0], basket);
  obs.objects[0].position = [obs.objects[0].position[0], obs.objects[0].position[1], obs.objects[0].position[2] + zMin - bounds.min[2]];
  return obs;
}

test("with the walls criterion a mug lying against the side bars inside the basket succeeds and the rim result is kept", () => {
  const obs = sidewaysAt(-0.168), walls = new TaskEvaluator("mug", { success: "walls" }), rim = new TaskEvaluator();
  walls.start(obs); rim.start(obs);
  assert.equal(walls.update(obs, 2).status, "success");
  assert.equal(walls.result.rimSettledFor, 0, "the original rim criterion was never met");
  assert.equal(rim.update(obs, 3).status, "running", "the default criterion is unchanged");
});

test("a mug pressed into the bars' contact layer still counts, one past the bars' middle does not", () => {
  // Soft contacts let a resting mug sink below a bar's inner face (-0.171); only the bar centre line (-0.175) is a limit.
  const pressed = new TaskEvaluator("mug", { success: "walls" }), obs = sidewaysAt(-0.1735); pressed.start(obs);
  assert.equal(pressed.update(obs, 2).status, "success");
  const through = new TaskEvaluator("mug", { success: "walls" }), past = sidewaysAt(-0.176); through.start(past);
  assert.equal(through.update(past, 3).status, "running");
});

test("the walls criterion agrees with the rim inside the opening", () => {
  const inside = new TaskEvaluator("mug", { success: "walls" }), ok = sidewaysAt(-0.12); inside.start(ok);
  assert.equal(inside.update(ok, 2).status, "success");
  assert.ok(Math.abs(inside.result.rimSettledFor! - 2) < 1e-9);
});

test("an evaluator without a target never succeeds, even with the mug resting in the basket, and times out at 120 s", () => {
  const obs = sidewaysAt(-0.12), none = new TaskEvaluator("none"); none.start(obs);
  assert.equal(none.target, null);
  assert.equal(none.update(obs, 3).status, "running");
  assert.equal(none.result.settledFor, 0);
  assert.equal(none.update(obs, 117.1).status, "failed"); assert.equal(none.result.reason, "timeout");
  assert.throws(() => new TaskEvaluator("none", { success: "walls" }), /mug/);
});
