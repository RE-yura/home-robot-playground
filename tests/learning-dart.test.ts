import assert from "node:assert/strict";
import { test } from "node:test";
import { DartNoise, DART_SIGMA, DART_WINDOW_SAMPLES, dartActive, perturbGoal } from "../src/learning/dart.ts";
import { ARM_LIMITS } from "../src/robot/hsr-model.ts";

test("DART offsets are seeded AR(1) with the configured spread during the approach and never move the gripper", () => {
  const run = (seed: number) => { const n = new DartNoise(seed); return Array.from({ length: 4000 }, () => n.next(true)); };
  const a = run(1234), b = run(1234);
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, run(1235));
  for (let i = 0; i < 5; i++) {
    const values = a.map(v => v[i]), std = Math.sqrt(values.reduce((s, v) => s + v * v, 0) / values.length);
    assert.ok(Math.abs(std - DART_SIGMA[i]) < 0.2 * DART_SIGMA[i], `channel ${i} std ${std}`);
    const lag = values.slice(1).reduce((s, v, k) => s + v * values[k], 0) / values.slice(1).reduce((s, _, k) => s + values[k] ** 2, 0);
    assert.ok(Math.abs(lag - 0.75) < 0.06, `channel ${i} lag-1 correlation ${lag}`);
  }
  assert.ok(a.every(v => v[5] === 0));
});

test("noise is injected only in the first 3 s of the teacher (the approach) and afterwards only decays", () => {
  assert.equal(DART_WINDOW_SAMPLES, 30);
  const n = new DartNoise(7); let last = [0, 0, 0, 0, 0, 0];
  for (let i = 0; i < DART_WINDOW_SAMPLES; i++) last = n.next(dartActive(i));
  const peak = Math.max(...last.map(Math.abs));
  assert.ok(peak > 0);
  let after = last;
  for (let i = DART_WINDOW_SAMPLES; i < DART_WINDOW_SAMPLES + 10; i++) { assert.equal(dartActive(i), false); after = n.next(dartActive(i)); }
  assert.ok(Math.max(...after.map(Math.abs)) <= peak * 0.75 ** 10 + 1e-12, "decayed to under 6 % before the close at 4 s");
});

test("a perturbed goal stays inside the joint limits", () => {
  const goal = { armJointTarget: [0.69, -2.62, 0, 1.22, 0] as [number, number, number, number, number], gripper: 0.8 };
  const out = perturbGoal(goal, [0.05, -0.1, 0.02, 0.1, -0.03, 0]);
  assert.deepEqual(out.armJointTarget, [0.69, -2.62, 0.02, 1.22, -0.03]);
  assert.equal(out.gripper, 0.8);
  assert.ok(out.armJointTarget.every((v, i) => v >= ARM_LIMITS[i][0] && v <= ARM_LIMITS[i][1]));
});

test("perturbed execution records the clean teacher goal", async () => {
  const { createSimulation } = await import("../src/simulation/mujoco.ts");
  const { prepareLearningScene } = await import("../src/learning/scene.ts");
  const { LearningCapture } = await import("../src/learning/capture.ts");
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 1000); sim.advance(2);
    const clean = { armJointTarget: [0.3, -0.5, 0, -1, 0] as [number, number, number, number, number], gripper: 0.8 };
    const capture = new LearningCapture(sim, { instruction: "Put the mug in the basket.", maxSamples: 2, actionForSample: () => clean,
      perturb: (goal, index) => { assert.equal(index, 0); return perturbGoal(goal, [0.01, 0.03, 0, 0, 0, 0]); } });
    capture.capture(["head", "hand"].map(view => ({ view: view as "head" | "hand", timestamp: sim.data.time, width: 320, height: 240, dataUrl: "data:image/jpeg;base64,/9j/AA==" })));
    assert.equal(sim.step(), true);
    assert.ok(Math.abs(sim.lastAction.armJointTarget![0] - 0.31) < 1e-12 && Math.abs(sim.lastAction.armJointTarget![1] + 0.47) < 1e-12, "the executed command is perturbed");
    assert.deepEqual(capture.latestSample!.action, [0.3, -0.5, 0, -1, 0, 0.8], "the recorded label is the clean goal");
    capture.stop();
  } finally { sim.dispose(); }
});
