import assert from "node:assert/strict";
import { test } from "node:test";
import { createSimulation } from "../src/simulation/mujoco.ts";
import { advancePhysicsFrame } from "../src/playground/physics-clock.ts";
import { LearningCapture } from "../src/learning/capture.ts";
import { LearnedRollout } from "../src/learning/learned-rollout.ts";
import { prepareLearningScene } from "../src/learning/scene.ts";
import type { CameraImage } from "../src/simulation/types.ts";

test("replay wall time cannot queue physics that fast-forwards on reset", async () => {
  const sim = await createSimulation(), clock = { accumulator: 0, generation: sim.resetGeneration };
  try {
    sim.restoreFrame(sim.physicsState());
    for (let frame = 0; frame < 100; frame++) advancePhysicsFrame(sim, 0.02, false, clock);
    sim.reset(); const start = sim.data.time;
    const steps = advancePhysicsFrame(sim, 0.016, false, clock);
    assert.equal(steps, 8, "only the current frame's wall time may advance physics");
    assert.ok(Math.abs(sim.data.time - start - 0.016) < 1e-8);
    advancePhysicsFrame(sim, 0.001, false, clock); sim.reset();
    assert.equal(advancePhysicsFrame(sim, 0.001, false, clock), 0, "Reset clears fractional elapsed time too");
  } finally { sim.dispose(); }
});

test("render-gated and paused wall time is discarded before normal cadence resumes", async () => {
  const sim = await createSimulation(), clock = { accumulator: 0, generation: sim.resetGeneration };
  try {
    sim.advance(1);
    const capture = new LearningCapture(sim, { instruction: "Put the mug in the basket.",
      actionForSample: () => ({ armJointTarget: [0.35, -0.2, 0.1, -1.3, 0], gripper: 0.7 }) });
    for (let frame = 0; frame < 100; frame++) assert.equal(advancePhysicsFrame(sim, 0.02, false, clock), 0);
    const start = sim.data.time;
    const pair: CameraImage[] = ["head", "hand"].map(view => ({ view: view as "head" | "hand", timestamp: start,
      width: 320, height: 240, dataUrl: "data:image/jpeg;base64,/9j/AA==" }));
    capture.capture(pair);
    assert.equal(advancePhysicsFrame(sim, 0.016, false, clock), 8);
    assert.ok(Math.abs(sim.data.time - start - 0.016) < 1e-8);
    capture.stop();
    for (let frame = 0; frame < 100; frame++) assert.equal(advancePhysicsFrame(sim, 0.02, true, clock), 0);
    assert.equal(advancePhysicsFrame(sim, 0.016, false, clock), 8);
  } finally { sim.dispose(); }
});

test("a frame ending exactly on a capture boundary refreshes transforms before mesh callbacks", async () => {
  const sim = await createSimulation(), clock = { accumulator: 0, generation: sim.resetGeneration };
  try {
    sim.advance(1);
    const capture = new LearningCapture(sim, { instruction: "Put the mug in the basket.", maxSamples: 2,
      actionForSample: () => ({ armJointTarget: [0.35, -0.2, 0.1, -1.3, 0], gripper: 0.7 }) });
    advancePhysicsFrame(sim, 0, false, clock);
    capture.capture(["head", "hand"].map(view => ({ view: view as "head" | "hand", timestamp: sim.data.time,
      width: 320, height: 240, dataUrl: "data:image/jpeg;base64,/9j/AA==" })));
    advancePhysicsFrame(sim, 0.05, false, clock); advancePhysicsFrame(sim, 0.05, false, clock);
    const positions = Array.from(sim.data.xpos), cameras = Array.from(sim.data.site_xpos);
    sim.mj.mj_kinematics(sim.model, sim.data);
    assert.deepEqual(Array.from(sim.data.xpos), positions, "no extra blocked step may be needed to refresh meshes");
    assert.deepEqual(Array.from(sim.data.site_xpos), cameras);
  } finally { sim.dispose(); }
});

test("learned terminal scoring pauses the current frame and later frames at the exact2ms step", async () => {
  const sim = await createSimulation(), clock = { accumulator: 0, generation: sim.resetGeneration };
  try {
    prepareLearningScene(sim, 42); sim.advance(2);
    const robot = sim.observation().robot!;
    const rollout = new LearnedRollout(sim, "Put the mug in the basket.", {
      infer: async () => Array.from({ length: 16 }, () => [...robot.joints, robot.gripper]), dispose() {},
    });
    rollout.capture(["head", "hand"].map(view => ({ view: view as "head" | "hand", timestamp: sim.data.time,
      width: 320, height: 240, dataUrl: "data:image/jpeg;base64,/9j/AA==" })));
    await Promise.resolve(); await Promise.resolve();
    // Synthetic fixture placement causes the real evaluator to fail on its next step.
    const body = sim.id("BODY", "mug"), q = sim.model.jnt_qposadr[sim.model.body_jntadr[body]];
    sim.data.qpos.set([2, 0.064, 2], q); sim.mj.mj_forward(sim.model, sim.data);
    const start = sim.data.time;
    assert.equal(advancePhysicsFrame(sim, 0.016, () => rollout.stopped, clock), 1);
    assert.equal(rollout.result.reason, "target-fell");
    assert.ok(Math.abs(sim.data.time - start - rollout.result.elapsed) < 1e-8);
    const terminal = sim.physicsState();
    for (let frame = 0; frame < 10; frame++) assert.equal(advancePhysicsFrame(sim, 0.05, () => rollout.stopped, clock), 0);
    assert.deepEqual(sim.physicsState(), terminal, "renderer wall time cannot move a finished learned trial");
    assert.equal(advancePhysicsFrame(sim, 0.002, false, clock), 1, "ordinary physical control can explicitly resume");
  } finally { sim.dispose(); }
});

test("explicit RTC playback speed scales wall time while retaining physical2ms steps", async () => {
  const sim = await createSimulation(), clock = { accumulator: 0, generation: sim.resetGeneration };
  try {
    sim.learningCapture = { playbackRate: .5, captureDue: () => false, capture() {}, stop() {} };
    const before = sim.data.time;
    assert.equal(advancePhysicsFrame(sim, .032, false, clock), 8);
    assert.ok(Math.abs(sim.data.time - before - .016) < 1e-8);
    sim.learningCapture = null;
    assert.equal(advancePhysicsFrame(sim, .016, false, clock), 8);
  } finally { sim.dispose(); }
});
