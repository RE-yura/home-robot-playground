import assert from "node:assert/strict";
import { test } from "node:test";
import { createSimulation } from "../src/simulation/mujoco.ts";
import { LearnedRollout, type LearnedInput, type LearnedPolicy } from "../src/learning/learned-rollout.ts";
import type { CameraImage } from "../src/simulation/types.ts";
import { prepareLearningScene } from "../src/learning/scene.ts";

const pair = (timestamp: number): CameraImage[] => ["head", "hand"].map(view => ({
  view: view as "head" | "hand", timestamp, width: 320, height: 240, dataUrl: "data:image/jpeg;base64,/9j/AA==",
}));
const chunk = () => Array.from({ length: 16 }, (_, i) => [0.31 + i * 0.001, -0.2, 0.1, -1.3, 0, 0.7]);
function pendingPolicy() {
  let resolve!: (value: number[][]) => void;
  let input: LearnedInput | null = null;
  const promise = new Promise<number[][]>(done => { resolve = done; });
  const policy: LearnedPolicy = { infer: value => { input = value; return promise; }, dispose() {} };
  return { policy, resolve, input: () => input };
}

test("page-owned policy survives Stop and a second task without resetting the physical scene", async () => {
  const sim = await createSimulation(); let disposed = 0;
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const generation = sim.resetGeneration, before = sim.physicsState();
    const pending = pendingPolicy(); pending.policy.dispose = () => { disposed++; };
    const first = new LearnedRollout(sim, "Put the mug in the basket.", pending.policy, { disposePolicy: false });
    first.capture(pair(sim.data.time)); first.stop();
    assert.equal(disposed, 0, "ending one task must not destroy the page-owned model");
    const second = new LearnedRollout(sim, "Put the tennis ball in the basket.", pending.policy, { disposePolicy: false });
    const arm = [...sim.armCommand]; pending.resolve(chunk()); await Promise.resolve(); await Promise.resolve();
    assert.equal(first.chunks.length, 0, "late first-task reply cannot execute in the second task");
    assert.deepEqual(sim.armCommand, arm); assert.equal(sim.resetGeneration, generation);
    assert.deepEqual(sim.physicsState(), before, "starting tasks preserves objects and physical time");
    second.stop(); assert.equal(disposed, 0); pending.policy.dispose(); assert.equal(disposed, 1);
  } finally { sim.dispose(); }
});

test("standalone diagnostic rollouts retain policy-disposal ownership by default", async () => {
  const sim = await createSimulation(); let disposed = 0;
  try {
    const p = pendingPolicy(); p.policy.dispose = () => { disposed++; };
    const r = new LearnedRollout(sim, "Put the mug in the basket.", p.policy);
    r.stop(); r.stop(); assert.equal(disposed, 1);
  } finally { sim.dispose(); }
});

test("learned inference freezes physics, exposes only fresh RGB/instruction/measured state, and replans after all sixteen 100ms actions", async () => {
  const sim = await createSimulation();
  try {
    sim.advance(1); const p = pendingPolicy(), measured = sim.observation().robot!;
    const rollout = new LearnedRollout(sim, "Put the mug in the basket.", p.policy);
    const snapshot = sim.physicsState();
    assert.equal(sim.step(), false);
    rollout.capture(pair(sim.data.time));
    sim.advance(1); assert.deepEqual(sim.physicsState(), snapshot);
    assert.deepEqual(Object.keys(p.input()!).sort(), ["images", "instruction", "state"]);
    assert.deepEqual(p.input()!.state, [...measured.joints, measured.gripper]);
    assert.equal(p.input()!.instruction, "Put the mug in the basket.");
    p.resolve(chunk()); await Promise.resolve(); await Promise.resolve();
    for (let i = 0; i < 16; i++) {
      assert.equal(sim.step(), true);
      assert.deepEqual(sim.lastAction.armJointTarget, [0.31 + i * 0.001, -0.2, 0.1, -1.3, 0]);
      sim.advance(0.098);
    }
    assert.equal(sim.step(), false);
    assert.ok(Math.abs(sim.data.time - snapshot.time - 1.6) < 1e-8);
    assert.equal(rollout.needsImages, true);
    assert.ok(Math.abs(rollout.result.elapsed - 1.6) < 1e-8, "evaluation runs on all 800 physical steps");
    assert.equal(rollout.chunks.length, 1);
  } finally { sim.dispose(); }
});

test("Stop, Reset and manual takeover discard late model replies and preserve gripper state", async () => {
  const sim = await createSimulation();
  try {
    for (const stop of [(r: LearnedRollout) => r.stop(), () => sim.reset(), () => sim.setKey("KeyW", true),
      () => sim.setTouchInput({ forward: 0.5 }), () => sim.applyAction({ gripper: 0.6 })]) {
      sim.reset(); sim.advance(1); const p = pendingPolicy();
      const r = new LearnedRollout(sim, "Put the mug in the basket.", p.policy);
      r.capture(pair(sim.data.time)); stop(r);
      const arm = [...sim.armCommand], opening = sim.opening;
      p.resolve(chunk()); await Promise.resolve(); await Promise.resolve();
      assert.equal(r.stopped, true); assert.equal(sim.learningCapture, null);
      assert.deepEqual(sim.armCommand, arm); assert.equal(sim.opening, opening);
      assert.equal(r.chunks.length, 0);
    }
  } finally { sim.dispose(); }
});

test("stale images and malformed actions stop learned control without applying any prefix", async () => {
  const sim = await createSimulation();
  try {
    sim.advance(1); const p = pendingPolicy(), r = new LearnedRollout(sim, "Put the mug in the basket.", p.policy);
    r.capture(pair(sim.data.time - 0.1));
    assert.equal(r.stopped, true); assert.match(r.error!, /fresh|time/i);
    for (const invalid of [chunk().slice(0, 15), chunk().map((row, i) => i === 15 ? [NaN, ...row.slice(1)] : row)]) {
      const p2 = pendingPolicy(), r2 = new LearnedRollout(sim, "Put the tennis ball in the basket.", p2.policy);
      const arm = [...sim.armCommand];r2.capture(pair(sim.data.time)); p2.resolve(invalid);
      await Promise.resolve(); await Promise.resolve();
      assert.equal(r2.stopped, true); assert.ok(r2.error); assert.deepEqual(sim.armCommand, arm);
      assert.equal(r2.chunks.length, 0);
    }
  } finally { sim.dispose(); }
});

test("a traced prediction preserves the exact model noise and executes all sixteen goals", async () => {
  const sim = await createSimulation();
  try {
    sim.advance(1);
    const actions = chunk(), initialNoise = Array.from(Float32Array.from({ length: 512 }, (_, i) => Math.sin(i)));
    const savedNoise = [...initialNoise];
    const policy: LearnedPolicy = { infer: async () => ({ actions, initialNoise }), dispose() {} };
    const r = new LearnedRollout(sim, "Put the mug in the basket.", policy);
    r.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve();
    assert.equal(r.stopped, false, r.error ?? undefined);
    assert.deepEqual(r.chunks[0].rawActions, actions);
    assert.deepEqual(r.chunks[0].initialNoise, savedNoise);
    initialNoise.fill(99);
    assert.deepEqual(r.chunks[0].initialNoise, savedNoise, "saved noise must not alias the policy reply");
    for (let i = 0; i < 16; i++) {
      assert.equal(sim.step(), true);
      assert.deepEqual(sim.lastAction.armJointTarget, actions[i].slice(0, 5));
      sim.advance(0.098);
    }
    assert.equal(sim.step(), false);
    assert.equal(r.needsImages, true);
    r.stop(); assert.equal(r.result.status, "running", "diagnostic stopping is not a completed failed episode");
  } finally { sim.dispose(); }
});

test("malformed traced noise stops before applying any learned goal", async () => {
  const sim = await createSimulation();
  try {
    for (const initialNoise of [Array(511).fill(0), Array(512).fill(NaN)]) {
      sim.reset(); sim.advance(1);
      const policy: LearnedPolicy = { infer: async () => ({ actions: chunk(), initialNoise }), dispose() {} };
      const r = new LearnedRollout(sim, "Put the mug in the basket.", policy);
      const arm = [...sim.armCommand];
      r.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve();
      assert.equal(r.stopped, true); assert.match(r.error!, /noise/i);
      assert.deepEqual(sim.armCommand, arm); assert.equal(r.chunks.length, 0);
    }
  } finally { sim.dispose(); }
});

test("stationary learned base drift is a terminal physical failure at its exact2ms step", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 42); sim.advance(2);
    const robot = sim.observation().robot!;
    const policy: LearnedPolicy = { infer: async () => Array.from({length:16}, () => [...robot.joints,robot.gripper]), dispose() {} };
    const rollout = new LearnedRollout(sim,"Put the mug in the basket.",policy);
    rollout.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve();
    // Test-only initial angular impulse reproduces actual physical displacement.
    sim.data.qvel[sim.model.jnt_dofadr[sim.id("JOINT","base_yaw")]] = 4;
    const start = sim.data.time; let steps = 0;
    while (!rollout.stopped && steps < 200) {
      if (sim.step()) steps++; else assert.equal(rollout.stopped,true);
    }
    assert.equal(rollout.stopped,true);
    assert.equal(rollout.result.status,"failed");
    assert.equal(rollout.result.reason,"base-drift");
    assert.equal(rollout.error,null,"physical failure is distinct from malformed model/infrastructure errors");
    assert.ok(Math.abs(rollout.result.elapsed-steps*.002)<1e-8);
    assert.ok(Math.abs(sim.data.time-start-rollout.result.elapsed)<1e-8);
    assert.equal(sim.learningCapture,null);
  } finally { sim.dispose(); }
});

test("stationary bounds remain2cm and.03rad; nonfinite base measurement stays an error", async () => {
  const sim = await createSimulation();
  try {
    for (const [joint,delta,expected] of [["base_x",.0199,false],["base_x",.0201,true],
      ["base_yaw",.0299,false],["base_yaw",.0301,true],["base_yaw",NaN,"error"]] as const) {
      prepareLearningScene(sim,42); sim.advance(2);
      const robot=sim.observation().robot!;
      const r=new LearnedRollout(sim,"Put the mug in the basket.",{
        infer:async () => Array.from({length:16},()=>[...robot.joints,robot.gripper]),dispose() {},
      });
      r.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve();
      // Synthetic pose injection checks only the unchanged guard boundary.
      sim.data.qpos[sim.joints[joint].q] += delta;
      sim.mj.mj_kinematics(sim.model,sim.data);
      assert.equal(r.captureDue(sim.data.time),Boolean(expected));
      if (expected === "error") {
        assert.match(r.error!,/Invalid measured base pose/); assert.equal(r.result.status,"running");
      } else {
        assert.equal(r.error,null); assert.equal(r.result.status,expected?"failed":"running");
        if(expected) assert.equal(r.result.reason,"base-drift");
      }
      r.stop();
    }
  } finally { sim.dispose(); }
});

test("an invalid initial base reference cannot become a completed physical failure after recovery", async () => {
  const sim=await createSimulation();
  try {
    prepareLearningScene(sim,42); sim.advance(2);
    const yaw=sim.q("base_yaw"); sim.data.qpos[sim.joints.base_yaw.q]=NaN;
    sim.mj.mj_kinematics(sim.model,sim.data);
    const r=new LearnedRollout(sim,"Put the mug in the basket.",{
      infer:async()=>chunk(),dispose() {},
    });
    sim.data.qpos[sim.joints.base_yaw.q]=yaw; sim.mj.mj_kinematics(sim.model,sim.data);
    r.capture(pair(sim.data.time));
    assert.equal(r.stopped,true); assert.equal(r.result.status,"running");
    assert.match(r.error!,/Invalid measured base pose/);
    assert.equal(r.chunks.length,0);
  } finally { sim.dispose(); }
});

test("paired learned predictions retain both noise draws and all four member outputs without aliases", async () => {
  const sim = await createSimulation();
  try {
    sim.advance(1);
    const prediction = { actions: chunk(), initialNoise: Array(512).fill(0.25), secondNoise: Array(512).fill(-0.5), memberActions: Array.from({ length: 4 }, (_, i) => Array(96).fill(i)) };
    const r = new LearnedRollout(sim, "Put the mug in the basket.", { infer: async () => prediction, dispose() {} });
    r.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve();
    assert.equal(r.stopped, false, r.error ?? undefined);
    assert.deepEqual(r.chunks[0].secondNoise, Array(512).fill(-0.5));
    assert.deepEqual(r.chunks[0].memberActions, Array.from({ length: 4 }, (_, i) => Array(96).fill(i)));
    prediction.secondNoise.fill(99); prediction.memberActions[0].fill(99);
    assert.equal(r.chunks[0].secondNoise![0], -0.5); assert.equal(r.chunks[0].memberActions![0][0], 0);
    r.stop();
  } finally { sim.dispose(); }
});

test("incomplete or nonfinite paired traces stop before applying a learned goal", async () => {
  const sim = await createSimulation();
  try {
    for (const extra of [{ secondNoise: Array(511).fill(0), memberActions: Array.from({ length: 4 }, () => Array(96).fill(0)) },
      { secondNoise: Array(512).fill(0), memberActions: Array.from({ length: 3 }, () => Array(96).fill(0)) },
      { secondNoise: Array(512).fill(0), memberActions: Array.from({ length: 4 }, () => Array(96).fill(NaN)) }]) {
      sim.reset(); sim.advance(1);
      const r = new LearnedRollout(sim, "Put the mug in the basket.", { infer: async () => ({ actions: chunk(), initialNoise: Array(512).fill(0), ...extra }), dispose() {} });
      const arm = [...sim.armCommand]; r.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve();
      assert.equal(r.stopped, true); assert.match(r.error!, /paired|trace/i);
      assert.deepEqual(sim.armCommand, arm); assert.equal(r.chunks.length, 0);
    }
  } finally { sim.dispose(); }
});

test("a one-member diagnostic trace (one member, no second noise) is kept; one member with a second noise is refused", async () => {
  const sim = await createSimulation();
  try {
    sim.advance(1);
    const single = { actions: chunk(), initialNoise: Array(512).fill(0.25), memberActions: [Array(96).fill(7)] };
    const r = new LearnedRollout(sim, "Put the mug in the basket.", { infer: async () => single, dispose() {} });
    r.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve();
    assert.equal(r.stopped, false, r.error ?? undefined); assert.deepEqual(r.chunks[0].memberActions, [Array(96).fill(7)]);
    assert.equal(r.chunks[0].secondNoise, undefined); r.stop();
    const mixed = new LearnedRollout(sim, "Put the mug in the basket.", { infer: async () => ({ ...single, secondNoise: Array(512).fill(0) }), dispose() {} });
    mixed.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve();
    assert.match(mixed.error!, /Incomplete/);
  } finally { sim.dispose(); }
});

test("a mug rollout can score with the walls criterion and keeps the rim time", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const p = pendingPolicy();
    const walls = new LearnedRollout(sim, "Put the mug in the basket.", p.policy, { success: "walls" });
    assert.equal(walls.result.rimSettledFor, 0); walls.stop();
    const rim = new LearnedRollout(sim, "Put the mug in the basket.", pendingPolicy().policy);
    assert.equal(rim.result.rimSettledFor, undefined); rim.stop();
  } finally { sim.dispose(); }
});

test("an unlearned instruction needs an explicit null target, reaches the policy as written and is scored with no success target", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    assert.throws(() => new LearnedRollout(sim, "Take the mug out of the basket.", pendingPolicy().policy), /target/);
    assert.throws(() => new LearnedRollout(sim, "コップを出して", pendingPolicy().policy, { target: null }), /instruction/);
    const p = pendingPolicy(), r = new LearnedRollout(sim, "Take the mug out of the basket.", p.policy, { target: null });
    assert.equal(r.evaluationTarget, null);
    for (let n = 0; n < 50 && !p.input(); n++) { if (r.needsImages) r.capture(["head", "hand"].map(view => ({ view: view as "head" | "hand", timestamp: sim.data.time, width: 320, height: 240, dataUrl: "data:image/jpeg;base64,/9j/AA==" })) as never); await new Promise(res => setImmediate(res)); }
    assert.equal(p.input()?.instruction, "Take the mug out of the basket.");
    r.stop();
    const learned = new LearnedRollout(sim, "Put the mug in the basket.", pendingPolicy().policy);
    assert.equal(learned.evaluationTarget, "mug"); learned.stop();
  } finally { sim.dispose(); }
});
