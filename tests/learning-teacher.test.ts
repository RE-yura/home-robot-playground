import assert from "node:assert/strict";
import { test } from "node:test";
import { createSimulation, type Simulation } from "../src/simulation/mujoco.ts";
import { prepareLearningScene } from "../src/learning/scene.ts";
import { StationaryTeacher } from "../src/learning/teacher.ts";
import { LearningCapture } from "../src/learning/capture.ts";
import { validateAction } from "../src/controllers/protocol.ts";
import { TaskEvaluator } from "../src/simulation/task.ts";
import type { Instruction } from "../src/learning/recording.ts";
import type { CameraImage } from "../src/simulation/types.ts";

function refresh(sim: Simulation) { sim.mj.mj_kinematics(sim.model, sim.data); }
function pair(time: number): CameraImage[] {
  return ["head", "hand"].map(view => ({ view: view as "head" | "hand", timestamp: time, width: 320, height: 240, dataUrl: "data:image/jpeg;base64,/9j/AA==" }));
}

for (const instruction of ["Put the mug in the basket.", "Put the tennis ball in the basket."] as Instruction[]) {
  for (const seed of [42, 7]) test(`physical teacher grasps, lifts, releases and settles: ${instruction} seed${seed}`, async () => {
    const sim = await createSimulation();
    try {
      prepareLearningScene(sim, seed); sim.advance(2); refresh(sim);
      const initial = sim.observation(), forces = Array.from(sim.model.actuator_forcerange);
      const teacher = new StationaryTeacher(sim, instruction);
      const capture = new LearningCapture(sim, { instruction, maxSamples: 1210, taskEvaluator: teacher.evaluator, actionForSample: () => {
        const state = sim.physicsState(), ctrl = Array.from(sim.data.ctrl);
        const action = teacher.next(sim.observation());
        assert.deepEqual(sim.physicsState(), state, "teacher callback cannot mutate dynamics");
        assert.deepEqual(Array.from(sim.data.ctrl), ctrl, "teacher returns targets instead of writing actuators");
        validateAction(action); return action;
      } });
      for (let i = 0; i < 1211 && !capture.stopped; i++) {
        sim.step(); capture.capture(pair(sim.data.time));
        if (teacher.result.status !== "running") { capture.stop(); break; }
        sim.advance(0.1);
      }
      assert.equal(capture.error, null);
      assert.equal(teacher.result.status, "success", JSON.stringify(teacher.result));
      assert.equal(teacher.result.grasped, true); assert.equal(teacher.result.lifted, true);
      assert.equal(teacher.result.teacherFailure, null);
      assert.equal(sim.gripperContactId, null);
      assert.deepEqual(Array.from(sim.model.actuator_forcerange), forces);
      assert.equal(sim.opening, 1);
      const otherId = instruction.includes("mug") ? "ball" : "mug", other = new TaskEvaluator(otherId);
      other.start(sim.observation()); assert.notEqual(other.update(sim.observation(), 2).status, "success");
      assert.ok(Math.hypot(sim.position(sim.baseId)[0] - initial.robot!.base.position[0], sim.position(sim.baseId)[2] - initial.robot!.base.position[2]) < 0.02);
      assert.equal(capture.episode!.kind, "calibration", "teacher verification does not certify training data");
    } finally { sim.dispose(); }
  });
}

test("Reset and physical chassis drift invalidate teacher goals", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim); sim.advance(2); refresh(sim);
    const resetTeacher = new StationaryTeacher(sim, "Put the mug in the basket.");
    sim.reset(); resetTeacher.next(sim.observation());
    assert.equal(resetTeacher.result.teacherFailure, "reset");
    prepareLearningScene(sim); sim.advance(2); refresh(sim);
    const driftTeacher = new StationaryTeacher(sim, "Put the mug in the basket.");
    sim.applyAction({ base: [1, 0, 0] }); sim.advance(0.4); refresh(sim);
    driftTeacher.next(sim.observation()); assert.equal(driftTeacher.result.teacherFailure, "base-drift");
    assert.equal(driftTeacher.result.status, "failed");
  } finally { sim.dispose(); }
});

test("opening a physical grasp before release fails and cannot recover", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim); sim.advance(2); refresh(sim);
    const teacher = new StationaryTeacher(sim, "Put the mug in the basket.");
    for (let i = 0; i < 80; i++) {
      refresh(sim); sim.applyAction(teacher.next(sim.observation())); sim.advance(0.1);
    }
    assert.equal(sim.gripperContactId, "mug");
    sim.applyAction({ gripper: 1 }); sim.advance(0.5); refresh(sim);
    teacher.next(sim.observation()); assert.equal(teacher.result.teacherFailure, "grasp-lost");
    sim.advance(1); refresh(sim); teacher.next(sim.observation());
    assert.equal(teacher.result.status, "failed");
  } finally { sim.dispose(); }
});

test("a missing or physically unreachable target fails instead of opening on schedule", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim); sim.advance(2); refresh(sim);
    const teacher = new StationaryTeacher(sim, "Put the mug in the basket.");
    const obs = sim.observation(); obs.objects = obs.objects.filter(o => o.id !== "mug");
    teacher.next(obs); assert.equal(teacher.result.reason, "target-missing");
    sim.reset(); sim.advance(2); refresh(sim);
    const unreachable = new StationaryTeacher(sim, "Put the tennis ball in the basket.");
    for (let i = 0; i < 61 && unreachable.result.status === "running"; i++) {
      refresh(sim); sim.applyAction(unreachable.next(sim.observation())); sim.advance(0.1);
    }
    assert.equal(unreachable.result.status, "failed");
    assert.ok(["grasp-missing", "base-drift", "wrong-grasp"].includes(unreachable.result.teacherFailure!));
  } finally { sim.dispose(); }
});

test("an actual bilateral grasp of the other object fails the requested instruction", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim); sim.advance(2); refresh(sim);
    const requested = new StationaryTeacher(sim, "Put the mug in the basket.");
    const ball = new StationaryTeacher(sim, "Put the tennis ball in the basket.");
    const capture = new LearningCapture(sim, { instruction: "Put the tennis ball in the basket.", maxSamples: 65, taskEvaluator: ball.evaluator,
      actionForSample: () => ball.next(sim.observation()) });
    for (let i = 0; i < 64; i++) { sim.step(); capture.capture(pair(sim.data.time)); sim.advance(0.1); }
    refresh(sim); assert.equal(sim.gripperContactId, "ball");
    requested.next(sim.observation()); assert.equal(requested.result.teacherFailure, "wrong-grasp");
    capture.stop();
  } finally { sim.dispose(); }
});

test("elapsed physics time ends an idle teacher and cannot resume its old plan", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim); sim.advance(2); refresh(sim);
    const teacher = new StationaryTeacher(sim, "Put the mug in the basket.");
    sim.advance(120); refresh(sim); teacher.next(sim.observation());
    assert.equal(teacher.result.status, "failed");
    assert.equal(teacher.result.reason ?? teacher.result.teacherFailure, "timeout");
    sim.advance(0.1); refresh(sim); teacher.next(sim.observation());
    assert.equal(teacher.result.status, "failed");
  } finally { sim.dispose(); }
});
