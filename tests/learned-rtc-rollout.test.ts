import test from "node:test";
import assert from "node:assert/strict";
import { createSimulation } from "../src/simulation/mujoco.ts";
import { prepareLearningScene } from "../src/learning/scene.ts";
import { LearnedRtcRollout } from "../src/learning/learned-rtc-rollout.ts";
import type { CameraImage } from "../src/simulation/types.ts";
const pair = (timestamp: number): CameraImage[] => ["head", "hand"].map(view => ({ view: view as "head" | "hand", timestamp, width: 320, height: 240, dataUrl: "data:image/jpeg;base64,/9j/AA==" }));
test("RTC keeps physical2ms steps running during the next model request", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const robot = sim.observation().robot!, goals = Array.from({ length: 16 }, () => [...robot.joints, robot.gripper]);
    let resolve!: (value: number[][]) => void; const requests: unknown[] = [];
    const policy = { infer(input: unknown, context: unknown) { requests.push({ input, context }); return new Promise<number[][]>(done => { resolve = done; }); }, dispose() {} };
    const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 4 });
    assert.equal(sim.step(), false); r.capture(pair(sim.data.time)); resolve(goals); await Promise.resolve(); await Promise.resolve();
    assert.equal(sim.step(), true); sim.advance(.098);
    // Next capture occurs on the100ms boundary; it blocks for its image only.
    assert.equal(sim.step(), false); r.capture(pair(sim.data.time));
    const time = sim.data.time; sim.advance(.2);
    assert.ok(Math.abs(sim.data.time - time - .2) < 1e-8, "inference must not freeze the physical world");
    assert.equal(r.inferencePending, true); assert.equal(requests.length, 2);
    resolve(goals); await Promise.resolve(); await Promise.resolve();
    assert.equal(r.stopped, false); assert.equal(r.chunks.length, 2);
    assert.equal(r.quality.physicsSteps, 150);
    assert.ok(Math.abs(r.quality.physicsSeconds - r.result.elapsed) < 1e-8);
    assert.ok(Number.isFinite(r.quality.peakUnintendedForce));
    assert.equal(r.playbackRate, .5); r.stop();
  } finally { sim.dispose(); }
});

test("RTC reset/Stop discard late replies without moving the actuator", async () => {
  const sim = await createSimulation();
  try {
    for (const reset of [false, true]) {
      prepareLearningScene(sim, 3000); sim.advance(2);
      let resolve!: (value: number[][]) => void;
      const policy = { infer() { return new Promise<number[][]>(done => { resolve = done; }); }, dispose() {} };
      const r = new LearnedRtcRollout(sim, "Put the tennis ball in the basket.", policy, { playbackRate: .5, delaySteps: 4, disposePolicy: false });
      r.capture(pair(sim.data.time)); if (reset) sim.reset(); else r.stop();
      const joints = [...sim.armCommand], opening = sim.opening;
      resolve(Array.from({ length: 16 }, () => [.6, -1, .2, -1.2, 0, .1])); await Promise.resolve(); await Promise.resolve();
      assert.equal(r.stopped, true); assert.equal(r.chunks.length, 0); assert.deepEqual(sim.armCommand, joints); assert.equal(sim.opening, opening);
      assert.equal(r.requests.length, 1);
      assert.equal(r.requests[0].status, "interrupted");
      assert.equal(r.requests[0].lateReturn, true);
      assert.ok(r.requests[0].returned);
      assert.ok(r.requests[0].interrupted);
      assert.equal(r.requests[0].handoff, null);
    }
  } finally { sim.dispose(); }
});

test("RTC reports exhausted coverage rather than silently holding the last goal forever", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const robot = sim.observation().robot!, goals = Array.from({ length: 16 }, () => [...robot.joints, robot.gripper]);
    let calls = 0;
    const policy = { infer() { return ++calls === 1 ? Promise.resolve(goals) : new Promise<number[][]>(() => {}); }, dispose() {} };
    const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 4 });
    r.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve();
    sim.advance(.1); assert.equal(sim.step(), false); r.capture(pair(sim.data.time)); sim.advance(1.6);
    assert.equal(r.stopped, true); assert.match(r.error!, /underflow/i); assert.equal(calls, 2);
    assert.ok(Math.abs(r.result.elapsed - 1.6) < 1e-8);
    assert.equal(r.requests.length, 2, "pending requests must survive queue exhaustion");
    assert.equal(r.requests[0].status, "accepted");
    assert.equal(r.requests[1].status, "interrupted");
    assert.equal(r.requests[1].returned, null);
    assert.equal(r.requests[1].continuation!.prefix.length, 15);
    assert.ok(Math.abs(r.requests[1].interrupted!.simulationTime - 3.6) < 1e-8);
    assert.match(r.requests[1].interrupted!.reason, /underflow/i);
  } finally { sim.dispose(); }
});

test("unguided continuous requests carry no continuation and replace only the not-yet-executed goals", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const robot = sim.observation().robot!, chunk = (grip: (i: number) => number) => Array.from({ length: 16 }, (_, i) => [...robot.joints, grip(i)]);
    let resolve!: (value: number[][]) => void; const argumentCounts: number[] = [];
    const policy = { infer(...args: unknown[]) { argumentCounts.push(args.length); return new Promise<number[][]>(done => { resolve = done; }); }, dispose() {} };
    const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 2, executionSteps: 8, guidance: false });
    const applied: number[] = [], apply = sim.applyAction.bind(sim);
    sim.applyAction = action => { applied.push(action.gripper!); return apply(action); };
    r.capture(pair(sim.data.time)); resolve(chunk(i => i === 0 ? 1 : 0)); await Promise.resolve(); await Promise.resolve();
    assert.equal(sim.step(), true); sim.advance(.098);
    assert.equal(sim.step(), false); r.capture(pair(sim.data.time));
    sim.advance(.2); resolve(chunk(() => .9)); await Promise.resolve(); await Promise.resolve();
    sim.advance(.75);
    // The unchanged public graph and worker can serve these requests.
    assert.deepEqual(argumentCounts, [1, 1]);
    assert.deepEqual(applied.slice(0, 3), [1, 0, 0], "goals already executed are never replaced");
    assert.deepEqual(applied.slice(3, 10), Array(7).fill(.9), "the new chunk takes over from the first not-yet-executed goal");
    r.stop();
  } finally { sim.dispose(); }
});

test("hybrid continuous execution runs a gripper-changing chunk to its end, then pauses physics for a fresh image", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const robot = sim.observation().robot!, chunk = (grip: (i: number) => number) => Array.from({ length: 16 }, (_, i) => [...robot.joints, grip(i)]);
    let resolve!: (value: number[][]) => void;
    const policy = { infer() { return new Promise<number[][]>(done => { resolve = done; }); }, dispose() {} };
    const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 2, executionSteps: 12, guidance: false, pauseForGripperChanges: true });
    const start = sim.data.time, flush = async () => { await Promise.resolve(); await Promise.resolve(); };
    const step = async () => { sim.advance(.1); if (r.needsImages) r.capture(pair(sim.data.time)); await flush(); };
    r.capture(pair(start)); resolve(chunk(() => 1)); await flush();
    while (r.requests.length < 2) await step();
    // This chunk closes the hand at its position8: it must run to its end instead of handing over at position12.
    resolve(chunk(i => i >= 8 ? 0 : 1)); await flush();
    while (r.requests.length < 3 && sim.data.time - start < 3) await step();
    assert.equal(r.error, null);
    const captured = r.requests[2].started.simulationTime;
    assert.ok(Math.abs(captured - start - 1.7) < 1e-8, `captured at the end of the closing chunk (${captured - start})`);
    assert.equal(r.pausedForGripper, true); assert.equal(r.gripperPauses, 1);
    sim.advance(.5); assert.ok(Math.abs(sim.data.time - captured) < 1e-8, "physics waits for the fresh plan");
    resolve(chunk(() => 0)); await flush(); sim.advance(.3);
    assert.equal(r.pausedForGripper, false); assert.ok(sim.data.time - captured > .25);
    assert.equal(r.chunks[2].skipped, 0, "the fresh plan starts at its first goal");
    r.stop();
  } finally { sim.dispose(); }
});

test("an offset model's chunk from capture c runs from c+2 while the queued goals cover the inference delay", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const robot = sim.observation().robot!, chunk = (grip: (i: number) => number) => Array.from({ length: 16 }, (_, i) => [...robot.joints, grip(i)]);
    let resolve!: (value: number[][]) => void;
    const policy = { infer() { return new Promise<number[][]>(done => { resolve = done; }); }, dispose() {} };
    const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 2, executionSteps: 12, guidance: false, chunkOffsetSteps: 2 });
    const applied: number[] = [], apply = sim.applyAction.bind(sim), flush = async () => { await Promise.resolve(); await Promise.resolve(); };
    sim.applyAction = action => { applied.push(action.gripper!); return apply(action); };
    r.capture(pair(sim.data.time)); resolve(chunk(i => .1 + i / 100)); await flush();
    const held = r.requests[0].input.state[5];
    assert.equal(sim.step(), true); sim.advance(.098);
    assert.equal(sim.step(), false); r.capture(pair(sim.data.time));
    sim.advance(.2); resolve(chunk(i => .5 + i / 100)); await flush(); sim.advance(.3);
    assert.deepEqual(applied.slice(0, 6), [held, held, .1, .5, .51, .52], "the first two goals hold the measured state; each reply starts two goals after its capture");
    assert.equal(r.chunks[1].skipped, 0);
    r.stop();
  } finally { sim.dispose(); }
});

test("an offset model is unguided and its execution horizon must keep the offset covered", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const policy = { infer() { return new Promise<number[][]>(() => {}); }, dispose() {} };
    assert.throws(() => new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 2, chunkOffsetSteps: 2 }), /offset/i);
    assert.throws(() => new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 2, executionSteps: 17, guidance: false, chunkOffsetSteps: 2 }), /horizon/i);
    assert.throws(() => new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 2, guidance: false, chunkOffsetSteps: 1.5 }), /offset/i);
    new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 2, executionSteps: 16, guidance: false, chunkOffsetSteps: 2 }).stop();
  } finally { sim.dispose(); }
});

test("hybrid execution of an offset model pauses at the end of the shifted chunk and holds the last goal until c+2", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const robot = sim.observation().robot!, chunk = (grip: (i: number) => number) => Array.from({ length: 16 }, (_, i) => [...robot.joints, grip(i)]);
    let resolve!: (value: number[][]) => void;
    const policy = { infer() { return new Promise<number[][]>(done => { resolve = done; }); }, dispose() {} };
    const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 2, executionSteps: 12, guidance: false, pauseForGripperChanges: true, chunkOffsetSteps: 2 });
    const start = sim.data.time, flush = async () => { await Promise.resolve(); await Promise.resolve(); };
    const step = async () => { sim.advance(.1); if (r.needsImages) r.capture(pair(sim.data.time)); await flush(); };
    r.capture(pair(start)); resolve(chunk(() => 1)); await flush();
    while (r.requests.length < 2) await step();
    resolve(chunk(i => i >= 8 ? 0 : 1)); await flush();
    while (r.requests.length < 3 && sim.data.time - start < 3) await step();
    assert.equal(r.error, null);
    const captured = r.requests[2].started.simulationTime;
    assert.ok(Math.abs(captured - start - 1.9) < 1e-8, `captured at the end of the shifted closing chunk (${captured - start})`);
    const applied: number[] = [], apply = sim.applyAction.bind(sim);
    sim.applyAction = action => { applied.push(action.gripper!); return apply(action); };
    resolve(chunk(() => .5)); await flush(); sim.advance(.35);
    assert.deepEqual(applied.slice(0, 3), [0, 0, .5], "the closed hand is held for the two goals before the fresh chunk");
    r.stop();
  } finally { sim.dispose(); }
});

test("a prefix-conditioned model is sent the committed goals; the public model never receives a continuation", async () => {
  const sim = await createSimulation();
  try {
    for (const conditioned of [true, false]) {
      prepareLearningScene(sim, 3000); sim.advance(2);
      const robot = sim.observation().robot!, chunk = Array.from({ length: 16 }, (_, i) => [...robot.joints, .1 + i / 100]);
      let resolve!: (value: number[][]) => void; const calls: unknown[][] = [];
      const policy = { infer(...args: unknown[]) { calls.push(args); return new Promise<number[][]>(done => { resolve = done; }); }, dispose() {} };
      const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 3, executionSteps: 12, guidance: false,
        ...(conditioned ? { prefixConditioning: { maxDelay: 10 } } : {}) });
      r.capture(pair(sim.data.time)); resolve(chunk); await Promise.resolve(); await Promise.resolve();
      assert.equal(sim.step(), true); sim.advance(.098); assert.equal(sim.step(), false); r.capture(pair(sim.data.time));
      if (conditioned) {
        assert.equal(calls[0][1], null, "nothing is queued before the first chunk");
        const sent = calls[1][1] as { prefix: number[][]; delaySteps: number };
        assert.equal(sent.delaySteps, 3); assert.deepEqual(sent.prefix.slice(0, 3).map(goal => goal[5]), [1, 2, 3].map(i => .1 + i / 100));
      } else assert.ok(calls.every(args => args.length === 1), "the unchanged public graph gets no continuation");
      r.stop();
    }
  } finally { sim.dispose(); }
});

test("prefix conditioning is unguided, unshifted and limited to the trained delay", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const policy = { infer() { return new Promise<number[][]>(() => {}); }, dispose() {} };
    const build = (options: object) => () => new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 3, executionSteps: 12, ...options });
    assert.throws(build({ prefixConditioning: { maxDelay: 10 } }), /prefix/i);
    assert.throws(build({ guidance: false, delaySteps: 9, executionSteps: 7, prefixConditioning: { maxDelay: 8 } }), /prefix/i);
    assert.throws(build({ guidance: false, chunkOffsetSteps: 2, prefixConditioning: { maxDelay: 10 } }), /prefix/i);
    build({ guidance: false, prefixConditioning: { maxDelay: 10 } })().stop();
  } finally { sim.dispose(); }
});

test("continuous execution keeps a one-member diagnostic trace and refuses a mixed one", async () => {
  const sim = await createSimulation();
  try {
    for (const second of [false, true]) {
      prepareLearningScene(sim, 3000); sim.advance(2);
      const robot = sim.observation().robot!, goals = Array.from({ length: 16 }, () => [...robot.joints, robot.gripper]);
      const trace = { actions: goals, initialNoise: Array(512).fill(0), memberActions: [Array(96).fill(0)], ...(second ? { secondNoise: Array(512).fill(0) } : {}) };
      const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", { infer: async () => trace, dispose() {} }, { playbackRate: .5, delaySteps: 3, executionSteps: 12, guidance: false });
      r.capture(pair(sim.data.time)); await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
      if (second) assert.match(r.error!, /Incomplete/);
      else { assert.equal(r.error, null); assert.equal(r.chunks.length, 1); }
      r.stop();
    }
  } finally { sim.dispose(); }
});

// Takeover collection: a fixed inference latency makes the trajectory independent of how fast the page infers.
const fixed = { playbackRate: .5, delaySteps: 5, executionSteps: 11, guidance: false, prefixConditioning: { maxDelay: 8 }, fixedLatencySteps: 5 };
async function fixedLatencyRun() {
  const sim = await createSimulation();
  prepareLearningScene(sim, 3000); sim.advance(2);
  const robot = sim.observation().robot!, chunk = (grip: (i: number) => number) => Array.from({ length: 16 }, (_, i) => [...robot.joints, grip(i)]);
  let resolve!: (value: number[][]) => void;
  const policy = { infer() { return new Promise<number[][]>(done => { resolve = done; }); }, dispose() {} };
  const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, fixed);
  const applied: number[] = [], apply = sim.applyAction.bind(sim), flush = async () => { await Promise.resolve(); await Promise.resolve(); };
  sim.applyAction = action => { applied.push(Math.round(action.gripper! * 100) / 100); return apply(action); };
  const start = sim.data.time;
  r.capture(pair(start)); resolve(chunk(i => .1 + i / 100)); await flush();
  sim.advance(.1); assert.equal(sim.step(), false); r.capture(pair(sim.data.time));
  return { sim, r, applied, start, flush, reply: () => resolve(chunk(i => .5 + i / 100)) };
}

test("fixed latency holds physics for a late reply", async () => {
  const { sim, r, applied, start, flush, reply } = await fixedLatencyRun();
  try {
    sim.advance(1);
    assert.ok(Math.abs(sim.data.time - start - .6) < 1e-8, `physics waits at capture + latency (${sim.data.time - start})`);
    assert.equal(sim.step(), false);
    reply(); await flush(); sim.advance(.1);
    assert.deepEqual(applied, [.1, .11, .12, .13, .14, .15, .55], "the late reply's goal 5 runs exactly at capture + latency");
    assert.equal(r.chunks[1].skipped, 5); assert.equal(r.error, null); r.stop();
  } finally { sim.dispose(); }
});

test("fixed latency defers an early reply", async () => {
  const { sim, r, applied, flush, reply } = await fixedLatencyRun();
  try {
    reply(); await flush();
    assert.equal(r.chunks.length, 1, "an early reply is not applied before capture + latency");
    sim.advance(.6);
    assert.deepEqual(applied, [.1, .11, .12, .13, .14, .15, .55]);
    assert.equal(r.chunks[1].skipped, 5); assert.equal(r.error, null); r.stop();
  } finally { sim.dispose(); }
});

test("handoff stops at a boundary before latching", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const robot = sim.observation().robot!, goals = Array.from({ length: 16 }, (_, i) => [...robot.joints, .1 + i / 100]);
    let calls = 0;
    const policy = { infer() { return ++calls === 1 ? Promise.resolve(goals) : new Promise<number[][]>(() => {}); }, dispose() {} };
    const t0 = sim.data.time, latched: number[] = [], begun: [string, number][] = [], apply = sim.applyAction.bind(sim);
    sim.applyAction = action => { latched.push(Math.round(action.gripper! * 100) / 100); return apply(action); };
    const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { ...fixed, disposePolicy: false,
      handoff: { reason: time => time - t0 > .3 - 1e-8 ? "x" : null, begin: (reason, time) => { begun.push([reason, time]); } } });
    r.capture(pair(t0)); await Promise.resolve(); await Promise.resolve();
    sim.advance(.1); assert.equal(sim.step(), false); r.capture(pair(sim.data.time)); sim.advance(.25);
    assert.equal(r.stopped, true); assert.equal(r.error, null);
    assert.equal(r.handedOver?.reason, "x"); assert.ok(Math.abs(r.handedOver!.time - t0 - .3) < 1e-8);
    assert.equal(begun.length, 1); assert.equal(begun[0][1], r.handedOver!.time);
    assert.deepEqual(latched, [.1, .11, .12], "no policy goal latches at or after the handoff boundary");
    assert.equal(r.requests[1].status, "interrupted"); assert.equal(r.requests[1].interrupted!.reason, "handoff");
  } finally { sim.dispose(); }
});

// Rule assist (opt-in): measured-only watchdog that returns the arm to its start pose and restarts the policy.
const assisted = { playbackRate: .5, delaySteps: 5, executionSteps: 11, guidance: false, prefixConditioning: { maxDelay: 8 }, assist: true, disposePolicy: false };
async function assistRun(options: object = {}, hold: (call: number) => boolean = () => false) {
  const sim = await createSimulation();
  prepareLearningScene(sim, 3000); sim.advance(2);
  const robot = sim.observation().robot!, start = [...robot.joints];
  const calls: { continuation: unknown; resolve: (v: number[][]) => void }[] = [];
  // Chunks alternate a closed and an open hand at the start pose: empty grasps that never lift.
  const chunk = (n: number) => Array.from({ length: 16 }, () => [...start, n % 2 ? 0 : 1]);
  const policy = { infer(_input: unknown, continuation: unknown) {
    const n = calls.length; let resolve!: (v: number[][]) => void;
    const reply = new Promise<number[][]>(done => { resolve = done; }); calls.push({ continuation, resolve });
    if (!hold(n)) resolve(chunk(n)); return reply;
  }, dispose() {} };
  const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { ...assisted, ...options });
  const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
  const step = async () => { sim.advance(.1); if (r.needsImages) r.capture(pair(sim.data.time)); await flush(); };
  r.capture(pair(sim.data.time)); await flush();
  return { sim, r, calls, start, step, flush, chunk };
}

test("assist restarts after two failed grasp cycles", async () => {
  const { sim, r, calls, start, step } = await assistRun();
  try {
    while (!r.assistEvents.length && sim.data.time < 20) await step();
    assert.equal(r.error, null);
    assert.deepEqual(r.assistEvents.map(e => e.reason), ["regrasp-loop"]);
    const at = r.assistEvents[0].time, before = calls.length;
    while (calls.length === before && sim.data.time < at + 5) await step();
    assert.ok(sim.data.time - at >= 2 - 1e-8, "the scripted return takes 2 s before the policy is asked again");
    assert.equal(calls[before].continuation, null, "the restart is a fresh first request");
    assert.ok(Math.max(...sim.observation().robot!.joints.map((q, i) => Math.abs(q - start[i]))) < 0.05);
    r.stop();
  } finally { sim.dispose(); }
});

test("restarts keep the evaluator clock", async () => {
  const { sim, r, step } = await assistRun();
  try {
    while (!r.assistEvents.length && sim.data.time < 20) await step();
    const elapsed = r.result.elapsed, time = sim.data.time; for (let i = 0; i < 30; i++) await step();
    assert.ok(sim.data.time - time > 2, "the run continues past the scripted return");
    assert.ok(Math.abs(r.result.elapsed - elapsed - (sim.data.time - time)) < 1e-6, "task time runs through the restart, never reset");
    r.stop();
  } finally { sim.dispose(); }
});

test("a restart drops the in-flight reply", async () => {
  // The stall rule fires at 2.5 s while the capture of 2.3 s (call 3) is still waiting for its reply.
  const held = 3;
  const { sim, r, calls, step, flush, chunk } = await assistRun({ assistStallSeconds: 2.5 }, n => n === held);
  try {
    while (!r.assistEvents.length && sim.data.time < 20) await step();
    assert.deepEqual(r.assistEvents.map(e => e.reason), ["stall"]);
    assert.equal(r.requests[held].status, "interrupted", "the request pending at the restart is interrupted");
    calls[held].resolve(chunk(1)); await flush();
    assert.equal(r.requests[held].lateReturn, true); assert.equal(r.error, null);
    r.stop();
  } finally { sim.dispose(); }
});

test("assist restarts at most twice", async () => {
  const { sim, r, step } = await assistRun({ assistStallSeconds: 4 });
  try {
    while (sim.data.time < 30 && !r.stopped) await step();
    assert.equal(r.assistEvents.length, 2); assert.equal(r.error, null);
    r.stop();
  } finally { sim.dispose(); }
});

// Late-reply grace (opt-in): when the queue runs out while a reply is pending, the last goal is held for up to
// lateHoldSteps boundaries before the run fails with an underflow.
async function lateRun(lateHoldSteps: number | undefined, replyAfter: number) {
  const sim = await createSimulation();
  prepareLearningScene(sim, 3000); sim.advance(2);
  const robot = sim.observation().robot!, chunk = (g: number) => Array.from({ length: 16 }, (_, i) => [...robot.joints, g + i / 100]);
  let resolve!: (v: number[][]) => void, calls = 0;
  const policy = { infer() { calls++; return calls <= 2 ? Promise.resolve(chunk(calls / 10)) : new Promise<number[][]>(done => { resolve = done; }); }, dispose() {} };
  const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy, { playbackRate: .5, delaySteps: 5, executionSteps: 11, guidance: false, prefixConditioning: { maxDelay: 8 }, ...(lateHoldSteps === undefined ? {} : { lateHoldSteps }) });
  const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
  r.capture(pair(sim.data.time)); await flush();
  sim.advance(.1); assert.equal(sim.step(), false); r.capture(pair(sim.data.time)); await flush();
  // The chunk captured at +0.1 s covers goals to +1.6 s; the next capture is at +1.2 s and its reply comes replyAfter
  // seconds after +1.7 s, when the queued goals have run out.
  sim.advance(1.1); assert.equal(sim.step(), false); r.capture(pair(sim.data.time));
  sim.advance(.5 + replyAfter); resolve(chunk(.5)); await flush(); sim.advance(.3);
  return { sim, r };
}

test("a late reply within the grace keeps the run going", async () => {
  const { sim, r } = await lateRun(3, .2);
  try { assert.equal(r.error, null); assert.equal(r.stopped, false); assert.equal(r.lateHolds, 2); r.stop(); } finally { sim.dispose(); }
});

test("without the grace, or beyond it, a late reply still ends the run with an underflow", async () => {
  for (const [hold, after] of [[undefined, .2], [1, .2]] as const) {
    const { sim, r } = await lateRun(hold, after);
    try { assert.match(r.error ?? "", /underflow/); } finally { sim.dispose(); }
  }
});

test("a continuous mug rollout can score with the walls criterion", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const policy = { infer: () => new Promise<never>(() => {}), dispose() {} };
    const r = new LearnedRtcRollout(sim, "Put the mug in the basket.", policy as never, { playbackRate: .5, delaySteps: 5, executionSteps: 11, guidance: false, prefixConditioning: { maxDelay: 8 }, success: "walls" });
    assert.equal(r.result.rimSettledFor, 0); r.stop();
  } finally { sim.dispose(); }
});

test("a continuous rollout of an unlearned instruction sends that text and has no success target", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    const inputs: { instruction: string }[] = [], policy = { infer: (input: { instruction: string }) => { inputs.push(input); return new Promise<never>(() => {}); }, dispose() {} };
    const options = { playbackRate: .5, delaySteps: 5, executionSteps: 11, guidance: false, prefixConditioning: { maxDelay: 8 } };
    assert.throws(() => new LearnedRtcRollout(sim, "Take the mug out of the basket.", policy as never, options), /target/);
    const r = new LearnedRtcRollout(sim, "Take the mug out of the basket.", policy as never, { ...options, target: null });
    assert.equal(r.evaluationTarget, null);
    if (r.needsImages) r.capture(pair(sim.data.time));
    assert.equal(inputs[0]?.instruction, "Take the mug out of the basket.");
    r.stop();
  } finally { sim.dispose(); }
});

test("a new continuous rollout waits for the policy's previous request before its first capture (planned tasks back to back)", async () => {
  const sim = await createSimulation();
  try {
    prepareLearningScene(sim, 3000); sim.advance(2);
    let release!: () => void; const idle = new Promise<void>(done => { release = done; }), calls: string[] = [];
    const policy = { whenIdle: idle, infer: (input: { instruction: string }) => { calls.push(input.instruction); return new Promise<never>(() => {}); }, dispose() {} };
    const r = new LearnedRtcRollout(sim, "Take the mug out of the basket.", policy as never, { playbackRate: .5, delaySteps: 5, executionSteps: 11, guidance: false, prefixConditioning: { maxDelay: 8 }, target: null });
    const before = sim.physicsState();
    assert.equal(r.needsImages, false, "no capture while the worker still runs the previous task's request");
    assert.equal(sim.step(), false, "physics holds until the first plan of this task"); assert.deepEqual(sim.physicsState(), before);
    release(); await new Promise(res => setImmediate(res));
    assert.equal(r.needsImages, true);
    r.capture(pair(sim.data.time)); assert.deepEqual(calls, ["Take the mug out of the basket."]);
    r.stop();
  } finally { sim.dispose(); }
});
