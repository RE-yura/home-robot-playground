import test from "node:test";
import assert from "node:assert/strict";
import { chooseRtcTiming, continuousExecutionSteps } from "../src/learning/rtc-runtime.ts";
test("RTC budgets warmed inference with visible slower playback and spare prediction tail", () => {
  const t = chooseRtcTiming([{ inferenceMs: 9000 }, { inferenceMs: 1800 }, { inferenceMs: 2000 }]);
  assert.equal(t.playbackRate, .3); assert.equal(t.delaySteps, 7);
  assert.ok(t.delaySteps < 16 - t.delaySteps);
  assert.equal(chooseRtcTiming([{ inferenceMs: 1000 }, { inferenceMs: 300 }, { inferenceMs: 400 }]).playbackRate, 1);
  assert.throws(() => chooseRtcTiming([{ inferenceMs: NaN }]), /timing|samples/i);
  assert.throws(() => chooseRtcTiming([{ inferenceMs: 1000 }, { inferenceMs: 9000 }, { inferenceMs: 9000 }]), /slow|budget/i);
});

test("continuous execution runs min(12, 16 - delay) goals per chunk unless a valid override replans sooner", () => {
  assert.equal(continuousExecutionSteps(5, null), 11);
  assert.equal(continuousExecutionSteps(3, null), 12);
  assert.equal(continuousExecutionSteps(5, "6"), 6);
  assert.equal(continuousExecutionSteps(5, "11"), 11);
  for (const bad of ["5", "12", "6.5", "x", ""]) assert.throws(() => continuousExecutionSteps(5, bad), /execution/);
  assert.throws(() => continuousExecutionSteps(8, null), /execution/);
});
