import test from "node:test";
import assert from "node:assert/strict";
import { rtcContextFeed, validateRtcIdentity, RTC_GRAPH_CANDIDATE, RTC_GRAPH_CANDIDATES } from "../src/learning/rtc-contract.ts";
import { LEARNED_EXTERNAL_IDENTITY } from "../src/learning/learned-onnx-contract.ts";

test("first RTC inference has no conditioning; subsequent requests pad only actual queued commands", () => {
  const initial = rtcContextFeed(null);
  assert.equal(initial.enabled, false); assert.ok(initial.weights.every(x => x === 0));
  const prefix = Array.from({ length: 12 }, (_, i) => [i, 0, 0, 0, 0, .5] as [number, number, number, number, number, number]);
  const actual = rtcContextFeed({ prefix, delaySteps: 4, horizon: 10 });
  assert.equal(actual.enabled, true); assert.equal(actual.prefix.length, 96);
  assert.equal(actual.prefix[6], 1); assert.equal(actual.prefix[72], 0);
  assert.deepEqual([...actual.weights.slice(0, 4)], [1, 1, 1, 1]);
  assert.ok(actual.weights.slice(4, 10).every(x => x > 0 && x < 1)); assert.ok(actual.weights.slice(10).every(x => x === 0));
  prefix[1][0] = 99; assert.equal(actual.prefix[6], 1);
});
test("RTC rejects absent coverage, invalid channels and incompatible exact file identities", () => {
  assert.throws(() => rtcContextFeed({ prefix: [], delaySteps: 4, horizon: 10 }), /prefix|coverage/i);
  assert.throws(() => rtcContextFeed({ prefix: [Array(6).fill(NaN)] as never, delaySteps: 0, horizon: 1 }), /finite/i);
  assert.throws(() => rtcContextFeed({ prefix: [Array(6).fill(0)] as never, delaySteps: 2, horizon: 1 }), /delay|coverage/i);
  const identity = { graph: RTC_GRAPH_CANDIDATE, weights: LEARNED_EXTERNAL_IDENTITY.weights };
  assert.equal(validateRtcIdentity(identity).graph.sha256, RTC_GRAPH_CANDIDATE.sha256);
  assert.throws(() => validateRtcIdentity({ ...identity, graph: { ...identity.graph, sha256: "0".repeat(64) } }), /identity/i);
});
test("RTC guidance-cap variants are explicit exact identities over the unchanged public weights", () => {
  assert.deepEqual(RTC_GRAPH_CANDIDATES.max10, RTC_GRAPH_CANDIDATE);
  for (const graph of Object.values(RTC_GRAPH_CANDIDATES)) {
    const identity = { graph: { bytes: graph.bytes, sha256: graph.sha256 }, weights: LEARNED_EXTERNAL_IDENTITY.weights };
    assert.equal(validateRtcIdentity(identity).graph.sha256, graph.sha256);
    assert.throws(() => validateRtcIdentity({ ...identity, weights: { ...identity.weights, sha256: "0".repeat(64) } }), /identity/i);
  }
  assert.equal(new Set(Object.values(RTC_GRAPH_CANDIDATES).map(g => g.sha256)).size, 3);
});
