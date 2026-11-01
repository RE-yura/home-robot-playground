import test from "node:test";
import assert from "node:assert/strict";
import { prefixFeed } from "../src/learning/prefix-conditioning.ts";
import type { SixValues } from "../src/learning/recording.ts";
const goal = (i: number): SixValues => [i, i + .1, i + .2, i + .3, i + .4, i + .5];

test("without a continuation, or after a held capture with nothing queued, the model is conditioned on no goals", () => {
  for (const continuation of [null, { prefix: [], delaySteps: 3, horizon: 0 }]) {
    const feed = prefixFeed(continuation, 10);
    assert.equal(feed.delay, 0); assert.equal(feed.prefix.length, 96); assert.equal(feed.mask.length, 16);
    assert.ok(feed.prefix.every(v => v === 0)); assert.ok(feed.mask.every(v => v === 0));
  }
});

test("the first delaySteps committed goals are fed in order and marked", () => {
  const feed = prefixFeed({ prefix: [0, 1, 2, 3, 4].map(goal), delaySteps: 3, horizon: 5 }, 10);
  assert.equal(feed.delay, 3);
  assert.deepEqual(Array.from(feed.mask), [1, 1, 1, ...Array(13).fill(0)]);
  assert.deepEqual(Array.from(feed.prefix.subarray(0, 18)), [0, 1, 2].flatMap(i => goal(i)).map(v => Math.fround(v)));
  assert.ok(feed.prefix.subarray(18).every(v => v === 0), "goals after the delay are left to the model");
});

test("a queue shorter than the delay, an untrained delay or a non-finite goal is refused", () => {
  assert.throws(() => prefixFeed({ prefix: [0, 1].map(goal), delaySteps: 3, horizon: 2 }, 10), /prefix/i);
  assert.throws(() => prefixFeed({ prefix: Array.from({ length: 12 }, (_, i) => goal(i)), delaySteps: 11, horizon: 10 }, 10), /delay/i);
  assert.throws(() => prefixFeed({ prefix: [[NaN, 0, 0, 0, 0, 0], goal(1), goal(2)], delaySteps: 3, horizon: 3 }, 10), /finite/i);
});
