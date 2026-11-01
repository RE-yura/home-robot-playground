import test from "node:test";
import assert from "node:assert/strict";
import { greedyAfterThinking, validateLanguageSeed, END_THINKING, THINKING_TOKEN_LIMIT, GOAL_TOKEN_LIMIT } from "../src/learning/language-thinking.ts";

test("the frozen full numeric RNG seed is supported without truncation to uint32", () => {
  assert.equal(validateLanguageSeed(20261019360), 20261019360);
  for (const value of [-1, .5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, "20261019360"]) assert.throws(() => validateLanguageSeed(value), /seed/);
});
test("native reasoning logits remain unchanged until the actual closing marker", () => {
  const logits = new Float32Array([4, 8, 3, 7, 5]);
  assert.equal(greedyAfterThinking(logits, [9, 8, 7]), null);
  assert.deepEqual([...logits], [4, 8, 3, 7, 5]);
});
test("reasoning that reaches its budget is closed: the end marker is forced, then the goal follows", () => {
  const logits = new Float32Array(END_THINKING + 8).fill(1); logits[3] = 9;
  assert.deepEqual(greedyAfterThinking(logits, Array(THINKING_TOKEN_LIMIT).fill(1)), { chosenToken: END_THINKING, score: 1, thinkingTokens: THINKING_TOKEN_LIMIT, forced: true });
  assert.equal(logits.filter(Number.isFinite).length, 1); assert.equal(logits[END_THINKING], 1);
  assert.throws(() => greedyAfterThinking(new Float32Array(END_THINKING + 8), Array(THINKING_TOKEN_LIMIT + 1).fill(1)), /reasoning token limit/);
});
test("after thinking the goal is decoded greedily: only the largest original neural score survives", () => {
  const logits = new Float32Array([99, 8, 3, 7, 5]);
  assert.deepEqual(greedyAfterThinking(logits, [9, END_THINKING]), { chosenToken: 0, score: 99, thinkingTokens: 2 });
  assert.deepEqual([...logits], [99, -Infinity, -Infinity, -Infinity, -Infinity]);
  const later = new Float32Array([1, 8, 3, 7, 5]);
  assert.equal(greedyAfterThinking(later, [9, END_THINKING, 4, 4])!.chosenToken, 1);
  assert.equal(GOAL_TOKEN_LIMIT, 192);
  assert.throws(() => greedyAfterThinking(new Float32Array([1, 2]), [END_THINKING, ...Array(GOAL_TOKEN_LIMIT).fill(1)]), /goal token limit/);
  assert.throws(() => greedyAfterThinking(new Float32Array([NaN, NaN]), [END_THINKING]), /finite/);
});
