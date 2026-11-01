import test from "node:test";
import assert from "node:assert/strict";
import { parseLanguageGoal, validateLanguageRequest, learnedTarget, LEARNED_INSTRUCTIONS, LANGUAGE_SYSTEM, STAGES_SYSTEM, parseStages, SKILL_SYSTEM, parseSkill } from "../src/learning/language-goals.ts";

const step = (skill: string, ja: string, en: string) => ({ skill, ja, en });
test("the model names a skill for every step; learned skills get the training sentence, other steps keep their own", () => {
  const mug = step("mug", "コップをかごに入れる", "Put the mug into the basket"), out = step("other", "コップをかごから出す", "Take the mug out of the basket.");
  assert.deepEqual(parseLanguageGoal(JSON.stringify({ status: "proposed", tasks: [mug, out] })),
    { status: "proposed", tasks: [{ ja: "コップをかごに入れる", en: "Put the mug in the basket." }, { ja: "コップをかごから出す", en: "Take the mug out of the basket." }] });
  assert.deepEqual(parseLanguageGoal(JSON.stringify({ status: "proposed", tasks: [step("ball", "ボールを入れる", "")] })).tasks, [{ ja: "ボールを入れる", en: "Put the tennis ball in the basket." }]);
  assert.deepEqual(parseLanguageGoal('{"status":"clarify","tasks":[]}'), { status: "clarify", tasks: [] });
  assert.equal(learnedTarget("Put the mug in the basket."), "mug"); assert.equal(learnedTarget("Put the tennis ball in the basket."), "ball");
  for (const en of ["put the mug in the basket.", "Put the mug in the basket", "Take the mug out of the basket.", " Put the mug in the basket."]) assert.equal(learnedTarget(en), null, en);
  assert.deepEqual(LEARNED_INSTRUCTIONS, { mug: "Put the mug in the basket.", ball: "Put the tennis ball in the basket." });
});
test("model text around the JSON object is ignored, but the object itself must match the schema exactly", () => {
  const goal = { status: "proposed", tasks: [step("other", "冷蔵庫を開ける", "Open the fridge.")] };
  const parsed = { status: "proposed", tasks: [{ ja: "冷蔵庫を開ける", en: "Open the fridge." }] };
  assert.deepEqual(parseLanguageGoal("```json\n" + JSON.stringify(goal) + "\n```"), parsed);
  assert.deepEqual(parseLanguageGoal("Here is the plan: " + JSON.stringify(goal) + " Done."), parsed);
  const five = Array.from({ length: 5 }, () => step("other", "コップを持つ", "Hold the mug."));
  for (const value of [
    { status: "proposed", tasks: [] }, { status: "proposed", tasks: five }, { status: "clarify", tasks: [step("other", "a", "b")] },
    { status: "unsupported", tasks: [] }, { status: "proposed", tasks: [{ skill: "other", ja: "コップ", en: "Hold the mug.", extra: 1 }] },
    { status: "proposed", tasks: [{ ja: "コップ", en: "Hold the mug." }] }, { status: "proposed", tasks: [step("fridge", "冷蔵庫", "Open it.")] },
    { status: "proposed", tasks: [step("other", "", "Hold the mug.")] }, { status: "proposed", tasks: [step("other", "コップ", "コップを持つ")] },
    { status: "proposed", tasks: [step("other", "コップ", "")] }, { status: "proposed", tasks: [step("other", "コップ", "x".repeat(161))] },
    { status: "proposed", tasks: [step("other", "コ\nップ", "Hold the mug.")] }, { status: "proposed", tasks: ["mug"] },
    { status: "proposed", tasks: [step("other", "コップ", "Hold the mug.")], note: "x" },
  ]) assert.throws(() => parseLanguageGoal(JSON.stringify(value)), /language goal/i, JSON.stringify(value));
  for (const text of ["", "no json here", "{not json}"]) assert.throws(() => parseLanguageGoal(text), /language goal/i, text);
});
test("stages of an unlearned step come from a separate short answer: a JSON array of 1-5 short Japanese lines", () => {
  assert.deepEqual(parseStages('["コップに近づく","掴んで持ち上げる","机に置く"]'), ["コップに近づく", "掴んで持ち上げる", "机に置く"]);
  assert.deepEqual(parseStages('```json\n["近づく"]\n```'), ["近づく"]);
  assert.deepEqual(parseStages('["コップを机の端に置く","コップを机の端に置く"]'), ["コップを机の端に置く"], "repeated lines are shown once");
  for (const text of ["", "[]", JSON.stringify(Array(6).fill("近づく")), '[""]', '["近\\nづく"]', JSON.stringify(["x".repeat(31)]), '"近づく"', "[1]", "{}"])
    assert.throws(() => parseStages(text), /stages/i, text);
  assert.match(STAGES_SYSTEM, /JSON array/); assert.doesNotMatch(LANGUAGE_SYSTEM, /stages/, "the plan prompt itself does not ask for stages");
});
test("a separate per-step skill answer overrides the plan's label; a learned label is dropped only when the model wrote other English", () => {
  const raw = (steps: object[]) => JSON.stringify({ status: "proposed", tasks: steps });
  const out = { skill: "mug", ja: "コップをかごから出す", en: "Take the mug out of the basket." };
  assert.deepEqual(parseLanguageGoal(raw([out]), ["other"]).tasks, [{ ja: out.ja, en: out.en }]);
  assert.deepEqual(parseLanguageGoal(raw([out])).tasks, [{ ja: out.ja, en: "Put the mug in the basket." }], "without the answer the plan's label stands");
  const sure = { skill: "mug", ja: "コップを置く", en: "Put the mug in the basket." };
  assert.deepEqual(parseLanguageGoal(raw([sure]), ["other"]).tasks, [{ ja: sure.ja, en: "Put the mug in the basket." }]);
  const near = { skill: "other", ja: "ボールをかごへ", en: "Put the ball into the basket." };
  assert.deepEqual(parseLanguageGoal(raw([near]), ["ball"]).tasks, [{ ja: near.ja, en: "Put the tennis ball in the basket." }]);
  assert.deepEqual(parseLanguageGoal(raw([near, out]), [null, "other"]).tasks.map(t => t.en), ["Put the ball into the basket.", "Take the mug out of the basket."]);
  assert.throws(() => parseLanguageGoal(raw([near]), ["ball", "mug"]), /language goal/i);
  for (const [text, skill] of [["mug", "mug"], [" Ball.", "ball"], ["other\n", "other"], ["OTHER", "other"]]) assert.equal(parseSkill(text), skill, text);
  for (const text of ["", "fridge", "maybe mug"]) assert.throws(() => parseSkill(text), /skill/i, text);
  assert.match(SKILL_SYSTEM, /mug, ball or other/);
});
test("the prompt keeps the clarification rule for missing objects and bare pronouns", () => {
  assert.match(LANGUAGE_SYSTEM, /bare pronoun/); assert.match(LANGUAGE_SYSTEM, /clarify/);
});
test("request validation preserves the exact instruction and rejects empty/oversized instructions", () => {
  assert.equal(validateLanguageRequest("コップの次にボールを運んで。"), "コップの次にボールを運んで。");
  assert.equal(validateLanguageRequest("  mug  "), "  mug  ");
  for (const value of ["", " \n ", "x".repeat(501), null, { instruction: "mug" }]) assert.throws(() => validateLanguageRequest(value), /instruction/i);
});
