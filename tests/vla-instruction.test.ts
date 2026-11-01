import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { PreTrainedTokenizer } from "@huggingface/transformers";
import { encodeVlaInstruction, verifyVlaTokenizer, VLA_TOKENIZER_FILES } from "../src/learning/vla-instruction.ts";

const file = (name: string) => readFileSync(new URL(`../public/models/vla-tokenizer/${name}`, import.meta.url));
const tokenizer = new PreTrainedTokenizer(JSON.parse(file("tokenizer.json").toString()), JSON.parse(file("tokenizer_config.json").toString()));
const encode = (text: string) => tokenizer.encode(text, { add_special_tokens: false });
// The two rows exported with prefix-rtc-v9 (browser-manifest.json): text + newline, right-padded with id 2 to 48.
const row = (ids: number[]) => ({ tokens: [...ids, ...Array(48 - ids.length).fill(2)], mask: [...ids.map(() => true), ...Array(48 - ids.length).fill(false)] });
const EXPORTED = {
  "Put the mug in the basket.": row([22030, 260, 283, 1433, 281, 260, 11831, 30, 198]),
  "Put the tennis ball in the basket.": row([22030, 260, 21059, 4405, 281, 260, 11831, 30, 198]),
};

test("the pinned training tokenizer files are the ones served", () => {
  for (const [name, pin] of Object.entries(VLA_TOKENIZER_FILES)) {
    const bytes = file(name);
    assert.equal(bytes.length, pin.bytes, name); assert.equal(createHash("sha256").update(bytes).digest("hex"), pin.sha256, name);
  }
});
test("the training tokenizer reproduces both exported instruction rows and encodes any other English instruction", () => {
  verifyVlaTokenizer(encode, EXPORTED);
  for (const [text, expected] of Object.entries(EXPORTED)) assert.deepEqual(encodeVlaInstruction(encode, text), expected);
  const out = encodeVlaInstruction(encode, "Take the mug out of the basket.");
  assert.equal(out.mask.filter(Boolean).length, 10); assert.equal(out.tokens.length, 48); assert.equal(out.tokens[9], 198); assert.equal(out.tokens[10], 2);
});
test("an instruction longer than the 48-token window is refused instead of truncated", () => {
  assert.throws(() => encodeVlaInstruction(encode, "pick ".repeat(60).trim() + "."), /48/);
});
test("a tokenizer that encodes either learned sentence differently is refused", () => {
  const shifted = (text: string) => { const ids = encode(text); return text.startsWith("Put the tennis") ? [ids[0] + 1, ...ids.slice(1)] : ids; };
  assert.throws(() => verifyVlaTokenizer(shifted, EXPORTED), /tokenizer/i);
  assert.throws(() => verifyVlaTokenizer(encode, { "Put the mug in the basket.": EXPORTED["Put the mug in the basket."] }), /tokenizer/i);
});
