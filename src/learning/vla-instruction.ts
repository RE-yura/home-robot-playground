// Language input of the VLA for instructions other than the two exported ones. The learned rows come from the
// browser manifest unchanged; any other English instruction is tokenized by the training tokenizer (SmolVLM2, copied
// from the checkpoint's processor) exactly like the LeRobot preprocessor: text + "\n", no special tokens, right-padded
// with id 2 to 48 tokens. The tokenizer is admitted only after it reproduces both exported rows.
export const VLA_TOKENIZER_PATH = "/models/vla-tokenizer/";
export const VLA_TOKENIZER_FILES = {
  "tokenizer.json": { bytes: 3548256, sha256: "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a" },
  "tokenizer_config.json": { bytes: 28626, sha256: "dd9ce2ab89a3dd881bd9378f1a79b943a064b9275a7e1706d5b7b47b68977913" },
} as const;
export type VlaLanguageRow = { tokens: number[]; mask: boolean[] };
const LENGTH = 48, PAD = 2;

export function encodeVlaInstruction(encode: (text: string) => number[], text: string): VlaLanguageRow {
  const ids = encode(text + "\n");
  if (!ids.length || ids.length > LENGTH || !ids.every(id => Number.isSafeInteger(id) && id >= 0)) throw new Error(`Instruction does not fit the VLA's 48-token window (${ids.length} tokens)`);
  return { tokens: [...ids, ...Array(LENGTH - ids.length).fill(PAD)], mask: [...ids.map(() => true), ...Array(LENGTH - ids.length).fill(false)] };
}

export function verifyVlaTokenizer(encode: (text: string) => number[], exported: Record<string, VlaLanguageRow>) {
  const names = ["Put the mug in the basket.", "Put the tennis ball in the basket."];
  if (!names.every(name => exported[name]) || names.some(name => JSON.stringify(encodeVlaInstruction(encode, name)) !== JSON.stringify(exported[name])))
    throw new Error("VLA tokenizer disagrees with the exported instruction tokens");
}
