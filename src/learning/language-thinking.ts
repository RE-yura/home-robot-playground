export const END_THINKING = 151668;
export const THINKING_TOKEN_LIMIT = 1024;
export const GOAL_TOKEN_LIMIT = 192;

export function validateLanguageSeed(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) throw Error("Invalid language sampling seed");
  return value;
}

// Leave native reasoning untouched; when it reaches its budget, close it by forcing the end marker (budget forcing).
// After the marker, decode the goal greedily: the largest original neural score is the only token left, so the same
// reasoning gives the same goal. Its schema is checked when it ends.
export function greedyAfterThinking(logits: Float32Array, generated: number[]) {
  const endThinking = generated.indexOf(END_THINKING);
  if (endThinking < 0) {
    if (generated.length > THINKING_TOKEN_LIMIT) throw new Error("Language reasoning token limit exceeded");
    if (generated.length < THINKING_TOKEN_LIMIT) return null;
    const score = Number.isFinite(logits[END_THINKING]) ? logits[END_THINKING] : 0;
    logits.fill(-Infinity); logits[END_THINKING] = score;
    return { chosenToken: END_THINKING, score, thinkingTokens: THINKING_TOKEN_LIMIT, forced: true };
  }
  if (endThinking > THINKING_TOKEN_LIMIT) throw new Error("Language reasoning token limit exceeded");
  if (generated.length - endThinking - 1 >= GOAL_TOKEN_LIMIT) throw new Error("Language goal token limit exceeded");
  let best = -1;
  for (let i = 0; i < logits.length; i++) if (Number.isFinite(logits[i]) && (best < 0 || logits[i] > logits[best])) best = i;
  if (best < 0) throw new Error("Language logits must be finite");
  const score = logits[best]; logits.fill(-Infinity); logits[best] = score;
  return { chosenToken: best, score, thinkingTokens: endThinking + 1 };
}
