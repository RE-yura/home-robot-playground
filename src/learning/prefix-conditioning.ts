import type { RtcContinuation } from "./learned-rtc-rollout.ts";

// Inputs of a model fine-tuned with training-time action conditioning (arXiv 2512.05964): the executor's goals that
// will run while the model infers, in physical units, and a 0/1 mask over the 16 goals of the new chunk.
// A capture with nothing queued (the first one, or one taken while physics waits at the end of the queue) conditions
// on no goals, which is the original unconditioned sampling.
export function prefixFeed(continuation: RtcContinuation | null, maxDelay: number) {
  const prefix = new Float32Array(96), mask = new Float32Array(16);
  const delay = continuation && continuation.prefix.length ? continuation.delaySteps : 0;
  if (!Number.isInteger(delay) || delay < 0 || delay > maxDelay) throw new Error("Prefix delay is outside the trained range");
  if (continuation && delay > continuation.prefix.length) throw new Error("The queued prefix is shorter than the inference delay");
  for (let i = 0; i < delay; i++) {
    const goal = continuation!.prefix[i];
    if (goal.length !== 6 || !goal.every(Number.isFinite)) throw new Error("Prefix goals must be six finite values");
    prefix.set(goal, i * 6); mask[i] = 1;
  }
  return { prefix, mask, delay };
}
