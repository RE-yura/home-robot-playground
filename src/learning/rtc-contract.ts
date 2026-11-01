import type { RtcContinuation } from "./learned-rtc-rollout.ts";
import { LEARNED_EXTERNAL_IDENTITY, type LearnedExternalIdentity } from "./learned-onnx-contract.ts";

// Private diagnostic candidates. They are not selected by the hosted auto-loader. Each is rebuilt from the shared
// public graph by tools/vla-train/rtc_onnx.py with the same weights; only the per-step guidance cap (max_gain) differs.
export const RTC_GRAPH_CANDIDATES = {
  max10: { bytes: 28680197, sha256: "5f3a109eacdcfcf57e60d9110d20beafd2c6f75ee91d6dfd82efb98b063ccebc", file: "graph.onnx" },
  max5: { bytes: 28680196, sha256: "e9fb035a105e047a9a78070dbbc2e3bcb83a7ac7ede2606405c2f2d5d1b834e7", file: "graph-max5.onnx" },
  max2: { bytes: 28680196, sha256: "7b0d46223dac217d558ecb0576bd7566ea352222d5e37bf4c5d2b3128ac30a54", file: "graph-max2.onnx" },
} as const;
export type RtcGraphVariant = keyof typeof RTC_GRAPH_CANDIDATES;
export const RTC_GRAPH_CANDIDATE = RTC_GRAPH_CANDIDATES.max10;
export function validateRtcIdentity(identity: LearnedExternalIdentity) {
  const w = LEARNED_EXTERNAL_IDENTITY.weights;
  if (!Object.values(RTC_GRAPH_CANDIDATES).some(g => identity.graph.bytes === g.bytes && identity.graph.sha256 === g.sha256) ||
    identity.weights.path !== w.path || identity.weights.bytes !== w.bytes || identity.weights.sha256 !== w.sha256) throw new Error("RTC diagnostic model identity mismatch");
  return identity;
}
export function rtcContextFeed(context: RtcContinuation | null) {
  const prefix = new Float32Array(96), weights = new Float32Array(16);
  if (context === null) return { prefix, weights, enabled: false };
  const { delaySteps: delay, horizon, prefix: previous } = context;
  if (!Array.isArray(previous) || previous.length < 1 || previous.length > 16 || previous.some(a => !Array.isArray(a) || a.length !== 6 || !a.every(Number.isFinite))) throw new Error("RTC prefix needs finite six-channel commands");
  if (!Number.isInteger(delay) || !Number.isInteger(horizon) || delay < 0 || delay > horizon || horizon > previous.length || horizon > 10) throw new Error("Invalid RTC delay/coverage");
  previous.forEach((a, i) => prefix.set(a, i * 6));
  if (!prefix.every(Number.isFinite)) throw new Error("RTC prefix exceeds finite Float32 range");
  for (let i = 0; i < horizon; i++) {
    const w = i < delay ? 1 : Math.fround(1 - (i - delay + 1) / (horizon - delay + 1));
    weights[i] = i < delay ? 1 : Math.fround(Math.fround(w * Math.fround(Math.expm1(w))) / Math.fround(Math.E - 1));
  }
  return { prefix, weights, enabled: true };
}
