import { RTC_GRAPH_CANDIDATES, type RtcGraphVariant } from "./rtc-contract.ts";

export function chooseRtcTiming(samples: { inferenceMs: number }[], coverageMs = 600) {
  if (samples.length < 3 || samples.some(s => !Number.isFinite(s.inferenceMs) || s.inferenceMs <= 0)) throw new Error("Three valid RTC timing samples required");
  if (!Number.isFinite(coverageMs) || coverageMs < 100 || coverageMs > 1000) throw new Error("Invalid RTC coverage budget");
  const warmLatencyMs = Math.max(...samples.slice(-2).map(s => s.inferenceMs));
  const playbackRate = Math.min(1, coverageMs / warmLatencyMs);
  if (playbackRate < .1) throw new Error("Inference too slow for the current RTC coverage budget");
  const delaySteps = Math.ceil(warmLatencyMs / 1000 * playbackRate * 10 - 1e-8) + 1;
  return { playbackRate, delaySteps, warmLatencyMs };
}

// Goals run per continuous chunk before the next capture: min(12, 16 - delay) by default. An override (?rtcExec=N) must
// keep one request in flight (N > delay) and the delay's goals queued (N <= 16 - delay); a smaller N replans more often
// without changing the playback rate.
export function continuousExecutionSteps(delaySteps: number, override: string | null) {
  const steps = override === null ? Math.min(12, 16 - delaySteps) : Number(override);
  if (override === "" || !Number.isInteger(steps) || steps <= delaySteps || steps > 16 - delaySteps)
    throw new Error(`推論中に進む物理時間に対して実行コマ数 ${override ?? steps} は使えません（execution steps must exceed the delay ${delaySteps} and leave it queued）`);
  return steps;
}

// The existing hosted model loader is deliberately unchanged. A local diagnostic
// server must explicitly supply this checked candidate; it is not published yet.
export async function loadRtcDiagnosticGraph(signal: AbortSignal, variant: RtcGraphVariant = "max10") {
  const candidate = RTC_GRAPH_CANDIDATES[variant];
  if (!candidate) throw new Error("Unknown RTC diagnostic graph");
  const response = await fetch(`/diagnostics/rtc/${candidate.file}`, { signal });
  if (!response.ok) throw new Error("RTC diagnostic graph is unavailable");
  const bytes = await response.arrayBuffer();
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), v => v.toString(16).padStart(2, "0")).join("");
  if (bytes.byteLength !== candidate.bytes || hash !== candidate.sha256) throw new Error("RTC diagnostic graph identity mismatch");
  return new File([bytes], "hsr-smolvla.onnx");
}
