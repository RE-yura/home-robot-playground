export function meanPair(a: Float32Array, b: Float32Array) {
  if (a.length !== 96 || b.length !== 96) throw new Error("Expected two 16 × 6 predictions");
  const mean = new Float32Array(96);
  for (let i = 0; i < mean.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) throw new Error("Nonfinite prediction");
    mean[i] = Math.fround(Math.fround(a[i] + b[i]) * 0.5);
    if (!Number.isFinite(mean[i])) throw new Error("Nonfinite float32 mean");
  }
  return mean;
}
