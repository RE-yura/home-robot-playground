import type { LearnedInput, LearnedPolicy, LearnedPrediction } from "./learned-rollout.ts";
import { learnedModel, validateLearnedManifest, type LearnedManifest, type LearnedReferenceCase } from "./learned-onnx-contract.ts";
import type { RtcContinuation } from "./learned-rtc-rollout.ts";

export class LearnedOnnxPolicy implements LearnedPolicy {
  readonly ready: Promise<{ provider: string; modelSha256: string; weightSha256: string; numThreads: number; crossOriginIsolated: boolean; sampling: string }>;
  private readonly worker: Worker;
  private nextId = 0;
  private disposed = false;
  whenIdle: Promise<void> = Promise.resolve();
  get isDisposed() { return this.disposed; }
  private readonly pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  constructor(model: File, manifest: LearnedManifest, weights: File, options: { rtc?: boolean; modelId?: string; members?: 1 | 4; noise?: "zero" } = {}) {
    const checked = validateLearnedManifest(manifest, learnedModel(options.modelId));
    this.worker = new Worker(new URL("./learned-onnx-worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = event => {
      const request = this.pending.get(event.data.id); if (!request) return;
      clearTimeout(request.timer); this.pending.delete(event.data.id);
      if (event.data.error) request.reject(new Error(event.data.error)); else request.resolve(event.data.value);
    };
    this.worker.onerror = event => { this.failAll(new Error(event.message || "Learned worker failed")); this.dispose(); };
    this.ready = this.request({ type: "load", model, weights, manifest: checked, rtc: options.rtc ?? false, modelId: options.modelId, ...(options.members ? { members: options.members } : {}), ...(options.noise ? { noise: options.noise } : {}) }, 300000) as typeof this.ready;
  }
  private request(message: object, timeout: number): Promise<unknown> {
    if (this.disposed) return Promise.reject(new Error("Learned inference cancelled"));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.failAll(new Error("Learned inference timed out")); this.dispose(); }, timeout);
      this.pending.set(id, { resolve, reject, timer }); this.worker.postMessage({ ...message, id });
    });
  }
  infer(input: LearnedInput, continuation?: RtcContinuation | null): Promise<LearnedPrediction> {
    const prediction = this.ready.then(() => this.request({ type: "infer", input, ...(continuation !== undefined ? { continuation } : {}) }, 180000) as Promise<LearnedPrediction>);
    this.whenIdle = prediction.then(() => {}, () => {});
    return prediction;
  }
  async parity(reference: LearnedReferenceCase) { await this.ready; return this.request({ type: "parity", reference }, 180000); }
  private failAll(error: Error) { for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(error); } this.pending.clear(); }
  dispose() { if (this.disposed) return; this.disposed = true; this.worker.terminate(); this.failAll(new Error("Learned inference cancelled")); }
}
