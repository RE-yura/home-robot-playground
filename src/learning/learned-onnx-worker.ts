/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/webgpu";
import { meanPair } from "./learned-antithetic-mean.ts";
import wasmUrl from "virtual:learned-wasm";
import { fetchLearnedWasm } from "./learned-wasm.ts";
import wasmModuleUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { compareLearnedActions, learnedModel, validateLearnedManifest, validateLearnedExternalIdentity, PUBLIC_LEARNED_MODEL, type LearnedManifest, type LearnedReferenceCase } from "./learned-onnx-contract.ts";
import type { LearnedInput } from "./learned-rollout.ts";
import type { RtcContinuation } from "./learned-rtc-rollout.ts";
import type { SixValues } from "./recording.ts";
import { rtcContextFeed, validateRtcIdentity, RTC_GRAPH_CANDIDATES } from "./rtc-contract.ts";
import { prefixFeed } from "./prefix-conditioning.ts";
import { encodeVlaInstruction, verifyVlaTokenizer, VLA_TOKENIZER_FILES, VLA_TOKENIZER_PATH } from "./vla-instruction.ts";

ort.env.wasm.wasmPaths = { mjs: wasmModuleUrl };
ort.env.wasm.numThreads = self.crossOriginIsolated ? 4 : 1; ort.env.wasm.proxy = false;
let session: ort.InferenceSession | null = null, manifest: LearnedManifest | null = null;
// prefixDelayLimit > 0 only for a model trained with action-prefix conditioning: its graph takes the committed goals.
// memberCount 1 is a speed diagnostic: one sample from the single-member graph, no antithetic mean.
// zeroNoise (diagnostic): every member starts its flow at zero noise instead of a Gaussian draw.
let busy = false, sharedGraph = false, rtcGraph = false, prefixDelayLimit = 0, memberCount = 4, zeroNoise = false;
const sharedOutputs = Array.from({ length: 4 }, (_, i) => `__ensemble_member_${i}/hsr_action_chunk`);
const names = ["head_rgb", "hand_rgb", "language_tokens", "language_mask", "hsr_state", "initial_noise"];
// Instructions outside the exported table (plan steps the VLA was never trained on) are tokenized by the pinned
// training tokenizer, loaded once and admitted only after it reproduces both exported rows.
let vlaEncode: Promise<(text: string) => number[]> | null = null;
async function loadVlaEncode(): Promise<(text: string) => number[]> {
  const [json, config] = await Promise.all(Object.entries(VLA_TOKENIZER_FILES).map(async ([name, pin]) => {
    const bytes = new Uint8Array(await (await fetch(VLA_TOKENIZER_PATH + name)).arrayBuffer());
    const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), b => b.toString(16).padStart(2, "0")).join("");
    if (bytes.length !== pin.bytes || sha256 !== pin.sha256) throw new Error(`VLA tokenizer file identity mismatch: ${name}`);
    return JSON.parse(new TextDecoder().decode(bytes));
  }));
  const { PreTrainedTokenizer } = await import("@huggingface/transformers");
  const tokenizer = new PreTrainedTokenizer(json, config), encode = (text: string) => tokenizer.encode(text, { add_special_tokens: false });
  verifyVlaTokenizer(encode, manifest!.instructions);
  return encode;
}
async function languageRow(instruction: string) {
  const row = (manifest!.instructions as Record<string, { tokens: number[]; mask: boolean[] } | undefined>)[instruction];
  if (row) return row;
  if (typeof instruction !== "string" || !/^[\x20-\x7e]{1,160}$/.test(instruction)) throw new Error("Invalid learned instruction");
  vlaEncode ??= loadVlaEncode().catch(error => { vlaEncode = null; throw error; });
  return encodeVlaInstruction(await vlaEncode, instruction);
}
function noise() {
  if (zeroNoise) return new Float32Array(512);
  const uniform = crypto.getRandomValues(new Uint32Array(512)), values = new Float32Array(512);
  for (let i = 0; i < 512; i += 2) {
    const radius = Math.sqrt(-2 * Math.log((uniform[i] + 1) / 4294967297)), angle = 2 * Math.PI * (uniform[i + 1] + 1) / 4294967297;
    values[i] = radius * Math.cos(angle); values[i + 1] = radius * Math.sin(angle);
  }
  return values;
}
async function image(dataUrl: string) {
  const bitmap = await createImageBitmap(await (await fetch(dataUrl)).blob(), { colorSpaceConversion: "none" });
  try {
    if (bitmap.width !== 320 || bitmap.height !== 240) throw new Error("Unexpected camera dimensions");
    const canvas = new OffscreenCanvas(320, 240), context = canvas.getContext("2d")!;
    context.drawImage(bitmap, 0, 0); const pixels = context.getImageData(0, 0, 320, 240).data;
    const rgb = new Float32Array(3 * 240 * 320);
    for (let i = 0; i < 240 * 320; i++) for (let c = 0; c < 3; c++) rgb[c * 240 * 320 + i] = pixels[i * 4 + c] / 255;
    return rgb;
  } finally { bitmap.close(); }
}
async function run(head: Float32Array, hand: Float32Array, tokens: number[], mask: boolean[], state: number[], initial: Float32Array, second?: Float32Array, continuation: RtcContinuation | null = null) {
  if (!session || !manifest) throw new Error("Learned model is not ready");
  if (head.length !== 230400 || hand.length !== 230400 || state.length !== 6 || initial.length !== 512 || tokens.length !== 48 || mask.length !== 48 ||
    !head.every(v => Number.isFinite(v) && v >= 0 && v <= 1) || !hand.every(v => Number.isFinite(v) && v >= 0 && v <= 1) ||
    !state.every(Number.isFinite) || !initial.every(Number.isFinite) ||
    (sharedGraph ? !second || second.length !== 512 || !second.every(Number.isFinite) : second !== undefined) || !tokens.every(v => Number.isSafeInteger(v) && v >= 0) || !mask.every(v => typeof v === "boolean"))
    throw new Error("Invalid learned input shapes/values");
  const context = rtcGraph ? rtcContextFeed(continuation) : null, committed = prefixDelayLimit ? prefixFeed(continuation, prefixDelayLimit) : null;
  const feed: Record<string, ort.Tensor> = { head_rgb: new ort.Tensor("float32", head, [1, 3, 240, 320]), hand_rgb: new ort.Tensor("float32", hand, [1, 3, 240, 320]),
    language_tokens: new ort.Tensor("int64", BigInt64Array.from(tokens, BigInt), [1, 48]),
    language_mask: new ort.Tensor("bool", Uint8Array.from(mask, Number), [1, 48]),
    hsr_state: new ort.Tensor("float32", Float32Array.from(state), [1, 6]), initial_noise: new ort.Tensor("float32", initial, [1, 16, 32]) };
  if (sharedGraph) feed.second_noise = new ort.Tensor("float32", second!, [1, 16, 32]);
  if (committed) {
    feed.action_prefix = new ort.Tensor("float32", committed.prefix, [1, 16, 6]);
    feed.prefix_mask = new ort.Tensor("float32", committed.mask, [1, 16]);
  }
  if (context) {
    feed.rtc_prefix = new ort.Tensor("float32", context.prefix, [1, 16, 6]);
    feed.rtc_weights = new ort.Tensor("float32", context.weights, [1, 16, 1]);
    feed.rtc_enabled = new ort.Tensor("bool", Uint8Array.of(Number(context.enabled)), []);
  }
  let outputs: Record<string, ort.Tensor> | null = null;
  try {
    outputs = await session.run(feed);
    const expected = sharedGraph ? sharedOutputs : ["hsr_action_chunk"];
    if (Object.keys(outputs).length !== expected.length || !expected.every(name => name in outputs!)) throw new Error("Unexpected learned output names");
    return expected.map(name => {
      const result = outputs![name];
      if (result.type !== "float32" || result.dims.join(",") !== "1,16,6" || !(result.data instanceof Float32Array) || !result.data.every(Number.isFinite))
        throw new Error("Unexpected learned output");
      return new Float32Array(result.data);
    });
  } finally { Object.values(feed).forEach(t => t.dispose()); if (outputs) Object.values(outputs).forEach(t => t.dispose()); }
}
type Request = { id: number; type: "load"; model: File; weights: File; manifest: unknown; rtc?: boolean; modelId?: string; members?: 1 | 4; noise?: "zero" } |
  { id: number; type: "infer"; input: LearnedInput; continuation?: RtcContinuation | null } | { id: number; type: "parity"; reference: LearnedReferenceCase };
self.onmessage = async (event: MessageEvent<Request>) => {
  const message = event.data;
  if (busy) { self.postMessage({ id: message.id, error: "Learned worker is busy" }); return; }
  busy = true;
  try {
    if (message.type === "load") {
      if (session) throw new Error("Learned model already loaded");
      const pinned = learnedModel(message.modelId);
      if (message.rtc && pinned !== PUBLIC_LEARNED_MODEL) throw new Error("RTC diagnostics use the public model only");
      manifest = validateLearnedManifest(message.manifest, pinned);
      if (!message.weights || message.weights.name !== "hsr-smolvla.weights") throw new Error("Exact external learned weights required");
      if (![pinned.graph.bytes, pinned.sharedGraph.bytes, ...(message.rtc ? Object.values(RTC_GRAPH_CANDIDATES).map(g => g.bytes) : [])].includes(message.model.size) || message.weights.size !== pinned.weights.bytes)
        throw new Error("Incorrect external model/weights size");
      const bytes = await message.model.arrayBuffer(), weightBytes = await message.weights.arrayBuffer();
      const hashFile = async (data: ArrayBuffer) => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", data)), v => v.toString(16).padStart(2, "0")).join("");
      const hash = await hashFile(bytes), weightHash = await hashFile(weightBytes);
      const loaded = { graph: { bytes: message.model.size, sha256: hash }, weights: { path: message.weights.name, bytes: message.weights.size, sha256: weightHash } };
      const identity = message.rtc ? validateRtcIdentity(loaded) : validateLearnedExternalIdentity(loaded, pinned);
      rtcGraph = message.rtc === true; sharedGraph = rtcGraph || hash === pinned.sharedGraph.sha256; prefixDelayLimit = pinned.prefixConditioning?.maxDelay ?? 0;
      memberCount = message.members ?? 4; zeroNoise = message.noise === "zero";
      if ((memberCount !== 1 && memberCount !== 4) || (memberCount === 1 && (sharedGraph || rtcGraph))) throw new Error("A one-member diagnostic needs the single-member graph");
      ort.env.wasm.wasmBinary = await fetchLearnedWasm(wasmUrl);
      session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ["webgpu"], graphOptimizationLevel: "all",
        externalData: [{ path: identity.weights.path, data: new Uint8Array(weightBytes) }] });
      const expectedInputs = [...names, ...(sharedGraph ? ["second_noise"] : []), ...(rtcGraph ? ["rtc_prefix", "rtc_weights", "rtc_enabled"] : []), ...(prefixDelayLimit ? ["action_prefix", "prefix_mask"] : [])], expectedOutputs = sharedGraph ? sharedOutputs : ["hsr_action_chunk"];
      if (session.inputNames.length !== expectedInputs.length || !expectedInputs.every(name => session!.inputNames.includes(name)) || session.outputNames.join(",") !== expectedOutputs.join(","))
        throw new Error("Learned model interface mismatch");
      self.postMessage({ id: message.id, value: { provider: "webgpu", modelId: pinned.id, modelSha256: hash, weightSha256: weightHash,
        numThreads: ort.env.wasm.numThreads, crossOriginIsolated: self.crossOriginIsolated,
        sampling: "two-independent-antithetic-pairs-f32", ...(rtcGraph ? { predictedActionsPerChunk: 16 } : { executedActionsPerChunk: 16 }), sharedGraph, rtcGraph } });
    } else if (message.type === "parity") {
      const r = message.reference, started = performance.now();
      const initial = Float32Array.from(r.initial_noise);
      // A prefix-conditioned reference may carry committed goals: a contiguous mask over the first goals.
      const delay = r.prefix_mask ? r.prefix_mask.filter(v => v === 1).length : 0;
      if (r.prefix_mask && (!r.action_prefix || r.action_prefix.length !== 96 || r.prefix_mask.length !== 16 || r.prefix_mask.some((v, i) => v !== (i < delay ? 1 : 0)))) throw new Error("Invalid reference prefix");
      const committed = delay ? { prefix: Array.from({ length: delay }, (_, i) => r.action_prefix!.slice(i * 6, i * 6 + 6) as SixValues), delaySteps: delay, horizon: delay } : null;
      const actions = (await run(Float32Array.from(r.head_rgb), Float32Array.from(r.hand_rgb), r.language_tokens, r.language_mask, r.hsr_state, initial, sharedGraph ? initial : undefined, committed))[0];
      const proof = compareLearnedActions(actions, Float32Array.from(r.expected), manifest!.actionNormalization.mean, manifest!.actionNormalization.std);
      self.postMessage({ id: message.id, value: { ...proof, inferenceMs: performance.now() - started, actions: Array.from(actions) } });
    } else {
      if (!rtcGraph && !prefixDelayLimit && message.continuation !== undefined) throw new Error("RTC continuation requires the explicit RTC diagnostic model or a prefix-conditioned model");
      const input = message.input;
      if (Object.keys(input).sort().join(",") !== "images,instruction,state" || input.images.length !== 2 ||
        !["head", "hand"].every(view => input.images.filter(image => image.view === view).length === 1) ||
        input.images[0].timestamp !== input.images[1].timestamp || input.state.length !== 6 || !input.state.every(Number.isFinite))
        throw new Error("Invalid synchronized learned input");
      const head = input.images.find(i => i.view === "head"), hand = input.images.find(i => i.view === "hand");
      if (!head || !hand) throw new Error("Missing learned RGB/instruction");
      const language = await languageRow(input.instruction);
      const [h, p] = await Promise.all([image(head.dataUrl), image(hand.dataUrl)]);
      const initialNoise = noise(), secondNoise = noise(), members: Float32Array[] = sharedGraph ?
        await run(h, p, language.tokens, language.mask, input.state, initialNoise, secondNoise, message.continuation ?? null) : [];
      if (!sharedGraph) for (const initial of memberCount === 1 ? [initialNoise] : [initialNoise, Float32Array.from(initialNoise, value => -value), secondNoise, Float32Array.from(secondNoise, value => -value)])
        members.push((await run(h, p, language.tokens, language.mask, input.state, initial, undefined, message.continuation ?? null))[0]);
      const actions = memberCount === 1 ? members[0] : meanPair(meanPair(members[0], members[1]), meanPair(members[2], members[3]));
      self.postMessage({ id: message.id, value: {
        actions: Array.from({ length: 16 }, (_, i) => Array.from(actions.subarray(i * 6, i * 6 + 6))),
        initialNoise: Array.from(initialNoise), ...(memberCount === 1 ? {} : { secondNoise: Array.from(secondNoise) }), memberActions: members.map(member => Array.from(member)),
      } });
    }
  } catch (error) {
    self.postMessage({ id: message.id, error: error instanceof Error ? error.message : String(error) });
    if (message.type === "load" && session) { await session.release(); session = null; }
  } finally { busy = false; }
};
