/// <reference lib="webworker" />
import { AutoTokenizer, AutoModelForCausalLM, TextStreamer, env, LogitsProcessor, LogitsProcessorList, random, type Tensor } from "@huggingface/transformers";
import wasmUrl from "virtual:learned-wasm";
import wasmModuleUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { fetchLearnedWasm } from "./learned-wasm.ts";
import { LANGUAGE_SYSTEM, STAGES_SYSTEM, SKILL_SYSTEM, validateLanguageRequest, parseLanguageGoal, parseStages, parseSkill } from "./language-goals.ts";
import { greedyAfterThinking, validateLanguageSeed, END_THINKING, THINKING_TOKEN_LIMIT, GOAL_TOKEN_LIMIT } from "./language-thinking.ts";
import { downloadLanguageParts, validateLanguageParts } from "./language-download.ts";
import { LANGUAGE_MODEL } from "./language-model.ts";

const MODEL = "qwen3-1.7b-external", receipts = new Map<string, { bytes: number; sha256: string }>();
let tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>, model: Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>>;
let eos = 0, busy = false;
env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = "/models/language/";
// Our cache verifies content even on cache hits; the library's cache would skip
// the verification hook. No silent replacement of mismatched cached weights.
env.useBrowserCache = false; env.useWasmCache = false;
env.backends.onnx.wasm!.numThreads = 1; env.backends.onnx.wasm!.wasmPaths = { mjs: wasmModuleUrl };
env.fetch = async (input, init) => {
  const url = new URL(String(input), self.location.href);
  if (url.origin !== self.location.origin || !url.pathname.startsWith(LANGUAGE_MODEL.path)) throw Error("Language model route is not admitted");
  const file = url.pathname.slice(LANGUAGE_MODEL.path.length), expected = LANGUAGE_MODEL.files[file as keyof typeof LANGUAGE_MODEL.files];
  if (!expected) throw Error("Unknown language model file");
  if (init?.method === "HEAD" || new Headers(init?.headers).has("Range")) return fetch(url, init);
  const cache = await caches.open("hsr-language-355-" + LANGUAGE_MODEL.revision);
  const saved = await cache.match(url.href);
  let response = saved;
  if (!response && expected[0] > 20 * 1024 * 1024) {
    const manifest = await fetch(LANGUAGE_MODEL.path + "manifest.json");
    if (!manifest.ok) throw Error("Language manifest download failed");
    const parts = validateLanguageParts(await manifest.json(), file);
    response = new Response(await downloadLanguageParts(parts, cache, undefined, (loaded,total) => self.postMessage({progress:{status:"language-download",loaded,total}})));
  }
  response ??= await fetch(url, init);
  if (!response.ok) throw Error(`Language model download failed (${response.status})`);
  const buffer = await response.clone().arrayBuffer();
  const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)), b => b.toString(16).padStart(2, "0")).join("");
  if (buffer.byteLength !== expected[0] || sha256 !== expected[1]) throw Error(`Language model identity mismatch: ${file}`);
  if (!saved && expected[0] <= 20 * 1024 * 1024) {
    try { await cache.put(url.href, response.clone()); } catch { /* Verified bytes remain usable without cache space. */ }
  }
  receipts.set(file, { bytes: buffer.byteLength, sha256 }); return response;
};
class GoalProcessor extends LogitsProcessor {
  branches: { prefix: number[]; tokens: number[]; scores: number[] }[] = [];
  promptLength = 0;
  generated: number[] = [];
  _call(ids: bigint[][], logits: Tensor): Tensor {
    if (logits.dims[0] !== 1 || logits.type !== "float32") throw Error("Unexpected language logit shape");
    this.generated = ids[0].slice(this.promptLength).map(Number);
    greedyAfterThinking(logits.data as Float32Array, this.generated);
    return logits;
  }
}
self.onmessage = async ({ data }) => {
  const id = data.id;
  if (busy) { self.postMessage({ id, error: "Language worker is busy" }); return; }
  busy = true;
  let diagnostic: unknown = null;
  try {
    if (data.type === "download") {
      // Fill the verified part cache only; "load" later reads (and re-verifies) the parts from it.
      let cache: Cache | undefined;
      try { cache = await caches.open("hsr-language-355-" + LANGUAGE_MODEL.revision); } catch { /* Without storage, "load" downloads directly. */ }
      if (cache) {
        const manifest = await fetch(LANGUAGE_MODEL.path + "manifest.json");
        if (!manifest.ok) throw Error("Language manifest download failed");
        const parts = await manifest.json();
        for (const [file, [bytes]] of Object.entries(LANGUAGE_MODEL.files)) if (bytes > 20 * 1024 * 1024)
          await downloadLanguageParts(validateLanguageParts(parts, file), cache, undefined, (loaded, total) => self.postMessage({ progress: { status: "language-download", loaded, total } }), false);
      }
      self.postMessage({ id, value: { downloaded: Boolean(cache) } });
    } else if (data.type === "load") {
      const began = performance.now(); env.backends.onnx.wasm!.wasmBinary = await fetchLearnedWasm(wasmUrl);
      const progress_callback = (progress: unknown) => self.postMessage({ progress });
      tokenizer = await AutoTokenizer.from_pretrained(MODEL, { progress_callback });
      model = await AutoModelForCausalLM.from_pretrained(MODEL, { device: "webgpu", dtype: "q4f16", use_external_data_format: true, progress_callback });
      eos = tokenizer.eos_token_id;
      if (!Number.isInteger(eos)) throw Error("Invalid language end token");
      if (receipts.size !== 6) throw Error("Incomplete language file identity");
      self.postMessage({ id, value: { upstream: LANGUAGE_MODEL.upstream, revision: LANGUAGE_MODEL.revision, provider: "webgpu", dtype: "q4f16", receipts: Object.fromEntries(receipts), initializeMs: performance.now() - began } });
    } else if (data.type === "interpret") {
      if (!model || !tokenizer) throw Error("Language model is not loaded");
      const instruction = validateLanguageRequest(data.instruction), began = performance.now();
      const options = { tokenize: true as const, return_dict: true as const, add_generation_prompt: true, enable_thinking: true };
      const encoded = tokenizer.apply_chat_template([{ role: "system", content: LANGUAGE_SYSTEM }, { role: "user", content: instruction }], options);
      const processor = new GoalProcessor(), list = new LogitsProcessorList(); processor.promptLength = encoded.input_ids.dims[1]; list.push(processor);
      const seed = validateLanguageSeed(data.seed);
      random.seed(seed);
      const streamedTokens: number[] = [];
      const streamer = new TextStreamer(tokenizer, { skip_prompt: true, callback_function: () => {}, token_callback_function: tokens => {
        const partialTokens = tokens.map(Number), offset = streamedTokens.length;
        if (offset + partialTokens.length > THINKING_TOKEN_LIMIT + GOAL_TOKEN_LIMIT) throw Error("Language stream budget exceeded");
        streamedTokens.push(...partialTokens);
        self.postMessage({ id, partialTokens, offset, branches: processor.branches, elapsedMs: performance.now() - began });
      } });
      const context = { instruction, seed, generationAttempted: true, inputTokenIds: Array.from(encoded.input_ids.data, Number) };
      self.postMessage({ id, diagnostic: { ...context, outputTokens: [], branches: [], elapsedMs: 0 } });
      let outputTokens: number[];
      try {
        const result = await model.generate({ ...encoded, max_new_tokens: THINKING_TOKEN_LIMIT + GOAL_TOKEN_LIMIT, do_sample: true,
          temperature: .6, top_k: 20, logits_processor: list, streamer }) as Tensor;
        outputTokens = Array.from(result.data.slice(encoded.input_ids.dims[1]), Number);
        if (JSON.stringify(outputTokens) !== JSON.stringify(streamedTokens)) throw Error("Language token stream mismatch");
      } catch (error) {
        diagnostic = { ...context, outputTokens: streamedTokens, branches: processor.branches, elapsedMs: performance.now() - began };
        throw error;
      }
      const endThinking = outputTokens.indexOf(END_THINKING);
      diagnostic = { ...context, outputTokens, branches: processor.branches, endThinking, elapsedMs: performance.now() - began };
      if (endThinking < 0 || outputTokens.at(-1) !== eos) throw Error("Language goal did not finish");
      const rawText = tokenizer.decode(outputTokens.slice(endThinking + 1), { skip_special_tokens: true });
      parseLanguageGoal(rawText);
      self.postMessage({ id, value: { ...(diagnostic as object), rawText } });
    } else if (data.type === "skill" || data.type === "stages") {
      // A short answer without reasoning, decoded greedily: whether one planned step is a learned skill, or the stages
      // of one unlearned step (for display only).
      if (!model || !tokenizer) throw Error("Language model is not loaded");
      const step = `${validateLanguageRequest(data.ja)} / ${validateLanguageRequest(data.en)}`;
      const options = { tokenize: true as const, return_dict: true as const, add_generation_prompt: true, enable_thinking: false };
      const skill = data.type === "skill";
      const encoded = tokenizer.apply_chat_template([{ role: "system", content: skill ? SKILL_SYSTEM : STAGES_SYSTEM }, { role: "user", content: step }], options);
      const result = await model.generate({ ...encoded, max_new_tokens: skill ? 4 : 96, do_sample: false }) as Tensor;
      const rawText = tokenizer.decode(Array.from(result.data.slice(encoded.input_ids.dims[1]), Number), { skip_special_tokens: true });
      self.postMessage({ id, value: skill ? { skill: parseSkill(rawText), rawText } : { stages: parseStages(rawText), rawText } });
    } else throw Error("Unknown language request");
  } catch (error) { self.postMessage({ id, error: String(error), diagnostic }); }
  finally { busy = false; }
};
