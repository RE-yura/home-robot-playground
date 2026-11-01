/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "virtual:learned-wasm";
import wasmModuleUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { fetchLearnedWasm } from "./learned-wasm.ts";
import { decodeRecognitionViews, prepareRecognitionView, validateRecognitionManifest, validateRecognitionOutput, verifyRecognitionGraph } from "./recognition-detector.ts";

ort.env.wasm.wasmPaths = { mjs: wasmModuleUrl };
ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
let session: ort.InferenceSession | null = null, busy = false;

async function nativeRgb(dataUrl: string) {
  if (!/^data:image\/jpeg;base64,[A-Za-z0-9+/]+={0,2}$/.test(dataUrl)) throw new Error("Native head JPEG required");
  const encoded = await (await fetch(dataUrl)).blob();
  const bitmap = await createImageBitmap(encoded, { imageOrientation: "none", premultiplyAlpha: "none", colorSpaceConversion: "none" });
  const canvas = new OffscreenCanvas(320, 240);
  try {
    if (bitmap.width !== 320 || bitmap.height !== 240) throw new Error("Native 320×240 head image required");
    const context = canvas.getContext("2d", { alpha: false, colorSpace: "srgb", willReadFrequently: true });
    if (!context) throw new Error("Native image decoder unavailable");
    context.drawImage(bitmap, 0, 0);
    const rgba = context.getImageData(0, 0, 320, 240, { colorSpace: "srgb" }).data, rgb = new Uint8Array(320 * 240 * 3);
    for (let i = 0; i < 320 * 240; i++) {
      if (rgba[i * 4 + 3] !== 255) throw new Error("Opaque RGB image required");
      rgb.set(rgba.subarray(i * 4, i * 4 + 3), i * 3);
    }
    return rgb;
  } finally { bitmap.close(); canvas.width = 0; canvas.height = 0; }
}
async function load() {
  if (session) throw new Error("Recognition model already loaded");
  const response = await fetch("/models/recognition-t7/manifest.json");
  if (!response.ok) throw new Error("認識モデルがまだ公開されていません。モデルの配置後に再読み込みしてください。");
  const text = await response.text();
  if (text.length > 32768) throw new Error("Recognition manifest is too large");
  let manifest;
  try { manifest = validateRecognitionManifest(JSON.parse(text)); }
  catch { throw new Error("認識モデルの情報がまだ配置されていないか、対応していません。"); }
  const graph = await fetch(manifest.graph.path);
  if (!graph.ok) throw new Error("認識モデルのファイルを読み込めませんでした。");
  const bytes = await graph.arrayBuffer(); await verifyRecognitionGraph(bytes, manifest.graph);
  ort.env.wasm.wasmBinary = await fetchLearnedWasm(wasmUrl);
  session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ["wasm"] });
  if (session.inputNames.join() !== "images" || session.outputNames.join() !== "output") throw new Error("Unexpected T7 model interface");
  return { provider: "wasm", modelSha256: manifest.graph.sha256 };
}
async function infer(dataUrl: string) {
  if (!session) throw new Error("Recognition model is not ready");
  const began = performance.now(), rgb = await nativeRgb(dataUrl), raws: Float32Array[] = [], inferenceMs: number[] = [];
  for (let view = 0; view < 5; view++) {
    const input = new ort.Tensor("float32", prepareRecognitionView(rgb, view), [1, 3, 416, 416]);
    let outputs: Record<string, ort.Tensor> | null = null;
    try {
      const issued = performance.now(); outputs = await session.run({ images: input }); inferenceMs.push(performance.now() - issued);
      if (Object.keys(outputs).join() !== "output") throw new Error("Unexpected recognition outputs");
      const raw = validateRecognitionOutput(outputs.output).slice();
      // A malformed view rejects the whole snapshot; no partial boxes are emitted.
      if (!raw.every(Number.isFinite)) throw new Error("Nonfinite recognition output");
      raws.push(raw);
    } finally { input.dispose(); if (outputs) Object.values(outputs).forEach(t => t.dispose()); }
  }
  return { detections: decodeRecognitionViews(raws), inferenceMs, elapsedMs: performance.now() - began };
}
self.onmessage = async (event: MessageEvent<{ type: "load" } | { type: "snapshot"; dataUrl: string }>) => {
  if (busy) return;
  busy = true;
  try {
    const value = event.data.type === "load" ? await load() : await infer(event.data.dataUrl);
    self.postMessage({ type: event.data.type === "load" ? "ready" : "result", value });
  } catch (error) {
    const message = String(error);
    try { await session?.release(); } catch { /* Preserve the original load/inference error. */ }
    session = null; self.postMessage({ type: "error", error: message });
  } finally { busy = false; }
};
