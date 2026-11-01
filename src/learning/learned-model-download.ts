/// <reference lib="dom" />
import { browserFileNames, PUBLIC_LEARNED_MODEL, type LearnedModelIdentity } from "./learned-onnx-contract.ts";

export const hostedModelPath = (model: LearnedModelIdentity) => `/models/${model.hostedName ?? `smolvla-${model.weights.sha256.slice(0, 12)}`}`;
export const HOSTED_MODEL_PATH = hostedModelPath(PUBLIC_LEARNED_MODEL);
export type HostedFile = { name: string; bytes: number; sha256: string; parts: { bytes: number; sha256: string }[] };
type ModelCache = Pick<Cache, "match" | "put" | "delete">;
const validHash = (value: unknown) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const validBytes = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) > 0;

export function validateHostedModel(value: unknown, model = PUBLIC_LEARNED_MODEL): HostedFile[] {
  const v = value as { schema?: unknown; files?: HostedFile[] } | null;
  if (v?.schema !== "hsr-hosted-model-v1" || !Array.isArray(v.files) || v.files.length !== 6) throw new Error("モデル配信情報が不正です");
  return browserFileNames.map(name => {
    const matches = v.files!.filter(file => file?.name === name), expected = name === "hsr-smolvla.onnx" ? model.graph : name === "hsr-smolvla.weights" ? model.weights : null;
    if (matches.length !== 1) throw new Error("モデル配信ファイルが不足・重複しています");
    const file = matches[0];
    if (!validBytes(file.bytes) || file.sha256 !== (expected?.sha256 ?? model.references[name]) ||
      (expected ? file.bytes !== expected.bytes : file.bytes > 25 * 1024 * 1024) ||
      !Array.isArray(file.parts) || !file.parts.length || file.parts.length > 128 ||
      file.parts.some(part => !part || !validBytes(part.bytes) || part.bytes > 20 * 1024 * 1024 || !validHash(part.sha256)) ||
      file.parts.reduce((sum, part) => sum + part.bytes, 0) !== file.bytes) throw new Error("検証済みモデルと配信情報が一致しません");
    return file;
  });
}
export async function matchesPart(blob: Blob, part: HostedFile["parts"][number]) {
  if (blob.size !== part.bytes) return false;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", await blob.arrayBuffer()));
  return Array.from(digest, value => value.toString(16).padStart(2, "0")).join("") === part.sha256;
}
export async function downloadLearnedFile(file: HostedFile, cache: ModelCache | undefined, signal: AbortSignal, progress?: (bytes: number) => void, model = PUBLIC_LEARNED_MODEL): Promise<File> {
  const blobs: Blob[] = []; let loaded = 0;
  for (let index = 0; index < file.parts.length; index++) {
    signal.throwIfAborted();
    const part = file.parts[index], url = `${hostedModelPath(model)}/${file.name}.${String(index).padStart(3, "0")}.bin`;
    let blob: Blob | undefined;
    try { blob = await (await cache?.match(url))?.blob(); } catch { /* Storage may be unavailable in private browsing. */ }
    if (blob && !await matchesPart(blob, part)) {
      blob = undefined; try { await cache?.delete(url); } catch { /* A fresh verified response can still be used. */ }
    }
    if (!blob) {
      const response = await fetch(url, { signal: AbortSignal.any([signal, AbortSignal.timeout(120000)]) });
      if (!response.ok) throw new Error(`モデルの取得に失敗しました (${response.status})`);
      blob = await response.blob();
      if (!await matchesPart(blob, part)) throw new Error(`取得したモデルのデータが一致しません (${file.name} / ${index}, ${blob.size} bytes)`);
      signal.throwIfAborted();
      try { await cache?.put(url, new Response(blob)); } catch { /* Quota errors must not prevent inference. */ }
    }
    signal.throwIfAborted(); blobs.push(blob); loaded += blob.size; progress?.(loaded);
  }
  signal.throwIfAborted(); return new File(blobs, file.name);
}
export async function loadSharedLearnedGraph(cache: ModelCache | undefined, signal: AbortSignal, model = PUBLIC_LEARNED_MODEL): Promise<File> {
  signal.throwIfAborted(); let blob: Blob | undefined, downloaded = false; const sharedArchive = model.sharedArchive;
  try { blob = await (await cache?.match(sharedArchive.url))?.blob(); } catch { /* Storage may be unavailable. */ }
  if (blob && !await matchesPart(blob, sharedArchive)) {
    blob = undefined; try { await cache?.delete(sharedArchive.url); } catch { /* A verified network response remains usable. */ }
  }
  if (!blob) {
    const response = await fetch(sharedArchive.url, { signal: AbortSignal.any([signal, AbortSignal.timeout(120000)]) });
    if (!response.ok) throw new Error(`モデルの取得に失敗しました (${response.status})`);
    blob = await response.blob(); downloaded = true;
    if (!await matchesPart(blob, sharedArchive)) throw new Error(`取得したモデルのデータが一致しません (shared graph, ${blob.size} bytes)`);
  }
  signal.throwIfAborted();
  const graph = await new Response(blob.stream().pipeThrough(new DecompressionStream("gzip"))).blob();
  if (!await matchesPart(graph, model.sharedGraph)) throw new Error("展開したモデルのデータが一致しません");
  signal.throwIfAborted();
  if (downloaded) try { await cache?.put(sharedArchive.url, new Response(blob)); } catch { /* Quota errors must not prevent verified use. */ }
  signal.throwIfAborted(); return new File([graph], "hsr-smolvla.onnx");
}
// options.singleGraph (one-member speed diagnostic) keeps the hosted single-member graph instead of the shared four-member graph.
export async function loadHostedLearnedFiles(signal: AbortSignal, progress: (loaded: number, total: number) => void, model = PUBLIC_LEARNED_MODEL, options: { singleGraph?: boolean } = {}): Promise<File[]> {
  const response = await fetch(`${hostedModelPath(model)}/manifest.json`, { signal: AbortSignal.any([signal, AbortSignal.timeout(30000)]) });
  if (!response.ok) throw new Error(`モデルの配信情報を取得できません (${response.status})`);
  const files = validateHostedModel(await response.json(), model), total = files.reduce((sum, file) => sum + file.bytes, 0) - (options.singleGraph ? 0 : model.graph.bytes - model.sharedGraph.bytes);
  let cache: ModelCache | undefined, loaded = 0;
  try { cache = await caches.open(`hsr-vla-${model.weights.sha256}`); } catch { /* Download directly when storage is unavailable. */ }
  const result: File[] = [];
  for (const file of files) {
    const selected = file.name === "hsr-smolvla.onnx" && !options.singleGraph ? await loadSharedLearnedGraph(cache, signal, model) :
      await downloadLearnedFile(file, cache, signal, bytes => progress(loaded + bytes, total), model);
    result.push(selected); loaded += selected.size; progress(loaded, total);
  }
  return result;
}
