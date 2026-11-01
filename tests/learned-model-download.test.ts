import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { downloadLearnedFile, loadHostedLearnedFiles, validateHostedModel, hostedModelPath, HOSTED_MODEL_PATH, type HostedFile } from "../src/learning/learned-model-download.ts";
import { browserFileNames, PUBLIC_LEARNED_MODEL } from "../src/learning/learned-onnx-contract.ts";

const digest = (value: string) => createHash("sha256").update(value).digest("hex");
const file: HostedFile = { name: "hsr-smolvla.weights", bytes: 6, sha256: digest("abcdef"),
  parts: ["abc", "def"].map(value => ({ bytes: value.length, sha256: digest(value) })) };
function memoryCache() {
  const entries = new Map<string, Response>();
  return { entries, async match(key: string) { return entries.get(key)?.clone(); },
    async put(key: string, value: Response) { entries.set(key, value); }, async delete(key: string) { return entries.delete(key); } };
}
const key = (index: number) => `${HOSTED_MODEL_PATH}/${file.name}.${String(index).padStart(3, "0")}.bin`;

test("a partial cached model resumes only the missing part, preserves order and then loads without network", async t => {
  const cache = memoryCache(); cache.entries.set(key(0), new Response("abc")); const requests: string[] = [];
  t.mock.method(globalThis, "fetch", async (url: string) => { requests.push(url); return new Response("def"); });
  const progress: number[] = [];
  const loaded = await downloadLearnedFile(file, cache, new AbortController().signal, bytes => progress.push(bytes));
  assert.equal(await loaded.text(), "abcdef"); assert.deepEqual(requests, [key(1)]); assert.equal(progress.at(-1), 6);
  requests.length = 0; assert.equal(await (await downloadLearnedFile(file, cache, new AbortController().signal)).text(), "abcdef");
  assert.deepEqual(requests, []);
});
test("corrupt cache is replaced from the network; a corrupt network response is never cached", async t => {
  const cache = memoryCache(); cache.entries.set(key(0), new Response("bad")); cache.entries.set(key(1), new Response("def"));
  t.mock.method(globalThis, "fetch", async () => new Response("abc"));
  assert.equal(await (await downloadLearnedFile(file, cache, new AbortController().signal)).text(), "abcdef");
  assert.equal(await cache.entries.get(key(0))!.clone().text(), "abc");
  cache.entries.clear(); t.mock.restoreAll(); t.mock.method(globalThis, "fetch", async () => new Response("bad"));
  await assert.rejects(downloadLearnedFile(file, cache, new AbortController().signal), /一致/);
  assert.equal(cache.entries.size, 0);
});
test("storage denial still permits a verified load and cancellation prevents returning a model", async t => {
  t.mock.method(globalThis, "fetch", async (url: string) => new Response(url.endsWith("000.bin") ? "abc" : "def"));
  const denied = { async match() { throw new Error("Storage denied"); }, async put() { throw new Error("Quota exceeded"); }, async delete() { return false; } };
  assert.equal(await (await downloadLearnedFile(file, denied, new AbortController().signal)).text(), "abcdef");
  const stop = new AbortController();
  await assert.rejects(downloadLearnedFile(file, undefined, stop.signal, () => stop.abort()), { name: "AbortError" });
});
test("hosted metadata cannot substitute a different model or omit one of the six verified files", () => {
  assert.throws(() => validateHostedModel({ files: [file] }));
  assert.throws(() => validateHostedModel({ schema: "hsr-hosted-model-v1", files: Array(6).fill(file) }));
});

const candidate = { ...PUBLIC_LEARNED_MODEL, id: "fixture-offset", manifestGraph: { bytes: 10, sha256: "1".repeat(64) },
  graph: { bytes: 6, sha256: digest("abcdef") }, sharedGraph: { bytes: 12, sha256: "3".repeat(64) },
  sharedArchive: { url: "/models/fixture/hsr-smolvla-shared.onnx.gz", bytes: 13, sha256: "4".repeat(64) },
  weights: { path: "hsr-smolvla.weights" as const, bytes: 6, sha256: digest("ghijkl") }, chunkOffsetSteps: 2 };
const hosted = (model: typeof candidate) => ({ schema: "hsr-hosted-model-v1", files: browserFileNames.map(name => {
  const pinned = name === "hsr-smolvla.onnx" ? model.graph : name === "hsr-smolvla.weights" ? model.weights : { bytes: 6, sha256: model.references[name] };
  return { name, bytes: pinned.bytes, sha256: pinned.sha256, parts: [{ bytes: pinned.bytes, sha256: "6".repeat(64) }] };
}) });
test("a candidate is served from its own path and its hosted metadata is checked against its own hashes", async t => {
  assert.equal(HOSTED_MODEL_PATH, hostedModelPath(PUBLIC_LEARNED_MODEL));
  assert.equal(hostedModelPath(candidate), `/models/smolvla-${digest("ghijkl").slice(0, 12)}`);
  assert.equal(hostedModelPath({ ...candidate, hostedName: "smolvla-fixture-e5" }), "/models/smolvla-fixture-e5", "a graph-only variant keeps its own path");
  assert.equal(validateHostedModel(hosted(candidate), candidate).length, 6);
  assert.throws(() => validateHostedModel(hosted(candidate)), /一致/);
  assert.throws(() => validateHostedModel(hosted(PUBLIC_LEARNED_MODEL as typeof candidate), candidate), /一致/);
  const requests: string[] = []; t.mock.method(globalThis, "fetch", async (url: string) => { requests.push(url); return new Response(url.endsWith("000.bin") ? "abc" : "def"); });
  await downloadLearnedFile(file, undefined, new AbortController().signal, undefined, candidate);
  assert.ok(requests.every(url => url.startsWith(`${hostedModelPath(candidate)}/`)), requests.join());
});

test("a one-member diagnostic downloads the verified single-member graph instead of the shared four-member graph", async t => {
  const contents: Record<string, string> = { "hsr-smolvla.onnx": "abcdef", "hsr-smolvla.weights": "ghijkl" };
  for (const [i, name] of ["browser-manifest.json", "browser-case-0.json", "browser-case-1.json", "browser-case-2.json"].entries()) contents[name] = `refs-${i}`;
  const model = { ...candidate, references: Object.fromEntries(Object.entries(contents).filter(([name]) => name.endsWith(".json")).map(([name, value]) => [name, digest(value)])) };
  const manifest = { schema: "hsr-hosted-model-v1", files: browserFileNames.map(name => ({ name, bytes: contents[name].length, sha256: digest(contents[name]),
    parts: [{ bytes: contents[name].length, sha256: digest(contents[name]) }] })) };
  const requests: string[] = [];
  t.mock.method(globalThis, "fetch", async (url: string) => {
    requests.push(url);
    if (url.endsWith("/manifest.json")) return new Response(JSON.stringify(manifest));
    const name = browserFileNames.find(item => url.endsWith(`/${item}.000.bin`)); return new Response(name ? contents[name] : "missing", { status: name ? 200 : 404 });
  });
  const files = await loadHostedLearnedFiles(new AbortController().signal, () => {}, model as typeof PUBLIC_LEARNED_MODEL, { singleGraph: true });
  assert.equal(await files[browserFileNames.indexOf("hsr-smolvla.onnx")].text(), "abcdef");
  assert.ok(!requests.includes(model.sharedArchive.url), "the shared graph is not fetched");
});
