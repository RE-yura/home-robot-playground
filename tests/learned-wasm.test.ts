import assert from "node:assert/strict";
import { test } from "node:test";
import { gzipSync } from "node:zlib";
import { readFile, mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { build } from "vite";
import { fetchLearnedWasm } from "../src/learning/learned-wasm.ts";
import { learnedWasmAsset } from "../tools/learned-wasm-asset.ts";

test("learned runtime restores gzip and raw WASM bytes and rejects failed transfers", async () => {
  const bytes = Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0]);
  const original = globalThis.fetch;
  try {
    for (const body of [bytes, gzipSync(bytes)]) {
      globalThis.fetch = async () => new Response(body);
      assert.deepEqual(await fetchLearnedWasm("/runtime"), bytes);
    }
    globalThis.fetch = async () => new Response(null, { status: 404 });
    await assert.rejects(fetchLearnedWasm("/runtime"), /404/);
    globalThis.fetch = async () => new Response(Uint8Array.from([31, 139, 0]));
    await assert.rejects(fetchLearnedWasm("/runtime"));
  } finally { globalThis.fetch = original; }
});

test("production asset restores the identical ONNX runtime below the Site asset limit", async () => {
  const directory = await mkdtemp(join(tmpdir(), "learned-wasm-"));
  const entry = join(directory, "entry.js");
  await writeFile(entry, 'export { default } from "virtual:learned-wasm";');
  let result;
  try {
    result = await build({ configFile: false, logLevel: "silent", plugins: [learnedWasmAsset()],
      build: { write: false, minify: false, lib: { entry, formats: ["es"], fileName: "runtime" } } });
  } finally { await rm(directory, { recursive: true, force: true }); }
  if (Array.isArray(result)) { assert.equal(result.length, 1); result = result[0]; }
  assert.ok("output" in result);
  const assets = result.output.filter(item => item.type === "asset");
  assert.equal(assets.length, 1);
  const asset = assets[0]; assert.equal(asset.type, "asset");
  assert.match(asset.fileName, /\.wasm\.gz$/);
  assert.ok(typeof asset.source !== "string" && asset.source.byteLength < 26214400);
  const compressed = new Uint8Array(asset.source);
  const original = await readFile(new URL(import.meta.resolve("onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm")));
  const restored = new Uint8Array(await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  assert.equal(createHash("sha256").update(restored).digest("hex"), createHash("sha256").update(original).digest("hex"));
  assert.ok(result.output.some(item => item.type === "chunk" && item.code.includes(asset.fileName)));
});
