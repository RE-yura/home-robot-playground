import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
// @ts-expect-error Node-only assembler is executed directly by the Node test runner.
import { assembleStaticSite, candidateModelBundle } from "../tools/assemble-static-site.mjs";

const sha = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
const part = (bytes: Buffer) => ({ bytes: bytes.length, sha256: sha(bytes) });

async function fixture() {
  const root = await mkdtemp(path.join(tmpdir(), "static-site-")), dist = path.join(root, "dist");
  await mkdir(path.join(dist, "assets"), { recursive: true });
  await writeFile(path.join(dist, "index.html"), "<!doctype html>");
  await writeFile(path.join(dist, "assets", "app.js"), "app");
  const graph = [Buffer.from("graph-0"), Buffer.from("graph-1")], config = Buffer.from("{}"), weights = Buffer.from("w");
  const vlaSource = path.join(root, "vla"), languageSource = path.join(root, "language");
  await mkdir(vlaSource); await mkdir(path.join(languageSource, "onnx"), { recursive: true });
  for (const [i, bytes] of graph.entries()) await writeFile(path.join(vlaSource, `graph.onnx.00${i}.bin`), bytes);
  await writeFile(path.join(languageSource, "config.json.000.bin"), config);
  await writeFile(path.join(languageSource, "onnx", "data.000.bin"), weights);
  const detector = Buffer.from("detector"), detectorFile = path.join(root, "model.onnx");
  await writeFile(detectorFile, detector);
  const options = {
    dist, out: path.join(root, "site"),
    bundles: [
      { path: "/models/vla-fixture/", source: vlaSource, wholeSinglePart: false,
        manifest: { schema: "fixture", files: [{ name: "graph.onnx", bytes: 14, sha256: sha(Buffer.concat(graph)), parts: graph.map(part) }] } },
      { path: "/models/language/fixture/", source: languageSource, wholeSinglePart: true,
        manifest: { path: "/models/language/fixture/", files: [{ name: "config.json", bytes: 2, parts: [part(config)] }, { name: "onnx/data", bytes: 1, parts: [part(weights)] }] } },
    ],
    files: [{ path: "/models/detector/model.onnx", source: detectorFile, ...part(detector) }],
  };
  return { root, options };
}

test("a static site holds the app and every verified model part at the paths the app fetches", async () => {
  const { options } = await fixture();
  await assembleStaticSite(options);
  const site = (relative: string) => readFile(path.join(options.out, relative), "utf8");
  assert.equal(await site("index.html"), "<!doctype html>");
  assert.equal(await site("models/vla-fixture/graph.onnx.001.bin"), "graph-1");
  assert.deepEqual(JSON.parse(await site("models/vla-fixture/manifest.json")), options.bundles[0].manifest);
  assert.equal(await site("models/language/fixture/config.json"), "{}", "single-part language files are also served by their original name");
  assert.equal(await site("models/language/fixture/onnx/data"), "w");
  assert.equal(await site("models/detector/model.onnx"), "detector");
  assert.deepEqual((await readdir(path.join(options.out, "models", "vla-fixture"))).sort(), ["graph.onnx.000.bin", "graph.onnx.001.bin", "manifest.json"]);
});

test("a model part that does not match its pinned hash stops the assembly", async () => {
  const { options } = await fixture();
  await writeFile(path.join(options.bundles[0].source, "graph.onnx.001.bin"), "graph-X");
  await assert.rejects(assembleStaticSite(options), /graph\.onnx\.001\.bin/);
});

test("an application build that already carries models or diagnostics is refused", async () => {
  for (const leak of ["models/smolvla-c63f9ccf6048", "diagnostics/rtc"]) {
    const { options } = await fixture();
    await mkdir(path.join(options.dist, leak), { recursive: true });
    await assert.rejects(assembleStaticSite(options), /must not contain/);
  }
});

test("files over the host's per-file limit and an existing output are refused", async () => {
  const { options } = await fixture();
  await assert.rejects(assembleStaticSite({ ...options, maxFileBytes: 7 }), /larger than/);
  const fresh = await fixture();
  await mkdir(fresh.options.out);
  await assert.rejects(assembleStaticSite(fresh.options), /already exists/);
});

test("a registered candidate model is placed at its hosted folder together with its shared graph archive", () => {
  const model = { id: "fixture", hostedName: "smolvla-fixture", sharedArchive: { url: "/models/smolvla-fixture/hsr-smolvla-shared.onnx.gz", bytes: 7, sha256: "a".repeat(64) } };
  const manifest = { schema: "hsr-hosted-model-v1", files: [] }, validated: unknown[] = [];
  const result = candidateModelBundle(model, manifest, "/src/candidate", (value: unknown, identity: unknown) => { validated.push([value, identity]); });
  assert.deepEqual(validated, [[manifest, model]], "the manifest is checked against the registry identity");
  assert.deepEqual(result.bundle, { path: "/models/smolvla-fixture/", source: "/src/candidate", manifest, wholeSinglePart: false });
  assert.deepEqual(result.files, [{ path: "/models/smolvla-fixture/hsr-smolvla-shared.onnx.gz", source: path.join("/src/candidate", "hsr-smolvla-shared.onnx.gz"), bytes: 7, sha256: "a".repeat(64) }]);
});
