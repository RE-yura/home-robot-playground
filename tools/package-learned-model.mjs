// Model bytes stay outside Git. Run after the application build for publication.
// --model <id> packages a registered local candidate instead (into its own path, with its shared-graph archive).
import { createHash } from "node:crypto";
import { mkdir, open, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { gzipSync } from "node:zlib";
import { browserFileNames, learnedModel } from "../src/learning/learned-onnx-contract.ts";
import { hostedModelPath } from "../src/learning/learned-model-download.ts";

const args = process.argv.slice(2);
if (![4, 6].includes(args.length) || args[0] !== "--source" || args[2] !== "--output" || (args.length === 6 && args[4] !== "--model"))
  throw new Error("Use --source <verified model directory> --output <dist directory> [--model <candidate id>]");
const model = learnedModel(args[5]), HOSTED_MODEL_PATH = hostedModelPath(model);
const source = path.resolve(args[1]), output = path.join(path.resolve(args[3]), HOSTED_MODEL_PATH.slice(1));
await mkdir(output, { recursive: true });
const files = [], partBytes = 20 * 1024 * 1024;
for (const name of browserFileNames) {
  const expected = name === "hsr-smolvla.onnx" ? model.graph : name === "hsr-smolvla.weights" ? model.weights : null;
  const original = path.join(source, name), size = (await stat(original)).size;
  if (expected && size !== expected.bytes) throw new Error(`Incorrect model size: ${name}`);
  const handle = await open(original, "r"), hash = createHash("sha256"), parts = [];
  try {
    for (let offset = 0; offset < size;) {
      const buffer = Buffer.alloc(Math.min(partBytes, size - offset)); let filled = 0;
      while (filled < buffer.length) {
        const read = await handle.read(buffer, filled, buffer.length - filled, offset + filled);
        if (!read.bytesRead) throw new Error(`Unexpected end of model: ${name}`);
        filled += read.bytesRead;
      }
      hash.update(buffer);
      const part = { bytes: buffer.length, sha256: createHash("sha256").update(buffer).digest("hex") };
      await writeFile(path.join(output, `${name}.${String(parts.length).padStart(3, "0")}.bin`), buffer, { flag: "wx" });
      parts.push(part); offset += buffer.length;
    }
  } finally { await handle.close(); }
  const sha256 = hash.digest("hex");
  if (sha256 !== (expected?.sha256 ?? model.references[name])) throw new Error(`Verified model hash differs: ${name}`);
  files.push({ name, bytes: size, sha256, parts });
}
if (model.sharedArchive.url.startsWith(`${HOSTED_MODEL_PATH}/`)) {
  const digest = value => createHash("sha256").update(value).digest("hex");
  const graph = await readFile(path.join(source, "hsr-smolvla-shared.onnx")), archive = gzipSync(graph, { level: 9 });
  if (digest(graph) !== model.sharedGraph.sha256 || graph.length !== model.sharedGraph.bytes) throw new Error("Verified shared graph hash differs");
  if (digest(archive) !== model.sharedArchive.sha256 || archive.length !== model.sharedArchive.bytes) throw new Error("Shared graph archive differs from its registered hash");
  await writeFile(path.join(output, path.basename(model.sharedArchive.url)), archive, { flag: "wx" });
}
await writeFile(path.join(output, "manifest.json"), JSON.stringify({ schema: "hsr-hosted-model-v1", files }, null, 2) + "\n", { flag: "wx" });
console.log(JSON.stringify({ modelPath: HOSTED_MODEL_PATH, verifiedFiles: files.length, parts: files.reduce((sum, file) => sum + file.parts.length, 0), bytes: files.reduce((sum, file) => sum + file.bytes, 0) }));
