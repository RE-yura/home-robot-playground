// Assemble a plain static site (app build + verified model parts) for a static host such as Cloudflare Workers static assets.
// Model bytes stay outside Git: every part is copied from a local source and checked against the pinned manifests.
// Usage: node tools/assemble-static-site.mjs --dist <app build without models> --vla <public VLA parts dir>
//          --language <language parts dir> --recognition <recognition model.onnx> --out <fresh output dir>
//          [--candidate <registered candidate id> --candidate-dir <its packaged folder with manifest.json>]
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { access, constants, copyFile, cp, mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024, MAX_FILES = 20000;

async function exists(file) { try { await access(file); return true; } catch { return false; } }
async function sha256(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}
async function copyVerified(source, target, expected) {
  await mkdir(path.dirname(target), { recursive: true });
  await copyFile(source, target, constants.COPYFILE_FICLONE);
  const bytes = (await stat(target)).size, digest = await sha256(target);
  if (bytes !== expected.bytes || digest !== expected.sha256) throw new Error(`${source} does not match its pinned size/hash`);
}
async function* walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(file); else yield file;
  }
}

// bundles: { path: "/models/x/", source, manifest: { files: [{ name, parts: [{ bytes, sha256 }] }] }, wholeSinglePart }
// files: { path, source, bytes, sha256 }
export async function assembleStaticSite({ dist, out, bundles, files, maxFileBytes = DEFAULT_MAX_FILE_BYTES }) {
  if (await exists(out)) throw new Error(`${out} already exists; use a fresh output directory`);
  for (const entry of await readdir(path.join(dist, "models")).catch(() => []))
    if (entry.startsWith("smolvla-") || entry === "language") throw new Error(`The app build must not contain models/${entry}`);
  if (await exists(path.join(dist, "diagnostics"))) throw new Error("The app build must not contain diagnostics/");
  try {
    await cp(dist, out, { recursive: true });
    for (const bundle of bundles) {
      const directory = path.join(out, bundle.path);
      for (const file of bundle.manifest.files) for (const [index, part] of file.parts.entries()) {
        const name = `${file.name}.${String(index).padStart(3, "0")}.bin`;
        await copyVerified(path.join(bundle.source, name), path.join(directory, name), part);
        if (bundle.wholeSinglePart && file.parts.length === 1) await copyVerified(path.join(bundle.source, name), path.join(directory, file.name), part);
      }
      await writeFile(path.join(directory, "manifest.json"), JSON.stringify(bundle.manifest));
    }
    for (const file of files) await copyVerified(file.source, path.join(out, file.path), file);
    let count = 0;
    for await (const file of walk(out)) {
      if (++count > MAX_FILES) throw new Error(`More than ${MAX_FILES} files`);
      if ((await stat(file)).size > maxFileBytes) throw new Error(`${path.relative(out, file)} is larger than ${maxFileBytes} bytes`);
    }
    return { files: count };
  } catch (error) { await rm(out, { recursive: true, force: true }); throw error; }
}

// A registered candidate model (learned-onnx-contract.ts) served next to the public one: its parts at its hosted folder,
// checked against the registry identity, plus its shared graph archive.
export function candidateModelBundle(model, manifest, source, validate) {
  validate(manifest, model);
  const folder = `/models/${model.hostedName ?? `smolvla-${model.weights.sha256.slice(0, 12)}`}`;
  return { bundle: { path: `${folder}/`, source, manifest, wholeSinglePart: false },
    files: [{ path: model.sharedArchive.url, source: path.join(source, path.basename(model.sharedArchive.url)), bytes: model.sharedArchive.bytes, sha256: model.sharedArchive.sha256 }] };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { validateHostedModel, HOSTED_MODEL_PATH } = await import("../src/learning/learned-model-download.ts");
  const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => index % 2 ? pairs : [...pairs, [value, all[index + 1]]], []));
  for (const key of ["--dist", "--vla", "--language", "--recognition", "--out"]) if (!args[key]) throw new Error("Use --dist --vla --language --recognition --out");
  const json = async file => JSON.parse(await readFile(new URL(`../${file}`, import.meta.url), "utf8"));
  const vla = await json("worker/model-manifest.json"), language = await json("worker/language-manifest.json"), recognition = await json("public/models/recognition-t7/manifest.json");
  validateHostedModel(vla);
  let candidate = null;
  if (args["--candidate"]) {
    const { learnedModel } = await import("../src/learning/learned-onnx-contract.ts");
    candidate = candidateModelBundle(learnedModel(args["--candidate"]), JSON.parse(await readFile(path.join(args["--candidate-dir"], "manifest.json"), "utf8")), args["--candidate-dir"], validateHostedModel);
  }
  const result = await assembleStaticSite({
    dist: args["--dist"], out: args["--out"],
    bundles: [
      { path: `${HOSTED_MODEL_PATH}/`, source: args["--vla"], manifest: vla, wholeSinglePart: false },
      { path: language.path, source: args["--language"], manifest: language, wholeSinglePart: true },
      ...(candidate ? [candidate.bundle] : []),
    ],
    files: [{ path: recognition.graph.path, source: args["--recognition"], bytes: recognition.graph.bytes, sha256: recognition.graph.sha256 }, ...(candidate?.files ?? [])],
  });
  console.log(`Assembled ${result.files} files into ${args["--out"]}`);
}
