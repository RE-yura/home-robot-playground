import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import type { Plugin } from "vite";

export function learnedWasmAsset(): Plugin {
  const id = "virtual:learned-wasm", resolved = "\0" + id;
  const path = fileURLToPath(import.meta.resolve("onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm"));
  let production = false;
  return {
    name: "learned-wasm-asset",
    configResolved(config) { production = config.command === "build"; },
    resolveId(source) { if (source === id) return resolved; },
    async load(source) {
      if (source !== resolved) return;
      if (!production) return `export default ${JSON.stringify("/@fs/" + path)};`;
      const ref = this.emitFile({ type: "asset", name: "ort-wasm-simd-threaded.asyncify.wasm.gz", source: gzipSync(await readFile(path)) });
      return `export default import.meta.ROLLUP_FILE_URL_${ref};`;
    },
  };
}
