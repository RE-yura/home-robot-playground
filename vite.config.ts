import react from "@vitejs/plugin-react";
import { defineConfig, defaultClientConditions } from "vite";
import { learnedWasmAsset } from "./tools/learned-wasm-asset.ts";
import { rawGzipAssets } from "./tools/raw-gzip-assets.ts";

export default defineConfig({
  plugins: [react(), learnedWasmAsset(), rawGzipAssets()],
  worker: { plugins: () => [learnedWasmAsset()] },
  // Supply each chosen runtime explicitly instead of ORT's automatic asset imports.
  resolve: { conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions] },
  optimizeDeps: { exclude: ["@mujoco/mujoco"] },
});
