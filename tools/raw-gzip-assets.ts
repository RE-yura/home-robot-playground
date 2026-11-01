import { createReadStream, statSync } from "node:fs";
import { resolve, sep } from "node:path";
import type { Connect, Plugin } from "vite";

// Production hosting serves *.gz assets as plain bytes; the app verifies their hashes and inflates them itself.
// Vite's static server would add Content-Encoding: gzip, so the browser inflated them first and the checks failed.
export function serveRawGzip(root: string): Connect.NextHandleFunction {
  const base = resolve(root);
  return (req, res, next) => {
    const path = decodeURIComponent((req.url ?? "").split("?")[0]);
    if (!path.endsWith(".gz")) return next();
    const file = resolve(base, "." + path);
    if (!file.startsWith(base + sep)) return next();
    let size: number;
    try { const stat = statSync(file); if (!stat.isFile()) return next(); size = stat.size; } catch { return next(); }
    res.setHeader("Content-Type", "application/octet-stream"); res.setHeader("Content-Length", String(size));
    createReadStream(file).pipe(res);
  };
}

export function rawGzipAssets(): Plugin {
  return {
    name: "raw-gzip-assets",
    configureServer(server) { server.middlewares.use(serveRawGzip(server.config.publicDir)); },
    configurePreviewServer(server) { server.middlewares.use(serveRawGzip(resolve(server.config.root, server.config.build.outDir))); },
  };
}
