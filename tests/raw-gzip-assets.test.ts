import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { gzipSync } from "node:zlib";
import { serveRawGzip } from "../tools/raw-gzip-assets.ts";

function request(root: string, url: string) {
  const res = Object.assign(new PassThrough(), { headers: {} as Record<string, string>, setHeader(name: string, value: string) { this.headers[name.toLowerCase()] = value; } });
  const chunks: Buffer[] = []; res.on("data", chunk => chunks.push(chunk));
  return new Promise<{ next: boolean; headers: Record<string, string>; body: Buffer }>(resolve => {
    res.on("end", () => resolve({ next: false, headers: res.headers, body: Buffer.concat(chunks) }));
    serveRawGzip(root)({ url } as never, res as never, () => resolve({ next: true, headers: res.headers, body: Buffer.alloc(0) }));
  });
}

test("gzip assets are served as their exact bytes without Content-Encoding, like production hosting", async () => {
  const root = mkdtempSync(join(tmpdir(), "raw-gz-")), bytes = gzipSync(Buffer.from("graph bytes"));
  mkdirSync(join(root, "assets")); writeFileSync(join(root, "assets", "model.onnx.gz"), bytes); writeFileSync(join(root, "plain.txt"), "x");
  const served = await request(root, "/assets/model.onnx.gz?v=1");
  assert.equal(served.next, false);
  assert.equal(served.headers["content-type"], "application/octet-stream");
  assert.equal(served.headers["content-encoding"], undefined);
  assert.deepEqual(served.body, bytes);
  assert.equal((await request(root, "/plain.txt")).next, true, "other files stay with Vite");
  assert.equal((await request(root, "/missing.gz")).next, true);
  assert.equal((await request(root, "/../outside.gz")).next, true, "no paths outside the served root");
});
