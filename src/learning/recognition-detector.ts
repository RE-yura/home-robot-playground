// Direct539 BGR/halfpixel preprocessing and650 fixed-five-view decode; no truth inputs.
export const RECOGNITION_LABELS = ["mug", "ball", "basket"] as const;
export const RECOGNITION_VIEWS = [[0, 0, 320, 240], [0, 0, 160, 120], [160, 0, 160, 120], [0, 120, 160, 120], [160, 120, 160, 120]] as const;
export type RecognitionDetection = { label: typeof RECOGNITION_LABELS[number]; score: number; box: number[]; sourceIndex: number };
export type RecognitionResult = { detections: RecognitionDetection[]; inferenceMs: number[]; elapsedMs: number };
export type RecognitionManifest = {
  schema: "recognition-t7-preview-v1"; variant: "T7"; trainingUpdates: 28800;
  graph: { path: "/models/recognition-t7/model.onnx"; sha256: string; bytes: number };
  input: { name: "images"; dtype: "float32"; shape: number[] };
  output: { name: "output"; dtype: "float32"; shape: number[] };
  classes: string[]; pipeline: "fixed-five-view-650";
};
function need(value: unknown, message: string): asserts value { if (!value) throw new Error(message); }
function object(value: unknown): Record<string, unknown> {
  need(value !== null && typeof value === "object" && !Array.isArray(value), "Invalid recognition manifest");
  return value as Record<string, unknown>;
}
function keys(value: Record<string, unknown>, names: string[]) { return Object.keys(value).sort().join() === [...names].sort().join(); }
function sameArray(value: unknown, expected: readonly (string | number)[]) { return Array.isArray(value) && value.length === expected.length && value.every((v, i) => v === expected[i]); }
export function validateRecognitionManifest(value: unknown): RecognitionManifest {
  const row = object(value), graph = object(row.graph), input = object(row.input), output = object(row.output);
  need(keys(row, ["schema", "variant", "trainingUpdates", "graph", "input", "output", "classes", "pipeline"]) &&
    row.schema === "recognition-t7-preview-v1" && row.variant === "T7" && row.trainingUpdates === 28800 &&
    row.pipeline === "fixed-five-view-650" && sameArray(row.classes, RECOGNITION_LABELS), "Unexpected T7 recognition identity");
  need(keys(graph, ["path", "sha256", "bytes"]) && graph.path === "/models/recognition-t7/model.onnx" &&
    typeof graph.sha256 === "string" && /^[0-9a-f]{64}$/.test(graph.sha256) && Number.isSafeInteger(graph.bytes) &&
    (graph.bytes as number) > 0 && (graph.bytes as number) <= 128 * 1024 * 1024, "Missing or invalid recognition model identity");
  for (const [port, name, shape] of [[input, "images", [1, 3, 416, 416]], [output, "output", [1, 3549, 8]]] as const) {
    need(keys(port, ["name", "dtype", "shape"]) && port.name === name && port.dtype === "float32" && sameArray(port.shape, shape), "Unexpected recognition model interface");
  }
  return value as RecognitionManifest;
}
export async function verifyRecognitionGraph(bytes: ArrayBuffer, graph: RecognitionManifest["graph"]) {
  need(bytes.byteLength === graph.bytes, "Recognition graph byte length changed");
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), v => v.toString(16).padStart(2, "0")).join("");
  need(hash === graph.sha256, "Recognition graph SHA-256 changed");
}
export function prepareRecognitionView(rgb: Uint8Array, ordinal: number): Float32Array {
  need(rgb instanceof Uint8Array && rgb.length === 320 * 240 * 3 && Number.isInteger(ordinal) && ordinal >= 0 && ordinal < 5, "Exact native RGB and fixed view required");
  const [ox, oy, width, height] = RECOGNITION_VIEWS[ordinal], scale = ordinal === 0 ? 1.3 : 2.6;
  const out = new Float32Array(3 * 416 * 416); out.fill(114);
  for (let y = 0; y < 312; y++) {
    const sy = Math.max(0, Math.min(height - 1, (y + .5) / scale - .5)), y0 = Math.floor(sy), y1 = Math.min(y0 + 1, height - 1), wy = sy - y0;
    for (let x = 0; x < 416; x++) {
      const sx = Math.max(0, Math.min(width - 1, (x + .5) / scale - .5)), x0 = Math.floor(sx), x1 = Math.min(x0 + 1, width - 1), wx = sx - x0;
      for (let c = 0; c < 3; c++) {
        const a = rgb[((oy + y0) * 320 + ox + x0) * 3 + c], b = rgb[((oy + y0) * 320 + ox + x1) * 3 + c];
        const d = rgb[((oy + y1) * 320 + ox + x0) * 3 + c], e = rgb[((oy + y1) * 320 + ox + x1) * 3 + c];
        const top = a + (b - a) * wx, bottom = d + (e - d) * wx;
        out[(2 - c) * 416 * 416 + y * 416 + x] = top + (bottom - top) * wy;
      }
    }
  }
  return out;
}
export function validateRecognitionOutput(value: unknown): Float32Array {
  const tensor = object(value);
  need(tensor.type === "float32" && sameArray(tensor.dims, [1, 3549, 8]) && tensor.data instanceof Float32Array && tensor.data.length === 28392, "Static Float32[1,3549,8] recognition output required");
  return tensor.data;
}
function iou(a: number[], b: number[]) {
  const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
  need(Number.isFinite(intersection) && Number.isFinite(union) && union > 0, "Finite NMS arithmetic required"); return intersection / union;
}
function decodeView(raw: Float32Array): RecognitionDetection[] {
  need(raw instanceof Float32Array && raw.length === 28392, "Static Float32 recognition output required");
  const candidates: { classId: number; score: number; box: number[]; sourceIndex: number }[] = [];
  for (let index = 0; index < 3549; index++) {
    const row = raw.subarray(index * 8, index * 8 + 8);
    need(row.every(Number.isFinite) && !row.subarray(4).some(v => v < 0 || v > 1), "Finite geometry and probability range required");
    const offset = index < 2704 ? 0 : index < 3380 ? 2704 : 3380, stride = offset === 0 ? 8 : offset === 2704 ? 16 : 32;
    const grid = index - offset, n = 416 / stride, cx = (row[0] + grid % n) * stride, cy = (row[1] + Math.floor(grid / n)) * stride;
    const w = Math.exp(row[2]) * stride, h = Math.exp(row[3]) * stride, box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    need(box.every(Number.isFinite) && w > 0 && h > 0 && Number.isFinite(w * h), "Finite positive decoded geometry required");
    let classId = 0;
    for (let k = 1; k < 3; k++) if (row[5 + k] > row[5 + classId]) classId = k;
    const score = row[4] * row[5 + classId];
    if (score > .3) candidates.push({ classId, score, box, sourceIndex: index });
  }
  candidates.sort((a, b) => b.score - a.score || a.sourceIndex - b.sourceIndex);
  const kept: typeof candidates = [];
  for (const candidate of candidates) if (!kept.some(v => v.classId === candidate.classId && iou(v.box, candidate.box) > .65)) kept.push(candidate);
  return kept.flatMap(v => {
    const box = v.box.map((value, i) => Math.max(0, Math.min(i % 2 === 0 ? 320 : 240, value / 1.3)));
    return box[0] < box[2] && box[1] < box[3] ? [{ label: RECOGNITION_LABELS[v.classId], score: v.score, box, sourceIndex: v.sourceIndex }] : [];
  });
}
export function decodeRecognitionViews(raws: Float32Array[]): RecognitionDetection[] {
  need(Array.isArray(raws) && raws.length === 5, "Exactly five atomic recognition views required");
  const candidates: RecognitionDetection[] = [];
  for (let v = 0; v < 5; v++) for (const d of decodeView(raws[v])) {
    const [ox, oy] = RECOGNITION_VIEWS[v], box = d.box.map((x, i) => Math.max(0, Math.min(i % 2 === 0 ? 320 : 240, (v === 0 ? x : x / 2) + (v === 0 ? 0 : i % 2 === 0 ? ox : oy))));
    candidates.push({ ...d, box, sourceIndex: v * 3549 + d.sourceIndex });
  }
  candidates.sort((a, b) => b.score - a.score || a.sourceIndex - b.sourceIndex);
  const kept: RecognitionDetection[] = [];
  for (const d of candidates) if (!kept.some(k => k.label === d.label && iou(k.box, d.box) > .65)) kept.push(d);
  return kept.map(d => {
    const box = d.box.map(v => Math.max(0, Math.fround(v)));
    need(box.every(Number.isFinite) && !box.some(v => Object.is(v, -0)) &&
      0 <= box[0] && box[0] < box[2] && box[2] <= 320 &&
      0 <= box[1] && box[1] < box[3] && box[3] <= 240, "Finite positive native F32 box required");
    return { ...d, box };
  });
}
