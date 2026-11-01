// Diagnostic only: replay a recorded continuous (RTC) learned run on CPU MuJoCo. Each recorded reply is delivered at
// its recorded simulation time with its recorded prediction, then the final mug pose, bounds against the basket walls
// and contacts are reported. Usage: node replay-rtc.ts record.json [--trace]
import { readFileSync } from "node:fs";
import { createSimulation } from "../../src/simulation/mujoco.ts";
import { prepareLearningScene } from "../../src/learning/scene.ts";
import { LearnedRtcRollout, type RtcPolicy } from "../../src/learning/learned-rtc-rollout.ts";
import { mugBounds, mugFitsBasket, mugInsideBasketWalls } from "../../src/simulation/mug-geometry.ts";
const [path, flag] = process.argv.slice(2);
const record = JSON.parse(readFileSync(path, "utf8"));
const sim = await createSimulation();
prepareLearningScene(sim, record.seed); sim.advance(2); sim.coordinateJointTargets = true; // as LearnedPilot
const requests = record.rtcRequests as { returned: { simulationTime: number } | null; prediction: unknown; continuation: unknown }[];
let k = 0, pending: { at: number; resolve: (v: unknown) => void; prediction: unknown } | null = null, maxState = 0;
const policy: RtcPolicy = { infer: async (input) => {
  const q = requests[k] as any, c = record.chunks.find((x: any) => Math.abs(x.timestamp - sim.data.time) < 1e-6);
  if (c) { const m = Math.max(...input.state.map((v: number, i: number) => Math.abs(v - c.input.state[i]))); if (m > 1e-6 && maxState <= 1e-6) console.log("first state mismatch", k, sim.data.time.toFixed(3), m.toExponential(2)); maxState = Math.max(maxState, m); }
  else console.log("no recorded chunk at", sim.data.time.toFixed(3), "request", k);
  k++; if (!q?.returned) return new Promise(() => {});
  return new Promise(resolve => { pending = { at: q.returned.simulationTime, resolve, prediction: q.prediction }; });
}, dispose() {} };
const { warmLatencyMs: _w, playbackRate, ...schedule } = record.rtcSchedule;
const r = new LearnedRtcRollout(sim, record.instruction, policy, { ...schedule, playbackRate, disposePolicy: false, success: (process.env.SUCCESS ?? record.successCriterion ?? "rim") as "rim" | "walls" });
const flush = () => new Promise(res => setImmediate(res));
const pair = (t: number) => ["head", "hand"].map(view => ({ view, timestamp: t, width: 320, height: 240, dataUrl: "data:image/jpeg;base64,/9j/AA==" }));
const report = (label: string) => {
  const o = sim.observation(), mug = o.objects.find(x => x.id === "mug")!, basket = o.furniture.find(x => x.id === "basket")!, rb = o.robot!;
  const b = mugBounds(mug, basket); const [x, , z] = mug.rotation; const upY = 1 - 2 * (x * x + z * z);
  console.log(label, JSON.stringify({ t: +sim.data.time.toFixed(2), status: r.result.status, reason: r.result.reason, settledFor: +r.result.settledFor.toFixed(2),
    walls: mugInsideBasketWalls(mug, basket), rim: mugFitsBasket(mug, basket), boundsMin: b.min.map(v => +v.toFixed(4)), boundsMax: b.max.map(v => +v.toFixed(4)), mugUpY: +upY.toFixed(3),
    speed: +Math.hypot(...mug.velocity).toFixed(4), gripperContact: rb.gripperContact, contactId: rb.contactId, opening: +rb.gripper.toFixed(3) }));
};
let lastTrace = -1;
for (let n = 0; n < 400000 && !r.stopped; n++) {
  if (pending && sim.data.time >= (pending as any).at - 1e-9) { const p = pending as any; pending = null; p.resolve(p.prediction); await flush(); continue; }
  if (r.needsImages) { r.capture(pair(sim.data.time) as never); await flush(); continue; }
  if (!sim.step()) { if (pending) { const p = pending as any; pending = null; p.resolve(p.prediction); await flush(); continue; } await flush(); continue; }
  if (flag === "--trace" && Math.floor(sim.data.time) !== lastTrace) { lastTrace = Math.floor(sim.data.time); if (lastTrace % 4 === 0) report("trace"); }
}
report("final"); console.log("requests", k, "of", requests.length, "max state mismatch", maxState.toExponential(2), "recorded", JSON.stringify(record.result), "replayed", JSON.stringify(r.result));
sim.dispose();
