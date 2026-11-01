import test from "node:test";
import assert from "node:assert/strict";
import { LanguagePlanner } from "../src/learning/language-planner.ts";

test("a hard deadline preserves generated token evidence and ignores late replies", async context => {
  const oldWorker = globalThis.Worker; let owned: FakeWorker;
  class FakeWorker {
    onmessage: (event: { data: unknown }) => void = () => {}; onerror = () => {};
    terminated = false; lastId = 0;
    constructor() { owned = this; }
    postMessage(message: { id: number; type: string }) {
      this.lastId = message.id;
      if (message.type === "load") this.onmessage({ data: { id: message.id, value: { loaded: true } } });
      else {
        this.onmessage({ data: { id: message.id, diagnostic: { instruction: "コップをかごへ", seed: 20261019360, generationAttempted: true, inputTokenIds: [10,11], outputTokens: [], branches: [] } } });
        this.onmessage({ data: { id: message.id, partialTokens: [7,8], offset: 0, elapsedMs: 25, branches: [] } });
      }
    }
    terminate() { this.terminated = true; }
  }
  globalThis.Worker = FakeWorker as unknown as typeof Worker;
  context.mock.timers.enable({ apis: ["setTimeout"] });
  const planner = new LanguagePlanner();
  try {
    await planner.ready;
    const pending = planner.interpret("コップをかごへ");
    await Promise.resolve(); await Promise.resolve();
    const assertion = assert.rejects(pending, (error: Error) => {
      const evidence = error.cause as { generationAttempted: boolean; outputTokens: number[]; seed: number };
      assert.match(error.message, /timed out/); assert.equal(evidence.generationAttempted, true);
      assert.deepEqual(evidence.outputTokens, [7,8]); assert.equal(evidence.seed, 20261019360); return true;
    });
    context.mock.timers.tick(60000); await assertion;
    assert.equal(owned!.terminated, true);
    owned!.onmessage({ data: { id: owned!.lastId, value: { late: true } } });
    await assert.rejects(planner.interpret("コップをかごへ"), /cancelled/);
  } finally { planner.dispose(); globalThis.Worker = oldWorker; context.mock.timers.reset(); }
});

test('a progressing initial model download may exceed four minutes; explicit disposal still cancels it', async context=>{
 const previous=globalThis.Worker;let owned:any;
 class PendingWorker {onmessage=(_e:any)=>{};onerror=()=>{};id=0;terminated=false;constructor(){owned=this;}postMessage(m:any){this.id=m.id;}terminate(){this.terminated=true;}}
 globalThis.Worker=PendingWorker as unknown as typeof Worker;context.mock.timers.enable({apis:['setTimeout']});
 const planner=new LanguagePlanner();let failure:unknown;planner.ready.catch(e=>{failure=e;});
 try {context.mock.timers.tick(240001);await Promise.resolve();assert.equal(owned.terminated,false);assert.equal(failure,undefined);planner.dispose();await assert.rejects(planner.ready,/cancelled/);assert.equal(owned.terminated,true);}
 finally {planner.dispose();globalThis.Worker=previous;context.mock.timers.reset();}
});

test("a deferred planner downloads first and starts the GPU model only after it is allowed to", async () => {
  const previous = globalThis.Worker, workers: any[] = [];
  class OrderedWorker {
    onmessage = (_e: any) => {}; onerror = () => {}; sent: string[] = []; ids = new Map<string, number>(); terminated = false;
    constructor() { workers.push(this); }
    postMessage(m: any) { this.sent.push(m.type); this.ids.set(m.type, m.id); }
    reply(type: string) { this.onmessage({ data: { id: this.ids.get(type), value: { type } } }); }
    terminate() { this.terminated = true; }
  }
  globalThis.Worker = OrderedWorker as unknown as typeof Worker;
  const flush = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };
  const planner = new LanguagePlanner(undefined, { deferLoad: true });
  try {
    await flush(); assert.deepEqual(workers.at(-1).sent, ["download"]);
    planner.startLoading(); await flush();
    assert.deepEqual(workers.at(-1).sent, ["download"], "the GPU model waits for the download to finish");
    workers.at(-1).reply("download"); await planner.downloaded; await flush();
    assert.deepEqual(workers.at(-1).sent, ["download", "load"]);
    workers.at(-1).reply("load"); await planner.ready;

    const waiting = new LanguagePlanner(undefined, { deferLoad: true });
    workers.at(-1).reply("download"); await waiting.downloaded; await flush();
    assert.deepEqual(workers.at(-1).sent, ["download"], "a downloaded model still waits for permission");
    const asked = waiting.interpret("コップをかごに入れて。"); asked.catch(() => {}); await flush();
    assert.deepEqual(workers.at(-1).sent, ["download", "load"], "asking for a plan allows the model to start");
    waiting.dispose(); await assert.rejects(waiting.ready, /cancelled/);
  } finally { planner.dispose(); globalThis.Worker = previous; }
});

test("an output that is not a valid goal is retried with the next two seeds, then reported", async () => {
  const oldWorker = globalThis.Worker; const seeds: number[] = []; let invalidUntil = 1;
  const goal = '{"status":"proposed","tasks":[{"skill":"other","ja":"コップをかごから出す","en":"Take the mug out of the basket."}]}';
  class FakeWorker {
    onmessage: (event: { data: unknown }) => void = () => {}; onerror = () => {};
    postMessage(message: { id: number; type: string; instruction?: string; seed?: number }) {
      if (message.type === "load") { this.onmessage({ data: { id: message.id, value: { loaded: true } } }); return; }
      if (message.type === "stages") { this.onmessage({ data: { id: message.id, value: { stages: ["コップに近づく"] } } }); return; }
      if (message.type === "skill") { this.onmessage({ data: { id: message.id, value: { skill: "other" } } }); return; }
      seeds.push(message.seed!);
      if (seeds.length <= invalidUntil) { this.onmessage({ data: { id: message.id, error: "Error: Invalid language goal" } }); return; }
      this.onmessage({ data: { id: message.id, value: { instruction: message.instruction, seed: message.seed, endThinking: 1, rawText: goal, branches: [], outputTokens: [1, 2], inputTokenIds: [3], elapsedMs: 10 } } });
    }
    terminate() {}
  }
  globalThis.Worker = FakeWorker as unknown as typeof Worker;
  const planner = new LanguagePlanner();
  try {
    await planner.ready;
    const receipt = await planner.interpret("コップをかごから出して", 100);
    assert.deepEqual(seeds, [100, 101]); assert.equal(receipt.seed, 101); assert.equal(receipt.goal.tasks[0].en, "Take the mug out of the basket.");
    seeds.length = 0; invalidUntil = 9;
    await assert.rejects(planner.interpret("コップをかごから出して", 100), /Invalid language goal/);
    assert.deepEqual(seeds, [100, 101, 102], "three seeds at most");
  } finally { planner.dispose(); globalThis.Worker = oldWorker; }
});

test("after the plan, each unlearned step gets its stages from a separate request; learned steps and failed answers get none", async () => {
  const oldWorker = globalThis.Worker; const requests: { type: string; ja?: string; en?: string }[] = [];
  const goal = JSON.stringify({ status: "proposed", tasks: [
    { skill: "mug", ja: "コップをかごに入れる", en: "Put the mug in the basket." },
    { skill: "other", ja: "コップをかごから出す", en: "Take the mug out of the basket." },
    { skill: "other", ja: "冷蔵庫を開ける", en: "Open the fridge." }] });
  class FakeWorker {
    onmessage: (event: { data: unknown }) => void = () => {}; onerror = () => {};
    postMessage(message: { id: number; type: string; instruction?: string; seed?: number; ja?: string; en?: string }) {
      requests.push({ type: message.type, ja: message.ja, en: message.en });
      if (message.type === "load") { this.onmessage({ data: { id: message.id, value: { loaded: true } } }); return; }
      if (message.type === "skill") { this.onmessage({ data: { id: message.id, error: "Error: Invalid plan skill answer" } }); return; }
      if (message.type === "stages") {
        if (message.en === "Open the fridge.") this.onmessage({ data: { id: message.id, error: "Error: Invalid plan stages" } });
        else this.onmessage({ data: { id: message.id, value: { stages: ["コップに近づく", "掴んで持ち上げる", "机に置く"] } } });
        return;
      }
      this.onmessage({ data: { id: message.id, value: { instruction: message.instruction, seed: message.seed, endThinking: 1, rawText: goal, branches: [], outputTokens: [1, 2], inputTokenIds: [3], elapsedMs: 10 } } });
    }
    terminate() {}
  }
  globalThis.Worker = FakeWorker as unknown as typeof Worker;
  const planner = new LanguagePlanner();
  try {
    await planner.ready;
    const receipt = await planner.interpret("コップを入れてから出して、冷蔵庫も開けて", 100);
    assert.deepEqual(requests.filter(r => r.type === "stages").map(r => r.en), ["Take the mug out of the basket.", "Open the fridge."]);
    assert.deepEqual(receipt.stages, [null, ["コップに近づく", "掴んで持ち上げる", "机に置く"], null]);
    assert.equal(receipt.goal.tasks.length, 3, "the plan itself is unchanged");
  } finally { planner.dispose(); globalThis.Worker = oldWorker; }
});

test("each planned step is also classified separately; the answer overrides the plan's label before stages are asked", async () => {
  const oldWorker = globalThis.Worker; const requests: { type: string; en?: string }[] = [];
  const goal = JSON.stringify({ status: "proposed", tasks: [
    { skill: "mug", ja: "コップをかごから出す", en: "Take the mug out of the basket." },
    { skill: "mug", ja: "コップをかごに入れる", en: "Put the mug in the basket." }] });
  class FakeWorker {
    onmessage: (event: { data: unknown }) => void = () => {}; onerror = () => {};
    postMessage(message: { id: number; type: string; instruction?: string; seed?: number; ja?: string; en?: string }) {
      requests.push({ type: message.type, en: message.en });
      if (message.type === "load") { this.onmessage({ data: { id: message.id, value: { loaded: true } } }); return; }
      if (message.type === "skill") { this.onmessage({ data: { id: message.id, value: { skill: message.en === "Take the mug out of the basket." ? "other" : "mug" } } }); return; }
      if (message.type === "stages") { this.onmessage({ data: { id: message.id, value: { stages: ["コップに近づく"] } } }); return; }
      this.onmessage({ data: { id: message.id, value: { instruction: message.instruction, seed: message.seed, endThinking: 1, rawText: goal, branches: [], outputTokens: [1, 2], inputTokenIds: [3], elapsedMs: 10 } } });
    }
    terminate() {}
  }
  globalThis.Worker = FakeWorker as unknown as typeof Worker;
  const planner = new LanguagePlanner();
  try {
    await planner.ready;
    const receipt = await planner.interpret("コップをかごから出して、また入れて", 100);
    assert.deepEqual(requests.filter(r => r.type === "skill").map(r => r.en), ["Take the mug out of the basket.", "Put the mug in the basket."]);
    assert.deepEqual(receipt.skills, ["other", "mug"]);
    assert.deepEqual(receipt.goal.tasks.map(t => t.en), ["Take the mug out of the basket.", "Put the mug in the basket."]);
    assert.deepEqual(requests.filter(r => r.type === "stages").map(r => r.en), ["Take the mug out of the basket."]);
    assert.deepEqual(receipt.stages, [["コップに近づく"], null]);
  } finally { planner.dispose(); globalThis.Worker = oldWorker; }
});
