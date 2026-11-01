import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import ts from "typescript";
import { PLAN_STEPS } from "../src/learning/plan-execution.ts";
import { learnedTarget } from "../src/learning/language-goals.ts";

// Run the actual panel and its hooks against a fictional language model, never the 1.4 GB worker.
const source = readFileSync(new URL("../src/learning/LanguagePlanPanel.tsx", import.meta.url), "utf8");
const tree = ts.createSourceFile("panel.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const body = tree.statements.filter(n => !ts.isImportDeclaration(n)).map(n => n.getText(tree)).join("\n").replaceAll("export ", "");
const code = ts.transpileModule(body, { compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2022 } }).outputText;
const STEPS: Record<string, { ja: string; en: string; stages?: string[] }> = { mug: { ja: "コップをかごに入れる", en: "Put the mug in the basket." }, ball: { ja: "ボールをかごに入れる", en: "Put the tennis ball in the basket." }, out: { ja: "コップをかごから出す", en: "Take the mug out of the basket.", stages: ["コップに近づく", "掴んで持ち上げる", "かごの外の机に置く"] } };
const proposed = (names: string[]) => {
  const tasks = names.map(name => ({ ja: STEPS[name].ja, en: STEPS[name].en }));
  return { rawText: JSON.stringify({ status: "proposed", tasks: names.map((name, i) => ({ skill: name === "out" ? "other" : name, ...tasks[i] })) }), goal: { status: "proposed", tasks }, stages: names.map(name => STEPS[name].stages ?? null) };
};

function setup(initial: Record<string, unknown> = {}) {
  const states: any[] = [], refs: any[] = [], effects: any[] = [], nodes: any[] = [], planners: FakePlanner[] = [];
  let stateAt = 0, refAt = 0, effectAt = 0;
  const executed: { receipt: unknown; requestedAt: number; plannedAt: number }[] = [], stops: number[] = [];
  const clock = { now: 1000 };
  class FakePlanner {
    disposed = false; ready = Promise.resolve(); downloaded = Promise.resolve(); requests: string[] = []; answer!: (value: unknown) => void;
    starts = 0; options: unknown;
    constructor(_progress: unknown, options: unknown) { this.options = options; planners.push(this); }
    startLoading() { this.starts++; }
    interpret(text: string) { this.requests.push(text); return new Promise(resolve => { this.answer = resolve; }); }
    dispose() { this.disposed = true; }
  }
  const equal = (a: unknown[] | undefined, b: unknown[] | undefined) => !!a && !!b && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  const environment = {
    LanguagePlanner: FakePlanner, PLAN_STEPS, learnedTarget, performance: { now: () => clock.now }, setInterval() { return 1; }, clearInterval() {},
    React: { createElement(type: unknown, props: any, ...children: unknown[]) { const n = { type, props: props ?? {}, children: children.flat() }; nodes.push(n); return n; } },
    useState(value: unknown) { const i = stateAt++; if (!(i in states)) states[i] = value; return [states[i], (v: any) => { states[i] = typeof v === "function" ? v(states[i]) : v; }]; },
    useRef(value: unknown) { const i = refAt++; return refs[i] ??= { current: value }; },
    useEffect(fn: () => (() => void) | undefined, deps?: unknown[]) { const i = effectAt++, old = effects[i]; if (!equal(old?.deps, deps)) effects[i] = { fn, deps, cleanup: old?.cleanup, pending: true }; },
  };
  const context = vm.createContext(environment); vm.runInContext(code, context);
  let props: Record<string, unknown> = { canExecute: true, stopEnabled: false, ...initial,
    onExecute(receipt: unknown, requestedAt: number, plannedAt: number) { executed.push({ receipt, requestedAt, plannedAt }); }, onStop() { stops.push(1); } };
  function render(next: Record<string, unknown> = {}) {
    props = { ...props, ...next }; context.props = props;
    stateAt = refAt = effectAt = 0; nodes.length = 0; vm.runInContext("LanguagePlanPanel(props)", context);
    for (const effect of effects) if (effect.pending) { effect.cleanup?.(); effect.cleanup = effect.fn(); effect.pending = false; }
  }
  render();
  const text = (node: any): string => node.children.map((c: any) => typeof c === "string" ? c : c && typeof c === "object" ? text(c) : "").join("");
  const button = (name: string) => nodes.find(n => n.type === "button" && text(n).trim() === name);
  const flush = async () => { for (let i = 0; i < 3; i++) await new Promise(resolve => setImmediate(resolve)); render(); };
  return { planners, executed, stops, render, button, flush, nodes, text, clock,
    async plan(name: "計画して実行" | "計画のみ", answer: unknown) {
      const control = button(name);
      if (control.props.type === "submit") nodes.find(n => n.type === "form").props.onSubmit({ preventDefault() {} }); else control.props.onClick();
      render(); await flush(); planners.at(-1)!.answer(answer); await flush();
    } };
}

test("the instruction starts as the single mug task and 計画して実行 runs the proposed plan once", async () => {
  const h = setup();
  assert.equal(h.nodes.find(n => n.type === "textarea").props.value, "コップをかごに入れて。");
  const receipt = proposed(["mug"]);
  await h.plan("計画して実行", receipt);
  assert.equal(h.planners[0].requests[0], "コップをかごに入れて。");
  assert.equal(h.executed.length, 1); assert.equal(h.executed[0].receipt, receipt);
  h.render(); h.render();
  assert.equal(h.executed.length, 1, "re-rendering must not start the same plan twice");
});

test("計画のみ shows the plan without moving the robot until この計画で実行 is pressed", async () => {
  const h = setup(), receipt = proposed(["mug", "ball"]);
  await h.plan("計画のみ", receipt);
  assert.equal(h.executed.length, 0);
  const tasks = h.nodes.filter(n => n.type === "strong").map(h.text);
  assert.deepEqual(tasks, ["コップをかごに入れる", "ボールをかごに入れる"], "the plan's own step names, in order");
  h.button("この計画で実行").props.onClick();
  assert.equal(h.executed.length, 1); assert.equal(h.executed[0].receipt, receipt);
});

test("計画して実行 waits for the robot to become ready instead of dropping the plan", async () => {
  const h = setup({ canExecute: false }), receipt = proposed(["mug"]);
  await h.plan("計画して実行", receipt);
  assert.equal(h.executed.length, 0);
  assert.equal(h.button("この計画で実行"), undefined, "a queued run must not also offer a manual start");
  h.render({ canExecute: true });
  assert.equal(h.executed.length, 1); assert.equal(h.executed[0].receipt, receipt);
});

test("計画して実行 does not start a plan the model asked to clarify", async () => {
  const h = setup();
  await h.plan("計画して実行", { rawText: "{}", goal: { status: "clarify", tasks: [] } });
  assert.equal(h.executed.length, 0);
});
test("any requested step is planned and run; unlearned steps are labelled and explained instead of refused", async () => {
  const h = setup(), receipt = proposed(["mug", "out"]);
  await h.plan("計画して実行", receipt);
  assert.equal(h.executed.length, 1, "the plan runs");
  const all = h.text(h.nodes.find(n => n.props["aria-label"] === "LLMの動作計画"));
  assert.ok(all.includes("コップをかごに入れる") && all.includes("学習済み"), all);
  assert.ok(all.includes("コップをかごから出す") && all.includes("未学習") && all.includes("Take the mug out of the basket."), all);
  assert.ok(all.includes("120秒"), all); assert.ok(!all.includes("対応範囲"), all);
  const stages = h.nodes.filter(n => n.props.className === "plan-steps plan-steps-unverified").map(h.text).join("|");
  for (const stage of ["コップに近づく", "掴んで持ち上げる", "かごの外の机に置く"]) assert.ok(stages.includes(stage), stages);
  assert.ok(stages.includes("未確認") && !stages.includes("✓"), stages);
});

test("停止 cancels planning, a queued run and the robot, and late answers cannot start anything", async () => {
  const planning = setup();
  planning.nodes.find(n => n.type === "form").props.onSubmit({ preventDefault() {} }); planning.render(); await planning.flush();
  planning.button("停止").props.onClick(); planning.render();
  assert.equal(planning.planners[0].disposed, true);
  planning.planners[0].answer(proposed(["mug"])); await planning.flush();
  assert.equal(planning.executed.length, 0);
  assert.equal(planning.stops.length, 0, "stopping only the language model must not report a robot stop");

  const queued = setup({ canExecute: false });
  await queued.plan("計画して実行", proposed(["mug"]));
  queued.button("停止").props.onClick(); queued.render({ canExecute: true });
  assert.equal(queued.executed.length, 0, "a stopped queued run must not start once the robot is ready");

  const running = setup({ stopEnabled: true });
  running.button("停止").props.onClick();
  assert.equal(running.stops.length, 1);
});

test("the planning model starts downloading when the panel opens and starts on the GPU only once warm", async () => {
  const h = setup({ warm: false }); await h.flush();
  assert.equal(h.planners.length, 1); assert.equal(JSON.stringify(h.planners[0].options), JSON.stringify({ deferLoad: true }));
  assert.equal(h.planners[0].starts, 0, "the robot model is still being verified");
  h.render({ warm: true });
  assert.equal(h.planners[0].starts, 1);
  await h.plan("計画のみ", proposed(["mug"]));
  assert.equal(h.planners.length, 1, "planning reuses the preloaded model");
});

test("the planning time stays fixed after the plan and is handed to the run", async () => {
  const h = setup();
  const form = h.nodes.find(n => n.type === "form");
  h.clock.now = 1000; h.button("計画のみ").props.onClick(); h.render(); await h.flush();
  h.clock.now = 13000; h.planners[0].answer(proposed(["mug"])); await h.flush();
  h.clock.now = 40000; h.render();
  const times = () => h.text(h.nodes.find(n => n.props.className === "plan-time"));
  assert.match(times(), /計画0:12/);
  h.button("この計画で実行").props.onClick();
  assert.deepEqual([h.executed[0].requestedAt, h.executed[0].plannedAt], [1000, 13000]);
  h.clock.now = 90000; h.render();
  assert.match(times(), /計画0:12/, "running the plan keeps the planning time");
  assert.ok(form);
});
