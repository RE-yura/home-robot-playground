import { learnedTarget, parseLanguageGoal, parseSkill, rawPlanSteps, validateLanguageRequest, type LanguageGoal, type PlanSkill } from "./language-goals.ts";
import { THINKING_TOKEN_LIMIT, GOAL_TOKEN_LIMIT } from "./language-thinking.ts";
export type LanguageReceipt = { instruction: string; seed: number; endThinking: number; rawText: string; branches: { prefix: number[]; tokens: number[]; scores: number[] }[]; outputTokens: number[]; inputTokenIds: number[]; elapsedMs: number; goal: LanguageGoal;
  // Per planned step: the stages of an unlearned step from a separate short request (null for learned or failed ones).
  stages?: (string[] | null)[];
  // Per planned step: the separate skill answer that overrides the plan's label (null when there was no valid answer).
  skills?: (PlanSkill | null)[] };
export class LanguagePlanner {
  readonly ready: Promise<unknown>;
  // With deferLoad the parts are fetched into the cache at once, but the GPU model starts only after startLoading()
  // (or the first interpret()), so it does not compete with another model that is still being verified.
  readonly downloaded: Promise<unknown>;
  private allowLoad: () => void = () => {};
  private worker: Worker;
  private nextId = 0;
  private disposed = false;
  private interpreting = false;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout>;
    diagnostic: { outputTokens: number[]; [key: string]: unknown } }>();
  constructor(progress?: (value: unknown) => void, { deferLoad = false }: { deferLoad?: boolean } = {}) {
    this.worker = new Worker(new URL("./language-worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = ({ data }) => {
      if (data.progress) { progress?.(data.progress); return; }
      const p = this.pending.get(data.id); if (!p) return;
      if (data.diagnostic && !data.error) { p.diagnostic = data.diagnostic; return; }
      if (data.partialTokens) {
        if (data.offset !== p.diagnostic.outputTokens.length || !Array.isArray(data.partialTokens) || data.partialTokens.some((v: number) => !Number.isSafeInteger(v) || v < 0) || data.offset + data.partialTokens.length > THINKING_TOKEN_LIMIT + GOAL_TOKEN_LIMIT) {
          this.dispose("Invalid language token stream"); return;
        }
        p.diagnostic.outputTokens.push(...data.partialTokens); p.diagnostic.branches = data.branches; p.diagnostic.elapsedMs = data.elapsedMs; return;
      }
      clearTimeout(p.timer); this.pending.delete(data.id);
      if (data.error) p.reject(new Error(data.error, { cause: data.diagnostic ?? p.diagnostic })); else p.resolve(data.value);
    };
    this.worker.onerror = error => this.dispose(error.message || "Language worker failed");
    if (!deferLoad) { this.downloaded = Promise.resolve(); this.ready = this.request({ type: "load" }, 900000); return; }
    this.downloaded = this.request({ type: "download" }, 900000);
    const allowed = new Promise<void>(resolve => { this.allowLoad = resolve; });
    this.ready = Promise.all([this.downloaded, allowed]).then(() => this.request({ type: "load" }, 900000));
  }
  startLoading() { this.allowLoad(); }
  private request(message: object, timeout: number): Promise<unknown> {
    if (this.disposed) return Promise.reject(new Error("Language interpretation cancelled"));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.dispose("Language interpretation timed out"), timeout);
      this.pending.set(id, { resolve, reject, timer, diagnostic: { request: message, generationAttempted: false, outputTokens: [] } }); this.worker.postMessage({ ...message, id });
    });
  }
  // An output that does not end in a valid goal is sampled again with the next seed, at most three seeds in all.
  async interpret(instruction: string, seed = 20261019360): Promise<LanguageReceipt> {
    validateLanguageRequest(instruction);
    if (this.interpreting) throw new Error("Language interpretation already pending");
    this.interpreting = true;
    try {
      this.startLoading(); await this.ready;
      for (let attempt = 0; ; attempt++) {
        let value: Omit<LanguageReceipt, "goal">;
        try { value = await this.request({ type: "interpret", instruction, seed: seed + attempt }, 60000) as Omit<LanguageReceipt, "goal">; }
        catch (error) { if (attempt < 2 && /Invalid language goal|did not finish|goal token limit/.test(String(error))) continue; throw error; }
        if (value.instruction !== instruction || value.seed !== seed + attempt || !Number.isFinite(value.elapsedMs) || value.elapsedMs < 0 || !Array.isArray(value.branches)) throw new Error("Invalid language receipt");
        const skills: (PlanSkill | null)[] = [];
        for (const step of rawPlanSteps(value.rawText)) {
          try { skills.push(parseSkill(((await this.request({ type: "skill", ja: step.ja, en: step.en }, 30000)) as { skill: string }).skill)); }
          catch (error) { if (this.disposed) throw error; skills.push(null); }
        }
        const goal = parseLanguageGoal(value.rawText, skills), stages: (string[] | null)[] = [];
        for (const step of goal.tasks) {
          if (learnedTarget(step.en) !== null) { stages.push(null); continue; }
          try { stages.push(((await this.request({ type: "stages", ja: step.ja, en: step.en }, 30000)) as { stages: string[] }).stages); }
          catch (error) { if (this.disposed) throw error; stages.push(null); }
        }
        return { ...value, goal, stages, skills };
      }
    } finally { this.interpreting = false; }
  }
  dispose(reason = "Language interpretation cancelled") {
    if (this.disposed) return;
    this.disposed = true; this.worker.terminate();
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(new Error(reason, { cause: p.diagnostic })); }
    this.pending.clear();
  }
}
