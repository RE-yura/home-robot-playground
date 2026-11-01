import { LearnedRollout, type LearnedPolicy } from './learned-rollout.ts';
import type { LearnedRtcRollout } from './learned-rtc-rollout.ts';
// Either the synchronous executor or the continuous (asynchronous) one; both expose the same task interface.
export type TaskRollout = LearnedRollout | LearnedRtcRollout;
import { learnedTarget, parseLanguageGoal } from './language-goals.ts';
import type { LanguageReceipt } from './language-planner.ts';
import type { Simulation } from '../simulation/mujoco.ts';
import type { TaskResult } from '../simulation/task.ts';

export const PLAN_STEPS = ['映像から接近', '把持・持ち上げ', 'かごへ運搬・解放', 'かご内の静止確認'] as const;
// target is the learned object, or null for a planned step the VLA never learned (attempted, no success criterion).
export type PlanTaskProgress = {
  label: string; instruction: string; target: "mug" | "ball" | null; status: 'queued' | 'running' | 'success' | 'failed' | 'stopped';
  step: number; observed: boolean[]; durations: number[]; startedAt: number | null; endedAt: number | null;
  stepStartedAt: number | null; recovering: boolean; result?: TaskResult; error?: string | null;
};
// This observer reads simulation sensors for UI only. The unchanged VLA receives
// RGB, canonical language and measured joints; no progress/simulator features.
export class PlanExecution {
  readonly tasks: PlanTaskProgress[];
  readonly receipt: LanguageReceipt;
  readonly requestedAt: number;
  private sim: Simulation;
  private policy: LearnedPolicy;
  private now: () => number;
  private factory: (sim: Simulation, instruction: string, policy: LearnedPolicy, options: {disposePolicy: boolean; target: "mug" | "ball" | null}) => TaskRollout;
  current = 0;
  status: 'ready' | 'running' | 'success' | 'failed' | 'stopped' = 'ready';
  endedAt: number | null = null;
  rollout: TaskRollout | null = null;
  readonly rollouts: TaskRollout[] = [];
  private heldFor = 0;
  private initialY = 0;
  private drops = 0;
  // A grasp and lift measured in this task; a release over the rim may count as a drop, but a mug that then settles in
  // the basket was carried there, so the measured steps stay checked.
  private liftedOnce = false;
  constructor(sim: Simulation, policy: LearnedPolicy, receipt: LanguageReceipt, requestedAt: number, now = () => performance.now(), factory: PlanExecution['factory'] = (s, i, p, o) => new LearnedRollout(s, i, p, o)) {
    const goal = parseLanguageGoal(receipt.rawText, receipt.skills);
    if (goal.status !== 'proposed' || JSON.stringify(goal) !== JSON.stringify(receipt.goal)) throw Error('A confirmed proposed LLM receipt is required');
    this.sim = sim; this.policy = policy; this.receipt = receipt; this.requestedAt = requestedAt; this.now = now; this.factory = factory;
    this.tasks = goal.tasks.map(step => ({label: step.ja, instruction: step.en, target: learnedTarget(step.en), status:'queued', step:0, observed:[false,false,false,false], durations:[0,0,0,0], startedAt:null, endedAt:null, stepStartedAt:null, recovering:false}));
  }
  get elapsedMs() {
    const task = this.tasks[this.current];
    const terminal = task.endedAt !== null && (task.status !== 'success' || this.current === this.tasks.length - 1);
    return Math.max(0, (this.endedAt ?? (terminal ? task.endedAt! : this.now())) - this.requestedAt);
  }
  stepElapsed(task: PlanTaskProgress, index: number) {
    return task.durations[index] + (task.status === 'running' && task.step === index && task.stepStartedAt !== null ? this.now() - task.stepStartedAt : 0);
  }
  start() { if (this.status !== 'ready') return; this.status = 'running'; this.beginTask(); }
  private beginTask() {
    const task = this.tasks[this.current];
    task.status = 'running'; task.startedAt = this.now(); task.stepStartedAt = task.startedAt;
    this.heldFor = 0; this.drops = 0; this.liftedOnce = false;
    this.initialY = this.sim.observation().objects.find(o => o.id === task.target)?.position[1] ?? 0;
    const r = this.factory(this.sim, task.instruction, this.policy, {disposePolicy:false, target: task.target});
    this.rollout = r; this.rollouts.push(r);
    const stop = r.stop.bind(r);
    r.stop = () => { stop(); if (this.rollout === r) this.finishTask(); };
    const afterStep = r.afterStep.bind(r);
    r.afterStep = dt => { afterStep(dt); this.observe(dt); };
  }
  private setStep(index: number) {
    const task = this.tasks[this.current]; if (task.step === index) return;
    if (task.stepStartedAt !== null) task.durations[task.step] += this.now() - task.stepStartedAt;
    task.step = index; task.stepStartedAt = this.now();
  }
  private observe(dt: number) {
    const task = this.tasks[this.current], r = this.rollout;
    if (this.status !== 'running' || !r || task.status !== 'running' || task.target === null) return;
    const observation = this.sim.observation(), target = observation.objects.find(o => o.id === task.target);
    const held = observation.robot?.gripperContact && observation.robot.contactId === task.target;
    this.heldFor = held ? this.heldFor + dt : 0;
    if (r.result.drops > this.drops) { this.drops = r.result.drops; task.recovering = true; task.observed[1] = false; task.observed[2] = false; this.setStep(1); }
    if (this.heldFor >= .2 - 1e-8) { task.recovering = false; task.observed[0] = true; if (task.step < 1) this.setStep(1); }
    if (this.heldFor >= .2 - 1e-8 && target && target.position[1] - this.initialY >= .15) { task.observed[1] = true; this.liftedOnce = true; this.setStep(2); }
    if (r.result.settledFor > 0) { task.recovering = false; task.observed[2] = true; if (this.liftedOnce) task.observed[0] = task.observed[1] = true; this.setStep(3); }
    else if (task.step === 3) { task.observed[2] = false; this.setStep(2); }
  }
  poll() {
    if (this.status !== 'running' || !this.rollout?.stopped) return;
    const task = this.tasks[this.current];
    this.finishTask();
    if (task.status !== 'success') { this.status = task.status as 'failed' | 'stopped'; this.endedAt = task.endedAt; return; }
    if (this.current + 1 === this.tasks.length) { this.status = 'success'; this.endedAt = task.endedAt; }
    else { this.current++; this.beginTask(); }
  }
  private finishTask() {
    const task = this.tasks[this.current], r = this.rollout;
    if (!r || task.status !== 'running') return;
    task.result = r.result; task.error = r.error;
    if (r.result.status === 'success' && !r.error) task.observed[3] = true;
    this.closeTask(r.error || r.result.status === 'failed' ? 'failed' : r.result.status === 'success' ? 'success' : 'stopped');
  }
  private closeTask(status: PlanTaskProgress['status']) {
    const task = this.tasks[this.current];
    if (task.stepStartedAt !== null) task.durations[task.step] += this.now() - task.stepStartedAt;
    task.stepStartedAt = null; task.endedAt = this.now(); task.status = status;
  }
  stop() {
    if (this.status !== 'running') return;
    this.rollout?.stop();
    this.status = this.tasks[this.current].status === 'failed' ? 'failed' : 'stopped'; this.endedAt = this.now();
  }
}
