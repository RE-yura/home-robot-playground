import type { Simulation } from "../simulation/mujoco.ts";
import type { CameraImage } from "../simulation/types.ts";
import type { SixValues } from "./recording.ts";
import { learnedTarget } from "./language-goals.ts";
import { ARM_LIMITS } from "../robot/hsr-model.ts";
import { TaskEvaluator, type MugSuccessCriterion } from "../simulation/task.ts";
import { validateLearningBasePose } from "./recording.ts";
import type { Vec3 } from "../simulation/types.ts";
export type LearnedInput = { images: CameraImage[]; instruction: string; state: SixValues };
// The two learned instructions score their own object. Any other instruction (a planned step the VLA never learned)
// must state target: null and is then scored without a success criterion.
export function evaluationTarget(instruction: string, target: "mug" | "ball" | null | undefined): "mug" | "ball" | null {
  if (typeof instruction !== "string" || !/^[\x20-\x7e]{1,160}$/.test(instruction)) throw new Error("Unsupported learned instruction");
  const learned = learnedTarget(instruction);
  if (target === undefined) { if (learned === null) throw new Error("An unlearned instruction needs target: null"); return learned; }
  if (target !== null && target !== learned) throw new Error("The evaluation target must match the learned instruction");
  return target;
}
export type LearnedPrediction = { actions: number[][]; initialNoise: number[]; secondNoise?: number[]; memberActions?: number[][] };
// A member trace is either the four antithetic members with the second noise, or (one-member speed diagnostic) a single
// member without a second noise.
export function memberTraceComplete(secondNoise: unknown, memberActions: unknown) {
  const rows = (value: unknown, count: number) => Array.isArray(value) && value.length === count &&
    value.every(row => Array.isArray(row) && row.length === 96 && row.every(Number.isFinite));
  if (secondNoise === undefined) return rows(memberActions, 1);
  return Array.isArray(secondNoise) && secondNoise.length === 512 && secondNoise.every(Number.isFinite) && rows(memberActions, 4);
}
export interface LearnedPolicy { infer(input: LearnedInput): Promise<number[][] | LearnedPrediction>; dispose(): void }
export type LearnedChunk = { timestamp: number; input: LearnedInput; rawActions: number[][]; actions: SixValues[]; inferenceMs: number; initialNoise?: number[]; secondNoise?: number[]; memberActions?: number[][] };
const EPS = 1e-8;
export class LearnedRollout {
  stopped = false;
  error: string | null = null;
  readonly chunks: LearnedChunk[] = [];
  private readonly sim: Simulation;
  private readonly instruction: string;
  private readonly policy: LearnedPolicy;
  private readonly generation: number;
  private readonly base: Vec3;
  private readonly evaluator: TaskEvaluator;
  private phase: "images" | "inference" | "execute" = "images";
  private actions: SixValues[] = [];
  private index = 0;
  private nextTime: number;
  private chunkTime = 0;
  private readonly disposePolicy: boolean;
  readonly executedActionsPerChunk: 4 | 16;
  get needsImages() { return !this.stopped && this.phase === "images"; }
  get inferencePending() { return !this.stopped && this.phase === "inference"; }
  get result() { return this.evaluator.result; }
  get successCriterion() { return this.evaluator.criterion; }
  get evaluationTarget() { return this.evaluator.target; }
  constructor(sim: Simulation, instruction: string, policy: LearnedPolicy, options: { disposePolicy?: boolean; executedActionsPerChunk?: 4 | 16; success?: MugSuccessCriterion; target?: "mug" | "ball" | null } = {}) {
    const target = evaluationTarget(instruction, options.target);
    const cadence = options.executedActionsPerChunk ?? 16;
    if (cadence !== 4 && cadence !== 16) throw new Error("Unsupported learned execution cadence");
    this.executedActionsPerChunk = cadence;
    sim.setController(null);
    this.sim = sim; this.instruction = instruction; this.policy = policy; this.disposePolicy = options.disposePolicy ?? true;
    this.generation = sim.resetGeneration; this.base = this.basePose(); this.nextTime = sim.data.time;
    this.evaluator = new TaskEvaluator(target ?? "none", options.success ? { success: options.success } : {});
    this.evaluator.start(sim.observation()); sim.learningCapture = this;
    sim.phase = "学習済みVLA · カメラ待ち";
  }
  private basePose(): Vec3 {
    const b = this.sim.position(this.sim.baseId); return [b[0], b[2], this.sim.q("base_yaw")];
  }
  private baseDrifted(): boolean {
    const pose = this.basePose();
    if (![...pose, ...this.base].every(Number.isFinite)) throw new Error("Invalid measured base pose");
    try { validateLearningBasePose(pose, this.base); }
    catch { return true; }
    return false;
  }
  private fail(error: unknown) {
    this.error = (error instanceof Error ? error.message : String(error)).slice(0, 1024);
    this.stop();
  }
  private latch(action: SixValues) {
    // This is an internal action; external applyAction remains a takeover hook.
    this.sim.learningCapture = null;
    try { this.sim.applyAction({ armJointTarget: action.slice(0, 5) as [number, number, number, number, number],
      gripper: action[5], base: [0, 0, 0], head: [0, 0] }); }
    finally { if (!this.stopped) this.sim.learningCapture = this; }
    this.sim.phase = "学習済みVLA · 操作実行";
  }
  captureDue(time: number): boolean {
    if (this.stopped) return false;
    try {
      if (this.sim.resetGeneration !== this.generation) throw new Error("Learned rollout reset");
      this.sim.mj.mj_kinematics(this.sim.model, this.sim.data);
      if (this.baseDrifted()) { this.evaluator.fail("base-drift"); this.stop(); return true; }
      if (this.phase !== "execute") return true;
      if (time < this.nextTime - EPS) return false;
      if (Math.abs(time - this.nextTime) > EPS) throw new Error("Learned action missed its 100ms boundary");
      this.index++;
      if (this.index === this.actions.length) { this.phase = "images"; this.sim.phase = "学習済みVLA · カメラ待ち"; return true; }
      this.latch(this.actions[this.index]);
      this.nextTime = this.chunkTime + (this.index + 1) / 10;
      return false;
    } catch (error) { this.fail(error); return true; }
  }
  capture(images: CameraImage[]) {
    if (!this.needsImages) return;
    try {
      const time = this.sim.data.time;
      if (!this.captureDue(time) || this.stopped) return;
      if (images.length !== 2 || !["head", "hand"].every(view => images.filter(i => i.view === view).length === 1) ||
        images.some(i => i.width !== 320 || i.height !== 240 || Math.abs(i.timestamp - time) > EPS ||
          !Number.isFinite(i.timestamp) || !/^data:image\/jpeg;base64,[A-Za-z0-9+/]+={0,2}$/.test(i.dataUrl)))
        throw new Error("Learned policy requires fresh synchronized camera images");
      const robot = this.sim.observation().robot!;
      const state = [...robot.joints, robot.gripper];
      if (state.length !== 6 || !state.every(Number.isFinite)) throw new Error("Invalid measured HSR state");
      const input: LearnedInput = { images: images.map(i => ({ ...i })), instruction: this.instruction, state: state as SixValues };
      this.phase = "inference"; this.sim.phase = "学習済みVLA · 推論中（物理を一時停止）";
      const started = performance.now();
      this.policy.infer(input).then(prediction => {
        if (this.stopped || this.sim.resetGeneration !== this.generation || this.sim.learningCapture !== this) return;
        try {
          const raw = Array.isArray(prediction) ? prediction : prediction?.actions;
          const initialNoise = Array.isArray(prediction) ? undefined : prediction?.initialNoise;
          const secondNoise = Array.isArray(prediction) ? undefined : prediction?.secondNoise;
          const memberActions = Array.isArray(prediction) ? undefined : prediction?.memberActions;
          if (!Array.isArray(raw) || raw.length !== 16 || raw.some(row => !Array.isArray(row) || row.length !== 6 || !row.every(Number.isFinite)))
            throw new Error("Learned output must contain 16×6 finite action values");
          if (!Array.isArray(prediction) && (!Array.isArray(initialNoise) || initialNoise.length !== 512 || !initialNoise.every(Number.isFinite)))
            throw new Error("Learned inference trace must contain 512 finite initial noise values");
          if ((secondNoise !== undefined || memberActions !== undefined) && !memberTraceComplete(secondNoise, memberActions))
            throw new Error("Incomplete paired learned inference trace");
          // Preserve raw outputs for parity/auditing; only actuator goals are saturated.
          this.actions = raw.slice(0, this.executedActionsPerChunk).map(row => row.map((value, i) => {
            const [lo, hi] = i === 5 ? [0, 1] : ARM_LIMITS[i]; return Math.max(lo, Math.min(hi, value));
          }) as SixValues);
          this.chunks.push({ timestamp: time, input, rawActions: raw.map(row => [...row]),
            actions: this.actions.map(row => [...row] as SixValues), inferenceMs: performance.now() - started,
            ...(initialNoise ? { initialNoise: [...initialNoise] } : {}),
            ...(secondNoise ? { secondNoise: [...secondNoise] } : {}), ...(memberActions ? { memberActions: memberActions.map(row => [...row]) } : {}) });
          this.index = 0; this.chunkTime = time; this.nextTime = time + 0.1; this.phase = "execute";
          this.latch(this.actions[0]);
        } catch (error) { this.fail(error); }
      }, error => { if (!this.stopped) this.fail(error); });
    } catch (error) { this.fail(error); }
  }
  afterStep(dt: number) {
    if (this.stopped) return;
    this.sim.mj.mj_kinematics(this.sim.model, this.sim.data);
    try {
      this.evaluator.update(this.sim.observation(), dt, this.baseDrifted() ? "base-drift" : undefined);
      if (this.result.status !== "running") this.stop();
    } catch (error) { this.fail(error); }
  }
  stop() {
    if (this.stopped) return;
    this.stopped = true;
    if (this.sim.learningCapture === this) { this.sim.learningCapture = null; this.sim.setController(null); }
    if (this.disposePolicy) this.policy.dispose();
  }
}
