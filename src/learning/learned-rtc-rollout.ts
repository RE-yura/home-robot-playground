import type { Simulation } from "../simulation/mujoco.ts";
import type { CameraImage, Vec3 } from "../simulation/types.ts";
import { TaskEvaluator, type MugSuccessCriterion } from "../simulation/task.ts";
import { ARM_LIMITS, ARM_NAMES } from "../robot/hsr-model.ts";
import { validateLearningBasePose, type SixValues } from "./recording.ts";
import { evaluationTarget, memberTraceComplete, type LearnedInput, type LearnedPrediction, type LearnedChunk } from "./learned-rollout.ts";
import { RtcActionQueue, type RtcTicket } from "./rtc-action-queue.ts";
import { TeacherQualityAudit } from "./quality.ts";
import { GraspCycleWatch, returnGoals } from "./rtc-assist.ts";
export type RtcContinuation = { prefix: SixValues[]; delaySteps: number; horizon: number };
// Private data collection: at each 100 ms action boundary `reason` may hand the robot over (e.g. to a teacher); the rollout
// then stops before latching another goal and calls `begin` at that exact time.
export type RtcHandoff = { reason(time: number): string | null; begin(reason: string, time: number): void };
export interface RtcPolicy { infer(input: LearnedInput, continuation?: RtcContinuation | null): Promise<LearnedPrediction | number[][]>; dispose(): void; readonly whenIdle?: Promise<void> }
type RtcMoment = { simulationTime: number; wallTimeMs: number };
type RtcRequest = {
  id: number; input: LearnedInput; continuation: RtcContinuation | null;
  started: RtcMoment; returned: RtcMoment | null; interrupted: (RtcMoment & { reason: string }) | null;
  status: "pending" | "accepted" | "failed" | "interrupted"; lateReturn: boolean; error: string | null;
  prediction: LearnedPrediction | number[][] | null; firstActionAt: number | null;
  handoff: (RtcMoment & { previousGoal: SixValues | null; goal: SixValues; delta: SixValues | null }) | null;
};
const EPS = 1e-8;

export class LearnedRtcRollout {
  stopped = false;
  error: string | null = null;
  readonly pauseDisplayDuringInference = false;
  readonly playbackRate: number;
  readonly chunks: (LearnedChunk & { continuation: RtcContinuation | null; acceptedAt: number; skipped: number })[] = [];
  readonly requests: RtcRequest[] = [];
  private readonly sim: Simulation;
  private readonly instruction: string;
  private readonly policy: RtcPolicy;
  private queue: RtcActionQueue;
  private readonly evaluator: TaskEvaluator;
  private readonly audit: TeacherQualityAudit;
  private readonly generation: number;
  private readonly base: Vec3;
  readonly delaySteps: number;
  readonly executionSteps: number | null;
  private readonly guidance: boolean;
  private readonly pauseForGripperChanges: boolean;
  // A model trained on goals starting this many steps after its observation (0 for the public model).
  readonly chunkOffsetSteps: number;
  // Trained with action-prefix conditioning: every request carries the goals that will run during inference.
  readonly prefixConditioning: { maxDelay: number } | null;
  // End time of a gripper-changing chunk, at which physics waits for a plan from a fresh image.
  private pauseAt: number | null = null;
  pausedForGripper = false;
  gripperPauses = 0;
  private readonly disposePolicy: boolean;
  private primed = false;
  private pending = false;
  private wantImage = true;
  private nextBoundary: number;
  private nextCapture: number;
  private lastGoal: SixValues | null = null;
  // Fixed latency (data collection only): a reply replaces goals exactly this many actions after its capture. A late reply
  // holds physics at that boundary and an early one waits for it, so the trajectory does not depend on inference speed.
  readonly fixedLatencySteps: number | null;
  private readonly handoff: RtcHandoff | null;
  handedOver: { reason: string; time: number } | null = null;
  private dueAt: number | null = null;
  private early: (() => void) | null = null;
  // Rule assist (opt-in, reported separately from learned-only results): a measured-only watchdog (rtc-assist.ts). Two
  // failed grasp cycles, or assistStallSeconds (35) without success since the (re)start, return the arm to its start
  // pose over 2 s with an open hand; the policy then restarts like the episode start. At most two restarts; the task
  // evaluator, audit and timer run on.
  private readonly assist: boolean;
  private readonly assistStallSeconds: number;
  readonly assistEvents: { time: number; reason: "regrasp-loop" | "stall" }[] = [];
  private readonly startJoints: number[];
  private readonly graspWatch = new GraspCycleWatch();
  private segmentStart: number;
  private scripted: SixValues[] | null = null;
  private waitingIdle = false;
  private epoch = 0;
  // Late-reply grace (opt-in): when the queued goals run out while a reply is pending, the actuator keeps the last goal
  // for up to lateHoldSteps 100 ms boundaries instead of ending the run; lateHolds counts the held boundaries.
  readonly lateHoldSteps: number;
  lateHolds = 0;
  private heldRun = 0;
  get needsImages() { return !this.stopped && !this.pending && !this.waitingIdle && this.wantImage; }
  get inferencePending() { return !this.stopped && this.pending; }
  get result() { return this.evaluator.result; }
  get successCriterion() { return this.evaluator.criterion; }
  get evaluationTarget() { return this.evaluator.target; }
  get quality() { return this.audit.metrics; }
  constructor(sim: Simulation, instruction: string, policy: RtcPolicy, options: { target?: "mug" | "ball" | null; playbackRate: number; delaySteps: number; executionSteps?: number; guidance?: boolean; pauseForGripperChanges?: boolean; chunkOffsetSteps?: number; prefixConditioning?: { maxDelay: number }; disposePolicy?: boolean; fixedLatencySteps?: number; handoff?: RtcHandoff; assist?: boolean; assistStallSeconds?: number; lateHoldSteps?: number; success?: MugSuccessCriterion }) {
    const target = evaluationTarget(instruction, options.target);
    if (!Number.isFinite(options.playbackRate) || options.playbackRate <= 0 || options.playbackRate > 1 || !Number.isInteger(options.delaySteps) || options.delaySteps < 0 || options.delaySteps > 10) throw new Error("Invalid RTC timing budget");
    sim.setController(null); this.sim = sim; this.instruction = instruction; this.policy = policy;
    // Default (null): capture again as soon as a reply is accepted. A fixed execution horizon instead waits until that many
    // actions of each chunk (counted from its capture) have run, leaving 16 - executionSteps queued goals as the prefix.
    const exec = options.executionSteps ?? null;
    // delaySteps already includes one step of margin. Guided sampling also needs the prefix to extend past the delay;
    // unguided sampling only needs the queued prefix to outlast the expected inference delay. An offset model's reply
    // starts offset goals after its capture, so the prefix must also cover those goals.
    const guided = options.guidance ?? true, offset = options.chunkOffsetSteps ?? 0;
    if (!Number.isInteger(offset) || offset < 0 || offset > 4 || (offset && guided)) throw new Error("Invalid RTC chunk offset (offset models are unguided)");
    const prefixModel = options.prefixConditioning ?? null;
    if (prefixModel && (guided || offset || !Number.isInteger(prefixModel.maxDelay) || options.delaySteps > prefixModel.maxDelay))
      throw new Error("Prefix conditioning is unguided, unshifted and limited to its trained delay");
    const minPrefix = guided ? options.delaySteps + 1 : Math.max(options.delaySteps, offset);
    if (exec !== null && (!Number.isInteger(exec) || exec <= options.delaySteps || 16 + offset - exec < minPrefix)) throw new Error("Invalid RTC execution horizon");
    const latency = options.fixedLatencySteps ?? null;
    if (latency !== null && (!Number.isInteger(latency) || latency < 1 || latency > options.delaySteps)) throw new Error("Fixed RTC latency must be 1..delaySteps actions");
    this.fixedLatencySteps = latency; this.handoff = options.handoff ?? null;
    const hold = options.lateHoldSteps ?? 0;
    if (!Number.isInteger(hold) || hold < 0 || hold > 5) throw new Error("Late-reply grace must be 0..5 actions");
    this.lateHoldSteps = hold;
    this.assist = options.assist ?? false; this.assistStallSeconds = options.assistStallSeconds ?? 35;
    this.startJoints = ARM_NAMES.map(name => sim.q(name)); this.segmentStart = sim.data.time;
    this.playbackRate = options.playbackRate; this.delaySteps = options.delaySteps; this.executionSteps = exec; this.guidance = guided; this.pauseForGripperChanges = options.pauseForGripperChanges ?? false; this.chunkOffsetSteps = offset; this.prefixConditioning = prefixModel; this.disposePolicy = options.disposePolicy ?? true;
    this.generation = sim.resetGeneration; this.base = this.basePose();
    this.queue = new RtcActionQueue(sim.data.time, offset); this.nextBoundary = this.nextCapture = sim.data.time;
    this.evaluator = new TaskEvaluator(target ?? "none", options.success ? { success: options.success } : {}); this.evaluator.start(sim.observation());
    // Independent physical audit only; these object/contact values never enter the policy.
    // An unlearned step has no target object; its force audit is reported against the mug, the primary object.
    this.audit = new TeacherQualityAudit(sim, target ?? "mug");
    // A previous continuous task (e.g. the preceding step of a plan) may leave its last request running in the worker;
    // the first capture waits for it, with physics held, instead of colliding with it.
    this.awaitIdle();
    sim.learningCapture = this;
  }
  private awaitIdle() {
    if (this.policy.whenIdle) { this.waitingIdle = true; this.policy.whenIdle.then(() => { this.waitingIdle = false; }, () => { this.waitingIdle = false; }); }
  }
  private basePose(): Vec3 { const p = this.sim.position(this.sim.baseId); return [p[0], p[2], this.sim.q("base_yaw")]; }
  private drifted() { try { validateLearningBasePose(this.basePose(), this.base); return false; } catch { return true; } }
  private fail(error: unknown) { this.error = String(error instanceof Error ? error.message : error).slice(0, 1024); this.stop(); }
  private moment(): RtcMoment { return { simulationTime: this.sim.data.time, wallTimeMs: performance.now() }; }
  // A chunk sampled from an image taken while the fingers were still moving may restart that transition, and a release
  // ramp only opens the hand in a chunk's last goals. Chunks whose gripper goals change run to their end; the next plan
  // then comes from an image taken with physics held, as in synchronous execution. Uses the policy's own goals only.
  private schedulePause(actions: SixValues[], firstIndex: number, endTime: number) {
    const grip = actions.slice(firstIndex).map(a => a[5]);
    if (!this.pauseForGripperChanges || Math.max(...grip) - Math.min(...grip) <= .3) { this.pauseAt = null; return; }
    this.pauseAt = endTime; this.nextCapture = Infinity;
  }
  private latch(action: SixValues) {
    this.sim.learningCapture = null;
    try { this.sim.applyAction({ armJointTarget: action.slice(0, 5) as [number, number, number, number, number], gripper: action[5], base: [0, 0, 0], head: [0, 0] }); }
    finally { if (!this.stopped) this.sim.learningCapture = this; }
    const request = this.requests.find(r => r.status === "accepted" && !r.handoff && r.firstActionAt !== null && Math.abs(r.firstActionAt - this.sim.data.time) < EPS);
    if (request) request.handoff = { ...this.moment(), previousGoal: this.lastGoal, goal: [...action], delta: this.lastGoal ? action.map((v, i) => v - this.lastGoal![i]) as SixValues : null };
    this.lastGoal = [...action];
  }
  captureDue(time: number): boolean {
    if (this.stopped) return false;
    try {
      if (this.sim.resetGeneration !== this.generation) throw new Error("RTC reset");
      this.sim.mj.mj_kinematics(this.sim.model, this.sim.data);
      if (this.drifted()) { this.evaluator.fail("base-drift"); this.stop(); return true; }
      if (!this.primed) return true;
      if (time >= this.nextBoundary - EPS) {
        if (Math.abs(time - this.nextBoundary) > EPS) throw new Error("RTC missed a100ms action boundary");
        const reason = this.handoff?.reason(time) ?? null;
        if (reason) { this.handedOver = { reason, time }; this.stop(); this.handoff!.begin(reason, time); return true; }
        if (this.scripted) {
          if (this.scripted.length) { this.latch(this.scripted.shift()!); this.nextBoundary += .1; return false; }
          this.restartPolicy(time); return true;
        }
        if (this.assist && this.assistEvents.length < 2) {
          const why = this.graspWatch.failedCycles >= 2 ? "regrasp-loop" : time - this.segmentStart >= this.assistStallSeconds - EPS ? "stall" : null;
          if (why) { this.beginReturn(why, time); this.latch(this.scripted!.shift()!); this.nextBoundary += .1; return false; }
        }
        if (this.dueAt !== null && time >= this.dueAt - EPS) {
          if (this.early) { const apply = this.early; this.early = null; apply(); if (this.stopped) return true; }
          else if (this.pending) return true;
        }
        if (this.pauseAt !== null && Math.abs(time - this.pauseAt) < EPS) {
          if (!this.pausedForGripper) { this.pausedForGripper = true; this.gripperPauses++; this.wantImage = true; }
          return true;
        }
        if (this.lateHoldSteps && this.pending && !this.queue.has(time) && this.heldRun < this.lateHoldSteps) {
          this.heldRun++; this.lateHolds++; this.nextBoundary += .1; return this.needsImages;
        }
        this.heldRun = 0;
        this.latch(this.queue.at(time)); this.nextBoundary += .1;
        if (!this.pending && time >= this.nextCapture - EPS) this.wantImage = true;
      }
      return this.needsImages;
    } catch (error) { this.fail(error); return true; }
  }
  private beginReturn(reason: "regrasp-loop" | "stall", time: number) {
    this.assistEvents.push({ time, reason }); this.epoch++;
    for (const request of this.requests) if (request.status === "pending") { request.status = "interrupted"; request.interrupted = { ...this.moment(), reason: "assist" }; }
    this.pending = false; this.dueAt = null; this.early = null; this.pauseAt = null; this.pausedForGripper = false; this.wantImage = false;
    this.scripted = returnGoals(ARM_NAMES.map(name => this.sim.q(name)), this.startJoints); this.graspWatch.reset();
  }
  // After the scripted return: a fresh queue and a first request without a prefix, physics held until its reply.
  private restartPolicy(time: number) {
    this.scripted = null; this.queue = new RtcActionQueue(time, this.chunkOffsetSteps); this.primed = false;
    this.nextBoundary = this.nextCapture = time; this.segmentStart = time; this.wantImage = true;
    this.awaitIdle();
  }
  capture(images: CameraImage[]) {
    if (!this.needsImages) return;
    try {
      const time = this.sim.data.time;
      if (!this.captureDue(time) || this.stopped) return;
      if (images.length !== 2 || !["head", "hand"].every(view => images.filter(i => i.view === view).length === 1) ||
        images.some(i => i.width !== 320 || i.height !== 240 || Math.abs(i.timestamp - time) > EPS || !Number.isFinite(i.timestamp) || !/^data:image\/jpeg;base64,[A-Za-z0-9+/]+={0,2}$/.test(i.dataUrl))) throw new Error("RTC requires fresh synchronized camera images");
      const robot = this.sim.observation().robot!, state = [...robot.joints, robot.gripper];
      if (state.length !== 6 || !state.every(Number.isFinite)) throw new Error("Invalid measured RTC state");
      const ticket: RtcTicket | null = this.primed ? this.queue.begin(time, this.pausedForGripper) : null;
      if (ticket && !this.pausedForGripper && ticket.prefix.length < (this.guidance ? this.delaySteps + 1 : Math.max(this.delaySteps, this.chunkOffsetSteps))) throw new Error("RTC has insufficient prediction coverage");
      const continuation = ticket ? { prefix: ticket.prefix, delaySteps: this.delaySteps, horizon: Math.min(10, ticket.prefix.length) } : null;
      const input: LearnedInput = { images: images.map(i => ({ ...i })), instruction: this.instruction, state: state as SixValues };
      const request: RtcRequest = { id: this.requests.length + 1, input, continuation, started: this.moment(), returned: null, interrupted: null,
        status: "pending", lateReturn: false, error: null, prediction: null, firstActionAt: null, handoff: null };
      this.requests.push(request);
      this.pending = true; this.wantImage = false; const started = request.started.wallTimeMs, epoch = this.epoch;
      this.dueAt = ticket && this.fixedLatencySteps !== null && !this.pausedForGripper ? time + this.fixedLatencySteps / 10 : null;
      // With guidance off the queue still aligns replies by capture time, but the model samples the chunk unconditioned:
      // no continuation is sent, so the unchanged public graph and worker serve the request. A prefix-conditioned model
      // instead receives the committed goals (null before the first chunk).
      (this.guidance || this.prefixConditioning ? this.policy.infer(input, continuation) : this.policy.infer(input)).then(prediction => {
        request.returned = this.moment(); request.prediction = structuredClone(prediction);
        if (this.stopped || epoch !== this.epoch || this.sim.resetGeneration !== this.generation || this.sim.learningCapture !== this) { request.lateReturn = true; return; }
        const returnedMs = request.returned.wallTimeMs;
        const apply = () => { try {
          const raw = Array.isArray(prediction) ? prediction : prediction.actions;
          if (!Array.isArray(raw) || raw.length !== 16 || raw.some(a => !Array.isArray(a) || a.length !== 6 || !a.every(Number.isFinite))) throw new Error("RTC output must contain16x6 finite goals");
          const trace = Array.isArray(prediction) ? null : prediction;
          if (trace && (!Array.isArray(trace.initialNoise) || trace.initialNoise.length !== 512 || !trace.initialNoise.every(Number.isFinite) ||
            !memberTraceComplete(trace.secondNoise, trace.memberActions))) throw new Error("Incomplete RTC paired inference trace");
          const actions = raw.map(a => a.map((v, i) => { const [lo, hi] = i === 5 ? [0, 1] : ARM_LIMITS[i]; return Math.max(lo, Math.min(hi, v)); }) as SixValues);
          let skipped = 0; const span = (this.chunkOffsetSteps + 16) / 10;
          if (ticket) { const accepted = this.queue.accept(ticket, actions, this.sim.data.time); skipped = accepted.skipped; this.nextCapture = this.executionSteps === null ? accepted.firstTime : Math.max(accepted.firstTime, ticket.timestamp + this.executionSteps / 10); request.firstActionAt = accepted.firstTime; request.status = "accepted"; this.pausedForGripper = false; this.schedulePause(actions, skipped, ticket.timestamp + span); }
          else {
            // An offset model does not predict its first goals; the robot holds its measured state until they start.
            this.queue.prime(actions, this.chunkOffsetSteps ? state : undefined); request.firstActionAt = time + this.chunkOffsetSteps / 10; request.status = "accepted";
            this.latch(this.queue.at(time)); this.primed = true; this.nextBoundary = this.nextCapture = time + .1; this.schedulePause(actions, 0, time + span);
          }
          this.chunks.push({ timestamp: time, input, rawActions: raw.map(a => [...a]), actions, inferenceMs: returnedMs - started, continuation,
            acceptedAt: this.sim.data.time, skipped, ...(trace ? { initialNoise: [...trace.initialNoise], ...(trace.secondNoise ? { secondNoise: [...trace.secondNoise] } : {}), memberActions: trace.memberActions!.map(a => [...a]) } : {}) });
          this.pending = false; this.dueAt = null;
        } catch (error) { request.status = "failed"; request.error = String(error); this.fail(error); } };
        if (this.dueAt !== null && this.sim.data.time < this.dueAt - EPS) this.early = apply; else apply();
      }, error => {
        request.returned = this.moment(); request.error = String(error);
        if (this.stopped || epoch !== this.epoch) request.lateReturn = true;
        else { request.status = "failed"; this.fail(error); }
      });
    } catch (error) { this.fail(error); }
  }
  afterStep(dt: number) {
    if (this.stopped) return;
    this.sim.mj.mj_kinematics(this.sim.model, this.sim.data);
    try {
      this.evaluator.update(this.sim.observation(), dt, this.drifted() ? "base-drift" : undefined);
      this.audit.update(dt, { ...this.result, phase: "rtc" });
      if (this.assist && this.primed && !this.scripted) this.graspWatch.update(this.sim.q("arm_lift_joint"), this.sim.gripperOpening());
      if (this.result.status !== "running") this.stop();
    }
    catch (error) { this.fail(error); }
  }
  stop() {
    if (this.stopped) return;
    this.stopped = true; this.queue.stop(); this.early = null; this.dueAt = null;
    for (const request of this.requests) if (request.status === "pending") {
      request.status = "interrupted";
      request.interrupted = { ...this.moment(), reason: this.error ?? (this.handedOver ? "handoff" : this.result.status === "running" ? "stopped" : this.result.status) };
    }
    this.audit.dispose();
    if (this.sim.learningCapture === this) { this.sim.learningCapture = null; this.sim.setController(null); }
    if (this.disposePolicy) this.policy.dispose();
  }
}
