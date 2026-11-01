import type { Simulation } from "../simulation/mujoco.ts";
import type { ArmJointTarget, CameraImage, CameraAcquisition, Vec3 } from "../simulation/types.ts";
import { LearningRecorder, DemonstrationRecorder, validateDemoCollection, validateLearningBasePose, type Instruction, type CapturedEpisode, type DemoCollection, type SixValues } from "./recording.ts";
import type { TaskEvaluator } from "../simulation/task.ts";

export interface LearningCaptureGate {
  readonly needsImages?: boolean;
  readonly inferencePending?: boolean;
  readonly playbackRate?: number;
  readonly pauseDisplayDuringInference?: boolean;
  captureDue(time: number): boolean;
  capture(images: CameraImage[], acquisition?: CameraAcquisition): void;
  stop(): void;
  afterStep?(dt: number): void;
}
type Options = { instruction: Instruction; maxSamples?: number; taskEvaluator?: TaskEvaluator;
  collection?: DemoCollection;
  actionForSample: (index: number, state: SixValues) => { armJointTarget: ArmJointTarget; gripper: number };
  // DART collection: the executed command is perturb(goal); the recorded action remains the clean goal.
  perturb?: (goal: { armJointTarget: ArmJointTarget; gripper: number }, index: number) => { armJointTarget: ArmJointTarget; gripper: number } };
const EPS = 1e-8;

export class LearningCapture implements LearningCaptureGate {
  stopped = false;
  error: string | null = null;
  episode: CapturedEpisode | null = null;
  private readonly sim: Simulation;
  private readonly options: Options;
  private readonly recorder: LearningRecorder | DemonstrationRecorder;
  private readonly startBase: Vec3;
  private readonly startTime: number;
  private readonly generation: number;
  private readonly maxSamples: number;
  private index = 0;
  private pending = false;
  private nextTime: number;
  private sample: { timestamp: number; state: SixValues; action: SixValues } | null = null;

  get latestSample() {
    return this.sample && { timestamp: this.sample.timestamp, state: [...this.sample.state] as SixValues, action: [...this.sample.action] as SixValues };
  }

  constructor(sim: Simulation, options: Options) {
    const maxSamples = options.maxSamples ?? 30;
    if (!Number.isInteger(maxSamples) || maxSamples < 1 || maxSamples > 1800) throw new Error("Invalid capture duration");
    if (options.collection) validateDemoCollection(options.collection, sim.seed);
    sim.setController(null);
    this.sim = sim; this.options = options; this.maxSamples = maxSamples;
    this.startBase = this.basePose(); this.startTime = sim.data.time; this.nextTime = this.startTime;
    this.generation = sim.resetGeneration;
    const recording = { instruction: options.instruction, seed: sim.seed, basePose: this.startBase,
      robotMassKg: sim.robotMassKg, armTorqueNm: sim.armTorqueNm };
    this.recorder = options.collection ? new DemonstrationRecorder({ ...recording, collection: options.collection }) : new LearningRecorder(recording);
    sim.learningCapture = this;
  }
  private basePose(): Vec3 {
    const b = this.sim.position(this.sim.baseId); return [b[0], b[2], this.sim.q("base_yaw")];
  }
  private fail(error: unknown) {
    this.error = (error instanceof Error ? error.message : String(error)).slice(0, 1024) || "Capture failed";
    this.stop();
  }
  afterStep(dt: number) {
    if (this.stopped || !this.options.taskEvaluator) return;
    this.sim.mj.mj_kinematics(this.sim.model, this.sim.data);
    this.options.taskEvaluator.update(this.sim.observation(), dt);
  }
  captureDue(time: number): boolean {
    if (this.stopped) return false;
    try {
      if (this.sim.resetGeneration !== this.generation) throw new Error("Capture reset");
      // mj_step integrates qpos after computing transforms. Refresh only
      // kinematics before meshes/cameras read the newly integrated state.
      this.sim.mj.mj_kinematics(this.sim.model, this.sim.data);
      validateLearningBasePose(this.basePose(), this.startBase);
      if (time < this.nextTime - EPS) return false;
      if (Math.abs(time - this.nextTime) > EPS) throw new Error("Capture missed its time boundary");
      if (this.pending) {
        this.recorder.complete(time); this.pending = false; this.index++;
        if (this.index >= this.maxSamples) this.stop();
      }
      return true;
    } catch (error) { this.fail(error); return true; }
  }
  capture(images: CameraImage[]) {
    if (this.stopped) return;
    try {
      if (!this.captureDue(this.sim.data.time) || this.stopped) return;
      const robot = this.sim.observation().robot!;
      const state = [...robot.joints, robot.gripper] as SixValues;
      const goal = this.options.actionForSample(this.index, [...state]);
      // Internal synchronous latching bypasses the external-action takeover hook.
      this.sim.learningCapture = null;
      const command = this.sim.applyAction({ ...(this.options.perturb ? this.options.perturb(goal, this.index) : goal), base: [0, 0, 0], head: [0, 0] });
      this.sim.learningCapture = this;
      const action = (this.options.perturb ? [...goal.armJointTarget, Math.max(0, Math.min(1, goal.gripper))] : [...command.armJointTarget!, command.gripper!]) as SixValues;
      this.recorder.begin(this.sim.data.time, state, action, images, this.basePose());
      this.sample = { timestamp: this.sim.data.time, state, action };
      this.pending = true; this.nextTime = this.startTime + (this.index + 1) / 10;
    } catch (error) { this.fail(error); }
  }
  stop() {
    if (this.stopped) return;
    this.stopped = true;
    if (this.sim.learningCapture === this) this.sim.learningCapture = null;
    if (!this.error && this.pending && Math.abs(this.sim.data.time - this.nextTime) < EPS) this.recorder.complete(this.sim.data.time);
    this.sim.setController(null);
    this.episode = this.recorder.finish(this.error);
  }
}
