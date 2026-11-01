import { Quaternion, Vector3 } from "three";
import type { Simulation } from "../simulation/mujoco.ts";
import { TaskEvaluator, type TaskResult } from "../simulation/task.ts";
import type { ArmJointTarget, Observation, Vec3 } from "../simulation/types.ts";
import { ARM_LIMITS } from "../robot/hsr-model.ts";
import { solveHSR, forwardHSR } from "../robot/hsr-ik.ts";
import { validateLearningBasePose, type Instruction } from "./recording.ts";
import { mugBounds } from "../simulation/mug-geometry.ts";

type Goal = { armJointTarget: ArmJointTarget; gripper: number };
export type TeacherResult = TaskResult & { teacherFailure: string | null; phase: string; grasped: boolean; lifted: boolean };
export type TeacherProfile = "legacy" | "stable-ball" | "ball-recovery" | "gentle-mug" | "gentle-mug-v2" | "gentle-mug-v3" | "gentle-mug-v4";
// gentle-mug: the legacy mug carry ends at full arm reach with the mug over the near rim, and the legacy lowering
// presses it onto the rim (~40 N); opening then releases that push and jerks the base (up to 30 mrad). gentle-mug
// lowers only until the mug bottom is 2 cm above the rim, opens without pressing, then bends the wrist (flex -0.25,
// roll +0.35) to push the mug off the rim into the basket before rising straight up.
const GENTLE = { rimTop: 0.205, rimGap: 0.02, maxDrop: 0.12, push: [0, 0, 0, -0.25, 0.35], retreat: 0.05 };
// gentle-mug-v2: the carry ends with the hand pitched +0.4 instead of 0. The mug then hangs about 4 cm further into
// the basket, its outline clears the near rim, and the wrist never passes the straight-wrist singularity where the
// IK swapped arm and wrist roll mid-carry (about 1.6 rad in 0.4 s). If the outline is inside the fence (0.5 cm
// margin) the mug is lowered 16 cm, below the rim, before the hand opens, so it stands instead of falling over the
// rim and rolling (CPU seeds 1000-1099: 96/100 successes, all upright; v1 94/100, 92 on their side).
const GENTLE_V2 = { carryPitch: 0.4, deepDrop: 0.16, fenceMargin: 0.005 };
// gentle-mug-v3: v2 sets the upright mug down where the arm reaches, about 1 cm from the near and side walls, so a
// policy that releases 1 cm off leaves it outside the success box (v10 synchronous 7/10). v3 swings arm roll -0.25 /
// wrist flex -0.15 over the whole release and opens only in its second half, tipping the mug toward the basket centre
// (CPU audit, normal seeds 1000-1089: 87/90, quality-accepted 81, median wall margin 5.9 cm vs v2 0.9 cm). Swinging
// the closed hand faster pressed the leaning mug onto an opening finger above the 100 N quality limit.
const GENTLE_V3 = { swing: [0, 0, -0.25, -0.15, 0] };
// gentle-mug-v4: v3 still leaves about 8 % of the mugs standing next to a wall (v12 synchronous: the one failure).
// A larger swing (-0.3 / -0.18) tips nearly all of them (CPU audit 1000-1044: 45/45, 1 upright, wall margin p10
// 1.5 cm vs 0.9 cm) at the cost of more opening-finger contacts above the 100 N quality limit (accepted 36/45).
const GENTLE_V4 = { swing: [0, 0, -0.3, -0.18, 0] };
const PHASES = ["above", "align", "close", "lift", "carry", "lower", "release", "settle"];


// Oracle teacher for labels/feasibility only. Object state never becomes a policy feature.
export class StationaryTeacher {
  private readonly sim: Simulation;
  private readonly targetId: "mug" | "ball";
  readonly evaluator: TaskEvaluator;
  private readonly generation: number;
  private readonly startTime: number;
  private readonly startBase: Vec3;
  private objectStart: Vec3;
  private lastTime: number;
  private phaseIndex = 0;
  private phaseStart: Vec3;
  private lowerJoints: ArmJointTarget | null = null;
  private goal: ArmJointTarget;
  private readonly initialJoints: ArmJointTarget;
  private readonly readyJoints: ArmJointTarget;
  private failure: string | null = null;
  private completed = false;
  private grasped = false;
  private lifted = false;
  private lostFor = 0;
  private readonly profile: TeacherProfile;
  private clearanceStart: Vec3 | null = null;
  private clearanceComplete = false;
  private drop = 0.12;
  // Takeover collection starts the gentle mug teacher at a later phase from a state the learned policy reached.
  private readonly entryOffset: number;

  constructor(sim: Simulation, instruction: Instruction, profile: TeacherProfile = "legacy", entry: { entryPhase: 0 | 1 | 3 | 4 | 5 } = { entryPhase: 0 }) {
    if (instruction !== "Put the mug in the basket." && instruction !== "Put the tennis ball in the basket.") throw new Error("Unsupported teacher instruction");
    if (!["legacy", "stable-ball", "ball-recovery", "gentle-mug", "gentle-mug-v2", "gentle-mug-v3", "gentle-mug-v4"].includes(profile)) throw new Error("Unsupported teacher profile");
    if (profile.startsWith("gentle-mug") ? instruction !== "Put the mug in the basket." : profile !== "legacy" && instruction !== "Put the tennis ball in the basket.") throw new Error("Unsupported ball teacher profile");
    this.profile = profile;
    this.sim = sim; this.targetId = instruction.includes("mug") ? "mug" : "ball";
    this.evaluator = new TaskEvaluator(this.targetId);
    const obs = sim.observation(), robot = obs.robot!;
    this.generation = sim.resetGeneration; this.startTime = this.lastTime = obs.timestamp;
    this.startBase = [robot.base.position[0], robot.base.position[2], robot.base.yaw];
    this.objectStart = [...obs.objects.find(o => o.id === this.targetId)!.position];
    this.phaseStart = [...robot.endEffectorPosition]; this.goal = [...robot.joints] as ArmJointTarget;
    this.initialJoints = [...this.goal];
    let ready: Vec3 = this.objectStart.map((v, i) => v - robot.base.position[i] + (i === 1 ? 0.18 : 0)) as Vec3;
    ready = new Vector3(...ready).applyAxisAngle(new Vector3(0,1,0),-robot.base.yaw).toArray() as Vec3;
    this.readyJoints = solveHSR(ready, [-robot.base.yaw, 0, 0], [0.4, -1.9, 0, 0.33, 0]) as ArmJointTarget;
    const entryPhase = entry.entryPhase;
    if (![0, 1, 3, 4, 5].includes(entryPhase) || (entryPhase && !profile.startsWith("gentle-mug"))) throw new Error("Teacher entry phases need a gentle-mug profile");
    if (entryPhase >= 3 && !(robot.gripperContact && robot.contactId === this.targetId)) throw new Error("Teacher entry past the grasp needs a held mug");
    this.entryOffset = entryPhase ? this.ends()[entryPhase - 1] : 0;
    this.grasped = entryPhase >= 3; this.lifted = entryPhase >= 4;
    this.evaluator.start(obs);
  }
  private ends() {
    const recovery = this.profile === "ball-recovery", stable = recovery || this.profile === "stable-ball", ball = this.targetId === "ball";
    return recovery ? [3, 6, 8, 11, 15, 18, 21, 115] : stable ? [2, 4, 6, 9, 13, 16, 19, 120] : [2, 4, 6, 9, ball ? 13 : 17, ball ? 13 : 20, ball ? 16 : 23, 120];
  }
  get result(): TeacherResult {
    const task = this.evaluator.result;
    return { ...task, elapsed: this.lastTime - this.startTime,
      status: this.failure ? "failed" : task.status === "success" && !this.completed ? "running" : task.status,
      teacherFailure: this.failure, phase: this.profile === "ball-recovery" && !this.clearanceComplete ? "clearance" : PHASES[this.phaseIndex], grasped: this.grasped, lifted: this.lifted };
  }
  // Acquisition may close at the canonical 2ms success before the next camera
  // sample or scheduled endpoint. This changes no goal or physical state.
  finishScoredSuccess(): boolean {
    if (this.failure || this.sim.resetGeneration !== this.generation || this.evaluator.result.status !== "success" || !this.grasped || !this.lifted) return false;
    this.lastTime = this.sim.data.time; this.completed = true;
    return true;
  }
  private hold(obs: Observation): Goal {
    return { armJointTarget: obs.robot!.joints.map((q, i) => Math.max(ARM_LIMITS[i][0], Math.min(ARM_LIMITS[i][1], q))) as ArmJointTarget,
      gripper: this.sim.opening };
  }
  private fail(reason: string, obs: Observation): Goal { this.failure = reason; return this.hold(obs); }

  next(obs: Observation): Goal {
    if (this.result.status !== "running") return this.hold(obs);
    if (this.sim.resetGeneration !== this.generation) return this.fail("reset", obs);
    const recovery = this.profile === "ball-recovery", stable = recovery || this.profile === "stable-ball", v4 = this.profile === "gentle-mug-v4", v3 = v4 || this.profile === "gentle-mug-v3", v2 = v3 || this.profile === "gentle-mug-v2", gentle = v2 || this.profile === "gentle-mug";
    const ends = this.ends();
    const robot = obs.robot!, dt = obs.timestamp - this.lastTime, totalElapsed = obs.timestamp - this.startTime, elapsed = totalElapsed - (recovery ? 5 : 0) + this.entryOffset;
    if (!Number.isFinite(dt) || dt < -1e-8) return this.fail("invalid-time", obs);
    this.lastTime = obs.timestamp;
    try { validateLearningBasePose([robot.base.position[0], robot.base.position[2], robot.base.yaw], this.startBase); }
    catch { return this.fail("base-drift", obs); }
    // Capture scores every physical step; this call validates the current observation without adding time twice.
    this.evaluator.update(obs, 0);
    if (this.evaluator.result.status === "failed") return this.hold(obs);
    if (totalElapsed >= 120 - 1e-8) return this.fail("timeout", obs);
    const object = obs.objects.find(o => o.id === this.targetId)!;
    const held = robot.gripperContact && robot.contactId === this.targetId;
    const releasing = elapsed >= ends[5] - 1e-8;
    if (robot.gripperContact && robot.contactId !== this.targetId && !(releasing && robot.contactId === "basket")) return this.fail("wrong-grasp", obs);
    if (recovery && !this.clearanceComplete) {
      if (totalElapsed < 2 - 1e-8) {
        this.goal = [...this.initialJoints];
        this.goal[0] = Math.min(0.69, this.initialJoints[0] + 0.18 * Math.min(1, (totalElapsed + 0.1) / 2));
        return { armJointTarget: [...this.goal], gripper: 1 };
      }
      if (totalElapsed < 5 - 1e-8) {
        this.clearanceStart ??= [...robot.endEffectorPosition];
        const u = Math.min(1, (totalElapsed - 2 + 0.1) / 3), smooth = u * u * (3 - 2 * u);
        const target: Vec3 = [object.position[0], object.position[1] + 0.4, object.position[2]];
        const world = this.clearanceStart.map((v, i) => v + (target[i] - v) * smooth) as Vec3;
        const local = new Vector3(...world).sub(new Vector3(...robot.base.position)).applyAxisAngle(new Vector3(0, 1, 0), -robot.base.yaw).toArray() as Vec3;
        for (let n = 0; n < 5; n++) this.goal = solveHSR(local, [-robot.base.yaw, -0.9, 0], this.goal) as ArmJointTarget;
        return { armJointTarget: [...this.goal], gripper: 1 };
      }
      this.clearanceComplete = true;
      this.objectStart = [...object.position]; this.phaseStart = [...robot.endEffectorPosition];
    }
    if (held) { this.grasped = true; this.lostFor = 0; this.lifted ||= object.position[1] > this.objectStart[1] + 0.15; }
    else if (this.grasped && elapsed < ends[5] - 1e-8) this.lostFor += dt;
    if (this.lostFor > 0.2 + 1e-8) return this.fail("grasp-lost", obs);
    if (elapsed >= ends[2] - 1e-8 && !this.grasped) return this.fail("grasp-missing", obs);
    if (elapsed >= ends[3] - 1e-8 && !this.lifted) return this.fail("lift-blocked", obs);
    if (this.evaluator.result.status === "success") {
      if (!this.grasped || !this.lifted) return this.fail("unverified-transport", obs);
      if (elapsed >= ends[6] - 1e-8) {
        this.completed = true;
        return this.hold(obs);
      }
    }
    const phase = ends.findIndex(end => elapsed < end - 1e-8);
    if (phase < 0) return this.fail("timeout", obs);
    if (phase !== this.phaseIndex) {
      if (phase === 6) {
        const basket = obs.furniture.find(o => o.id === "basket")!;
        const local = new Vector3(...object.position).sub(new Vector3(...basket.position)).applyQuaternion(new Quaternion(...basket.rotation).invert());
        const above = this.targetId === "mug" ? Math.abs(local.x) <= 0.18 && Math.abs(local.z) <= 0.16 : Math.abs(local.x) + 0.0335 <= 0.21 && Math.abs(local.z) + 0.0335 <= 0.19;
        if (!held || !above) return this.fail("release-blocked", obs);
      }
      if (phase === 5 || (stable && phase === 6)) this.lowerJoints = [...robot.joints] as ArmJointTarget;
      if (gentle && phase === 5) {
        const basket = obs.furniture.find(o => o.id === "basket")!;
        const b = mugBounds(object, basket), m = GENTLE_V2.fenceMargin;
        const inside = b.min[0] >= -0.18 + m && b.max[0] <= 0.18 - m && b.min[2] >= -0.16 + m && b.max[2] <= 0.16 - m;
        this.drop = v2 && inside ? GENTLE_V2.deepDrop : Math.max(0, Math.min(GENTLE.maxDrop, b.min[1] - GENTLE.rimTop - GENTLE.rimGap));
      }
      this.phaseIndex = phase; this.phaseStart = [...robot.endEffectorPosition];
    }
    const begin = phase ? ends[phase - 1] : 0, duration = phase === 7 ? 3 : ends[phase] - begin;
    const ratio = (elapsed - begin + 0.1) / duration;
    const u = ratio >= 1 - 1e-8 ? 1 : ratio, smooth = u * u * (3 - 2 * u);
    const ball = this.targetId === "ball", carry: Vec3 = ball ? [0.18, 0.78, stable ? 0.465 : 0.50] : [0.25, 0.82, 0.40];
    const [x, y, z] = this.objectStart;
    const target: Vec3 = phase === 0 ? [x, y + (recovery ? 0.07 : 0.18), z] : phase <= 2 ? [x, y + (ball ? 0 : 0.015), z] : phase === 3 ? [x, y + 0.3, z] : phase === 4 || phase === 6 ? carry : [0.18, 0.9, 0.42];
    if (recovery && (phase === 1 || phase === 2)) {
      // A sphere at the fingertips is squeezed outward. Seat it 1cm toward the palm.
      const inward = new Vector3(0, 0, 0.01).applyQuaternion(new Quaternion(...robot.endEffectorRotation));
      target[0] += inward.x; target[1] += inward.y; target[2] += inward.z;
    }
    if (phase === 0 && !ball) {
      this.goal = this.initialJoints.map((v, i) => v + (this.readyJoints[i] - v) * smooth) as ArmJointTarget;
    } else if (gentle && phase >= 5) {
      // Lower, hold while opening; settle pushes with the wrist in its first half and rises by drop + retreat in the second.
      const first = Math.min(1, 2 * u), second = Math.max(0, 2 * u - 1), s1 = first * first * (3 - 2 * first), s2 = second * second * (3 - 2 * second);
      // Depth below the lowered pose: drop while lowering and opening; the settle rise is drop + retreat (even at drop 0).
      const depth = phase === 5 ? this.drop * smooth : phase === 6 ? this.drop : this.drop - s2 * (this.drop + GENTLE.retreat);
      // v3: the hand swings over the whole release (closed for its first half) and keeps the swing afterwards.
      const swing = v3 ? (phase === 6 ? smooth : phase === 7 ? 1 : 0) : 0;
      this.goal = this.lowerJoints!.map((q, i) => Math.max(ARM_LIMITS[i][0], Math.min(ARM_LIMITS[i][1], q + (phase === 7 ? GENTLE.push[i] * s1 : 0) + (v4 ? GENTLE_V4 : GENTLE_V3).swing[i] * swing))) as ArmJointTarget;
      this.goal[0] = Math.max(0, Math.min(ARM_LIMITS[0][1], this.lowerJoints![0] - depth));
    } else if (stable && phase === 6) {
      // Opening a loaded grasp must not also move the arm through another IK solve.
      this.goal = [...this.lowerJoints!];
    } else if ((phase === 5 || phase === 6) && (!ball || stable)) {
      this.goal = [...this.lowerJoints!];
      this.goal[0] = Math.max(0, this.lowerJoints![0] - (ball ? 0.05 : 0.12) * (phase === 5 ? smooth : 1));
    } else {
      const world = this.phaseStart.map((v, i) => v + (target[i] - v) * smooth) as Vec3;
      const relative = world.map((v, i) => v - robot.base.position[i]) as Vec3;
      const local = new Vector3(...relative).applyAxisAngle(new Vector3(0,1,0),-robot.base.yaw).toArray() as Vec3;
      const compensated = [...local] as Vec3;
      for (let n = 0; n < 5; n++) {
        this.goal = solveHSR(compensated, [-robot.base.yaw, ball ? -0.9 : phase <= 3 ? -0.4 : phase === 4 ? -0.4 * (1-smooth) + (v2 ? GENTLE_V2.carryPitch * smooth : 0) : v2 ? GENTLE_V2.carryPitch : 0, 0], this.goal) as ArmJointTarget;
        if (recovery && (phase === 1 || phase === 2)) {
          // Label-only position correction for the residual of the 5-DOF pose solver.
          const actual = forwardHSR(this.goal).position.toArray();
          compensated.forEach((v, i) => { compensated[i] = v + Math.max(-0.03, Math.min(0.03, local[i] - actual[i])); });
        }
      }
    }
    return { armJointTarget: [...this.goal], gripper: recovery && phase === 0 ? 0 : phase < 2 ? ball ? 0.5 : 0.8 : phase < 6 ? 0 : phase === 6 ? (v3 ? Math.max(0, Math.min(1, 2 * u - 1)) : u) : 1 };
  }
}
