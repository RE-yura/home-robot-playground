import type { ArmJointTarget, Quaternion, RobotAction, Vec3 } from "../simulation/types.ts";
import { ARM_LIMITS, ARM_NAMES } from "../robot/hsr-model.ts";

export const CONTROL_LIMITS = {
  baseSpeed: 0.25, baseYawRate: 0.7, handSpeed: 0.18, handAngularRate: 0.7, headRate: 1,
  world: { x: [-3.25, 3.25], y: [0.05, 2.6], z: [-3.10, 2.70] },
  base: { x: [-3.05, 3.05], z: [-2.90, 2.50] },
  handLocal: { x: [-1.2, 1.2], y: [0.05, 2.1], z: [-1.2, 1.2] },
} as const;

export function stopAction(): RobotAction {
  return { base: [0, 0, 0], translation: [0, 0, 0], rotation: [0, 0, 0], head: [0, 0] };
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${name} must be an object`);
  return value as Record<string, unknown>;
}
function fields(value: Record<string, unknown>, allowed: readonly string[], name: string) {
  for (const key of Object.keys(value)) if (!allowed.includes(key)) throw new Error(`Unknown ${name} field: ${key}`);
}
function numeric(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name} must be finite`);
  return value;
}
function vector(value: unknown, size: number, name: string): number[] {
  if (!Array.isArray(value) || value.length !== size) throw new Error(`${name} must have ${size} numbers`);
  return Array.from(value, (v, i) => numeric(v, `${name}[${i}]`));
}
const clamp = (value: number, limit: number) => Math.max(-limit, Math.min(limit, value));
function limitVector(value: number[], limit: number): number[] {
  const magnitude = Math.hypot(...value), scale = magnitude > limit ? limit / magnitude : 1;
  return value.map(v => v * scale);
}
function bounded(value: number, range: readonly [number, number], name: string) {
  if (value < range[0] || value > range[1]) throw new Error(`${name} is outside the room/workspace bounds`);
}

export function validateArmJointTarget(input: unknown): ArmJointTarget {
  const target = vector(input, 5, "armJointTarget");
  target.forEach((value, i) => {
    const [lo, hi] = ARM_LIMITS[i];
    if (value < lo || value > hi) throw new Error(`armJointTarget ${ARM_NAMES[i]} is outside joint limits`);
  });
  return target as ArmJointTarget;
}

/** Shared boundary for manual, built-in and external actions. Velocities are per second. */
export function validateAction(input: unknown): RobotAction {
  const value = record(input, "action"), action: RobotAction = {};
  fields(value, ["base", "translation", "rotation", "head", "gripper", "handTarget", "baseTarget", "armJointTarget"], "action");
  if (value.armJointTarget !== undefined) {
    if (value.translation !== undefined || value.rotation !== undefined || value.handTarget !== undefined)
      throw new Error("armJointTarget cannot be combined with Cartesian hand commands");
    action.armJointTarget = validateArmJointTarget(value.armJointTarget);
  }
  if (value.base !== undefined) {
    const base = vector(value.base, 3, "base"), linear = limitVector(base.slice(0, 2), CONTROL_LIMITS.baseSpeed);
    action.base = [linear[0], linear[1], clamp(base[2], CONTROL_LIMITS.baseYawRate)];
  }
  if (value.translation !== undefined) action.translation = limitVector(vector(value.translation, 3, "translation"), CONTROL_LIMITS.handSpeed) as Vec3;
  if (value.rotation !== undefined) action.rotation = limitVector(vector(value.rotation, 3, "rotation"), CONTROL_LIMITS.handAngularRate) as Vec3;
  if (value.head !== undefined) action.head = vector(value.head, 2, "head").map(v => clamp(v, CONTROL_LIMITS.headRate)) as [number, number];
  if (value.gripper !== undefined) action.gripper = Math.max(0, Math.min(1, numeric(value.gripper, "gripper")));
  if (value.baseTarget !== undefined) {
    const goal = vector(value.baseTarget, 3, "baseTarget") as Vec3;
    bounded(goal[0], CONTROL_LIMITS.base.x, "baseTarget X"); bounded(goal[1], CONTROL_LIMITS.base.z, "baseTarget Z");
    goal[2] = Math.atan2(Math.sin(goal[2]), Math.cos(goal[2]));
    action.baseTarget = goal;
  }
  if (value.handTarget !== undefined) {
    const target = record(value.handTarget, "handTarget");
    fields(target, ["position", "rotation", "frame"], "handTarget");
    if (target.frame !== "world" && target.frame !== "base") throw new Error("handTarget.frame must be world or base");
    const position = vector(target.position, 3, "handTarget.position") as Vec3;
    const bounds = target.frame === "world" ? CONTROL_LIMITS.world : CONTROL_LIMITS.handLocal;
    bounded(position[0], bounds.x, "handTarget X"); bounded(position[1], bounds.y, "handTarget Y"); bounded(position[2], bounds.z, "handTarget Z");
    action.handTarget = { position, frame: target.frame };
    if (target.rotation !== undefined) {
      const rotation = vector(target.rotation, 4, "handTarget.rotation"), magnitude = Math.hypot(...rotation);
      if (magnitude < 1e-8) throw new Error("handTarget.rotation must be a non-zero quaternion");
      action.handTarget.rotation = rotation.map(v => v / magnitude) as Quaternion;
    }
  }
  return action;
}
