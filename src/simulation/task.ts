import { TABLE_HEIGHT } from "./table.ts";
import { mugBounds, mugFitsBasket, mugInsideBasketWalls } from "./mug-geometry.ts";
import type { Observation, Quaternion, Vec3 } from "./types.ts";

export type TaskStatus = "ready" | "running" | "success" | "failed";
export type TaskFailureReason = "target-missing" | "basket-missing" | "target-fell" | "robot-tipped" | "base-drift" | "timeout";
// rimSettledFor: with the "walls" mug criterion, the settled time under the original rim-opening criterion.
export type TaskResult = { status: TaskStatus; elapsed: number; collisions: number; drops: number; settledFor: number; reason?: TaskFailureReason; rimSettledFor?: number };
// "rim" (default, original): the mug fits the rim opening. "walls" (user decision 2026-10-28 for the mug goal): the mug
// lies within the inner faces of the side bars, below the rim; a mug fallen against the bars inside the basket counts.
export type MugSuccessCriterion = "rim" | "walls";

function rotate(point: Vec3, quaternion: Quaternion): Vec3 {
  const length = Math.hypot(...quaternion), [x, y, z, w] = quaternion.map(v => v / length);
  const [a, b, c] = point, tx = 2 * (y * c - z * b), ty = 2 * (z * a - x * c), tz = 2 * (x * b - y * a);
  return [a + w * tx + y * tz - z * ty, b + w * ty + z * tx - x * tz, c + w * tz + x * ty - y * tx];
}
function ready(): TaskResult { return { status: "ready", elapsed: 0, collisions: 0, drops: 0, settledFor: 0 }; }

export class TaskEvaluator {
  readonly name: string;
  // "none": a planned step the robot never learned; it cannot succeed and ends by a physical failure or the timeout.
  private targetId: "mug" | "ball" | "none";
  private metrics = ready();
  private collisionActive = false;
  private lifted = false;
  private dropActive = false;
  private contactLostFor = 0;
  readonly criterion: MugSuccessCriterion;

  constructor(targetId: "mug" | "ball" | "none" = "mug", options: { success?: MugSuccessCriterion } = {}) {
    this.targetId = targetId; this.criterion = options.success ?? "rim";
    if (this.criterion !== "rim" && (this.criterion !== "walls" || targetId !== "mug")) throw new Error("The walls success criterion is for the mug task");
    this.name = targetId === "mug" ? "コップをかごへ入れる" : targetId === "ball" ? "テニスボールをかごへ入れる" : "未学習の作業";
  }
  get target(): "mug" | "ball" | null { return this.targetId === "none" ? null : this.targetId; }

  get result(): TaskResult { return { ...this.metrics }; }
  reset(): TaskResult {
    this.metrics = ready(); if (this.criterion === "walls") this.metrics.rimSettledFor = 0; this.collisionActive = false; this.lifted = false; this.dropActive = false; this.contactLostFor = 0;
    return this.result;
  }
  start(observation: Observation): TaskResult {
    this.reset(); this.metrics.status = "running";
    return this.update(observation, 0);
  }
  fail(reason: TaskFailureReason): TaskResult {
    if (this.metrics.status !== "running") return this.result;
    this.metrics.status = "failed"; this.metrics.reason = reason; this.metrics.settledFor = 0;
    if (this.criterion === "walls") this.metrics.rimSettledFor = 0;
    return this.result;
  }
  update(observation: Observation, dt: number, physicalFailure?: TaskFailureReason): TaskResult {
    if (!Number.isFinite(dt) || dt < 0) throw new Error("Task time must be a finite nonnegative number");
    if (this.metrics.status !== "running") return this.result;
    this.metrics.elapsed += dt;
    if (this.targetId === "none") {
      const collisionForce = observation.robot?.collision?.force ?? observation.robot?.tableForce ?? 0;
      if (!this.collisionActive && collisionForce > 25) this.metrics.collisions++;
      if (collisionForce > 25) this.collisionActive = true; else if (collisionForce < 15) this.collisionActive = false;
      if (physicalFailure) return this.fail(physicalFailure);
      if (observation.robot?.base.tipped || (observation.robot?.base.tilt ?? 0) > Math.PI / 3) return this.fail("robot-tipped");
      if (this.metrics.elapsed >= 120) return this.fail("timeout");
      return this.result;
    }
    const target = observation.objects.find(object => object.id === this.targetId), basket = observation.furniture.find(item => item.id === "basket");
    if (!target) return this.fail("target-missing");
    if (!basket) return this.fail("basket-missing");
    const robot = observation.robot, held = Boolean(robot?.gripperContact && (robot.contactId === this.targetId || robot.contactId === undefined));
    const collisionForce = robot?.collision?.force ?? robot?.tableForce ?? 0;
    if (!this.collisionActive && collisionForce > 25) this.metrics.collisions++;
    // Hysteresis prevents solver noise around the threshold from creating repeated episodes.
    if (collisionForce > 25) this.collisionActive = true;
    else if (collisionForce < 15) this.collisionActive = false;
    if (physicalFailure) return this.fail(physicalFailure);
    if (robot?.base.tipped || (robot?.base.tilt ?? 0) > Math.PI / 3 ||
      (robot?.base.rotation && rotate([0, 1, 0], robot.base.rotation)[1] < 0.5)) return this.fail("robot-tipped");
    const ballFits = (horizontalOnly = false) => {
      const [x, y, z, w] = basket.rotation;
      const local = rotate(target.position.map((v, i) => v - basket.position[i]) as Vec3, [-x, -y, -z, w]);
      // Bars have gaps: use their physical outer envelope, not the ideal inner planes.
      return local.every((v, i) => Number.isFinite(v) && ((horizontalOnly && i === 1) ||
        (v - 0.0335 >= [-0.21, -0.017, -0.19][i] && v + 0.0335 <= [0.21, 0.175, 0.19][i])));
    };
    const fits = this.targetId === "mug" ? mugFitsBasket(target, basket) : ballFits();
    const aboveBasket = this.targetId === "mug" ? mugFitsBasket(target, basket, true) : ballFits(true);
    const bottom = this.targetId === "mug" ? mugBounds(target).min[1] : target.position[1] - 0.0335;
    // A release above the basket is intentional; initial tabletop settling has never been held.
    if (held) { this.lifted ||= bottom > TABLE_HEIGHT + 0.02; this.dropActive = false; this.contactLostFor = 0; }
    else if (this.lifted) this.contactLostFor += dt;
    if (this.lifted && !held && this.contactLostFor >= 0.1 - 1e-8 && !aboveBasket && target.velocity[1] < -0.05 && !this.dropActive) {
      this.metrics.drops++; this.dropActive = true;
    }
    if (bottom <= 0.015 && !held) {
      if (!this.dropActive) this.metrics.drops++;
      return this.fail("target-fell");
    }
    const slow = !held && Math.hypot(...target.velocity) < 0.03, walls = this.criterion === "walls";
    const settled = slow && (walls ? mugInsideBasketWalls(target, basket) : fits);
    this.metrics.settledFor = settled ? this.metrics.settledFor + dt : 0;
    if (walls) this.metrics.rimSettledFor = slow && fits ? (this.metrics.rimSettledFor ?? 0) + dt : 0;
    if (this.metrics.settledFor >= 2 - 1e-8) this.metrics.status = "success";
    else if (this.metrics.elapsed >= 120) return this.fail("timeout");
    return this.result;
  }
}
