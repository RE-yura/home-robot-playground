import type { SixValues } from "./recording.ts";

// Rule assist (opt-in, reported separately from learned-only results). Reads measured robot channels only.
// A failed grasp cycle: the measured opening falls below 0.45 and later rises above 0.75 while the arm-lift joint never
// rose more than 0.10 m above its value at the close. Opening changes inside one close count once.
export class GraspCycleWatch {
  failedCycles = 0;
  private closedLift: number | null = null;
  private lifted = false;
  update(lift: number, opening: number) {
    if (this.closedLift === null) {
      if (opening < 0.45) { this.closedLift = lift; this.lifted = false; }
      return;
    }
    if (lift > this.closedLift + 0.10) this.lifted = true;
    if (opening > 0.75) { if (!this.lifted) this.failedCycles++; this.closedLift = null; }
  }
  reset() { this.failedCycles = 0; this.closedLift = null; this.lifted = false; }
}

// Twenty 100 ms goals easing (smoothstep) from the measured joints back to the episode's start joints, hand open.
export function returnGoals(from: number[], to: number[], steps = 20): SixValues[] {
  return Array.from({ length: steps }, (_, k) => {
    const u = (k + 1) / steps, s = u * u * (3 - 2 * u);
    return [...from.map((q, i) => k === steps - 1 ? to[i] : q + (to[i] - q) * s), 1] as SixValues;
  });
}
