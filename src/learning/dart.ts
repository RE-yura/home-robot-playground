import { ARM_LIMITS } from "../robot/hsr-model.ts";
import { createRandom } from "../simulation/random.ts";
import type { ArmJointTarget } from "../simulation/types.ts";
import type { SixValues } from "./recording.ts";

// DART (Laskey et al. 2017) for the gentle mug teacher: during its first 3 s (moving above the mug and the first half of
// the descent) the executed goal carries a low-frequency AR(1) offset; the recorded label stays the clean teacher goal, so
// the policy learns to correct from slightly wrong poses. Afterwards the offset only decays (under 6 % by the close at
// 4 s): the time-scheduled teacher cannot re-aim, and noise into the close made it miss the mug (CPU, seeds 1000-1049:
// 43/50 with this window, 26/50 when the noise reached into the close).
export const DART_VERSION = "hsr-mug-gentle-dart-v1" as const;
export const DART_VERSIONS: readonly string[] = [DART_VERSION, "hsr-mug-gentle-dart-v2", "hsr-mug-gentle-dart-v3", "hsr-mug-gentle-dart-v4"];
export const DART_SIGMA: SixValues = [0.01, 0.03, 0.03, 0.03, 0.03, 0]; // arm lift m, four arm/wrist joints rad, gripper none
export const DART_RHO = 0.75; // per 100 ms goal
export const DART_WINDOW_SAMPLES = 30;
export const dartActive = (sampleIndex: number) => sampleIndex < DART_WINDOW_SAMPLES;

export class DartNoise {
  private readonly random: () => number;
  private offset: SixValues = [0, 0, 0, 0, 0, 0];
  private spare: number | null = null;
  private readonly sigma: SixValues;
  private readonly rho: number;
  constructor(seed: number, sigma: SixValues = DART_SIGMA, rho = DART_RHO) {
    if (!Number.isInteger(seed) || seed < 0) throw new Error("DART noise needs an integer seed");
    this.sigma = sigma; this.rho = rho;
    // A stream separate from the scene layout's random numbers for the same seed.
    this.random = createRandom((seed ^ 0x5da27e11) >>> 0);
  }
  private gaussian() {
    if (this.spare !== null) { const value = this.spare; this.spare = null; return value; }
    const radius = Math.sqrt(-2 * Math.log(1 - this.random())), angle = 2 * Math.PI * this.random();
    this.spare = radius * Math.sin(angle); return radius * Math.cos(angle);
  }
  next(innovate: boolean): SixValues {
    const scale = Math.sqrt(1 - this.rho * this.rho);
    this.offset = this.offset.map((value, i) => this.rho * value + (innovate && this.sigma[i] ? scale * this.sigma[i] * this.gaussian() : 0)) as SixValues;
    return [...this.offset] as SixValues;
  }
}

export function perturbGoal(goal: { armJointTarget: ArmJointTarget; gripper: number }, offset: SixValues) {
  return { armJointTarget: goal.armJointTarget.map((q, i) => Math.max(ARM_LIMITS[i][0], Math.min(ARM_LIMITS[i][1], q + offset[i]))) as ArmJointTarget,
    gripper: goal.gripper };
}
