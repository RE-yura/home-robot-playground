import type { Simulation } from "../simulation/mujoco.ts";
import { TIME_STEP } from "../simulation/scene.ts";

type Clock = { accumulator: number; generation: number };
export function advancePhysicsFrame(sim: Simulation, delta: number, paused: boolean | (() => boolean), clock: Clock): number {
  if (clock.generation !== sim.resetGeneration) { clock.accumulator = 0; clock.generation = sim.resetGeneration; }
  let steps = 0;
  const isPaused = () => typeof paused === "function" ? paused() : paused;
  if (isPaused()) clock.accumulator = 0;
  else {
    clock.accumulator += Math.min(delta, 0.05) * (sim.learningCapture?.playbackRate ?? 1);
    while (clock.accumulator >= TIME_STEP) {
      if (isPaused() || !sim.step()) { clock.accumulator = 0; break; }
      clock.accumulator -= TIME_STEP; steps++;
    }
  }
  // A frame may finish on a boundary without attempting one more step.
  // Gate/refresh here, before the subsequent robot/object mesh callbacks.
  sim.learningCapture?.captureDue(sim.data.time);
  return steps;
}
