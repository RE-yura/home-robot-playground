import type { Simulation } from "../simulation/mujoco.ts";
import { createRandom } from "../simulation/random.ts";
import { BASE_HOME } from "../robot/hsr-model.ts";

// Stationary tabletop task; the ordinary room Reset keeps its original layout.
export const LEARNING_SCENE_VERSION = "hsr-stationary-v5" as const;
export function prepareLearningScene(sim: Simulation, seed = 42) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error("Learning seed must be a uint32");
  sim.reset(seed);
  const random = createRandom(seed);
  for (const [name, position] of [
    ["mug", [-0.078, 0.463, 0.38]],
    ["ball", [-0.06, 0.4335, 0.49]],
    // Keep the opened right fingertip clear of the bin during the mug descent.
    // v4's 4mm object jitter could jam it against the wall before bilateral grasp.
    ["room_basket", [0.30, 0.425, 0.42]],
  ] as const) {
    const body = sim.id("BODY", name), q = sim.model.jnt_qposadr[sim.model.body_jntadr[body]];
    const dx = seed === 42 ? 0 : (random() * 2 - 1) * 0.004;
    const dz = seed === 42 ? 0 : (random() * 2 - 1) * 0.003;
    sim.data.qpos.set([position[0] + dx, position[1], position[2] + dz], q);
  }
  // Match the bin's initial heading to the reachable release region. Its full
  // physical fence/floor geometry, mass and freely simulated motion are unchanged.
  const basket = sim.id("BODY", "room_basket");
  const basketQ = sim.model.jnt_qposadr[sim.model.body_jntadr[basket]];
  sim.data.qpos.set([Math.cos(-0.1), 0, Math.sin(-0.1), 0], basketQ + 3);
  sim.baseTarget = [0, 1.08, -0.1];
  sim.data.qpos[sim.joints.base_yaw.q] = -0.1;
  sim.data.qpos[sim.joints.base_x.q] = -BASE_HOME[0];
  sim.data.qpos[sim.joints.base_z.q] = 1.08 - BASE_HOME[1];
  // A fixed oblique view sees both candidates beside the initial arm. The
  // physical camera attachment/intrinsics and instruction-independent scene stay identical.
  sim.headTarget = [-45 * Math.PI / 180, -0.6];
  sim.data.qpos[sim.joints.tilt.q] = sim.headTarget[0];
  sim.data.qpos[sim.joints.pan.q] = sim.headTarget[1];
  sim.mj.mj_forward(sim.model, sim.data);
  sim.setController(null);
  return { id: LEARNING_SCENE_VERSION, seed };
}
