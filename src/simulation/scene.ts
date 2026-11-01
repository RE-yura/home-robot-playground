import { createRandom } from "./random.ts";
import { TABLE_HEIGHT } from "./table.ts";
import type { ObjectType, Vec3 } from "./types.ts";

export const TIME_STEP = 0.002;
export const OBJECTS: { id: string; type: ObjectType; position: Vec3 }[] = [
  { id: "cube", type: "cube", position: [-0.55, TABLE_HEIGHT + 0.34, -0.26] },
  { id: "ball", type: "ball", position: [-0.23, TABLE_HEIGHT + 0.34, 0.2] },
  { id: "can", type: "can", position: [-0.53, TABLE_HEIGHT + 0.36, 0.1] },
  { id: "mug", type: "mug", position: [0, TABLE_HEIGHT + 0.31, 0.44] },
  { id: "bottle", type: "bottle", position: [-0.22, TABLE_HEIGHT + 0.38, -0.25] },
  { id: "plate", type: "plate", position: [0.09, TABLE_HEIGHT + 0.30, -0.24] },
  { id: "baseball", type: "baseball", position: [0.65, 0.3, 1.6] },
  { id: "tea", type: "tea", position: [2.34, 0.52, -2.39] },
  { id: "juice", type: "juice", position: [2.55, 0.88, -2.39] },
  { id: "milk", type: "milk", position: [2.78, 0.89, -2.39] },
];

// Seed 42 preserves the validated grasp scene. Other seeds vary safe slots.
export function createObjects(seed: number) {
  const random = createRandom(seed);
  return OBJECTS.map(({ id, type, position }) => ({
    id, type,
    position: position.map((v, axis) => v + (seed === 42 || ["tea", "juice", "milk"].includes(type) ? 0 :
      (random() - 0.5) * [0.09, 0.15, 0.02][axis])) as Vec3,
  }));
}
