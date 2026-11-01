import { Quaternion as ThreeQuaternion, Vector3 } from "three";
import type { Quaternion, Vec3 } from "./types.ts";

export type MugPose = { position: Vec3; rotation: Quaternion };
export type MugBounds = { min: Vec3; max: Vec3 };

function box(center: Vec3, half: Vec3, yaw = 0) {
  const rotation = new ThreeQuaternion().setFromAxisAngle(new Vector3(0, 1, 0), yaw);
  return { center: new Vector3(...center), halfAxes: [new Vector3(half[0], 0, 0), new Vector3(0, half[1], 0), new Vector3(0, 0, half[2])].map(axis => axis.applyQuaternion(rotation)) };
}
// Match the compound collision shapes in household.ts, including the open ring and handle gaps.
const tangentHalf = 0.05 * Math.tan(Math.PI / 24) * 1.02;
const BOXES = [
  ...Array.from({ length: 24 }, (_, i) => {
    const angle = i * Math.PI * 2 / 24;
    return box([0.05 * Math.cos(angle), 0, 0.05 * Math.sin(angle)], [0.005, 0.06, tangentHalf], -angle);
  }),
  box([0.078, -0.038, 0], [0.024, 0.006, 0.009]), box([0.078, 0.038, 0], [0.024, 0.006, 0.009]),
  box([0.1, 0, 0], [0.006, 0.038, 0.009]),
];

// Exact coordinate support of every collision primitive, expressed in world or the supplied frame.
export function mugBounds(mug: MugPose, frame?: MugPose): MugBounds {
  const inverse = new ThreeQuaternion(...(frame?.rotation ?? [0, 0, 0, 1])).normalize().invert();
  const rotation = inverse.clone().multiply(new ThreeQuaternion(...mug.rotation).normalize());
  const position = new Vector3(...mug.position).sub(new Vector3(...(frame?.position ?? [0, 0, 0]))).applyQuaternion(inverse);
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  const include = (center: Vector3, extent: Vec3) => {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], center.getComponent(axis) - extent[axis]);
      max[axis] = Math.max(max[axis], center.getComponent(axis) + extent[axis]);
    }
  };
  for (const shape of BOXES) {
    const center = shape.center.clone().applyQuaternion(rotation).add(position);
    const axes = shape.halfAxes.map(axis => axis.clone().applyQuaternion(rotation));
    include(center, [0, 1, 2].map(i => axes.reduce((sum, axis) => sum + Math.abs(axis.getComponent(i)), 0)) as Vec3);
  }
  const up = new Vector3(0, 1, 0).applyQuaternion(rotation);
  const cylinder = new Vector3(0, -0.056, 0).applyQuaternion(rotation).add(position);
  include(cylinder, [0, 1, 2].map(i => {
    const vertical = up.getComponent(i);
    return 0.007 * Math.abs(vertical) + 0.05 * Math.sqrt(Math.max(0, 1 - vertical * vertical));
  }) as Vec3);
  return { min, max };
}

// Centre lines of the basket's side bars (x ±0.195, z ±0.175; inner faces 4 mm inside). A mug resting against a bar
// sinks a fraction of a millimetre into its contact layer, so the inner face itself would reject it; passing the centre
// line needs going through or over a bar. The original success box (mugFitsBasket) is the rim opening, about 1 cm
// inside the bars, so a mug lying against the bars below the rim fails there.
export const BASKET_WALLS = { x: 0.195, z: 0.175 };
export function mugInsideBasketWalls(mug: MugPose, basket: MugPose): boolean {
  const { min, max } = mugBounds(mug, basket);
  return min.every(Number.isFinite) && max.every(Number.isFinite) && min[0] >= -BASKET_WALLS.x && max[0] <= BASKET_WALLS.x &&
    min[2] >= -BASKET_WALLS.z && max[2] <= BASKET_WALLS.z && min[1] >= -0.017 && max[1] <= 0.175;
}

export function mugFitsBasket(mug: MugPose, basket: MugPose, horizontalOnly = false, margin = 0): boolean {
  const { min, max } = mugBounds(mug, basket);
  const lower = [-0.18 + margin, -0.017, -0.16 + margin], upper = [0.18 - margin, 0.175, 0.16 - margin];
  return min.every((value, i) => Number.isFinite(value) && Number.isFinite(max[i]) &&
    ((horizontalOnly && i === 1) || (value >= lower[i] && max[i] <= upper[i])));
}
