import { Quaternion, Vector3 } from "three";
import { forwardHSR } from "./hsr-ik.ts";
import { ARM_LIMITS } from "./hsr-model.ts";
import type { Quaternion as Rotation, Vec3 } from "../simulation/types.ts";

const UP = new Vector3(0, 1, 0);
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

// Solve one world-space hand pose with planar chassis + five HSR arm joints.
// Explicit chassis input locks those base axes; damping prefers arm motion when it can satisfy both goals.
export function solveWholeBody(position: Vec3, rotation: Rotation, base: Vec3, arm: readonly number[], lockTranslation: boolean, lockYaw: boolean, height = 0, support: Rotation = [0, 0, 0, 1]) {
  const q = [...base, ...arm], desired = new Quaternion(...rotation).invert();
  const axes = q.map((_, i) => i).filter(i => i >= 3 || (i < 2 ? !lockTranslation : !lockYaw));
  const residual = (values: number[]) => {
    const pose = forwardHSR(values.slice(3)), yaw = new Quaternion().setFromAxisAngle(UP, values[2]).multiply(new Quaternion(...support));
    pose.position.applyQuaternion(yaw).add(new Vector3(values[0], height, values[1]));
    const difference = pose.rotation.premultiply(yaw).multiply(desired);
    if (difference.w < 0) difference.set(-difference.x, -difference.y, -difference.z, -difference.w);
    const magnitude = Math.hypot(difference.x, difference.y, difference.z);
    const scale = (magnitude > 1e-8 ? 2 * Math.atan2(magnitude, difference.w) / magnitude : 2) * 0.4;
    return [...pose.position.toArray().map((v, i) => v - position[i]), difference.x * scale, difference.y * scale, difference.z * scale];
  };
  for (let iteration = 0; iteration < 10; iteration++) {
    const error = residual(q);
    if (Math.hypot(...error) < 0.0001) break;
    const derivatives = axes.map(i => {
      const shifted = [...q]; shifted[i] += 1e-5;
      return residual(shifted).map((v, row) => (v - error[row]) / 1e-5);
    });
    const matrix = derivatives.map((a, i) => [...derivatives.map((b, j) =>
      a.reduce((sum, v, row) => sum + v * b[row], 0) + (i === j ? axes[i] < 2 ? 0.02 : axes[i] === 2 ? 0.006 : 0.0004 : 0)),
      -a.reduce((sum, v, row) => sum + v * error[row], 0)]);
    for (let col = 0; col < axes.length; col++) {
      let pivot = col;
      for (let row = col + 1; row < axes.length; row++) if (Math.abs(matrix[row][col]) > Math.abs(matrix[pivot][col])) pivot = row;
      [matrix[col], matrix[pivot]] = [matrix[pivot], matrix[col]];
      const divisor = matrix[col][col];
      for (let j = col; j <= axes.length; j++) matrix[col][j] /= divisor;
      for (let row = 0; row < axes.length; row++) if (row !== col) {
        const factor = matrix[row][col];
        for (let j = col; j <= axes.length; j++) matrix[row][j] -= factor * matrix[col][j];
      }
    }
    axes.forEach((axis, i) => {
      const limit = axis < 2 ? 0.03 : axis === 2 ? 0.08 : 0.2;
      q[axis] += clamp(matrix[i][axes.length], -limit, limit);
      if (axis >= 3) q[axis] = clamp(q[axis], ARM_LIMITS[axis - 3][0], ARM_LIMITS[axis - 3][1]);
    });
  }
  return { base: q.slice(0, 3) as Vec3, arm: q.slice(3) };
}
