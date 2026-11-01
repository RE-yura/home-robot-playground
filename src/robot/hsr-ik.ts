import { Euler, Quaternion, Vector3 } from "three";
import { ARM_LIMITS } from "./hsr-model.ts";
import type { Vec3 } from "../simulation/types.ts";

const ROS_TO_WORLD = new Quaternion(-0.5, 0.5, 0.5, 0.5);
// At zero requested attitude, the fingers point forward and the jaws close horizontally.
export const HAND_FORWARD = new Quaternion(Math.SQRT1_2, Math.SQRT1_2, 0, 0);
const clamp = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n));

export function forwardHSR(q: readonly number[]) {
  const position = new Vector3(0, 0, 0.34 + q[0]), rotation = new Quaternion();
  const offset = (x: number, y: number, z: number) => position.add(new Vector3(x, y, z).applyQuaternion(rotation));
  const turn = (axis: Vector3, angle: number) => rotation.multiply(new Quaternion().setFromAxisAngle(axis, angle));
  offset(0.141, 0.078, 0);
  turn(new Vector3(0, -1, 0), q[1]); offset(0.005, 0, 0.345);
  turn(new Vector3(0, 0, 1), q[2]); turn(new Vector3(0, -1, 0), q[3]);
  turn(new Vector3(0, 0, 1), q[4]); offset(0.012, 0, 0.1405);
  turn(new Vector3(0, 0, 1), Math.PI); offset(0, 0, 0.08);
  return { position: position.applyQuaternion(ROS_TO_WORLD), rotation: rotation.premultiply(ROS_TO_WORLD) };
}

export function targetRotation([yaw, pitch, roll]: Vec3) {
  return new Quaternion().setFromEuler(new Euler(pitch, yaw, roll, "YXZ")).multiply(HAND_FORWARD);
}

// Five joints cannot realize every six-dimensional pose. Priority follows position or attitude input.
export function solveHSR(target: Vec3, attitude: Vec3, current: readonly number[], preferAttitude = false) {
  const desired = targetRotation(attitude), q = [...current];
  const weight = preferAttitude ? 1 : 0.06;
  const residual = (joints: number[]) => {
    const pose = forwardHSR(joints), difference = pose.rotation.multiply(desired.clone().invert());
    if (difference.w < 0) difference.set(-difference.x, -difference.y, -difference.z, -difference.w);
    const magnitude = Math.hypot(difference.x, difference.y, difference.z);
    const scale = magnitude > 1e-8 ? 2 * Math.atan2(magnitude, difference.w) / magnitude : 2;
    return [...pose.position.toArray().map((v, i) => v - target[i]),
      difference.x * scale * weight, difference.y * scale * weight, difference.z * scale * weight];
  };
  for (let iteration = 0; iteration < 12; iteration++) {
    const error = residual(q);
    if (Math.hypot(...error) < 0.0001) break;
    const derivatives = q.map((_, i) => {
      const shifted = [...q]; shifted[i] += 1e-5;
      return residual(shifted).map((v, row) => (v - error[row]) / 1e-5);
    });
    const matrix = derivatives.map((a, i) => [...derivatives.map((b, j) =>
      a.reduce((sum, v, row) => sum + v * b[row], 0) + (i === j ? 0.0004 : 0)),
      -a.reduce((sum, v, row) => sum + v * error[row], 0)]);
    for (let col = 0; col < 5; col++) {
      let pivot = col;
      for (let row = col + 1; row < 5; row++) if (Math.abs(matrix[row][col]) > Math.abs(matrix[pivot][col])) pivot = row;
      [matrix[col], matrix[pivot]] = [matrix[pivot], matrix[col]];
      const divisor = matrix[col][col];
      for (let j = col; j <= 5; j++) matrix[col][j] /= divisor;
      for (let row = 0; row < 5; row++) if (row !== col) {
        const factor = matrix[row][col];
        for (let j = col; j <= 5; j++) matrix[row][j] -= factor * matrix[col][j];
      }
    }
    q.forEach((value, i) => { q[i] = clamp(value + clamp(matrix[i][5], -0.2, 0.2), ARM_LIMITS[i][0], ARM_LIMITS[i][1]); });
  }
  return q;
}
