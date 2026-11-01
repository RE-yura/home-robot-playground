export type Vec3 = [number, number, number];
export type Quaternion = [number, number, number, number]; // x, y, z, w
export type ArmJointTarget = [number, number, number, number, number]; // ARM_NAMES: lift metres, then radians
export type ObjectType = "cube" | "ball" | "baseball" | "can" | "mug" | "bottle" | "plate" | "tea" | "juice" | "milk";

// Velocities are metres/second and radians/second. Omitted motion fields stop that motion.
// base = local [strafe right, forward, yaw]; hand velocities are world-space.
export type RobotAction = {
  base?: Vec3;
  translation?: Vec3;
  rotation?: Vec3;
  head?: [number, number]; // tilt / pan rates
  gripper?: number; // 0 = closed, 1 = open
  handTarget?: { position: Vec3; rotation?: Quaternion; frame: "world" | "base" };
  armJointTarget?: ArmJointTarget; // Mutually exclusive with Cartesian hand commands.
  baseTarget?: Vec3; // world X / Z / yaw, approached at the same speed limit
};
export type CameraImage = { view: "head" | "hand"; timestamp: number; width: number; height: number; dataUrl: string };
// Renderer provenance stays outside the policy's five-field image payload.
export type CameraAcquisition = Readonly<{ captureId: string; generation: number; acquisitionSequence: number; timestamp: number }>;
export type ObjectObservation = {
  id: string;
  type: ObjectType;
  position: Vec3;
  rotation: Quaternion;
  velocity: Vec3;
};
export type RobotObservation = {
  base: { position: Vec3; yaw: number; rotation?: Quaternion; tilt?: number; tipped?: boolean };
  joints: number[];
  gripper: number;
  endEffectorPosition: Vec3;
  endEffectorRotation: Quaternion;
  gripperContact: boolean; // Both fingers touch the same object; no welded attachment.
  tableForce: number;
  contactId?: string | null;
  collision?: { force: number; count: number; selfCount: number };
};
export type Observation = {
  timestamp: number;
  objects: ObjectObservation[];
  furniture: { id: string; position: Vec3; rotation: Quaternion }[];
  robot?: RobotObservation;
};
export interface Controller { getAction(observation: Observation): RobotAction; abort?(): void }
