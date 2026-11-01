import type { CameraImage, Observation, RobotAction } from "../simulation/types.ts";
import { validateArmJointTarget } from "../controllers/protocol.ts";

export type PhysicsState = { time: number; qpos: number[]; qvel: number[]; act: number[] };
export type RecordingMetadata = {
  modelId: string; seed: number; robotMassKg: number; armTorqueNm: number;
  fixedTimestep: number; nq: number; nv: number;
};
export type RecordedFrame = {
  elapsed: number; state: PhysicsState; observation: Observation; action: RobotAction; images?: CameraImage[];
};
export type RecordedSession = {
  version: 1; metadata: RecordingMetadata; frames: RecordedFrame[]; duration: number;
  stoppedReason: "finished" | "duration-limit" | "size-limit";
};

const MAX_BYTES = 64 * 1024 * 1024, MAX_DURATION = 180, MAX_FRAMES = 3601;
const SAMPLE_INTERVAL = 0.05, CAMERA_INTERVAL = 1, EPSILON = 1e-8;
const encoder = new TextEncoder();

function check(condition: unknown, label: string): asserts condition {
  if (!condition) throw new Error(`Invalid recording: ${label}`);
}
function fields(value: unknown, keys: string[], label: string): asserts value is Record<string, unknown> {
  check(value !== null && typeof value === "object" && !Array.isArray(value), label);
  check(Object.keys(value).every(key => keys.includes(key)), `${label} schema`);
}
function number(value: unknown, label: string): asserts value is number {
  check(typeof value === "number" && Number.isFinite(value), `${label} must be a finite number`);
}
function positive(value: unknown, max: number, label: string) {
  number(value, label); check(value > 0 && value <= max, `${label} limit`);
}
function integer(value: unknown, min: number, max: number, label: string) {
  number(value, label); check(Number.isInteger(value) && value >= min && value <= max, `${label} limit`);
}
function string(value: unknown, label: string) {
  check(typeof value === "string" && value.length > 0 && value.length <= 128, label);
}
function vector(value: unknown, length: number, label: string): asserts value is number[] {
  check(Array.isArray(value) && value.length === length, `${label} dimension`);
  value.forEach(entry => number(entry, label));
}
function quaternion(value: unknown, label: string) {
  vector(value, 4, label); check(Math.abs(Math.hypot(...value) - 1) < 0.02, `${label} quaternion`);
}
function boundedArray(value: unknown, max: number, label: string): asserts value is unknown[] {
  check(Array.isArray(value) && value.length <= max, `${label} length limit`);
}
function validateMetadata(value: unknown): asserts value is RecordingMetadata {
  fields(value, ["modelId", "seed", "robotMassKg", "armTorqueNm", "fixedTimestep", "nq", "nv"], "metadata");
  string(value.modelId, "model ID"); integer(value.seed, 0, 0xffffffff, "seed");
  positive(value.robotMassKg, 1000, "robot mass"); positive(value.armTorqueNm, 1000, "arm torque");
  positive(value.fixedTimestep, 0.1, "fixed timestep");
  integer(value.nq, 1, 4096, "qpos dimension"); integer(value.nv, 1, 4096, "qvel dimension");
}
function validateState(value: unknown, metadata: RecordingMetadata): asserts value is PhysicsState {
  fields(value, ["time", "qpos", "qvel", "act"], "state");
  number(value.time, "state time"); check(value.time >= 0, "state time");
  vector(value.qpos, metadata.nq, "qpos"); vector(value.qvel, metadata.nv, "qvel");
  boundedArray(value.act, 4096, "act"); value.act.forEach(entry => number(entry, "act"));
}
function validateAction(value: unknown): asserts value is RobotAction {
  fields(value, ["base", "translation", "rotation", "head", "gripper", "handTarget", "baseTarget", "armJointTarget"], "action");
  if (value.armJointTarget !== undefined) {
    check(value.translation === undefined && value.rotation === undefined && value.handTarget === undefined, "mixed joint/Cartesian action");
    validateArmJointTarget(value.armJointTarget);
  }
  for (const key of ["base", "translation", "rotation", "baseTarget"]) {
    if (value[key] !== undefined) vector(value[key], 3, `action ${key}`);
  }
  if (value.head !== undefined) vector(value.head, 2, "action head");
  if (value.gripper !== undefined) {
    number(value.gripper, "gripper"); check(value.gripper >= 0 && value.gripper <= 1, "gripper range");
  }
  if (value.handTarget !== undefined) {
    fields(value.handTarget, ["position", "rotation", "frame"], "hand target");
    vector(value.handTarget.position, 3, "hand target position");
    if (value.handTarget.rotation !== undefined) quaternion(value.handTarget.rotation, "hand target rotation");
    check(value.handTarget.frame === "world" || value.handTarget.frame === "base", "hand target frame");
  }
}
function validateObservation(value: unknown): asserts value is Observation {
  fields(value, ["timestamp", "objects", "furniture", "robot"], "observation");
  number(value.timestamp, "observation time"); check(value.timestamp >= 0, "observation time");
  boundedArray(value.objects, 128, "objects"); boundedArray(value.furniture, 128, "furniture");
  for (const object of value.objects) {
    fields(object, ["id", "type", "position", "rotation", "velocity"], "object");
    string(object.id, "object ID");
    check(["cube", "ball", "baseball", "can", "mug", "bottle", "plate", "tea", "juice", "milk"].includes(object.type as string), "object type");
    vector(object.position, 3, "object position"); quaternion(object.rotation, "object rotation"); vector(object.velocity, 3, "object velocity");
  }
  for (const item of value.furniture) {
    fields(item, ["id", "position", "rotation"], "furniture"); string(item.id, "furniture ID");
    vector(item.position, 3, "furniture position"); quaternion(item.rotation, "furniture rotation");
  }
  if (value.robot === undefined) return;
  const robot = value.robot;
  fields(robot, ["base", "joints", "gripper", "endEffectorPosition", "endEffectorRotation", "gripperContact", "tableForce", "contactId", "collision"], "robot");
  fields(robot.base, ["position", "yaw", "rotation", "tilt", "tipped"], "robot base");
  vector(robot.base.position, 3, "base position"); number(robot.base.yaw, "base yaw");
  if (robot.base.rotation !== undefined) quaternion(robot.base.rotation, "base rotation");
  if (robot.base.tilt !== undefined) number(robot.base.tilt, "base tilt");
  if (robot.base.tipped !== undefined) check(typeof robot.base.tipped === "boolean", "base tipped");
  boundedArray(robot.joints, 512, "robot joints"); robot.joints.forEach(joint => number(joint, "robot joint"));
  number(robot.gripper, "robot gripper"); check(robot.gripper >= 0 && robot.gripper <= 1, "robot gripper range");
  vector(robot.endEffectorPosition, 3, "end effector position"); quaternion(robot.endEffectorRotation, "end effector rotation");
  check(typeof robot.gripperContact === "boolean", "gripper contact"); number(robot.tableForce, "table force");
  if (robot.contactId !== undefined && robot.contactId !== null) string(robot.contactId, "contact ID");
  if (robot.collision !== undefined) {
    fields(robot.collision, ["force", "count", "selfCount"], "collision");
    number(robot.collision.force, "collision force"); check(robot.collision.force >= 0, "collision force");
    integer(robot.collision.count, 0, 100000, "collision count"); integer(robot.collision.selfCount, 0, 100000, "self collision count");
  }
}
function validateImages(value: unknown): asserts value is CameraImage[] {
  boundedArray(value, 2, "camera images"); check(value.length === 2, "camera pair");
  const views = new Set();
  for (const image of value) {
    fields(image, ["view", "timestamp", "width", "height", "dataUrl"], "camera image");
    check(image.view === "head" || image.view === "hand", "camera view"); check(!views.has(image.view), "camera pair views"); views.add(image.view);
    number(image.timestamp, "camera time"); check(image.timestamp >= 0, "camera time");
    integer(image.width, 1, 1024, "camera width"); integer(image.height, 1, 1024, "camera height");
    check(typeof image.dataUrl === "string" && image.dataUrl.length <= 4 * 1024 * 1024 &&
      /^data:image\/(jpeg|png);base64,[A-Za-z0-9+/]+={0,2}$/.test(image.dataUrl), "camera image format or size");
  }
}
function validateFrame(value: unknown, metadata: RecordingMetadata): asserts value is RecordedFrame {
  fields(value, ["elapsed", "state", "observation", "action", "images"], "frame");
  number(value.elapsed, "frame elapsed"); check(value.elapsed >= 0 && value.elapsed <= MAX_DURATION, "frame duration limit");
  validateState(value.state, metadata); validateObservation(value.observation); validateAction(value.action);
  if (value.images !== undefined) validateImages(value.images);
}
function validateSession(value: unknown, expectedNq?: number, expectedNv?: number): asserts value is RecordedSession {
  fields(value, ["version", "metadata", "frames", "duration", "stoppedReason"], "session");
  check(value.version === 1, "unsupported version"); validateMetadata(value.metadata);
  check(expectedNq === undefined || expectedNq === value.metadata.nq, "model qpos dimension mismatch");
  check(expectedNv === undefined || expectedNv === value.metadata.nv, "model qvel dimension mismatch");
  number(value.duration, "duration"); check(value.duration >= 0 && value.duration <= MAX_DURATION, "duration limit");
  check(["finished", "duration-limit", "size-limit"].includes(value.stoppedReason as string), "stop reason");
  boundedArray(value.frames, MAX_FRAMES, "frames");
  let firstTime = 0, lastElapsed = -Infinity, lastImageTime = -Infinity, actLength = -1;
  for (const frame of value.frames) {
    validateFrame(frame, value.metadata);
    if (lastElapsed === -Infinity) { firstTime = frame.state.time; actLength = frame.state.act.length; check(frame.elapsed === 0, "first frame elapsed"); }
    else check(frame.elapsed - lastElapsed >= SAMPLE_INTERVAL - EPSILON, "frame cadence");
    check(Math.abs(frame.state.time - firstTime - frame.elapsed) < EPSILON, "state time alignment");
    check(frame.state.act.length === actLength, "act dimension mismatch");
    if (frame.images) {
      check(frame.elapsed - lastImageTime >= CAMERA_INTERVAL - EPSILON, "camera cadence"); lastImageTime = frame.elapsed;
    }
    lastElapsed = frame.elapsed;
  }
  check(Math.abs(value.duration - (value.frames.length ? lastElapsed : 0)) < EPSILON, "duration does not match frames");
}

export function importRecordedSession(text: string, expectedNq?: number, expectedNv?: number): RecordedSession {
  check(typeof text === "string" && text.length <= MAX_BYTES && encoder.encode(text).byteLength <= MAX_BYTES, "file size limit");
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error("Invalid recording: JSON syntax"); }
  validateSession(value, expectedNq, expectedNv);
  return value;
}
export function exportRecordedSession(session: RecordedSession): string {
  validateSession(session); const text = JSON.stringify(session);
  check(encoder.encode(text).byteLength <= MAX_BYTES, "file size limit"); return text;
}

export class SessionRecorder {
  private session: RecordedSession;
  private startTime: number | null = null;
  private lastImageTime = -Infinity;
  private lastImageSourceTime = -Infinity;
  private bytes: number;
  private closed = false;

  constructor(metadata: RecordingMetadata) {
    validateMetadata(metadata);
    this.session = { version: 1, metadata: { ...metadata }, frames: [], duration: 0, stoppedReason: "finished" };
    this.bytes = encoder.encode(JSON.stringify(this.session)).byteLength + 512;
  }
  append(state: PhysicsState, observation: Observation, action: RobotAction, images?: CameraImage[]): boolean {
    if (this.closed) return false;
    number(state.time, "state time"); check(state.time >= 0, "state time");
    const elapsed = state.time - (this.startTime ?? state.time), last = this.session.frames.at(-1);
    check(!last || state.time >= last.state.time, "state time must not move backward");
    if (elapsed > MAX_DURATION + EPSILON) return this.stop("duration-limit");
    if (last && elapsed - last.elapsed < SAMPLE_INTERVAL - EPSILON) return true;
    const frame: RecordedFrame = { elapsed: Math.min(elapsed, MAX_DURATION), state, observation, action };
    let imageSourceTime = -Infinity;
    if (images && elapsed - this.lastImageTime >= CAMERA_INTERVAL - EPSILON) {
      validateImages(images); imageSourceTime = Math.min(...images.map(image => image.timestamp));
      if (imageSourceTime > this.lastImageSourceTime) frame.images = images;
    }
    validateFrame(frame, this.session.metadata);
    check(!last || state.act.length === last.state.act.length, "act dimension mismatch");
    const text = JSON.stringify(frame), bytes = encoder.encode(text).byteLength + 1;
    if (this.bytes + bytes > MAX_BYTES || this.session.frames.length >= MAX_FRAMES) return this.stop("size-limit");
    this.session.frames.push(JSON.parse(text) as RecordedFrame); this.session.duration = frame.elapsed; this.bytes += bytes;
    this.startTime ??= state.time;
    if (frame.images) { this.lastImageTime = elapsed; this.lastImageSourceTime = imageSourceTime; }
    return elapsed >= MAX_DURATION - EPSILON ? this.stop("duration-limit") : true;
  }
  private stop(reason: RecordedSession["stoppedReason"]): false {
    this.closed = true; this.session.stoppedReason = reason; return false;
  }
  finish(): RecordedSession {
    this.closed = true; return structuredClone(this.session);
  }
}

// Restore this snapshot while physics is paused. Actions are a trace, not re-simulated inputs.
export function playbackFrame(session: RecordedSession, elapsed: number): RecordedFrame | null {
  number(elapsed, "playback time");
  if (!session.frames.length) return null;
  let low = 0, high = session.frames.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (session.frames[middle].elapsed <= elapsed + EPSILON) low = middle;
    else high = middle - 1;
  }
  return session.frames[low];
}
