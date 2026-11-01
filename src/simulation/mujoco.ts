import { Euler, Matrix4, Quaternion as ThreeQuaternion, Vector3 } from "three";
import loadMujoco, { type MainModule, type MjModel, type MjData, type DoubleBuffer } from "@mujoco/mujoco";
import { SCENE } from "../robot/model.ts";
import { ARM_NAMES, ARM_SPEEDS, BASE_HOME, HAND_OPEN, HAND_CLOSED, HSR_LINKS } from "../robot/hsr-model.ts";
import { forwardHSR, HAND_FORWARD, solveHSR, targetRotation } from "../robot/hsr-ik.ts";
import { solveWholeBody } from "../robot/hsr-whole-body.ts";

import type { Vec3, Observation, Quaternion, RobotAction, Controller, CameraImage, CameraAcquisition } from "./types.ts";
import { createObjects, TIME_STEP, OBJECTS } from "./scene.ts";
import { TABLE_HEIGHT } from "./table.ts";
import { validateAction } from "../controllers/protocol.ts";
import { TaskEvaluator } from "./task.ts";
import type { PhysicsState } from "../recording/session.ts";
import type { LearningCaptureGate } from "../learning/capture.ts";
import { MOVABLE_FURNITURE, FURNITURE, FRIDGE_DOOR, KITCHEN_DOOR, OVEN_DOOR } from "./environment.ts";
type Stage = { name: string; duration: number; target?: Vec3; opening?: number; baseZ?: number; handleArc?: number; kitchenArc?: number; ovenArc?: number };
const DT = TIME_STEP;
const clamp = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n));
const ARM_HOME_TARGET: Vec3 = [-0.078, 0.98, -0.42];
const ARM_HOME = solveHSR(ARM_HOME_TARGET, [0, 0, 0], [0.4, -0.6, 0, -0.97, 0]);
const FRIDGE_GRASP_TARGET: Vec3 = [-0.078, 0.645, -0.525];
function fridgeGraspBase(angle: number): Vec3 {
  const fridge = FURNITURE.find(f => f.id === "fridge")!, c = Math.cos(angle), s = Math.sin(angle);
  const [x, , z] = FRIDGE_DOOR.handle, [tx, , tz] = FRIDGE_GRASP_TARGET;
  return [fridge.position[0] + FRIDGE_DOOR.hinge[0] + (x - tx) * c + (z - tz) * s,
    fridge.position[2] + FRIDGE_DOOR.hinge[2] - (x - tx) * s + (z - tz) * c, angle];
}
const KITCHEN_GRASP_TARGET: Vec3 = [-0.078, 0.78, -0.525];
function kitchenGraspBase(angle: number): Vec3 {
  const kitchen = FURNITURE.find(f => f.id === "kitchen")!, c = Math.cos(angle), s = Math.sin(angle);
  const [x, , z] = KITCHEN_DOOR.handle, [tx, , tz] = KITCHEN_GRASP_TARGET;
  return [kitchen.position[0] + KITCHEN_DOOR.hinge[0] + (x - tx) * c - (z - tz) * s,
    kitchen.position[2] + KITCHEN_DOOR.hinge[2] + (x - tx) * s + (z - tz) * c, -angle];
}
function ovenGraspTarget(angle: number): Vec3 {
  const [ , y, z] = OVEN_DOOR.handle;
  return [-0.078, OVEN_DOOR.hinge[1] + y * Math.cos(angle) - z * Math.sin(angle), -0.525];
}
function ovenGraspBase(angle: number): Vec3 {
  const kitchen = FURNITURE.find(f => f.id === "kitchen")!, [, y, z] = OVEN_DOOR.handle;
  return [kitchen.position[0] + OVEN_DOOR.hinge[0] + 0.078,
    kitchen.position[2] + OVEN_DOOR.hinge[2] + y * Math.sin(angle) + z * Math.cos(angle) + 0.525, 0];
}
import { ARM_KEYS, BASE_KEYS, SHIFT_KEYS, CONTROL_KEYS, TOUCH_STOP, type CameraView, type TouchInput } from "./input.ts";
export { CONTROL_KEYS, TOUCH_STOP, type CameraView, type TouchInput } from "./input.ts";

export class Simulation {
  readonly model: MjModel;
  readonly data: MjData;
  readonly keys = new Set<string>();
  touchInput: TouchInput = { ...TOUCH_STOP };
  readonly joints: Record<string, { q: number; actuator: number }> = {};
  readonly cubeId: number;
  readonly mugId: number;
  readonly baseId: number;
  readonly tcpId: number;
  readonly cameraId: number;
  readonly handCameraId: number;
  readonly tableId: number;
  private readonly floorId: number;
  private readonly gravityDofs: Record<string, number> = {};
  private readonly leftFingerGeoms = new Set<number>();
  private readonly rightFingerGeoms = new Set<number>();
  readonly objectBodies: Map<number, string>;
  seed = 42;
  target: Vec3 = [...ARM_HOME_TARGET];
  baseTarget: Vec3 = [...BASE_HOME];
  headTarget = [-22 * Math.PI / 180, 0];
  wristTarget: Vec3 = [0, 0, 0];
  opening = 1;
  armCommand: number[] = [];
  coordinateJointTargets = false;
  private handCommand = HAND_OPEN;
  private previousOpening = 1;
  private controlTicks = 0;
  private attitudePriority = false;
  private coordinatedPose: { position: Vec3; rotation: Quaternion } | null = null;
  private coordinatedBaseVelocity: Vec3 = [0, 0, 0];
  private armReturn: { start: number[]; elapsed: number; duration: number } | null = null;
  private headReturn: { start: number; elapsed: number; duration: number } | null = null;
  program: { stages: Stage[]; index: number; elapsed: number; start: Vec3; startArc?: number } | null = null;
  phase = "手動操作";
  maxTablePenetration = 0;
  tableForce = 0;
  peakTableForce = 0;
  armTorqueNm = 40;
  contactTargetY = TABLE_HEIGHT - 0.1;
  private holdForceSum = 0;
  private holdForceSteps = 0;
  private readonly contactForce: DoubleBuffer;
  private readonly doorDof: number;
  private readonly kitchenDofs: number[];
  minHoldHeight = Infinity;
  bilateralContact = false;
  gripperContactId: string | null = null;
  physicsMs = 0;
  readonly task = new TaskEvaluator();
  controller: Controller | null = null;
  lastAction: RobotAction = {};
  cameraImages: CameraImage[] = [];
  private readonly cameraCaptureSession = crypto.randomUUID();
  private readonly issuedCameraFrames = new WeakSet<CameraImage>();
  private cameraCaptureSequence = 0;
  private cameraCapture: { images: CameraImage[]; acquisition: CameraAcquisition } | null = null;
  get cameraAcquisition(): CameraAcquisition | null {
    return !this.replaying && this.cameraCapture?.images === this.cameraImages ? this.cameraCapture.acquisition : null;
  }
  publishCameraImages(images: CameraImage[]): CameraAcquisition {
    if (this.replaying || images.some(image => this.issuedCameraFrames.has(image)) ||
      images.length !== 2 || !["head", "hand"].every(view => images.filter(i => i.view === view).length === 1) ||
      images.some(i => i.width !== 320 || i.height !== 240 || !Number.isFinite(i.timestamp) || Math.abs(i.timestamp - this.data.time) > 1e-8 ||
        !/^data:image\/jpeg;base64,[A-Za-z0-9+/]+={0,2}$/.test(i.dataUrl))) throw new Error("Fresh rendered camera pair required; replay images cannot be issued");
    const acquisition = Object.freeze({ captureId: `${this.cameraCaptureSession}:${++this.cameraCaptureSequence}`,
      generation: this.resetGeneration, acquisitionSequence: this.cameraCaptureSequence, timestamp: this.data.time });
    const published = images.map(image => Object.freeze({ ...image }));
    Object.freeze(published);
    for (const image of [...images, ...published]) this.issuedCameraFrames.add(image);
    this.cameraImages = published;
    this.cameraCapture = { images: published, acquisition };
    return acquisition;
  }
  imageCaptureEnabled = false;
  learningCapture: LearningCaptureGate | null = null;
  afterStep: (() => void) | null = null;
  replaying = false;
  resetGeneration = 0;
  collisionForce = 0;
  collisionCount = 0;
  selfContactCount = 0;
  private heldAction: RobotAction | null = null;
  private actionUntil = 0;
  private actorTicks = 0;
  controlView: CameraView = "head";
  overviewRotation: Quaternion = [0, 0, 0, 1];
  readonly mj: MainModule;

  constructor(mj: MainModule) {
    this.mj = mj;
    this.model = mj.MjModel.from_xml_string(SCENE);
    this.data = new mj.MjData(this.model);
    for (const name of [...ARM_NAMES, "base_x", "base_z", "base_yaw", "hand_motor_joint", "head_pan_joint", "head_tilt_joint"]) {
      const j = this.id("JOINT", name);
      this.joints[name] = { q: this.model.jnt_qposadr[j], actuator: this.id("ACTUATOR", name) };
    }
    for (const link of HSR_LINKS) if (link.joint.type !== "fixed" && !this.joints[link.joint.name]) {
      this.joints[link.joint.name] = { q: this.model.jnt_qposadr[this.id("JOINT", link.joint.name)], actuator: -1 };
    }
    for (const name of ["base_y", "base_pitch", "base_roll"]) {
      this.joints[name] = { q: this.model.jnt_qposadr[this.id("JOINT", name)], actuator: -1 };
    }
    for (const name of [...ARM_NAMES, "torso_lift_joint", "head_pan_joint", "head_tilt_joint"]) {
      this.gravityDofs[name] = this.model.jnt_dofadr[this.id("JOINT", name)];
    }
    this.joints.pan = this.joints.head_pan_joint; this.joints.tilt = this.joints.head_tilt_joint;
    const hinge = this.id("JOINT", "fridge_hinge");
    this.joints.fridge_hinge = { q: this.model.jnt_qposadr[hinge], actuator: -1 };
    this.doorDof = this.model.jnt_dofadr[hinge];
    this.kitchenDofs = ["kitchen_hinge", "oven_hinge"].map(name => {
      const joint = this.id("JOINT", name);
      this.joints[name] = { q: this.model.jnt_qposadr[joint], actuator: -1 };
      return this.model.jnt_dofadr[joint];
    });
    this.cubeId = this.id("BODY", "cube"); this.baseId = this.id("BODY", "base");
    this.mugId = this.id("BODY", "mug");
    this.tcpId = this.id("SITE", "tcp"); this.cameraId = this.id("SITE", "head_camera");
    this.handCameraId = this.id("SITE", "hand_camera");
    this.tableId = this.id("GEOM", "table");
    this.floorId = this.id("GEOM", "floor");
    this.objectBodies = new Map(OBJECTS.map(({ id }) => [this.id("BODY", id), id]));
    this.objectBodies.set(this.id("BODY", "room_basket"), "basket");
    this.objectBodies.set(this.id("BODY", "room_tv_plant"), "tv_plant");
    this.objectBodies.set(this.id("BODY", "fridge_door"), "fridge_door");
    for (const name of ["kitchen_door", "oven_door"]) this.objectBodies.set(this.id("BODY", name), name);
    for (const [prefix, geoms] of [["hand_l_", this.leftFingerGeoms], ["hand_r_", this.rightFingerGeoms]] as const) {
      const bodies = new Set(HSR_LINKS.filter(link => link.name.startsWith(prefix)).map(link => this.id("BODY", link.name)));
      for (let i = 0; i < this.model.ngeom; i++) if (bodies.has(this.model.geom_bodyid[i])) geoms.add(i);
    }
    this.contactForce = new mj.DoubleBuffer(6);
    this.reset();
  }
  id(kind: "BODY" | "JOINT" | "ACTUATOR" | "SITE" | "GEOM", name: string) {
    const e = this.mj.mjtObj;
    const type = { BODY: e.mjOBJ_BODY, JOINT: e.mjOBJ_JOINT, ACTUATOR: e.mjOBJ_ACTUATOR, SITE: e.mjOBJ_SITE, GEOM: e.mjOBJ_GEOM }[kind].value;
    const id = this.mj.mj_name2id(this.model, type, name);
    if (id < 0) throw new Error("Missing model element: " + name);
    return id;
  }
  bodyRotation(id: number): Quaternion {
    const q = id * 4, r = this.data.xquat;
    return [r[q + 1], r[q + 2], r[q + 3], r[q]];
  }
  get baseTilt() { const r = new ThreeQuaternion(...this.bodyRotation(this.baseId)); return Math.acos(clamp(new Vector3(0, 1, 0).applyQuaternion(r).y, -1, 1)); }
  get tipped() { return this.baseTilt > Math.PI / 4; }
  physicsState(): PhysicsState { return { time: this.data.time, qpos: Array.from(this.data.qpos), qvel: Array.from(this.data.qvel), act: Array.from(this.data.act) }; }
  restoreFrame(state: PhysicsState, observation?: Observation) {
    if (state.qpos.length !== this.model.nq || state.qvel.length !== this.model.nv || state.act.length !== this.model.na ||
      state.time < 0 || ![state.time, ...state.qpos, ...state.qvel, ...state.act].every(Number.isFinite)) throw new Error("記録のモデルが一致しません。");
    if (!this.replaying) this.setController(null);
    this.replaying = true;
    this.data.time = state.time; this.data.qpos.set(state.qpos); this.data.qvel.set(state.qvel); this.data.act.set(state.act);
    this.mj.mj_forward(this.model, this.data);
    this.phase = "記録を再生";
    if (observation?.robot) {
      this.tableForce = observation.robot.tableForce;
      this.gripperContactId = observation.robot.contactId ?? null;
      this.bilateralContact = observation.robot.gripperContact;
      this.collisionForce = observation.robot.collision?.force ?? 0;
      this.collisionCount = observation.robot.collision?.count ?? 0;
      this.selfContactCount = observation.robot.collision?.selfCount ?? 0;
    } else this.readContacts();
  }
  setController(controller: Controller | null) {
    const capture = this.learningCapture; this.learningCapture = null; capture?.stop();
    if (this.heldAction?.armJointTarget) this.phase = "姿勢を保持";
    this.controller?.abort?.(); this.controller = controller; this.heldAction = null;
    this.keys.clear(); this.touchInput = { ...TOUCH_STOP }; this.program = null;
    this.armReturn = null; this.headReturn = null; this.stopCoordinatedMotion();
    const b = this.position(this.baseId); this.baseTarget = [b[0], b[2], this.q("base_yaw")];
    this.target = this.localTCP(); this.wristTarget = this.actualWristAngles();
    this.armCommand = ARM_NAMES.map(n => this.q(n)); this.lastAction = {};
  }
  applyAction(action: RobotAction) {
    this.learningCapture?.stop();
    const command = validateAction(action);
    if (this.replaying) throw new Error("再生を終了してリセットしてください。");
    const previous = this.heldAction;
    const jointTransition = Boolean(previous?.armJointTarget) !== Boolean(command.armJointTarget);
    if (jointTransition || ((previous?.translation?.some(Boolean) || previous?.rotation?.some(Boolean) || previous?.handTarget) &&
      !command.translation?.some(Boolean) && !command.rotation?.some(Boolean) && !command.handTarget)) {
      this.stopCoordinatedMotion(); this.target = this.localTCP(); this.wristTarget = this.actualWristAngles();
      if (jointTransition) {
        this.armCommand = ARM_NAMES.map(n => this.q(n));
        if (previous?.armJointTarget) this.phase = "姿勢を保持";
      }
    }
    if (command.armJointTarget && !previous?.armJointTarget) {
      this.program = null; this.armReturn = null; this.attitudePriority = false;
      for (const key of ARM_KEYS) this.keys.delete(key);
      this.touchInput.lift = 0; this.touchInput.reach = 0;
      this.phase = "関節目標を実行";
    }
    if ((previous?.base?.some(Boolean) || previous?.baseTarget) && !command.baseTarget && !command.base?.some(Boolean)) {
      const b = this.position(this.baseId); this.baseTarget = [b[0], b[2], this.q("base_yaw")];
    }
    this.heldAction = command; this.actionUntil = this.data.time + 0.5;
    return command;
  }
  q(name: string): number { return this.data.qpos[this.joints[name].q]; }
  ctrl(name: string, value: number) { this.data.ctrl[this.joints[name].actuator] = value; }
  position(id: number): Vec3 { return Array.from(this.data.xpos.slice(id * 3, id * 3 + 3)) as Vec3; }
  cube(): Vec3 { return this.position(this.cubeId); }
  tcp(): Vec3 { return Array.from(this.data.site_xpos.slice(this.tcpId * 3, this.tcpId * 3 + 3)) as Vec3; }
  siteRotation(id: number): Quaternion {
    const r = this.data.site_xmat, k = id * 9;
    const matrix = new Matrix4().set(r[k], r[k + 1], r[k + 2], 0, r[k + 3], r[k + 4], r[k + 5], 0,
      r[k + 6], r[k + 7], r[k + 8], 0, 0, 0, 0, 1);
    return new ThreeQuaternion().setFromRotationMatrix(matrix).toArray();
  }
  tcpRotation(): Quaternion { return this.siteRotation(this.tcpId); }
  get rotationMode() { return SHIFT_KEYS.some(k => this.keys.has(k)); }
  actualWristAngles(): Vec3 {
    const rotation = new ThreeQuaternion(...this.bodyRotation(this.baseId)).invert()
      .multiply(new ThreeQuaternion(...this.tcpRotation())).multiply(HAND_FORWARD.clone().invert());
    const euler = new Euler().setFromQuaternion(rotation, "YXZ");
    return [euler.y, euler.x, euler.z];
  }
  gripperOpening() { return clamp((this.q("hand_motor_joint") - HAND_CLOSED) / (HAND_OPEN - HAND_CLOSED), 0, 1); }
  targetError() { return Math.hypot(...this.localTCP().map((v, i) => v - this.target[i])); }
  attitudeError() {
    const actual = new ThreeQuaternion(...this.tcpRotation());
    const desired = new ThreeQuaternion(...this.bodyRotation(this.baseId)).multiply(targetRotation(this.wristTarget));
    return actual.angleTo(desired);
  }
  localTCP(): Vec3 {
    return new Vector3(...this.tcp()).sub(new Vector3(...this.position(this.baseId)))
      .applyQuaternion(new ThreeQuaternion(...this.bodyRotation(this.baseId)).invert()).toArray();
  }
  get robotMassKg(): number { return this.model.body_subtreemass[this.baseId]; }
  setParameters(massKg: number, torqueNm: number) {
    const scale = massKg / this.robotMassKg;
    for (let body = 1; body < this.model.nbody; body++) {
      if (this.model.body_rootid[body] !== this.baseId) continue;
      this.model.body_mass[body] *= scale;
      for (let axis = 0; axis < 3; axis++) this.model.body_inertia[body * 3 + axis] *= scale;
    }
    this.mj.mj_setConst(this.model, this.data);
    this.armTorqueNm = clamp(torqueNm, 5, 100);
    for (const name of ARM_NAMES.slice(1)) {
      const index = this.joints[name].actuator * 2;
      this.model.actuator_forcerange[index] = -this.armTorqueNm;
      this.model.actuator_forcerange[index + 1] = this.armTorqueNm;
    }
    this.reset();
  }
  meanTableForce(): number { return this.holdForceSteps ? this.holdForceSum / this.holdForceSteps : 0; }
  observation(): Observation {
    return {
      timestamp: this.data.time,
      objects: createObjects(this.seed).map(({ id, type }) => {
        const body = this.id("BODY", id), joint = this.model.body_jntadr[body];
        const dof = this.model.jnt_dofadr[joint], q = body * 4, r = this.data.xquat;
        return { id, type, position: this.position(body), rotation: [r[q + 1], r[q + 2], r[q + 3], r[q]],
          velocity: Array.from(this.data.qvel.slice(dof, dof + 3)) as Vec3 };
      }),
      furniture: MOVABLE_FURNITURE.map(({ id }) => {
        const body = this.id("BODY", `room_${id}`), q = body * 4, r = this.data.xquat;
        return { id, position: this.position(body), rotation: [r[q + 1], r[q + 2], r[q + 3], r[q]] };
      }),
      robot: { base: { position: this.position(this.baseId), yaw: this.q("base_yaw"), rotation: this.bodyRotation(this.baseId), tilt: this.baseTilt, tipped: this.tipped },
        joints: ARM_NAMES.map(n => this.q(n)), gripper: this.gripperOpening(),
        endEffectorPosition: this.tcp(), endEffectorRotation: this.tcpRotation(), gripperContact: this.bilateralContact, tableForce: this.tableForce, contactId: this.gripperContactId,
        collision: { force: this.collisionForce, count: this.collisionCount, selfCount: this.selfContactCount } },
    };
  }
  reset(seed = this.seed) {
    this.coordinateJointTargets = false;
    this.resetGeneration++; this.cameraImages = []; this.cameraCapture = null;
    this.collisionForce = 0; this.collisionCount = 0; this.selfContactCount = 0;
    this.setController(null); this.replaying = false; this.task.reset(); this.actorTicks = 0;
    this.seed = seed;
    this.mj.mj_resetData(this.model, this.data); this.keys.clear();
    this.touchInput = { ...TOUCH_STOP };
    for (const object of createObjects(seed)) {
      const body = this.id("BODY", object.id), q = this.model.jnt_qposadr[this.model.body_jntadr[body]];
      object.position.forEach((v, axis) => { this.data.qpos[q + axis] = v; });
    }
    this.target = [...ARM_HOME_TARGET]; this.baseTarget = [...BASE_HOME]; this.headTarget = [-22 * Math.PI / 180, 0];
    this.wristTarget = [0, 0, 0]; this.opening = 1; this.program = null; this.phase = "手動操作";
    this.maxTablePenetration = 0; this.minHoldHeight = Infinity; this.bilateralContact = false; this.gripperContactId = null;
    this.tableForce = 0; this.peakTableForce = 0; this.holdForceSum = 0; this.holdForceSteps = 0;
    this.armReturn = null; this.headReturn = null;
    this.coordinatedPose = null;
    this.coordinatedBaseVelocity = [0, 0, 0];
    this.armCommand = [...ARM_HOME];
    ARM_NAMES.forEach((name, i) => { this.data.qpos[this.joints[name].q] = this.armCommand[i]; });
    this.data.qpos[this.model.jnt_qposadr[this.id("JOINT", "torso_lift_joint")]] = this.armCommand[0] * 0.5;
    for (const [name, value] of [["hand_motor_joint", HAND_OPEN], ["hand_l_proximal_joint", HAND_OPEN], ["hand_r_proximal_joint", HAND_OPEN],
      ["hand_l_distal_joint", -HAND_OPEN - 0.087], ["hand_r_distal_joint", -HAND_OPEN - 0.087]] as const) {
      this.data.qpos[this.model.jnt_qposadr[this.id("JOINT", name)]] = value;
    }
    this.handCommand = HAND_OPEN; this.previousOpening = 1; this.controlTicks = 0; this.attitudePriority = false;
    this.data.qpos[this.joints.tilt.q] = this.headTarget[0];
    this.data.qpos[this.joints.pan.q] = this.headTarget[1];
    this.mj.mj_forward(this.model, this.data); this.applyControls(); this.mj.mj_forward(this.model, this.data);
  }
  startProbe(kind: "grasp" | "contact" | "door" | "kitchen" | "oven") {
    this.reset();
    if (kind === "kitchen" || kind === "oven") {
      const oven = kind === "oven", target = oven ? ovenGraspTarget(0) : KITCHEN_GRASP_TARGET;
      this.baseTarget = oven ? ovenGraspBase(0) : kitchenGraspBase(0);
      this.data.qpos[this.joints.base_x.q] = this.baseTarget[0] - BASE_HOME[0];
      this.data.qpos[this.joints.base_z.q] = this.baseTarget[1] - BASE_HOME[1];
      this.wristTarget = [0, 0, Math.PI / 2];
      this.applyControls(); this.mj.mj_forward(this.model, this.data);
      this.program = { stages: [
        { name: "横向きの取手へ接近", duration: 3, target: [-0.078, 0.9, -0.525] },
        { name: "取手の高さに合わせる", duration: 3, target },
        { name: "取手を掴む", duration: 2, opening: 0 },
        { name: "取手を引く", duration: 6, ...(oven ? { ovenArc: 20 * Math.PI / 180 } : { kitchenArc: 45 * Math.PI / 180 }) },
        { name: "扉を開いて保持", duration: 3 },
        { name: "扉を押して閉じる", duration: 6, ...(oven ? { ovenArc: 0 } : { kitchenArc: 0 }) },
        { name: "取手を離す", duration: 1, opening: 1 },
        { name: "扉から離れる", duration: 2, target: [-0.078, 0.9, -0.42] },
      ], index: 0, elapsed: 0, start: [...this.target], startArc: 0 };
      return;
    }
    if (kind === "door") {
      // A slight diagonal pinch keeps both loaded finger pads on the thin vertical handle.
      this.wristTarget[2] = 0.25;
      this.baseTarget = fridgeGraspBase(0);
      this.data.qpos[this.joints.base_x.q] = this.baseTarget[0] - BASE_HOME[0];
      this.data.qpos[this.joints.base_z.q] = this.baseTarget[1] - BASE_HOME[1];
      this.applyControls(); this.mj.mj_forward(this.model, this.data);
      this.program = { stages: [
        { name: "取手へ接近", duration: 2, target: [-0.078, 0.9, -0.525] },
        { name: "取手の高さに合わせる", duration: 2, target: FRIDGE_GRASP_TARGET },
        { name: "取手を掴む", duration: 2, opening: 0 },
        { name: "取手を引く", duration: 6, handleArc: 40 * Math.PI / 180 },
        { name: "扉を開いて保持", duration: 3 },
        { name: "扉を押して閉じる", duration: 6, handleArc: 0 },
        { name: "取手を離す", duration: 1, opening: 1 },
        { name: "扉から離れる", duration: 2, target: [-0.078, 0.9, -0.42] },
      ], index: 0, elapsed: 0, start: [...this.target], startArc: 0 };
      return;
    }
    // Stand farther back so the horizontal hand can reach the low tabletop.
    this.baseTarget[1] += 0.15;
    const mug = this.position(this.mugId), z = mug[2] - this.baseTarget[1], x = mug[0] - BASE_HOME[0];
    if (kind === "contact") {
      this.baseTarget[0] -= 0.30;
      this.wristTarget[1] = -0.35;
    }
    const above: Vec3 = kind === "grasp" ? [x, TABLE_HEIGHT + 0.25, z] : [-0.078, TABLE_HEIGHT + 0.25, -0.72];
    const grasp: Vec3 = [x, TABLE_HEIGHT + 0.063, z];
    const stages: Stage[] = [{ name: "物体の落下待ち", duration: 1 }, { name: "接近", duration: 2, target: above }];
    if (kind === "grasp") stages.push(
      { name: "指を合わせる", duration: 2, target: grasp }, { name: "閉じる", duration: 2, opening: 0 },
      { name: "持ち上げる", duration: 2, target: [x, TABLE_HEIGHT + 0.215, z] }, { name: "3秒保持", duration: 3 },
      { name: "机の内側へ運ぶ", duration: 2, baseZ: BASE_HOME[1] },
      { name: "開いて解放", duration: 1, opening: 1 }, { name: "落下を確認", duration: 1 });
    else stages.push(
      { name: "机へ押し付ける", duration: 3, target: [-0.078, this.contactTargetY, -0.72] }, { name: "接触を保持", duration: 3 },
      { name: "机から離す", duration: 2, target: above });
    this.program = { stages, index: 0, elapsed: 0, start: [...this.target] };
  }
  get armReturning() { return this.armReturn !== null; }
  get headReturning() { return this.headReturn !== null; }
  private stopCoordinatedMotion() {
    if (!this.coordinatedPose) return;
    this.coordinatedPose = null;
    this.coordinatedBaseVelocity = [0, 0, 0];
    const b = this.position(this.baseId);
    if (!BASE_KEYS.slice(0, 4).some(k => this.keys.has(k)) && !this.touchInput.forward && !this.touchInput.strafe) {
      this.baseTarget[0] = b[0]; this.baseTarget[1] = b[2];
    }
    if (!this.keys.has("KeyQ") && !this.keys.has("KeyE") && !this.touchInput.turn) this.baseTarget[2] = this.q("base_yaw");
  }
  returnArmHome() {
    if (this.replaying) return;
    if (this.controller || this.heldAction || this.learningCapture) this.setController(null);
    for (const key of ARM_KEYS) this.setKey(key, false);
    this.setTouchInput({ lift: 0, reach: 0 });
    this.program = null; this.phase = "アームを標準姿勢へ"; this.attitudePriority = false;
    this.armCommand = ARM_NAMES.map(n => this.q(n));
    this.target = this.localTCP(); this.wristTarget = this.actualWristAngles();
    this.armReturn = { start: [...this.armCommand], elapsed: 0,
      duration: Math.max(3, ...ARM_HOME.map((v, i) => 3 * Math.abs(v - this.armCommand[i]) / ARM_SPEEDS[i])) };
  }
  centerHead() {
    if (this.replaying) return;
    if (this.controller || this.heldAction || this.learningCapture) this.setController(null);
    const start = this.q("pan");
    this.headTarget[1] = start;
    this.headReturn = { start, elapsed: 0, duration: Math.max(2.5, Math.abs(start) * 3) };
  }
  private cancelArmReturn() {
    if (!this.armReturn) return;
    this.armReturn = null; this.phase = "手動操作";
    this.target = this.localTCP(); this.wristTarget = this.actualWristAngles();
    this.armCommand = ARM_NAMES.map(n => this.q(n));
  }
  setWristAngle(axis: number, angle: number) {
    if (this.replaying) return;
    if (this.controller || this.heldAction || this.learningCapture) this.setController(null);
    for (const key of ARM_KEYS) this.setKey(key, false);
    this.stopCoordinatedMotion();
    this.cancelArmReturn();
    this.program = null; this.phase = "手動操作";
    this.attitudePriority = true;
    this.target = this.localTCP();
    this.wristTarget[axis] = clamp(angle, axis === 1 ? -Math.PI / 2 : -Math.PI, axis === 1 ? Math.PI / 2 : Math.PI);
  }
  setKey(code: string, down: boolean) {
    if (this.replaying) return;
    if (!CONTROL_KEYS.includes(code) || this.keys.has(code) === down) return;
    if (down && !SHIFT_KEYS.includes(code) && (this.controller || this.heldAction || this.learningCapture)) this.setController(null);
    const wasRotating = this.rotationMode, hadArmKey = ARM_KEYS.some(k => this.keys.has(k));
    if (down) {
      if (ARM_KEYS.includes(code)) this.cancelArmReturn();
      if (code.startsWith("Arrow") && this.headReturn) { this.headReturn = null; this.headTarget[1] = this.q("pan"); }
      this.keys.add(code);
      if (!SHIFT_KEYS.includes(code)) { this.program = null; if (!this.armReturn) this.phase = "手動操作"; }
      if (code === "Space") this.opening = this.opening > 0.5 ? 0 : 1;
    }
    else this.keys.delete(code);
    if (!this.program && !this.armReturn && (ARM_KEYS.includes(code) || SHIFT_KEYS.includes(code))) {
      if (wasRotating !== this.rotationMode || !ARM_KEYS.some(k => this.keys.has(k)) || (!hadArmKey && down)) {
        this.stopCoordinatedMotion();
        this.wristTarget = this.actualWristAngles();
        this.target = this.localTCP(); this.armCommand = ARM_NAMES.map(n => this.q(n));
        if (ARM_KEYS.some(k => this.keys.has(k))) {
          this.coordinatedPose = { position: this.tcp(), rotation: this.tcpRotation() }; this.phase = "台車・アーム協調";
        } else this.phase = "手動操作";
      }
      this.attitudePriority = this.rotationMode;
    }
    if (!down && BASE_KEYS.includes(code) && !BASE_KEYS.some(k => this.keys.has(k)) &&
      !this.touchInput.forward && !this.touchInput.strafe && !this.touchInput.turn) {
      const b = this.position(this.baseId); this.baseTarget = [b[0], b[2], this.q("base_yaw")];
    }
  }
  setTouchInput(input: Partial<TouchInput>) {
    if (this.replaying) return;
    if (Object.values(input).some(Boolean) && (this.controller || this.heldAction || this.learningCapture)) this.setController(null);
    const previous = this.touchInput, next = { ...previous, ...input };
    for (const axis of Object.keys(next) as (keyof TouchInput)[]) next[axis] = clamp(next[axis], -1, 1);
    const wasArm = Boolean(previous.lift || previous.reach), arm = Boolean(next.lift || next.reach);
    if (arm) {
      this.cancelArmReturn();
      for (const key of ARM_KEYS) this.setKey(key, false);
      this.stopCoordinatedMotion();
    }
    if (Object.values(next).some(Boolean)) { this.program = null; if (!this.armReturn) this.phase = "手動操作"; }
    if (wasArm !== arm && !ARM_KEYS.some(k => this.keys.has(k))) {
      this.target = this.localTCP(); this.wristTarget = this.actualWristAngles();
      this.armCommand = ARM_NAMES.map(n => this.q(n)); this.attitudePriority = false;
    }
    if ((previous.forward || previous.strafe || previous.turn) && !next.forward && !next.strafe && !next.turn &&
      !BASE_KEYS.some(k => this.keys.has(k))) {
      const b = this.position(this.baseId); this.baseTarget = [b[0], b[2], this.q("base_yaw")];
    }
    this.touchInput = next;
  }
  humanAction(): RobotAction {
    const key = (a: string, b: string) => Number(this.keys.has(a)) - Number(this.keys.has(b));
    const forward = clamp(key("KeyW", "KeyS") + this.touchInput.forward, -1, 1);
    const strafe = clamp(key("KeyD", "KeyA") + this.touchInput.strafe, -1, 1);
    const norm = Math.max(1, Math.hypot(forward, strafe));
    const action: RobotAction = { base: [strafe / norm * 0.25, forward / norm * 0.25, clamp(key("KeyQ", "KeyE") + this.touchInput.turn, -1, 1) * 0.7],
      head: [key("ArrowUp", "ArrowDown"), key("ArrowLeft", "ArrowRight")], gripper: this.opening };
    const camera = new ThreeQuaternion(...(this.controlView === "overview" ? this.overviewRotation : this.siteRotation(this.controlView === "head" ? this.cameraId : this.handCameraId)));
    if (ARM_KEYS.some(k => this.keys.has(k))) {
      const v = this.rotationMode ? new Vector3(key("KeyI", "KeyK"), key("KeyJ", "KeyL"), key("KeyU", "KeyO")) : new Vector3(key("KeyL", "KeyJ"), key("KeyI", "KeyK"), key("KeyO", "KeyU"));
      v.multiplyScalar((this.rotationMode ? 0.7 : 0.18) / Math.max(1, v.length())).applyQuaternion(camera);
      if (this.rotationMode) action.rotation = v.toArray(); else action.translation = v.toArray();
    }
    if (this.touchInput.lift || this.touchInput.reach) {
      const n = Math.max(1, Math.hypot(this.touchInput.lift, this.touchInput.reach));
      action.handTarget = { frame: "base", position: [this.target[0], this.target[1] + this.touchInput.lift / n * 0.12 * DT, this.target[2] - this.touchInput.reach / n * 0.12 * DT] };
    }
    return validateAction(action);
  }
  private applyControls() {
    // Position-actuator offset supplies desired velocity feed-forward without changing force limits.
    const velocity = this.coordinatedBaseVelocity;
    this.ctrl("base_x", this.baseTarget[0] - BASE_HOME[0] + velocity[0] * 180 / 800);
    this.ctrl("base_z", this.baseTarget[1] - BASE_HOME[1] + velocity[1] * 180 / 800);
    this.ctrl("base_yaw", this.baseTarget[2] + velocity[2] * 35 / 150);
    const bias = (name: string) => this.data.qfrc_bias[this.gravityDofs[name]];
    ARM_NAMES.forEach((n, i) => this.ctrl(n, this.armCommand[i] + (bias(n) + (i === 0 ? 0.5 * bias("torso_lift_joint") : 0)) / (i === 0 ? 1200 : 220)));
    this.ctrl("hand_motor_joint", this.handCommand);
    this.ctrl("tilt", this.headTarget[0] + bias("head_tilt_joint") / 30); this.ctrl("pan", this.headTarget[1] + bias("head_pan_joint") / 30);
  }
  step() {
    if (this.replaying || this.learningCapture?.captureDue(this.data.time)) return false;
    const p = this.program;
    if (p) {
      const stage = p.stages[p.index]; this.phase = stage.name;
      const t = clamp(p.elapsed / stage.duration, 0, 1), s = t * t * (3 - 2 * t);
      if (stage.target) this.target = stage.target.map((v, i) => p.start[i] + (v - p.start[i]) * s) as Vec3;
      if (stage.opening !== undefined) this.opening = stage.opening;
      if (stage.baseZ !== undefined) this.baseTarget[1] += clamp(stage.baseZ - this.baseTarget[1], -0.1 * DT, 0.1 * DT);
      if (stage.handleArc !== undefined) this.baseTarget = fridgeGraspBase((p.startArc ?? 0) + (stage.handleArc - (p.startArc ?? 0)) * s);
      if (stage.kitchenArc !== undefined) this.baseTarget = kitchenGraspBase((p.startArc ?? 0) + (stage.kitchenArc - (p.startArc ?? 0)) * s);
      if (stage.ovenArc !== undefined) {
        const angle = (p.startArc ?? 0) + (stage.ovenArc - (p.startArc ?? 0)) * s;
        this.baseTarget = ovenGraspBase(angle); this.target = ovenGraspTarget(angle);
      }
      if (stage.name === "3秒保持") this.minHoldHeight = Math.min(this.minHoldHeight, this.position(this.mugId)[1]);
      p.elapsed += DT;
      if (p.elapsed >= stage.duration) {
        if (stage.target) this.target = [...stage.target];
        if (stage.handleArc !== undefined) p.startArc = stage.handleArc;
        if (stage.kitchenArc !== undefined) p.startArc = stage.kitchenArc;
        if (stage.ovenArc !== undefined) p.startArc = stage.ovenArc;
        p.index++; p.elapsed = 0; p.start = [...this.target];
        if (p.index === p.stages.length) { this.program = null; this.phase = "検証完了"; }
      }
    }
    if (this.controller && this.actorTicks % 10 === 0) this.applyAction(this.controller.getAction(this.observation()));
    if (this.heldAction && this.data.time > this.actionUntil) {
      if (this.heldAction.armJointTarget) { this.armCommand = ARM_NAMES.map(n => this.q(n)); this.phase = "姿勢を保持"; }
      this.heldAction = null; this.stopCoordinatedMotion();
      const b = this.position(this.baseId); this.baseTarget = [b[0], b[2], this.q("base_yaw")];
      this.target = this.localTCP(); this.wristTarget = this.actualWristAngles();
    }
    const action = this.heldAction ?? this.humanAction(); this.lastAction = action;
    const [strafe, forward, turnInput] = action.base ?? [0, 0, 0], yaw = this.q("base_yaw");
    const previousBase = [...this.baseTarget];
    if (action.baseTarget) {
      const delta = new Vector3(action.baseTarget[0] - this.baseTarget[0], 0, action.baseTarget[1] - this.baseTarget[1]).clampLength(0, 0.25 * DT);
      this.baseTarget[0] += delta.x; this.baseTarget[1] += delta.z;
      const turn = Math.atan2(Math.sin(action.baseTarget[2] - this.baseTarget[2]), Math.cos(action.baseTarget[2] - this.baseTarget[2]));
      this.baseTarget[2] += clamp(turn, -0.7 * DT, 0.7 * DT);
    }
    this.baseTarget[0] += (-Math.sin(yaw) * forward + Math.cos(yaw) * strafe) * DT;
    this.baseTarget[1] += (-Math.cos(yaw) * forward - Math.sin(yaw) * strafe) * DT;
    this.baseTarget[2] += turnInput * DT;
    if (action.gripper !== undefined) this.opening = action.gripper;
    if (action.handTarget) {
      this.stopCoordinatedMotion();
      const target = action.handTarget;
      const inv = new ThreeQuaternion(...this.bodyRotation(this.baseId)).invert();
      this.target = target.frame === "base" ? [...target.position] : new Vector3(...target.position)
        .sub(new Vector3(...this.position(this.baseId))).applyQuaternion(inv).toArray();
      if (target.rotation) {
        const r = new ThreeQuaternion(...target.rotation);
        if (target.frame === "world") r.premultiply(inv);
        const e = new Euler().setFromQuaternion(r.multiply(HAND_FORWARD.clone().invert()), "YXZ");
        this.wristTarget = [e.y, e.x, e.z];
      }
    }
    if ((action.translation?.some(Boolean) || action.rotation?.some(Boolean)) && !this.coordinatedPose) {
      this.coordinatedPose = { position: this.tcp(), rotation: this.tcpRotation() };
    }
    const coordinated = this.coordinatedPose;
    if (coordinated) {
      this.phase = "台車・アーム協調";
      if (forward || strafe) { this.coordinatedBaseVelocity[0] = 0; this.coordinatedBaseVelocity[1] = 0; }
      if (turnInput) this.coordinatedBaseVelocity[2] = 0;
      const base = new Vector3(...this.position(this.baseId)), turn = new ThreeQuaternion().setFromAxisAngle(new Vector3(0, 1, 0), this.baseTarget[2] - previousBase[2]);
      coordinated.position = new Vector3(...coordinated.position).sub(base).applyQuaternion(turn).add(base)
        .add(new Vector3(this.baseTarget[0] - previousBase[0], 0, this.baseTarget[1] - previousBase[1])).toArray();
      coordinated.rotation = new ThreeQuaternion(...coordinated.rotation).premultiply(turn).toArray();
      if (action.rotation) {
        const axis = new Vector3(...action.rotation), length = axis.length();
        if (length > 0) coordinated.rotation = new ThreeQuaternion(...coordinated.rotation)
          .premultiply(new ThreeQuaternion().setFromAxisAngle(axis.normalize(), length * DT)).toArray();
      }
      if (action.translation) coordinated.position = new Vector3(...coordinated.position).addScaledVector(new Vector3(...action.translation), DT).toArray();
      // Keep blocked motion from accumulating distant goals against furniture.
      const actual = new Vector3(...this.tcp()), lead = new Vector3(...coordinated.position).sub(actual).clampLength(0, 0.06);
      coordinated.position = actual.add(lead).toArray();
      const measured = new ThreeQuaternion(...this.tcpRotation()), desired = new ThreeQuaternion(...coordinated.rotation), angle = measured.angleTo(desired);
      if (angle > 0.2) coordinated.rotation = measured.slerp(desired, 0.2 / angle).toArray();
    }
    this.headTarget[0] = clamp(this.headTarget[0] + (action.head?.[0] ?? 0) * DT, -1.57, 0.52);
    this.headTarget[1] = clamp(this.headTarget[1] + (action.head?.[1] ?? 0) * DT, -3.84, 1.75);
    if (this.headReturn) {
      const r = this.headReturn, t = clamp((r.elapsed += DT) / r.duration, 0, 1), s = t * t * (3 - 2 * t);
      this.headTarget[1] = r.start * (1 - s);
      if (t === 1) this.headReturn = null;
    }
    if (this.armReturn) {
      const r = this.armReturn, t = clamp((r.elapsed += DT) / r.duration, 0, 1), s = t * t * (3 - 2 * t);
      this.armCommand = ARM_HOME.map((v, i) => r.start[i] + (v - r.start[i]) * s);
      const pose = forwardHSR(this.armCommand), euler = new Euler().setFromQuaternion(pose.rotation.multiply(HAND_FORWARD.clone().invert()), "YXZ");
      this.target = pose.position.toArray(); this.wristTarget = [euler.y, euler.x, euler.z];
      if (t === 1) {
        this.armReturn = null; this.target = [...ARM_HOME_TARGET]; this.wristTarget = [0, 0, 0]; this.phase = "手動操作";
      }
    } else if (this.controlTicks++ % 10 === 0) {
      let angles: number[];
      if (action.armJointTarget) angles = action.armJointTarget;
      else if (coordinated) {
        const result = solveWholeBody(coordinated.position, coordinated.rotation, this.baseTarget, this.armCommand, Boolean(forward || strafe || action.baseTarget), Boolean(turnInput || action.baseTarget), this.position(this.baseId)[1], new ThreeQuaternion().setFromEuler(new Euler(this.q("base_pitch"), 0, this.q("base_roll"), "XYZ")).toArray());
        const start = [...this.baseTarget];
        const delta = new Vector3(result.base[0] - this.baseTarget[0], 0, result.base[1] - this.baseTarget[1]).clampLength(0, 0.25 * DT * 10);
        this.baseTarget[0] += delta.x; this.baseTarget[1] += delta.z;
        this.baseTarget[2] += clamp(result.base[2] - this.baseTarget[2], -0.7 * DT * 10, 0.7 * DT * 10);
        const b = this.position(this.baseId);
        if (!forward && !strafe) {
          this.baseTarget[0] = clamp(this.baseTarget[0], b[0] - 0.06, b[0] + 0.06);
          this.baseTarget[1] = clamp(this.baseTarget[1], b[2] - 0.06, b[2] + 0.06);
        }
        if (!turnInput) this.baseTarget[2] = clamp(this.baseTarget[2], yaw - 0.15, yaw + 0.15);
        this.coordinatedBaseVelocity = this.baseTarget.map((v, i) =>
          (i < 2 ? forward || strafe : turnInput) ? 0 : (v - start[i]) / (DT * 10)) as Vec3;
        const local = new Vector3(...coordinated.position).sub(new Vector3(...b)).applyQuaternion(new ThreeQuaternion(...this.bodyRotation(this.baseId)).invert());
        this.target = local.toArray();
        const rotation = new ThreeQuaternion(...this.bodyRotation(this.baseId)).invert()
          .multiply(new ThreeQuaternion(...coordinated.rotation)).multiply(HAND_FORWARD.clone().invert());
        const euler = new Euler().setFromQuaternion(rotation, "YXZ"); this.wristTarget = [euler.y, euler.x, euler.z];
        angles = result.arm;
      } else angles = solveHSR(this.target, this.wristTarget, this.armCommand, this.attitudePriority);
      if (action.armJointTarget && this.coordinateJointTargets) {
        // Keep coupled joint goals on one path while respecting every joint's speed limit.
        const fraction = 1 / Math.max(1, ...angles.map((v, i) => Math.abs(v - this.armCommand[i]) / (ARM_SPEEDS[i] * DT * 10)));
        this.armCommand = angles.map((v, i) => this.armCommand[i] + (v - this.armCommand[i]) * fraction);
      } else this.armCommand = angles.map((v, i) => this.armCommand[i] + clamp(v - this.armCommand[i], -ARM_SPEEDS[i] * DT * 10, ARM_SPEEDS[i] * DT * 10));
    }
    const handTarget = HAND_CLOSED + this.opening * (HAND_OPEN - HAND_CLOSED);
    if (this.opening !== this.previousOpening) {
      const measured = this.q("hand_motor_joint"), previousTarget = HAND_CLOSED + this.previousOpening * (HAND_OPEN - HAND_CLOSED);
      // Coordinated joint targets preserve loaded closing pressure; legacy retargets and true reversals retain the measured anchor.
      if ((Math.abs(this.opening - this.previousOpening) > 0.01 && !(action.armJointTarget && this.coordinateJointTargets)) || (handTarget - measured) * (previousTarget - measured) < 0) this.handCommand = measured;
      this.previousOpening = this.opening;
    }
    this.handCommand += clamp(handTarget - this.handCommand, -DT, DT);
    // Approximate the seal attraction over the final 4.6 degrees; the door has no motor.
    const seal = clamp(this.q("fridge_hinge") / 0.08, 0, 1);
    this.data.qfrc_applied[this.doorDof] = -1.4 * (1 - seal * seal * (3 - 2 * seal));
    for (const [i, name] of ["kitchen_hinge", "oven_hinge"].entries()) {
      const closing = clamp(this.q(name) / 0.08, 0, 1);
      this.data.qfrc_applied[this.kitchenDofs[i]] = -(i ? 1.2 : 0.5) * (1 - closing * closing * (3 - 2 * closing));
    }
    this.applyControls(); this.mj.mj_step(this.model, this.data); this.readContacts();
    this.learningCapture?.afterStep?.(DT);
    if (p) this.lastAction = { ...action, baseTarget: [...this.baseTarget], handTarget: { frame: "base", position: [...this.target], rotation: targetRotation(this.wristTarget).toArray() } };
    if (++this.actorTicks % 10 === 0) this.task.update(this.observation(), DT * 10);
    if (this.actorTicks % 25 === 0) this.afterStep?.();
    return true;
  }
  private readContacts() {
    const contacts = this.data.contact;
    const left = new Set<number>(), right = new Set<number>();
    this.tableForce = 0; this.collisionForce = 0; this.collisionCount = 0; this.selfContactCount = 0;
    for (let i = 0; i < contacts.size(); i++) {
      const c = contacts.get(i)!;
      if ((c.geom1 === this.tableId && this.model.geom_contype[c.geom2] === 1) ||
          (c.geom2 === this.tableId && this.model.geom_contype[c.geom1] === 1))
      {
        this.maxTablePenetration = Math.max(this.maxTablePenetration, -c.dist);
        this.mj.mj_contactForce(this.model, this.data, i, this.contactForce);
        this.tableForce += Math.max(0, this.contactForce.GetView()[0]);
      }
      const bodyA = this.model.geom_bodyid[c.geom1], bodyB = this.model.geom_bodyid[c.geom2];
      const robotA = this.model.body_rootid[bodyA] === this.baseId, robotB = this.model.body_rootid[bodyB] === this.baseId;
      const fingerContact = this.leftFingerGeoms.has(c.geom1) || this.leftFingerGeoms.has(c.geom2) || this.rightFingerGeoms.has(c.geom1) || this.rightFingerGeoms.has(c.geom2);
      const contactObject = this.objectBodies.get(robotA ? bodyB : bodyA);
      // Opening takes time: remaining contact with the held object is still an intentional release.
      const graspContact = fingerContact && contactObject !== undefined && (this.opening < 0.5 || this.gripperContactId === contactObject);
      if (robotA && robotB) this.selfContactCount++;
      else if ((robotA || robotB) && c.geom1 !== this.floorId && c.geom2 !== this.floorId && !graspContact) {
        this.mj.mj_contactForce(this.model, this.data, i, this.contactForce);
        this.collisionForce += Math.max(0, this.contactForce.GetView()[0]); this.collisionCount++;
      }
      for (const [geoms, touching] of [[this.leftFingerGeoms, left], [this.rightFingerGeoms, right]] as const) {
        const other = geoms.has(c.geom1) ? c.geom2 : geoms.has(c.geom2) ? c.geom1 : -1;
        if (other < 0) continue;
        const body = this.model.geom_bodyid[other];
        if (this.objectBodies.has(body)) touching.add(body);
      }
      c.delete();
    }
    contacts.delete();
    const touched = Array.from(left).find(body => right.has(body));
    this.gripperContactId = touched === undefined ? null : this.objectBodies.get(touched)!;
    this.bilateralContact = this.gripperContactId !== null;
    this.peakTableForce = Math.max(this.peakTableForce, this.tableForce);
    if (this.phase === "接触を保持") { this.holdForceSum += this.tableForce; this.holdForceSteps++; }
  }
  advance(seconds: number) { for (let n = 0; n < Math.round(seconds / DT); n++) if (!this.step()) break; }
  dispose() { this.learningCapture?.stop(); this.controller?.abort?.(); this.afterStep = null; this.contactForce.delete(); this.data.delete(); this.model.delete(); }
}

export async function createSimulation(options: Parameters<typeof loadMujoco>[0] = {}) {
  return new Simulation(await loadMujoco(options));
}
