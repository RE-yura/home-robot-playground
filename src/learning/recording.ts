import { ARM_NAMES } from "../robot/hsr-model.ts";
import { validateArmJointTarget } from "../controllers/protocol.ts";
import type { CameraImage, Vec3 } from "../simulation/types.ts";

export type SixValues = [number, number, number, number, number, number];
export type Instruction = "Put the mug in the basket." | "Put the tennis ball in the basket.";
export type LearningSample = { index: number; timestamp: number; sourceTimestamp: number;
  state: SixValues; action: SixValues; images: { head: string; hand: string } };
const headFocal = 120 / Math.tan(32 * Math.PI / 180), handFocal = 160 / Math.tan(1);
const MANIFEST = {
  axisNames: [...ARM_NAMES, "gripper_opening"], units: ["m", "rad", "rad", "rad", "rad", "ratio"],
  sampleHz: 10, chunkLength: 16, actionSemantics: "joint-gripper-goal-held-next-interval",
  robotModelId: "hsr-household-6dof-v1",
  normalization: "uncomputed-training-split-only",
  basePolicy: { id: "lerobot/smolvla_base", revision: "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6" },
  processor: { id: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", revision: "7b375e1b73b11138ff12fe22c8f2822d8fe03467" },
  lerobotRevision: "ca69a2068462a37f7cdcb74180927a2f863d2bf7",
  cameras: { width: 320, height: 240, format: "RGB", origin: "top-left", opticalForward: "-Z",
    head: { site: "head_camera", parentLink: "head_tilt_link", position: [-0.079796, 0.022, 0.215172],
      rotationWxyz: [0.5, 0.5, -0.5, -0.5], intrinsics: [headFocal, 0, 160, 0, headFocal, 120, 0, 0, 1] },
    hand: { site: "hand_camera", parentLink: "hand_palm_link", position: [0.039, 0, -0.0045],
      rotationWxyz: [0, Math.SQRT1_2, Math.SQRT1_2, 0], intrinsics: [handFocal, 0, 160, 0, handFocal, 120, 0, 0, 1] } },
};
export type LearningEpisode = { schema: "hsr-learning-v1"; kind: "calibration"; error: string | null; instruction: Instruction; seed: number;
  manifest: typeof MANIFEST; physics: { robotMassKg: number; armTorqueNm: number; fixedTimestep: number };
  startTime: number; startBase: Vec3; duration: number; samples: LearningSample[];
  images: Record<string, CameraImage>; audit: { basePoses: Vec3[] } };
export type DemoCollection = { episodeId: string; purpose: "teacher-demonstration"; sourceCommit: string;
  teacherVersion: "hsr-stationary-teacher-v4" | "hsr-ball-teacher-v1" | "hsr-ball-reseat-v1" | "hsr-mug-gentle-v1" | "hsr-mug-gentle-dart-v1" | "hsr-mug-gentle-v2" | "hsr-mug-gentle-dart-v2" | "hsr-mug-gentle-v3" | "hsr-mug-gentle-dart-v3" | "hsr-mug-gentle-v4" | "hsr-mug-gentle-dart-v4"; sceneVersion: "hsr-stationary-v4" | "hsr-stationary-v5" | "hsr-stationary-v6-recovery" | "hsr-stationary-v7-ball-regrasp" | "hsr-stationary-v8-physical-ball-loss" | "hsr-stationary-v9-empty-carry-loss" | "hsr-stationary-v10-shallow-reseat" | "hsr-stationary-v11-no-reset-sequence" | "hsr-stationary-v12-takeover";
  qualityVersion: "hsr-teacher-quality-v2"; seedRegistrySha256: string };
export type DemonstrationEpisode = Omit<LearningEpisode, "schema" | "kind"> & {
  schema: "hsr-demonstration-v1"; kind: "demonstration"; collection: DemoCollection };
export type CapturedEpisode = LearningEpisode | DemonstrationEpisode;
type Options = { instruction: Instruction; seed: number; basePose: Vec3; robotMassKg: number; armTorqueNm: number };
const MAX_BYTES = 64 * 1024 * 1024, MAX_SAMPLES = 1800, EPS = 1e-8;
const encoder = new TextEncoder();
function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Invalid learning recording: ${message}`);
}
function fields(value: unknown, keys: string[], name: string): asserts value is Record<string, unknown> {
  check(value !== null && typeof value === "object" && !Array.isArray(value), name);
  check(Object.keys(value).every(key => keys.includes(key)) && keys.every(key => key in value), `${name} schema`);
}
function finite(value: unknown): asserts value is number { check(typeof value === "number" && Number.isFinite(value), "finite number"); }
function vector(value: unknown, length: number): asserts value is number[] {
  check(Array.isArray(value) && value.length === length, "channel count"); Array.from(value).forEach(finite);
}
function six(value: unknown, action: boolean): asserts value is SixValues {
  vector(value, 6); check(value[5] >= 0 && value[5] <= 1, "gripper range");
  if (action) validateArmJointTarget(value.slice(0, 5));
}
export function validateLearningBasePose(base: unknown, start: Vec3) {
  vector(base, 3); const angle = Math.atan2(Math.sin(base[2] - start[2]), Math.cos(base[2] - start[2]));
  check(Math.hypot(base[0] - start[0], base[1] - start[1]) <= 0.02 + EPS && Math.abs(angle) <= 0.03 + EPS, "base drift");
}
function image(value: unknown, view: string, time: number): asserts value is CameraImage {
  fields(value, ["view", "timestamp", "width", "height", "dataUrl"], "image"); finite(value.timestamp);
  check(value.view === view && Math.abs(value.timestamp - time) < EPS, "image view/time alignment");
  check(value.width === 320 && value.height === 240, "image dimensions");
  check(typeof value.dataUrl === "string" && value.dataUrl.length <= 4 * 1024 * 1024 &&
    /^data:image\/jpeg;base64,[A-Za-z0-9+/]+={0,2}$/.test(value.dataUrl), "image data");
}
function matches(value: unknown, expected: unknown): boolean {
  if (expected === null || typeof expected !== "object") return value === expected;
  if (value === null || typeof value !== "object" || Array.isArray(value) !== Array.isArray(expected)) return false;
  const a = value as Record<string, unknown>, b = expected as Record<string, unknown>;
  return Object.keys(a).length === Object.keys(b).length && Object.keys(b).every(key => key in a && matches(a[key], b[key]));
}
const keys = (index: number) => ({ head: `head/${String(index).padStart(6, "0")}.jpg`, hand: `hand/${String(index).padStart(6, "0")}.jpg` });

export const takeoverSeed = (seed: number) => (seed >= 1150 && seed <= 1199) || (seed >= 1600 && seed <= 1999);
export function validateDemoCollection(collection: unknown, seed: unknown): asserts collection is DemoCollection {
  fields(collection, ["episodeId", "purpose", "sourceCommit", "teacherVersion", "sceneVersion", "qualityVersion", "seedRegistrySha256"], "collection");
  const c = collection;
  check(typeof c.episodeId === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(c.episodeId), "collection identity");
  check(c.purpose === "teacher-demonstration", "collection purpose");
  check(typeof c.sourceCommit === "string" && /^[0-9a-f]{40}$/.test(c.sourceCommit) && typeof c.seedRegistrySha256 === "string" && /^[0-9a-f]{64}$/.test(c.seedRegistrySha256), "collection provenance");
  const legacy = c.teacherVersion === "hsr-stationary-teacher-v4" && ["hsr-stationary-v4", "hsr-stationary-v5", "hsr-stationary-v6-recovery", "hsr-stationary-v11-no-reset-sequence"].some(version=>c.sceneVersion===version);
  const ball = c.teacherVersion === "hsr-ball-teacher-v1" && ["hsr-stationary-v5", "hsr-stationary-v7-ball-regrasp", "hsr-stationary-v8-physical-ball-loss", "hsr-stationary-v9-empty-carry-loss"].some(version=>c.sceneVersion===version);
  const reseat = c.teacherVersion === "hsr-ball-reseat-v1" && c.sceneVersion === "hsr-stationary-v10-shallow-reseat";
  // v12: a teacher segment that takes over from the learned policy in continuous execution, on train seeds never used
  // for the mug before (1150-1199, 1600-1999).
  const gentle = (c.teacherVersion === "hsr-mug-gentle-v1" && ["hsr-stationary-v5", "hsr-stationary-v6-recovery", "hsr-stationary-v12-takeover"].some(version=>c.sceneVersion===version)) ||
    ((c.teacherVersion === "hsr-mug-gentle-v2" || c.teacherVersion === "hsr-mug-gentle-v3" || c.teacherVersion === "hsr-mug-gentle-v4") && ["hsr-stationary-v5", "hsr-stationary-v6-recovery"].some(version=>c.sceneVersion===version));
  // DART: the gentle mug teacher with noise-injected execution and clean labels, normal scene, train seeds only (it may
  // repeat seeds of the plain gentle collection; the scene differs only by the executed noise).
  const dart = (c.teacherVersion === "hsr-mug-gentle-dart-v1" || c.teacherVersion === "hsr-mug-gentle-dart-v2" || c.teacherVersion === "hsr-mug-gentle-dart-v3" || c.teacherVersion === "hsr-mug-gentle-dart-v4") && c.sceneVersion === "hsr-stationary-v5";
  check((legacy || ball || reseat || gentle || dart) && c.qualityVersion === "hsr-teacher-quality-v2", "collection version");
  check(typeof seed === "number" && Number.isInteger(seed) && ((seed >= 1000 && seed <= 1999) || (seed >= 2000 && seed <= 2099)), "reserved collection seed");
  if (c.sceneVersion === "hsr-stationary-v12-takeover") check(takeoverSeed(seed as number), "takeover seed");
  if (dart) check((seed as number) >= 1000 && (seed as number) <= 1999, "DART seed");
}

function validatePayload(value: unknown, demonstration: boolean): asserts value is CapturedEpisode {
  fields(value, ["schema", "kind", "error", "instruction", "seed", "manifest", "physics", "startTime", "startBase", "duration", "samples", "images", "audit", ...(demonstration ? ["collection"] : [])], "episode");
  check(value.schema === (demonstration ? "hsr-demonstration-v1" : "hsr-learning-v1") && value.kind === (demonstration ? "demonstration" : "calibration"), "version/kind");
  if (demonstration) validateDemoCollection(value.collection, value.seed);
  check(value.error === null || (typeof value.error === "string" && value.error.length > 0 && value.error.length <= 1024), "error");
  check(value.instruction === "Put the mug in the basket." || value.instruction === "Put the tennis ball in the basket.", "instruction");
  if (demonstration && ["hsr-ball-teacher-v1", "hsr-ball-reseat-v1"].includes((value.collection as DemoCollection).teacherVersion)) check(value.instruction === "Put the tennis ball in the basket.", "ball teacher instruction");
  if (demonstration && ["hsr-mug-gentle-v1", "hsr-mug-gentle-dart-v1", "hsr-mug-gentle-v2", "hsr-mug-gentle-dart-v2", "hsr-mug-gentle-v3", "hsr-mug-gentle-dart-v3", "hsr-mug-gentle-v4", "hsr-mug-gentle-dart-v4"].includes((value.collection as DemoCollection).teacherVersion)) check(value.instruction === "Put the mug in the basket.", "mug teacher instruction");
  finite(value.seed); check(Number.isInteger(value.seed) && value.seed >= 0 && value.seed <= 0xffffffff, "seed");
  check(matches(value.manifest, MANIFEST), "manifest units/calibration/revision");
  fields(value.physics, ["robotMassKg", "armTorqueNm", "fixedTimestep"], "physics");
  for (const key of ["robotMassKg", "armTorqueNm"]) { finite(value.physics[key]); check(value.physics[key] > 0 && value.physics[key] <= 1000, "physics bounds"); }
  check(value.physics.fixedTimestep === 0.002, "physics timestep");
  vector(value.startBase, 3); finite(value.startTime); check(value.startTime >= 0, "start time");
  finite(value.duration);
  check(Array.isArray(value.samples) && value.samples.length <= MAX_SAMPLES, "sample duration limit");
  check(Math.abs(value.duration - value.samples.length / 10) < EPS, "duration");
  fields(value.audit, ["basePoses"], "audit"); check(Array.isArray(value.audit.basePoses) && value.audit.basePoses.length === value.samples.length, "base audit");
  const allKeys = value.samples.flatMap((_: unknown, i: number) => Object.values(keys(i)));
  fields(value.images, allKeys, "images");
  const { startTime, images, startBase } = value, basePoses = value.audit.basePoses as unknown[];
  value.samples.forEach((sample: unknown, i: number) => {
    fields(sample, ["index", "timestamp", "sourceTimestamp", "state", "action", "images"], "sample");
    check(sample.index === i, "sample index"); finite(sample.timestamp); finite(sample.sourceTimestamp);
    check(Math.abs(sample.timestamp - i / 10) < EPS && Math.abs(sample.sourceTimestamp - startTime - i / 10) < EPS, "sample time cadence");
    six(sample.state, false); six(sample.action, true); fields(sample.images, ["head", "hand"], "sample images");
    for (const view of ["head", "hand"] as const) {
      const key = keys(i)[view]; check(sample.images[view] === key, "image reference"); image(images[key], view, sample.sourceTimestamp);
    }
    validateLearningBasePose(basePoses[i], startBase as Vec3);
  });
  check(encoder.encode(JSON.stringify(value)).byteLength <= MAX_BYTES, "size limit");
}

export function validateLearningEpisode(value: unknown): asserts value is LearningEpisode { validatePayload(value, false); }
export function validateDemonstrationEpisode(value: unknown): asserts value is DemonstrationEpisode { validatePayload(value, true); }

class RecordingSession {
  private episode: CapturedEpisode;
  private pending: { sample: LearningSample; images: CameraImage[]; base: Vec3; bytes: number } | null = null;
  private closed = false;
  private bytes = 4096;
  private started = false;
  constructor(options: Options & { collection?: DemoCollection }) {
    const calibration: LearningEpisode = { schema: "hsr-learning-v1", kind: "calibration", error: null, instruction: options.instruction, seed: options.seed,
      manifest: structuredClone(MANIFEST), physics: { robotMassKg: options.robotMassKg, armTorqueNm: options.armTorqueNm, fixedTimestep: 0.002 },
      startTime: 0, startBase: [...options.basePose], duration: 0, samples: [], images: {}, audit: { basePoses: [] } };
    this.episode = options.collection ? { ...calibration, schema: "hsr-demonstration-v1", kind: "demonstration", collection: structuredClone(options.collection) } : calibration;
    validatePayload(this.episode, this.episode.kind === "demonstration");
  }
  begin(time: number, state: SixValues, action: SixValues, images: CameraImage[], basePose: Vec3) {
    check(!this.closed, "closed"); check(!this.pending, "pending interval"); finite(time); check(time >= 0, "time");
    const index = this.episode.samples.length, start = this.started ? this.episode.startTime : time;
    check(index < MAX_SAMPLES, "duration limit"); check(Math.abs(time - start - index / 10) < EPS, "time cadence");
    six(state, false); six(action, true); validateLearningBasePose(basePose, this.episode.startBase);
    check(Array.isArray(images) && images.length === 2, "image pair");
    const pair = [images.find(v => v.view === "head"), images.find(v => v.view === "hand")];
    image(pair[0], "head", time); image(pair[1], "hand", time);
    const sample: LearningSample = { index, timestamp: index / 10, sourceTimestamp: time, state: [...state], action: [...action], images: keys(index) };
    const bytes = encoder.encode(JSON.stringify({ sample, pair, basePose })).byteLength + 512;
    check(this.bytes + bytes <= MAX_BYTES, "size limit");
    this.pending = { sample, images: structuredClone(pair) as CameraImage[], base: [...basePose], bytes };
    this.episode.startTime = start; this.started = true;
  }
  complete(time: number) {
    check(!this.closed, "closed"); check(this.pending, "no pending interval"); finite(time);
    check(Math.abs(time - this.pending.sample.sourceTimestamp - 0.1) < EPS, "interval completion time");
    const { sample, images, base, bytes } = this.pending;
    this.episode.samples.push(sample); this.episode.audit.basePoses.push(base);
    for (const image of images) this.episode.images[sample.images[image.view]] = image;
    this.bytes += bytes; this.episode.duration = this.episode.samples.length / 10; this.pending = null;
  }
  finish(error: string | null = null): CapturedEpisode {
    if (!this.closed) this.episode.error = error;
    this.closed = true; this.pending = null; validatePayload(this.episode, this.episode.kind === "demonstration"); return structuredClone(this.episode);
  }
}

// Both APIs create a new session; neither accepts an existing episode to relabel.
export class LearningRecorder extends RecordingSession {
  constructor(options: Options) { super({ ...options, collection: undefined }); }
  override finish(error: string | null = null): LearningEpisode { const value = super.finish(error); validateLearningEpisode(value); return value; }
}
export class DemonstrationRecorder extends RecordingSession {
  constructor(options: Options & { collection: DemoCollection }) { super(options); }
  override finish(error: string | null = null): DemonstrationEpisode { const value = super.finish(error); validateDemonstrationEpisode(value); return value; }
}
