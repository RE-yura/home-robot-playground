// One collector request. Without HSR_TEACHER_VERSION / HSR_SCENE_VERSION it is exactly the legacy request.
import { LEARNING_SCENE_VERSION } from "../../src/learning/scene.ts";
import type { DemoCollection, Instruction } from "../../src/learning/recording.ts";

const TEACHERS = ["hsr-stationary-teacher-v4", "hsr-mug-gentle-v1", "hsr-mug-gentle-dart-v1", "hsr-mug-gentle-v2", "hsr-mug-gentle-dart-v2", "hsr-mug-gentle-v3", "hsr-mug-gentle-dart-v3", "hsr-mug-gentle-v4", "hsr-mug-gentle-dart-v4"] as const, SCENES = [LEARNING_SCENE_VERSION, "hsr-stationary-v6-recovery"] as const;
export type CollectionRequest = { seed: number; instruction: Instruction; collection: DemoCollection };

export function collectionRequest({ target, seed, env, sourceHead, seedRegistrySha256, episodeId }:
  { target: "mug" | "ball"; seed: number; env: Record<string, string | undefined>; sourceHead: string; seedRegistrySha256: string; episodeId: string }): CollectionRequest {
  const teacherVersion = env.HSR_TEACHER_VERSION ?? "hsr-stationary-teacher-v4", sceneVersion = env.HSR_SCENE_VERSION ?? LEARNING_SCENE_VERSION;
  if (!(TEACHERS as readonly string[]).includes(teacherVersion)) throw new Error(`Unsupported collector teacher ${teacherVersion}`);
  if (!(SCENES as readonly string[]).includes(sceneVersion)) throw new Error(`Unsupported collector scene ${sceneVersion}`);
  return { seed, instruction: target === "mug" ? "Put the mug in the basket." : "Put the tennis ball in the basket.",
    collection: { episodeId, purpose: "teacher-demonstration", sourceCommit: sourceHead, teacherVersion: teacherVersion as DemoCollection["teacherVersion"],
      sceneVersion: sceneVersion as DemoCollection["sceneVersion"], qualityVersion: "hsr-teacher-quality-v2", seedRegistrySha256 } };
}
