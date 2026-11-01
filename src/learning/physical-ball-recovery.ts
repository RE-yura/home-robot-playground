// Offline preparation only. No object is moved after the ordinary v5 initialization.
import type { Simulation } from '../simulation/mujoco.ts';
import type { ArmJointTarget, Vec3 } from '../simulation/types.ts';
import { prepareBallTransportScene, RecoveryPreparationError } from './recovery-scene.ts';
import { StationaryTeacher } from './teacher.ts';
import { TeacherQualityAudit } from './quality.ts';
import { ObjectPreservationAudit } from './preservation.ts';
import { validateLearningBasePose } from './recording.ts';
import { TABLE_HEIGHT } from '../simulation/table.ts';

export const PHYSICAL_BALL_RECOVERY_SCENE_VERSION = "hsr-stationary-v8-physical-ball-loss" as const;

export function preparePhysicalBallRecovery(sim: Simulation, seed: number) {
 if (!Number.isInteger(seed) || !((seed >= 1000 && seed <= 1999) || (seed >= 2000 && seed <= 2099))) throw new Error('Physical recovery requires reserved collection seeds');
 if (sim.learningCapture) throw new Error('Cannot initialize physical recovery during capture');
 prepareBallTransportScene(sim, seed);
 const origin = sim.observation(), b = origin.robot!.base, startBase: Vec3 = [b.position[0], b.position[2], b.yaw];
 const teacher = new StationaryTeacher(sim, 'Put the tennis ball in the basket.', 'stable-ball');
 const quality = new TeacherQualityAudit(sim, 'ball'), preservation = new ObjectPreservationAudit(sim, 'ball');
 const releaseX = seed % 2 === 0 ? -.01 : .01;
 let goal = { armJointTarget: [...origin.robot!.joints] as ArmJointTarget, gripper: 1 };
 let heldFor = 0, triggeredAt: number | null = null, openedAt: number | null = null, restoredAt: number | null = null, lostAt: number | null = null;
 let heldBeforeOpeningSeconds = 0, stableSeconds = 0, tableSupported = false, failure: string | null = null;
 const ballBody = sim.id('BODY', 'ball');
 const trace: { seconds: number; phase: string; held: boolean; tableSupported: boolean; ball: Vec3; velocity: Vec3; goal: number[] }[] = [];
 const baseAudit = { physicsSteps: 0, maxTranslation: 0, maxYaw: 0 };
 try {
  for (let n = 0; n < 20000; n++) {
   const elapsed = n * .002, obs = sim.observation(), ball = obs.objects.find(o => o.id === 'ball')!;
   const held = obs.robot!.gripperContact && obs.robot!.contactId === 'ball';
   heldFor = held ? heldFor + .002 : 0;
   if (triggeredAt === null && heldFor >= .2 && ball.position[1] >= .65 && ball.position[0] >= releaseX) triggeredAt = elapsed;
   let phase = 'teacher-grasp-lift';
   if (triggeredAt === null) {
    if (n % 50 === 0) goal = teacher.next(obs);
    if (teacher.result.status !== 'running') throw new Error(`Preparation teacher failed: ${teacher.result.teacherFailure ?? teacher.result.reason}`);
   } else {
    const openingTime = triggeredAt + 2, restorationTime = openingTime + .3;
    if (elapsed < openingTime - 1e-8) {
     phase = 'held-settling';
     if (!held) throw new Error('Lost grasp before disclosed opening');
     goal.gripper = 0;
    } else if (elapsed < restorationTime - 1e-8) {
     phase = 'opening'; openedAt ??= elapsed; heldBeforeOpeningSeconds = Math.max(heldBeforeOpeningSeconds, heldFor);
     goal.gripper = 1;
    } else {
     phase = 'landing'; restoredAt ??= elapsed; goal.gripper = 0;
    }
   }
   sim.applyAction({ ...goal, base: [0, 0, 0], head: [0, 0] }); sim.step(); sim.mj.mj_kinematics(sim.model, sim.data);
   const after = sim.observation(), a = after.robot!.base, target = after.objects.find(o => o.id === 'ball')!;
   const nowHeld = after.robot!.gripperContact && after.robot!.contactId === 'ball';
   tableSupported = false;
   const contacts = sim.data.contact;
   try { for (let i = 0; i < contacts.size(); i++) {
    const contact = contacts.get(i)!;
    try { if (contact.dist <= .003 && ((contact.geom1 === sim.tableId && sim.model.geom_bodyid[contact.geom2] === ballBody) ||
     (contact.geom2 === sim.tableId && sim.model.geom_bodyid[contact.geom1] === ballBody))) tableSupported = true; }
    finally { contact.delete(); }
   } } finally { contacts.delete(); }
   if (openedAt !== null && held && !nowHeld && lostAt === null) lostAt = elapsed + .002;
   quality.update(.002, { phase, drops: 0 }); preservation.update(sim, .002);
   baseAudit.physicsSteps++; baseAudit.maxTranslation = Math.max(baseAudit.maxTranslation, Math.hypot(a.position[0] - startBase[0], a.position[2] - startBase[1]));
   baseAudit.maxYaw = Math.max(baseAudit.maxYaw, Math.abs(Math.atan2(Math.sin(a.yaw - startBase[2]), Math.cos(a.yaw - startBase[2]))));
   validateLearningBasePose([a.position[0], a.position[2], a.yaw], startBase);
   if (triggeredAt === null) teacher.evaluator.update(after, .002);
   if (n % 50 === 0) trace.push({ seconds: elapsed + .002, phase, held: nowHeld, tableSupported, ball: [...target.position], velocity: [...target.velocity], goal: [...goal.armJointTarget, goal.gripper] });
   if (target.position[1] < .4) throw new Error('Physical drop fell off the table');
   stableSeconds = restoredAt !== null && lostAt !== null && tableSupported && Math.abs(target.position[1] - (TABLE_HEIGHT + .0335)) <= .003 && !nowHeld && sim.gripperOpening() < .35 && Math.hypot(...target.velocity) <= .003 ? stableSeconds + .002 : 0;
   if (stableSeconds >= .3 - 1e-8) break;
   if (restoredAt !== null && elapsed > restoredAt + 8) throw new Error('Physical drop did not settle within the declared budget');
  }
  if (openedAt === null || restoredAt === null || lostAt === null || heldBeforeOpeningSeconds < .2 || stableSeconds < .3 - 1e-8) throw new Error('Physical grasp-loss preparation incomplete');
  if (quality.metrics.peakUnintendedForce >= 100 || quality.metrics.above25Seconds >= .1 || [quality.metrics.peakTargetFingerForce, quality.metrics.peakTargetEnvironmentForce].some(v => !Number.isFinite(v) || v >= 1000) || Object.values(preservation.metrics.objects).some(o => o.maxDisplacement > .01 || o.minHeight < o.start[1] - .03)) throw new Error('Physical drop preparation exceeded unchanged quality limits');
 } catch (error) { failure = String(error); }
 finally { quality.dispose(); sim.setController(null); }
 const initial = sim.observation();
 const scene = { id: PHYSICAL_BALL_RECOVERY_SCENE_VERSION, seed, preparation: { profile: 'actual-teacher-grasp-release' as const, origin, seconds: baseAudit.physicsSteps * .002, releaseX, triggeredAt, openedAt, restoredAt, lostAt, heldBeforeOpeningSeconds, stableSeconds, tableSupported, measuredOpening: sim.gripperOpening(), baseAudit, quality: quality.metrics, preservation: preservation.metrics, trace } };
 if (failure) throw new RecoveryPreparationError(failure, { scene, initial });
 return scene;
}
