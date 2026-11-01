import type {Simulation} from '../simulation/mujoco.ts';
import type {prepareEmptyCarryBallRecovery} from './empty-carry-ball-recovery.ts';
import type {prepareShallowBallRecovery} from './shallow-ball-recovery.ts';
import type {preparePhysicalBallRecovery} from './physical-ball-recovery.ts';
import type {ArmJointTarget} from '../simulation/types.ts';
import {createRandom} from '../simulation/random.ts';
import {prepareLearningScene} from './scene.ts';
import {TeacherQualityAudit} from './quality.ts';
import {ObjectPreservationAudit} from './preservation.ts';
import {validateLearningBasePose} from './recording.ts';
import type {Vec3} from '../simulation/types.ts';

export const RECOVERY_SCENE_VERSION='hsr-stationary-v6-recovery' as const;
export const BALL_RECOVERY_SCENE_VERSION='hsr-stationary-v7-ball-regrasp' as const;

export class RecoveryPreparationError extends Error {
 readonly evidence:{scene:ReturnType<typeof prepareRecoveryScene>|ReturnType<typeof prepareBallRecoveryScene>|ReturnType<typeof preparePhysicalBallRecovery>|ReturnType<typeof prepareEmptyCarryBallRecovery>|ReturnType<typeof prepareShallowBallRecovery>;initial:ReturnType<Simulation['observation']>};
 constructor(message:string,evidence:RecoveryPreparationError['evidence']){super(message);this.evidence=evidence;}
}

export function prepareBallTransportScene(sim:Simulation,seed:number) {
 const scene=prepareLearningScene(sim,seed);sim.advance(2);
 // Reset restores the legacy command mode, so enable this after initialization.
 sim.coordinateJointTargets=true;
 return scene;
}

// Offline world initialization: a synthetic gravity drop, not a recorded grasp loss.
// Placement happens before the first physics step; recovery never repositions an object.
export function prepareBallRecoveryScene(sim:Simulation,seed:number) {
 if(!Number.isInteger(seed)||!((seed>=1000&&seed<=1999)||(seed>=2000&&seed<=2099)))throw new Error('Ball recovery requires reserved collection seeds');
 if(sim.learningCapture)throw new Error('Cannot initialize a ball recovery world during capture');
 prepareLearningScene(sim,seed);sim.coordinateJointTargets=true;
 const random=createRandom(seed^0x42414c4c),initialBall:Vec3=[.015+(random()*2-1)*.008,.54,.44+(random()*2-1)*.015];
 const body=sim.id('BODY','ball'),q=sim.model.jnt_qposadr[sim.model.body_jntadr[body]];
 sim.data.qpos.set(initialBall,q);sim.mj.mj_forward(sim.model,sim.data);
 const initialPenetrations:{other:string;distance:number}[]=[],contacts=sim.data.contact;
 try{for(let i=0;i<contacts.size();i++){
  const c=contacts.get(i)!;
  try{
   const a=sim.model.geom_bodyid[c.geom1],b=sim.model.geom_bodyid[c.geom2];
   if((a===body||b===body)&&c.dist< -1e-6)initialPenetrations.push({other:sim.mj.mj_id2name(sim.model,sim.mj.mjtObj.mjOBJ_BODY.value,a===body?b:a)??'world',distance:c.dist});
  }finally{c.delete();}
 }}finally{contacts.delete();}
 const armJointTarget=[.54,-1.65,-1,-1,.6].map((value,i)=>value+(random()*2-1)*[.01,.02,.04,.03,.04][i])as ArmJointTarget;
 const gripper=.15+random()*.05,quality=new TeacherQualityAudit(sim,'ball'),settlingPreservation=new ObjectPreservationAudit(sim,'ball');
 const settlingMotion=Object.fromEntries(Object.keys(settlingPreservation.metrics.objects).map(id=>[id,{maxHorizontal:0,maxRise:0}]));
 let preservation:ObjectPreservationAudit|null=null;
 const base=sim.observation().robot!.base,startBase:Vec3=[base.position[0],base.position[2],base.yaw];
 const baseAudit={physicsSteps:0,maxTranslation:0,maxYaw:0};let lowestBallY=initialBall[1],failure:string|null=null;
 try{
  if(initialPenetrations.length)throw new Error('Ball recovery initialization overlaps existing geometry');
  for(let i=0;i<2000;i++){
   // Ordinary objects start above their support surfaces. Record their natural
   // world settling separately, before measuring motion caused by arm preparation.
   if(i===1000)preservation=new ObjectPreservationAudit(sim,'ball');
   if(i>=1000&&i%50===0)sim.applyAction({armJointTarget,gripper,base:[0,0,0],head:[0,0]});
   sim.advance(.002);sim.mj.mj_kinematics(sim.model,sim.data);
   quality.update(.002,{phase:i<1000?'world-settling':'preparation',drops:0});
   (preservation??settlingPreservation).update(sim,.002);
   const obs=sim.observation(),b=obs.robot!.base,pose:Vec3=[b.position[0],b.position[2],b.yaw];
   if(i<1000)for(const object of obs.objects){
    const motion=settlingMotion[object.id];if(!motion)continue;
    const start=settlingPreservation.metrics.objects[object.id].start;
    motion.maxHorizontal=Math.max(motion.maxHorizontal,Math.hypot(object.position[0]-start[0],object.position[2]-start[2]));
    motion.maxRise=Math.max(motion.maxRise,object.position[1]-start[1]);
   }
   lowestBallY=Math.min(lowestBallY,obs.objects.find(o=>o.id==='ball')!.position[1]);
   baseAudit.physicsSteps++;baseAudit.maxTranslation=Math.max(baseAudit.maxTranslation,Math.hypot(pose[0]-startBase[0],pose[1]-startBase[1]));
   baseAudit.maxYaw=Math.max(baseAudit.maxYaw,Math.abs(Math.atan2(Math.sin(pose[2]-startBase[2]),Math.cos(pose[2]-startBase[2]))));
   validateLearningBasePose(pose,startBase);
  }
 }catch(error){failure=String(error);}
 finally{quality.dispose();sim.setController(null);}
 const initial=sim.observation(),robot=initial.robot!,ball=initial.objects.find(o=>o.id==='ball')!;
 const scene={id:BALL_RECOVERY_SCENE_VERSION,seed,preparation:{profile:'synthetic-gravity-drop-empty-hand',seconds:4,settlingSeconds:2,initialBall,lowestBallY,initialPenetrations,
  armJointTarget,gripper,measuredOpening:sim.gripperOpening(),baseAudit,quality:quality.metrics,settlingMotion,settlingPreservation:settlingPreservation.metrics,preservation:preservation?.metrics??null}};
 const evidence={scene,initial};
 if(failure)throw new RecoveryPreparationError(failure,evidence);
 for(const [id,audit]of Object.entries(settlingPreservation.metrics.objects)){
  // v7 support planes: table items, the floor baseball, and fridge shelf drinks.
  const minimum=id==='baseball'?0:['cube','can','mug','bottle','plate'].includes(id) ? .4 : audit.start[1]-.03;
  const motion=settlingMotion[id];
  if(motion.maxHorizontal>.01||motion.maxRise>.01||audit.minHeight<minimum||audit.maxDisplacement>Math.hypot(.01,Math.max(.01,audit.start[1]-minimum))+1e-8)throw new RecoveryPreparationError('Ball recovery world settling exceeded object quality limits',evidence);
 }
 if(robot.gripperContact||robot.gripper>=.35||ball.position[1]<.4||initialBall[1]-ball.position[1]<.09||Math.hypot(...ball.velocity)>.03)throw new RecoveryPreparationError('Ball recovery requires a landed ball and empty closed gripper',evidence);
 if(quality.metrics.peakUnintendedForce>=100||quality.metrics.above25Seconds>=.1||[quality.metrics.peakTargetFingerForce,quality.metrics.peakTargetEnvironmentForce].some(f=>!Number.isFinite(f)||f<0||f>=1000)||!preservation||Object.values(preservation.metrics.objects).some(o=>o.maxDisplacement>.01||o.minHeight<o.start[1]-.03))throw new RecoveryPreparationError('Ball recovery preparation exceeded physical quality limits',evidence);
 return scene;
}

// Offline demonstration initialization only; learned execution keeps the v5 scene.
export function prepareRecoveryScene(sim:Simulation,seed:number) {
 if(!Number.isInteger(seed)||!((seed>=1000&&seed<=1999)||(seed>=2000&&seed<=2099)))throw new Error('Recovery requires reserved collection seeds');
 prepareLearningScene(sim,seed);sim.advance(2);
 const random=createRandom(seed^0x5245434f),nearMiss=seed%2===0;
 const centre=nearMiss?[.25,-1.57,-2.05,.35,2.27]:sim.observation().robot!.joints;
 const widths=nearMiss?[.01,.015,.015,.015,.015]:[.025,.1,.06,.06,.06];
 const armJointTarget=centre.map((q,i)=>q+(random()*2-1)*widths[i])as ArmJointTarget;
 const gripper=.15+random()*.1;
 const quality=new TeacherQualityAudit(sim,'mug'),preservation=new ObjectPreservationAudit(sim,'');
 const base=sim.observation().robot!.base,startBase=[base.position[0],base.position[2],base.yaw]as [number,number,number];
 const baseAudit={physicsSteps:0,maxTranslation:0,maxYaw:0};
 try{
  for(let i=0;i<1000;i++){
   if(i%50===0)sim.applyAction({armJointTarget,gripper});
   sim.advance(.002);sim.mj.mj_kinematics(sim.model,sim.data);
   quality.update(.002,{phase:'preparation',drops:0});preservation.update(sim,.002);
   const b=sim.observation().robot!.base,pose=[b.position[0],b.position[2],b.yaw];
   baseAudit.physicsSteps++;baseAudit.maxTranslation=Math.max(baseAudit.maxTranslation,Math.hypot(pose[0]-startBase[0],pose[1]-startBase[1]));
   baseAudit.maxYaw=Math.max(baseAudit.maxYaw,Math.abs(Math.atan2(Math.sin(pose[2]-startBase[2]),Math.cos(pose[2]-startBase[2]))));
   validateLearningBasePose(pose,startBase);
  }
 }finally{quality.dispose();sim.setController(null);}
 sim.mj.mj_kinematics(sim.model,sim.data);
 const robot=sim.observation().robot!;
 validateLearningBasePose([robot.base.position[0],robot.base.position[2],robot.base.yaw],startBase);
 const scene={id:RECOVERY_SCENE_VERSION,seed,preparation:{profile:nearMiss?'near-miss':'home',seconds:2,armJointTarget,gripper,measuredOpening:sim.gripperOpening(),baseAudit,quality:quality.metrics,preservation:preservation.metrics}};
 const evidence={scene,initial:sim.observation()};
 if(robot.gripperContact||sim.gripperOpening()>=.35)throw new RecoveryPreparationError('Recovery requires an empty closed gripper',evidence);
 if(quality.metrics.peakUnintendedForce>=100||quality.metrics.above25Seconds>=.1||[quality.metrics.peakTargetFingerForce,quality.metrics.peakTargetEnvironmentForce].some(f=>!Number.isFinite(f)||f<0||f>=1000)||Object.values(preservation.metrics.objects).some(o=>o.maxDisplacement>.01||o.minHeight<o.start[1]-.03))throw new RecoveryPreparationError('Recovery preparation exceeded physical quality limits',evidence);
 return scene;
}
