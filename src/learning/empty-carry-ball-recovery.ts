// Offline pose coverage only. Preparation is excluded from captured policy labels.
import {Vector3} from 'three';
import type {Simulation} from '../simulation/mujoco.ts';
import type {ArmJointTarget,Vec3,Observation} from '../simulation/types.ts';
import {createRandom} from '../simulation/random.ts';
import {solveHSR} from '../robot/hsr-ik.ts';
import {TABLE_HEIGHT} from '../simulation/table.ts';
import {preparePhysicalBallRecovery} from './physical-ball-recovery.ts';
import {RecoveryPreparationError} from './recovery-scene.ts';
import {TeacherQualityAudit} from './quality.ts';
import {ObjectPreservationAudit} from './preservation.ts';
import {validateLearningBasePose} from './recording.ts';
export const EMPTY_CARRY_BALL_RECOVERY_SCENE_VERSION='hsr-stationary-v9-empty-carry-loss' as const;
export function emptyCarryPose(seed:number,robot:NonNullable<Observation['robot']>){
 const base=robot.base,rng=createRandom((seed^0x73a4c91d)>>>0),desiredWorldHand:Vec3=[.02+.16*rng(),.74+.12*rng(),.44+.05*rng()],fraction=.2+.8*rng(),opening=seed%2;
 const startJoints=[...robot.joints]as ArmJointTarget;
 const local=new Vector3(...desiredWorldHand).sub(new Vector3(...base.position)).applyAxisAngle(new Vector3(0,1,0),-base.yaw).toArray()as Vec3;
 let solved=[...startJoints]as ArmJointTarget;for(let k=0;k<5;k++)solved=solveHSR(local,[-base.yaw,-.9,0],solved)as ArmJointTarget;
 const goalJoints=startJoints.map((q,i)=>q+(solved[i]-q)*fraction)as ArmJointTarget;
 return {desiredWorldHand,fraction,opening,startJoints,goalJoints};
}
export function prepareEmptyCarryBallRecovery(sim:Simulation,seed:number){
 if(!Number.isInteger(seed)||!((seed>=1000&&seed<=1999)||(seed>=2000&&seed<=2099)))throw new Error('Empty carry requires reserved collection seeds');
 if(sim.learningCapture)throw new Error('Cannot initialize empty carry during capture');
 let physicalLoss:ReturnType<typeof preparePhysicalBallRecovery>|null=null,physicalInitial:ReturnType<Simulation['observation']>|null=null;
 let emptyCarry:{desiredWorldHand:Vec3;fraction:number;opening:number;startJoints:ArmJointTarget;goalJoints:ArmJointTarget;seconds:number;tableSupported:boolean;maxTargetDisplacement:number;quality:TeacherQualityAudit['metrics'];preservation:ObjectPreservationAudit['metrics'];baseAudit:{reference:Vec3;physicsSteps:number;maxTranslation:number;maxYaw:number};trace:{seconds:number;state:number[];goal:number[];ball:Vec3;velocity:Vec3;held:boolean}[]}|null=null;
 let failure:string|null=null;
 try{
  try{physicalLoss=preparePhysicalBallRecovery(sim,seed);physicalInitial=sim.observation();}
  catch(error){if(error instanceof RecoveryPreparationError){physicalLoss=error.evidence.scene as ReturnType<typeof preparePhysicalBallRecovery>;physicalInitial=error.evidence.initial;}throw error;}
  const initial=physicalInitial,robot=initial.robot!,originBase=physicalLoss.preparation.origin.robot!.base,startBase:Vec3=[originBase.position[0],originBase.position[2],originBase.yaw],ballStart=initial.objects.find(o=>o.id==='ball')!.position;
  const {desiredWorldHand,fraction,opening,startJoints,goalJoints}=emptyCarryPose(seed,robot);
  const quality=new TeacherQualityAudit(sim,'ball'),preservation=new ObjectPreservationAudit(sim,'ball');
  emptyCarry={desiredWorldHand,fraction,opening,startJoints,goalJoints,seconds:0,tableSupported:false,maxTargetDisplacement:0,quality:quality.metrics,preservation:preservation.metrics,baseAudit:{reference:[...startBase],physicsSteps:0,maxTranslation:0,maxYaw:0},trace:[]};
  try{
   for(let n=0;n<1500;n++){
    const u=(n+1)/1500,s=u*u*(3-2*u),joints=startJoints.map((q,i)=>q+(goalJoints[i]-q)*s)as ArmJointTarget;
    sim.applyAction({armJointTarget:joints,gripper:opening,base:[0,0,0],head:[0,0]});sim.step();sim.mj.mj_kinematics(sim.model,sim.data);
    const obs=sim.observation(),b=obs.robot!.base,ball=obs.objects.find(o=>o.id==='ball')!;
    quality.update(.002,{phase:'empty-carry-preparation',drops:0});preservation.update(sim,.002);
    emptyCarry.seconds=(n+1)*.002;emptyCarry.baseAudit.physicsSteps++;
    emptyCarry.baseAudit.maxTranslation=Math.max(emptyCarry.baseAudit.maxTranslation,Math.hypot(b.position[0]-startBase[0],b.position[2]-startBase[1]));
    emptyCarry.baseAudit.maxYaw=Math.max(emptyCarry.baseAudit.maxYaw,Math.abs(Math.atan2(Math.sin(b.yaw-startBase[2]),Math.cos(b.yaw-startBase[2]))));
    emptyCarry.maxTargetDisplacement=Math.max(emptyCarry.maxTargetDisplacement,Math.hypot(...ball.position.map((v,i)=>v-ballStart[i])));
    if(n%50===0||n===1499)emptyCarry.trace.push({seconds:(n+1)*.002,state:[...obs.robot!.joints,obs.robot!.gripper],goal:[...joints,opening],ball:[...ball.position],velocity:[...ball.velocity],held:obs.robot!.gripperContact});
    validateLearningBasePose([b.position[0],b.position[2],b.yaw],startBase);
    if(obs.robot!.gripperContact||emptyCarry.maxTargetDisplacement>.01||ball.position[1]<.4||quality.metrics.peakUnintendedForce>=100||quality.metrics.above25Seconds>=.1||[quality.metrics.peakTargetFingerForce,quality.metrics.peakTargetEnvironmentForce].some(v=>!Number.isFinite(v)||v>=1000)||Object.values(preservation.metrics.objects).some(o=>o.maxDisplacement>.01||o.minHeight<o.start[1]-.03))throw new Error('Empty carry preparation violated unchanged contact/preservation limits');
   }
   const contacts=sim.data.contact,body=sim.id('BODY','ball');
   try{for(let i=0;i<contacts.size();i++){const c=contacts.get(i)!;try{if(c.dist<=.003&&((c.geom1===sim.tableId&&sim.model.geom_bodyid[c.geom2]===body)||(c.geom2===sim.tableId&&sim.model.geom_bodyid[c.geom1]===body)))emptyCarry.tableSupported=true;}finally{c.delete();}}}finally{contacts.delete();}
   const ball=sim.observation().objects.find(o=>o.id==='ball')!;
   if(Math.abs(sim.observation().robot!.gripper-opening)>.03||!emptyCarry.tableSupported||Math.abs(ball.position[1]-(TABLE_HEIGHT+.0335))>.003||Math.hypot(...ball.velocity)>.003)throw new Error('Empty carry target is not stationary on the tabletop');
  }finally{quality.dispose();sim.setController(null);}
 }catch(error){failure=String(error);}
 const scene={id:EMPTY_CARRY_BALL_RECOVERY_SCENE_VERSION,seed,physicalLoss,physicalInitial,emptyCarry};
 if(failure)throw new RecoveryPreparationError(failure,{scene,initial:sim.observation()});
 return scene;
}
