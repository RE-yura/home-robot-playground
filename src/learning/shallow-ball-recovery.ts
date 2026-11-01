import {Vector3} from 'three';
import type {Simulation} from '../simulation/mujoco.ts';
import type {ArmJointTarget,Vec3} from '../simulation/types.ts';
import {createRandom} from '../simulation/random.ts';
import {solveHSR} from '../robot/hsr-ik.ts';
import {prepareLearningScene} from './scene.ts';
import {RecoveryPreparationError} from './recovery-scene.ts';
import {TeacherQualityAudit} from './quality.ts';
import {ObjectPreservationAudit} from './preservation.ts';
import {validateLearningBasePose} from './recording.ts';

export const SHALLOW_BALL_RECOVERY_SCENE_VERSION='hsr-stationary-v10-shallow-reseat' as const;
export function shallowRecoveryPose(seed:number,ball:Vec3,yaw:number) {
 const random=createRandom((seed^0x5348414c)>>>0);
 return {position:[ball[0],ball[1]+.025+.02*random(),ball[2]] as Vec3,
  attitude:[-yaw+(random()*2-1)*.1,-.8+.15*random(),(random()*2-1)*.1] as Vec3,gripper:0};
}

// Offline initialization only. No object is moved after the first physics step.
export function prepareShallowBallRecovery(sim:Simulation,seed:number,purpose:'teacher-demonstration'|'calibration'='teacher-demonstration') {
 if(purpose==='calibration'){
  if(seed!==7&&seed!==42)throw Error('Shallow preparation requires registered calibration seeds');
 }else if(purpose!=='teacher-demonstration'||!Number.isInteger(seed)||!((seed>=1000&&seed<=1999)||(seed>=2000&&seed<=2099)))throw Error('Shallow recovery requires reserved collection seeds');
 if(sim.learningCapture)throw Error('Cannot initialize shallow recovery during capture');
 prepareLearningScene(sim,seed);
 const random=createRandom(seed^0x42414c4c),initialBall:Vec3=[.015+(random()*2-1)*.008,.54,.44+(random()*2-1)*.015];
 const body=sim.id('BODY','ball'),q=sim.model.jnt_qposadr[sim.model.body_jntadr[body]];
 sim.data.qpos.set(initialBall,q);sim.mj.mj_forward(sim.model,sim.data);
 const initialPenetrations:{other:string;distance:number}[]=[],contacts=sim.data.contact;
 try{for(let i=0;i<contacts.size();i++){
  const c=contacts.get(i)!;
  try{const a=sim.model.geom_bodyid[c.geom1],b=sim.model.geom_bodyid[c.geom2];
   if((a===body||b===body)&&c.dist< -1e-6)initialPenetrations.push({other:sim.mj.mj_id2name(sim.model,sim.mj.mjtObj.mjOBJ_BODY.value,a===body?b:a)??'world',distance:c.dist});
  }finally{c.delete();}
 }}finally{contacts.delete();}
 const settlingQuality=new TeacherQualityAudit(sim,'ball'),settlingPreservation=new ObjectPreservationAudit(sim,'ball');
 const settlingMotion=Object.fromEntries(Object.keys(settlingPreservation.metrics.objects).map(id=>[id,{maxHorizontal:0,maxRise:0}]));
 const base=sim.observation().robot!.base,startBase:Vec3=[base.position[0],base.position[2],base.yaw];
 const baseAudit={reference:startBase,physicsSteps:0,maxTranslation:0,maxYaw:0};
 const trace:{time:number;phase:string;base:{position:Vec3;yaw:number};held:boolean;heldFor:number;opening:number;goal:number[];joints:number[];target:Vec3;velocity:Vec3;tcp:Vec3}[]=[];
 const moves:{phase:string;position:Vec3;seconds:number;base:{position:Vec3;yaw:number};start:ArmJointTarget;end:ArmJointTarget}[]=[];
 let quality:TeacherQualityAudit|null=null,preservation:ObjectPreservationAudit|null=null,failure:string|null=null;
 let groundHeight=initialBall[1],preparationPeakHeight=groundHeight,heldFor=0,prepSteps=0;
 let pose=shallowRecoveryPose(seed,initialBall,base.yaw),closingJoints:ArmJointTarget|null=null;
 let beforePreparation=sim.observation();
 const auditBase=()=>{
  const b=sim.observation().robot!.base,pose:Vec3=[b.position[0],b.position[2],b.yaw];
  baseAudit.physicsSteps++;baseAudit.maxTranslation=Math.max(baseAudit.maxTranslation,Math.hypot(pose[0]-startBase[0],pose[1]-startBase[1]));
  baseAudit.maxYaw=Math.max(baseAudit.maxYaw,Math.abs(Math.atan2(Math.sin(pose[2]-startBase[2]),Math.cos(pose[2]-startBase[2]))));
  validateLearningBasePose(pose,startBase);
 };
 const checkQuality=()=>{
  const metrics=[settlingQuality.metrics,...(quality?[quality.metrics]:[])];
  if(metrics.some(m=>!Number.isFinite(m.peakUnintendedForce)||m.peakUnintendedForce>=100||
   [m.peakTargetFingerForce,m.peakTargetEnvironmentForce].some(f=>!Number.isFinite(f)||f<0||f>=1000))||
   metrics.reduce((sum,m)=>sum+m.above25Seconds,0)>=.1)throw Error('Shallow preparation exceeded physical force limits');
 };
 try{
  if(initialPenetrations.length)throw Error('Shallow initialization overlaps existing geometry');
  // Preserve the original probe's ordinary settling order: no post-step
  // kinematics refresh until2s. These audits never write simulator state.
  for(let i=0;i<1000;i++){
   if(!sim.step())throw Error('Shallow settling was interrupted');
   settlingQuality.update(.002,{phase:'world-settling',drops:0});settlingPreservation.update(sim,.002);auditBase();checkQuality();
   for(const object of sim.observation().objects){
    const motion=settlingMotion[object.id];if(!motion)continue;
    const start=settlingPreservation.metrics.objects[object.id].start;
    motion.maxHorizontal=Math.max(motion.maxHorizontal,Math.hypot(object.position[0]-start[0],object.position[2]-start[2]));
    motion.maxRise=Math.max(motion.maxRise,object.position[1]-start[1]);
   }
  }
  sim.coordinateJointTargets=true;sim.mj.mj_kinematics(sim.model,sim.data);
  beforePreparation=sim.observation();const ball=beforePreparation.objects.find(o=>o.id==='ball')!;
  if(Math.abs(ball.position[1]-.4335)>.003||Math.hypot(...ball.velocity)>.003)throw Error('Shallow preparation requires a stationary ball on the table');
  groundHeight=preparationPeakHeight=ball.position[1];pose=shallowRecoveryPose(seed,ball.position,beforePreparation.robot!.base.yaw);
  quality=new TeacherQualityAudit(sim,'ball');preservation=new ObjectPreservationAudit(sim,'ball');
  const step=(phase:string,armJointTarget:ArmJointTarget,gripper:number)=>{
   sim.applyAction({armJointTarget,gripper,base:[0,0,0],head:[0,0]});
   if(!sim.step())throw Error('Shallow preparation was interrupted');
   sim.mj.mj_kinematics(sim.model,sim.data);prepSteps++;
   const obs=sim.observation(),robot=obs.robot!,ball=obs.objects.find(o=>o.id==='ball')!,held=robot.gripperContact&&robot.contactId==='ball';
   heldFor=held?heldFor+.002:0;preparationPeakHeight=Math.max(preparationPeakHeight,ball.position[1]);
   quality!.update(.002,{phase,drops:0});preservation!.update(sim,.002);auditBase();checkQuality();
   if(ball.position[1]<.4||Object.values(preservation!.metrics.objects).some(o=>o.maxDisplacement>.01||o.minHeight<o.start[1]-.03))throw Error('Shallow preparation displaced an object');
   if(prepSteps%50===0)trace.push({time:obs.timestamp,phase,base:{position:[...robot.base.position],yaw:robot.base.yaw},held,heldFor,opening:robot.gripper,goal:[...armJointTarget,gripper],joints:[...robot.joints],target:[...ball.position],velocity:[...ball.velocity],tcp:[...robot.endEffectorPosition]});
  };
  const move=(phase:string,position:Vec3,seconds:number)=>{
   const robot=sim.observation().robot!,start=[...robot.joints] as ArmJointTarget;
   const local=new Vector3(...position).sub(new Vector3(...robot.base.position)).applyAxisAngle(new Vector3(0,1,0),-robot.base.yaw).toArray();
   let end=[...start] as ArmJointTarget;for(let i=0;i<5;i++)end=solveHSR(local,pose.attitude,end) as ArmJointTarget;
   moves.push({phase,position:[...position],seconds,base:{position:[...robot.base.position],yaw:robot.base.yaw},start,end:[...end]});
   for(let n=0;n<seconds/.002;n++){
    const u=(n+1)/(seconds/.002),s=u*u*(3-2*u);
    step(phase,start.map((v,i)=>v+(end[i]-v)*s) as ArmJointTarget,.5);
   }
  };
  move('hover',[pose.position[0],groundHeight+.18,pose.position[2]],3);
  move('shallow-approach',pose.position,2);
  closingJoints=[...sim.observation().robot!.joints] as ArmJointTarget;
  for(let n=0;n<500;n++)step('shallow-close',closingJoints,0);
 }catch(error){failure=String(error);}
 finally{settlingQuality.dispose();quality?.dispose();sim.setController(null);}
 const initial=sim.observation(),robot=initial.robot!,ball=initial.objects.find(o=>o.id==='ball')!;
 let tableSupported=false;const support=sim.data.contact;
 try{for(let i=0;i<support.size();i++){const c=support.get(i)!;try{
  if(c.dist<=.003&&((c.geom1===sim.tableId&&sim.model.geom_bodyid[c.geom2]===body)||(c.geom2===sim.tableId&&sim.model.geom_bodyid[c.geom1]===body)))tableSupported=true;
 }finally{c.delete();}}}finally{support.delete();}
 const measuredOffset=robot.endEffectorPosition[1]-ball.position[1],speed=Math.hypot(...ball.velocity);
 const scene={id:SHALLOW_BALL_RECOVERY_SCENE_VERSION,seed,preparation:{profile:'synthetic-gravity-drop-shallow-contact',purpose,seconds:8,settlingSeconds:2,initialBall,initialPenetrations,
  groundHeight,preparationPeakHeight,heightGain:preparationPeakHeight-groundHeight,pose,heldFor,measuredOffset,speed,tableSupported,measuredOpening:sim.gripperOpening(),
  quality:quality?.metrics??null,settlingQuality:settlingQuality.metrics,preservation:preservation?.metrics??null,settlingPreservation:settlingPreservation.metrics,
  settlingMotion,baseAudit,beforePreparation,moves,closingJoints,trace}};
 const evidence={scene,initial};
 if(failure)throw new RecoveryPreparationError(failure,evidence);
 for(const [id,audit]of Object.entries(settlingPreservation.metrics.objects)){
  const minimum=id==='baseball'?0:['cube','can','mug','bottle','plate'].includes(id)?.4:audit.start[1]-.03;
  const motion=settlingMotion[id];
  if(motion.maxHorizontal>.01||motion.maxRise>.01||audit.minHeight<minimum||audit.maxDisplacement>Math.hypot(.01,Math.max(.01,audit.start[1]-minimum))+1e-8)throw new RecoveryPreparationError('Shallow world settling exceeded object limits',evidence);
 }
 if(heldFor<.2||preparationPeakHeight-groundHeight>=.01||measuredOffset<.025||measuredOffset>.045||!tableSupported||speed>.003)throw new RecoveryPreparationError('Shallow contact prerequisite not established',evidence);
 return scene;
}
