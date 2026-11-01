import type {DoubleBuffer} from '@mujoco/mujoco';
import type {Simulation} from '../simulation/mujoco.ts';
import {mugBounds} from '../simulation/mug-geometry.ts';
import type {Vec3} from '../simulation/types.ts';
import type {TeacherResult} from './teacher.ts';

type Contact = {robotBodies:number;finger:boolean;floor:boolean;objectId:string|null;normalForce:number;grasping:boolean};
// Audit only: closed fingers touching the basket/other object are never an intended grasp.
export function unintendedContactForce(contacts:Contact[],targetId:string):number {
 return contacts.reduce((sum,c)=>sum+(c.robotBodies===1&&!c.floor&&!(c.finger&&c.objectId===targetId&&c.grasping)?Math.max(0,c.normalForce):0),0);
}

export class TeacherQualityAudit {
 readonly metrics = {schema:'hsr-teacher-quality-v2',physicsSteps:0,physicsSeconds:0,maxOtherMotion:0,minOtherHeight:Infinity,
  peakUnintendedForce:0,peakRobotEnvironmentForce:0,peakTargetFingerForce:0,targetFingerImpulseNs:0,peakTargetEnvironmentForce:0,targetEnvironmentImpulseNs:0,above25Seconds:0,impulseNs:0,prematureDrops:0,
  drops:[] as {time:number;phase:string;contactId:string|null;position:Vec3;velocity:Vec3;basketBounds:ReturnType<typeof mugBounds>|null}[],
  phases:{} as Record<string,{peak:number;above25Seconds:number;impulseNs:number;targetEnvironmentPeak:number;targetEnvironmentImpulseNs:number;targetFingerPeak:number}>,
  pairs:{} as Record<string,{peak:number;impulseNs:number}>,settlement:{physicsSteps:0,physicsSeconds:0,peakUnintendedForce:0}};
 private readonly sim:Simulation;
 private readonly targetId:'mug'|'ball';
 private readonly otherId:string;
 private readonly otherStart:Vec3;
 private readonly floor:number;
 private readonly targetBody:number;
 private readonly names:string[];
 private readonly fingers:Set<number>;
 private readonly force:DoubleBuffer;
 constructor(sim:Simulation,targetId:'mug'|'ball') {
  this.sim=sim;this.targetId=targetId;this.otherId=targetId==='mug'?'ball':'mug';
  this.otherStart=[...sim.observation().objects.find(o=>o.id===this.otherId)!.position];
  this.floor=sim.id('GEOM','floor');this.targetBody=sim.id('BODY',targetId);this.force=new sim.mj.DoubleBuffer(6);
  this.names=Array.from({length:sim.model.nbody},(_,i)=>sim.mj.mj_id2name(sim.model,sim.mj.mjtObj.mjOBJ_BODY.value,i)??String(i));
  this.fingers=new Set(this.names.flatMap((n,i)=>/^hand_[lr]_/.test(n)?[i]:[]));
 }
 update(dt:number,result:Pick<TeacherResult,'phase'|'drops'>&Partial<Pick<TeacherResult,'settledFor'>>) {
  if(Math.abs(dt-.002)>1e-10)throw new Error('Quality audit requires each 2ms completed physics step');
  const sim=this.sim,obs=sim.observation(),other=obs.objects.find(o=>o.id===this.otherId)!,m=this.metrics;
  m.physicsSteps++;m.physicsSeconds+=dt;
  m.maxOtherMotion=Math.max(m.maxOtherMotion,Math.hypot(...other.position.map((v,i)=>v-this.otherStart[i])));
  m.minOtherHeight=Math.min(m.minOtherHeight,other.position[1]);
  const contacts=sim.data.contact;let total=0,allForce=0,targetFingerForce=0,targetEnvironmentForce=0;
  try{for(let i=0;i<contacts.size();i++){
   const c=contacts.get(i)!;
   try{
    const a=sim.model.geom_bodyid[c.geom1],b=sim.model.geom_bodyid[c.geom2],ra=sim.model.body_rootid[a]===sim.baseId,rb=sim.model.body_rootid[b]===sim.baseId;
    if((a===this.targetBody&&!rb)||(b===this.targetBody&&!ra)){
     sim.mj.mj_contactForce(sim.model,sim.data,i,this.force);targetEnvironmentForce+=Math.max(0,this.force.GetView()[0]);
    }
    if(ra===rb||c.geom1===this.floor||c.geom2===this.floor)continue;
    sim.mj.mj_contactForce(sim.model,sim.data,i,this.force);
    const force=unintendedContactForce([{robotBodies:1,finger:this.fingers.has(a)||this.fingers.has(b),floor:false,
     objectId:sim.objectBodies.get(ra?b:a)??null,normalForce:this.force.GetView()[0],grasping:sim.opening<.5||sim.gripperContactId===this.targetId}],this.targetId);
    total+=force;const normal=Math.max(0,this.force.GetView()[0]);allForce+=normal;
    if((this.fingers.has(a)||this.fingers.has(b))&&sim.objectBodies.get(ra?b:a)===this.targetId)targetFingerForce+=normal;
    if(force>0){const key=[this.names[a],this.names[b]].sort().join('|'),pair=m.pairs[key]??={peak:0,impulseNs:0};
     pair.peak=Math.max(pair.peak,force);pair.impulseNs+=force*dt;}
   }finally{c.delete();}
  }}finally{contacts.delete();}
  m.peakRobotEnvironmentForce=Math.max(m.peakRobotEnvironmentForce,allForce);m.peakTargetFingerForce=Math.max(m.peakTargetFingerForce,targetFingerForce);m.targetFingerImpulseNs+=targetFingerForce*dt;
  m.peakTargetEnvironmentForce=Math.max(m.peakTargetEnvironmentForce,targetEnvironmentForce);m.targetEnvironmentImpulseNs+=targetEnvironmentForce*dt;
  m.peakUnintendedForce=Math.max(m.peakUnintendedForce,total);m.impulseNs+=total*dt;if(total>25)m.above25Seconds+=dt;
  // The canonical scorer runs first. Count its current consecutive settlement
  // interval independently of the teacher's release/settle phase labels.
  if(Number.isFinite(result.settledFor)&&result.settledFor!>0){
   m.settlement.physicsSteps++;m.settlement.physicsSeconds+=dt;m.settlement.peakUnintendedForce=Math.max(m.settlement.peakUnintendedForce,total);
  }else m.settlement={physicsSteps:0,physicsSeconds:0,peakUnintendedForce:0};
  const p=m.phases[result.phase]??={peak:0,above25Seconds:0,impulseNs:0,targetEnvironmentPeak:0,targetEnvironmentImpulseNs:0,targetFingerPeak:0};
  p.peak=Math.max(p.peak,total);p.impulseNs+=total*dt;if(total>25)p.above25Seconds+=dt;
  p.targetEnvironmentPeak=Math.max(p.targetEnvironmentPeak,targetEnvironmentForce);p.targetEnvironmentImpulseNs+=targetEnvironmentForce*dt;p.targetFingerPeak=Math.max(p.targetFingerPeak,targetFingerForce);
  if(result.drops>m.drops.length){
   const target=obs.objects.find(o=>o.id===this.targetId)!,basket=obs.furniture.find(o=>o.id==='basket')!;
   if(!['release','settle'].includes(result.phase))m.prematureDrops+=result.drops-m.drops.length;
   m.drops.push({time:obs.timestamp,phase:result.phase,contactId:sim.gripperContactId,position:[...target.position],velocity:[...target.velocity],
    basketBounds:this.targetId==='mug'?mugBounds(target,basket):null});
  }
 }
 dispose(){this.force.delete();}
}
