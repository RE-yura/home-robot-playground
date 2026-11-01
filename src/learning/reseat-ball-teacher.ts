import {Quaternion,Vector3} from "three";
import type {Simulation} from "../simulation/mujoco.ts";
import type {ArmJointTarget,Observation,Vec3} from "../simulation/types.ts";
import {TaskEvaluator} from "../simulation/task.ts";
import {solveHSR} from "../robot/hsr-ik.ts";
import {ARM_LIMITS} from "../robot/hsr-model.ts";
import {validateLearningBasePose} from "./recording.ts";
import type {TeacherResult} from "./teacher.ts";

type Goal={armJointTarget:ArmJointTarget;gripper:number};
const ENDS=[3,6,10,13,16,120],PHASES=["reseat","lift","carry","lower","release","settle"],EPS=1e-8;
const finite=(v:readonly number[],size:number)=>v.length===size&&v.every(Number.isFinite);
function finiteObservation(obs:Observation) {
  const r=obs.robot,b=obs.objects.find(o=>o.id==="ball"),basket=obs.furniture.find(o=>o.id==="basket");
  return Number.isFinite(obs.timestamp)&&Boolean(r&&finite(r.base.position,3)&&Number.isFinite(r.base.yaw)&&finite(r.joints,5)&&Number.isFinite(r.gripper)&&
    finite(r.endEffectorPosition,3)&&finite(r.endEffectorRotation,4)&&Math.hypot(...r.endEffectorRotation)>0&&Number.isFinite(r.collision?.force??r.tableForce))&&
    (!b||finite(b.position,3)&&finite(b.velocity,3)&&finite(b.rotation,4)&&Math.hypot(...b.rotation)>0)&&
    (!basket||finite(basket.position,3)&&finite(basket.rotation,4)&&Math.hypot(...basket.rotation)>0);
}

// Offline demonstration teacher only. Oracle object state never enters a policy.
// The caller supplies the independently audited preparation and scores every2ms.
export class ReseatBallTeacher {
  readonly evaluator=new TaskEvaluator("ball");
  private readonly sim:Simulation;
  private readonly attitude:Vec3;
  private readonly generation:number;
  private readonly startTime:number;
  private readonly startBase:Vec3;
  private readonly groundHeight:number;
  private readonly seat:Vec3;
  private lastControlTime:number|null=null;
  private lastPhysicsTime:number;
  private phase=-1;
  private start:ArmJointTarget;
  private end:ArmJointTarget;
  private releaseJoints:ArmJointTarget|null=null;
  private opening=.5;
  private heldFor=0;
  private lostFor=0;
  private grasped=false;
  private lifted=false;
  private failure:string|null=null;

  constructor(sim:Simulation,attitude:Vec3) {
    if(attitude.length!==3||!attitude.every(Number.isFinite))throw Error("Invalid reseat attitude");
    this.sim=sim;this.attitude=[...attitude];this.generation=sim.resetGeneration;
    const obs=sim.observation(),robot=obs.robot!,ball=obs.objects.find(o=>o.id==="ball");
    if(!ball||!robot)throw Error("Reseating requires the ball and robot");
    if(!finiteObservation(obs))throw Error("Invalid reseat observation");
    this.startTime=this.lastPhysicsTime=obs.timestamp;this.startBase=[robot.base.position[0],robot.base.position[2],robot.base.yaw];
    this.groundHeight=ball.position[1];this.start=this.end=[...robot.joints] as ArmJointTarget;
    this.seat=new Vector3(...ball.position).add(new Vector3(0,0,.01).applyQuaternion(new Quaternion(...robot.endEffectorRotation))).toArray();
    this.evaluator.start(obs);
  }

  get result():TeacherResult {
    const task=this.evaluator.result;
    return {...task,elapsed:this.lastPhysicsTime-this.startTime,status:this.failure?"failed":task.status,
      teacherFailure:this.failure,phase:PHASES[Math.max(0,this.phase)],grasped:this.grasped,lifted:this.lifted};
  }

  // LearningCapture updates evaluator first; this method never adds score time.
  afterStep(obs:Observation,dt:number) {
    if(!Number.isFinite(dt)||Math.abs(dt-.002)>1e-10)throw Error("Reseating audit requires each2ms physics step");
    if(!finiteObservation(obs)){this.failure="invalid-observation";return;}
    if(Math.abs(obs.timestamp-this.lastPhysicsTime-dt)>EPS){this.failure="invalid-time";return;}
    this.lastPhysicsTime=obs.timestamp;
    const robot=obs.robot!,ball=obs.objects.find(o=>o.id==="ball");
    try{validateLearningBasePose([robot.base.position[0],robot.base.position[2],robot.base.yaw],this.startBase);}
    catch{this.failure="base-drift";return;}
    const held=robot.gripperContact&&robot.contactId==="ball";
    this.heldFor=held?this.heldFor+dt:0;
    if(this.heldFor>=.2-EPS){this.grasped=true;this.lifted ||= Boolean(ball&&ball.position[1]>this.groundHeight+.15);}
    if(this.phase>=2&&this.phase<4){this.lostFor=held?0:this.lostFor+dt;if(this.lostFor>.2+EPS)this.failure="grasp-lost";}
    if(this.evaluator.result.status==="success"&&(!this.grasped||!this.lifted))this.failure="unverified-transport";
  }

  private hold(obs:Observation):Goal {
    const measured=obs.robot?.joints,joints=measured&&finite(measured,5)?measured:this.end;
    return {armJointTarget:joints.map((v,i)=>Math.max(ARM_LIMITS[i][0],Math.min(ARM_LIMITS[i][1],v))) as ArmJointTarget,gripper:this.opening};
  }
  private fail(reason:string,obs:Observation):Goal {this.failure=reason;return this.hold(obs);}

  next(obs:Observation):Goal {
    if(this.result.status!=="running")return this.hold(obs);
    if(this.sim.resetGeneration!==this.generation)return this.fail("reset",obs);
    if(!finiteObservation(obs))return this.fail("invalid-observation",obs);
    const robot=obs.robot!;
    try{validateLearningBasePose([robot.base.position[0],robot.base.position[2],robot.base.yaw],this.startBase);}
    catch{return this.fail("base-drift",obs);}
    this.evaluator.update(obs,0);
    if(this.evaluator.result.status==="failed")return this.hold(obs);
    if(this.evaluator.result.status==="success"&&(!this.grasped||!this.lifted))return this.fail("unverified-transport",obs);
    const elapsed=obs.timestamp-this.startTime;
    if(!Number.isFinite(elapsed)||elapsed< -EPS)return this.fail("invalid-time",obs);
    const sample=Math.round(elapsed*10),phase=ENDS.findIndex(end=>sample<end*10);
    if(phase<0)return this.fail("timeout",obs);
    if(robot.gripperContact&&robot.contactId!=="ball"&&!(phase>=4&&robot.contactId==="basket"))return this.fail("wrong-grasp",obs);
    if(Math.abs(obs.timestamp-(this.lastControlTime===null?this.startTime:this.lastControlTime+.1))>EPS)return this.fail("missed-control-boundary",obs);
    this.lastControlTime=obs.timestamp;
    if(phase!==this.phase){
      if(phase===2&&!(this.heldFor>=.2-EPS&&obs.objects.find(o=>o.id==="ball")!.position[1]>this.groundHeight+.15))return this.fail("secure-lift-missing",obs);
      if(phase===4){
        const ball=obs.objects.find(o=>o.id==="ball")!,basket=obs.furniture.find(o=>o.id==="basket")!;
        const local=new Vector3(...ball.position).sub(new Vector3(...basket.position)).applyQuaternion(new Quaternion(...basket.rotation).invert());
        if(!robot.gripperContact||robot.contactId!=="ball"||Math.abs(local.x)+.0335>.21||Math.abs(local.z)+.0335>.19)return this.fail("release-blocked",obs);
        this.releaseJoints=[...robot.joints] as ArmJointTarget;
      }
      this.phase=phase;this.start=[...robot.joints] as ArmJointTarget;this.end=[...this.start];
      if(phase<=2){
        const target:Vec3=phase===0?this.seat:phase===1?[robot.endEffectorPosition[0],robot.endEffectorPosition[1]+.3,robot.endEffectorPosition[2]]:[.18,.78,.465];
        const local=new Vector3(...target).sub(new Vector3(...robot.base.position)).applyAxisAngle(new Vector3(0,1,0),-robot.base.yaw).toArray();
        for(let n=0;n<5;n++)this.end=solveHSR(local,this.attitude,this.end) as ArmJointTarget;
      }
    }
    const begin=this.phase?ENDS[this.phase-1]:0,duration=ENDS[this.phase]-begin;
    const u=Math.min(1,(sample-begin*10+1)/(duration*10)),s=u*u*(3-2*u);
    this.opening=this.phase===0?.5:this.phase<4?0:this.phase===4?u:1;
    const joints:ArmJointTarget=this.phase>=4?[...this.releaseJoints!]:this.start.map((v,i)=>v+(this.end[i]-v)*s) as ArmJointTarget;
    if(this.phase===3)joints[0]=Math.max(0,this.start[0]-.05*s);
    return {armJointTarget:joints,gripper:this.opening};
  }
}
