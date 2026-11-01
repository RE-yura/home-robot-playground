"""Demonstration certification. Oracle audit never becomes a policy feature."""
import base64
import hashlib
import io
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 64 * 1024 * 1024
QUALITY_POLICY = "hsr-demo-quality-v2"
# Simulator numeric outlier gate, not a hardware-safe force specification.
TARGET_CONTACT_OUTLIER_N = 1000


def recovery_preparation_rejections(scene, initial):
    prep=scene.get('preparation',{});quality=prep.get('quality',{});preservation=prep.get('preservation',{})
    reasons=[]
    robot=initial.get('robot',{});opening=robot.get('gripper')
    if (robot.get('gripperContact')is not False or type(opening)not in (int,float) or
        not math.isfinite(opening)or not 0<=opening<.35 or prep.get('measuredOpening')!=opening):
        reasons.append('recovery-start-is-not-empty-and-closed')
    if (prep.get('seconds')!=2 or quality.get('schema')!='hsr-teacher-quality-v2' or
        quality.get('physicsSteps')!=1000 or type(quality.get('physicsSeconds'))not in (int,float)or
        not math.isfinite(quality['physicsSeconds'])or abs(quality['physicsSeconds']-2)>1e-8 or any(type(quality.get(k))not in (int,float)or
        not math.isfinite(quality[k])or not 0<=quality[k]<limit for k,limit in [('peakUnintendedForce',100),('above25Seconds',.1),('peakTargetFingerForce',TARGET_CONTACT_OUTLIER_N),('peakTargetEnvironmentForce',TARGET_CONTACT_OUTLIER_N)])):
        reasons.append('recovery-preparation-audit-or-contact-limit-failed')
    base=prep.get('baseAudit',{})
    if (base.get('physicsSteps')!=1000 or any(type(base.get(k))not in(int,float)or
        not math.isfinite(base[k])or not 0<=base[k]<=limit+1e-8 for k,limit in [('maxTranslation',.02),('maxYaw',.03)])):
        reasons.append('recovery-preparation-base-audit-or-drift-failed')
    objects=preservation.get('objects',{})
    if (preservation.get('schema')!='hsr-object-preservation-v1'or preservation.get('physicsSteps')!=1000 or
        set(objects)!={o['id']for o in initial.get('objects',[])}):
        reasons.append('recovery-preparation-object-audit-missing')
    else:
        for value in objects.values():
            displacement,height=value.get('maxDisplacement'),value.get('minHeight');start=value.get('start')
            if (type(displacement)not in (int,float)or type(height)not in (int,float)or
                not math.isfinite(displacement)or not math.isfinite(height)or not 0<=displacement<=.01 or
                not isinstance(start,list)or len(start)!=3 or any(type(x)not in(int,float)or not math.isfinite(x)for x in start)or height<start[1]-.03):
                reasons.append('recovery-preparation-object-moved-or-fell');break
    return reasons


def ball_recovery_preparation_rejections(scene, initial):
    prep=scene.get('preparation',{});quality=prep.get('quality',{});base=prep.get('baseAudit',{})
    def number(v):return type(v)in(int,float)and math.isfinite(v)
    def vector(v):return isinstance(v,list)and len(v)==3 and all(number(x)for x in v)
    reasons=[];robot=initial.get('robot',{});opening=robot.get('gripper')
    if robot.get('gripperContact')is not False or not number(opening)or not 0<=opening<.35 or prep.get('measuredOpening')!=opening:
        reasons.append('ball-regrasp-start-is-not-empty-and-closed')
    ball=next((o for o in initial.get('objects',[])if o.get('id')=='ball'),{})
    start=prep.get('initialBall');position=ball.get('position');velocity=ball.get('velocity');lowest=prep.get('lowestBallY')
    if (scene.get('id')!='hsr-stationary-v7-ball-regrasp'or prep.get('profile')!='synthetic-gravity-drop-empty-hand'or
        prep.get('seconds')!=4 or prep.get('settlingSeconds')!=2 or prep.get('initialPenetrations')!=[] or
        not vector(start)or not .007-1e-8<=start[0]<=.023+1e-8 or abs(start[1]-.54)>1e-8 or not .425-1e-8<=start[2]<=.455+1e-8 or
        not vector(position)or not vector(velocity)or not number(lowest)or not .4<=lowest<=position[1]+1e-8 or
        position[1]<.4 or start[1]-position[1]<.09 or math.hypot(*velocity)>.03):
        reasons.append('ball-regrasp-gravity-drop-or-landing-invalid')
    if (quality.get('schema')!='hsr-teacher-quality-v2'or quality.get('physicsSteps')!=2000 or
        not number(quality.get('physicsSeconds'))or abs(quality['physicsSeconds']-4)>1e-8 or
        any(not number(quality.get(k))or not 0<=quality[k]<limit for k,limit in [('peakUnintendedForce',100),('above25Seconds',.1),('peakTargetFingerForce',TARGET_CONTACT_OUTLIER_N),('peakTargetEnvironmentForce',TARGET_CONTACT_OUTLIER_N)])):
        reasons.append('ball-regrasp-preparation-audit-or-contact-limit-failed')
    if base.get('physicsSteps')!=2000 or any(not number(base.get(k))or not 0<=base[k]<=limit+1e-8 for k,limit in [('maxTranslation',.02),('maxYaw',.03)]):
        reasons.append('ball-regrasp-preparation-base-audit-or-drift-failed')
    identities={o['id']for o in initial.get('objects',[])if o['id']!='ball'}
    settling_motion=prep.get('settlingMotion',{})
    if set(settling_motion)!=identities or any(not number(v.get(k))or not 0<=v[k]<=.01 for v in settling_motion.values()for k in ['maxHorizontal','maxRise']):
        reasons.append('ball-regrasp-world-settling-motion-invalid')
    for name in ['settlingPreservation','preservation']:
        audit=prep.get(name)or {};objects=audit.get('objects',{})
        if audit.get('schema')!='hsr-object-preservation-v1'or audit.get('physicsSteps')!=1000 or set(objects)!=identities:
            reasons.append('ball-regrasp-preparation-object-audit-missing:'+name);continue
        for identity,value in objects.items():
            displacement,height,start=value.get('maxDisplacement'),value.get('minHeight'),value.get('start')
            if (not vector(start)or not number(displacement)or displacement<0 or not number(height)or
                (name=='preservation'and(displacement>.01 or height<start[1]-.03))):
                reasons.append('ball-regrasp-preparation-object-moved-or-fell:'+name);break
            if name=='settlingPreservation':
                minimum=0 if identity=='baseball'else .4 if identity in ('cube','can','mug','bottle','plate')else start[1]-.03
                if height<minimum or displacement>math.hypot(.01,max(.01,start[1]-minimum))+1e-8:
                    reasons.append('ball-regrasp-world-settling-support-or-displacement-invalid');break
    return reasons



def physical_ball_preparation_rejections(scene,initial):
 def number(v):return type(v)in(int,float)and math.isfinite(v)
 def vector(v,n=3):return isinstance(v,list)and len(v)==n and all(number(x)for x in v)
 p=scene.get('preparation',{});reasons=[]
 robot=initial.get('robot',{});ball=next((o for o in initial.get('objects',[])if o.get('id')=='ball'),{})
 opening=robot.get('gripper');position=ball.get('position');velocity=ball.get('velocity')
 if (robot.get('gripperContact')is not False or not number(opening)or not 0<=opening<.35 or p.get('measuredOpening')!=opening or
     not vector(position)or abs(position[1]-.4335)>.003 or p.get('tableSupported')is not True or not vector(velocity)or math.hypot(*velocity)>.003):
  reasons.append('physical-loss-start-is-not-empty-closed-and-stationary-on-table')
 times=[p.get(k)for k in ['triggeredAt','openedAt','lostAt','restoredAt','seconds']]
 if (scene.get('id')!='hsr-stationary-v8-physical-ball-loss'or p.get('profile')!='actual-teacher-grasp-release'or
     not all(number(v)for v in times)or not 0<=times[0]<times[1]<=times[2]<=times[3]<times[4]<=40 or
     abs(times[1]-times[0]-2)>1e-8 or abs(times[3]-times[1]-.3)>1e-8 or times[4]-times[3]>8.004 or
     not number(p.get('heldBeforeOpeningSeconds'))or p['heldBeforeOpeningSeconds']<.2 or
     not number(p.get('stableSeconds'))or not .3-1e-8<=p['stableSeconds']<=times[4]-times[3]+1e-8):
  reasons.append('actual-grasp-open-loss-restoration-or-settlement-missing')
 seconds=p.get('seconds');steps=round(seconds/.002)if number(seconds)else -1
 quality=p.get('quality',{});base=p.get('baseAudit',{});preservation=p.get('preservation',{})
 if (steps<=0 or not number(seconds)or abs(steps*.002-seconds)>1e-8 or quality.get('schema')!='hsr-teacher-quality-v2'or
     quality.get('physicsSteps')!=steps or not number(quality.get('physicsSeconds'))or abs(quality['physicsSeconds']-seconds)>1e-8 or
     any(not number(quality.get(k))or not 0<=quality[k]<limit for k,limit in [('peakUnintendedForce',100),('above25Seconds',.1),('peakTargetFingerForce',1000),('peakTargetEnvironmentForce',1000)])):
  reasons.append('physical-loss-preparation-quality-or-continuous-audit-invalid')
 if base.get('physicsSteps')!=steps or any(not number(base.get(k))or not 0<=base[k]<=limit+1e-8 for k,limit in [('maxTranslation',.02),('maxYaw',.03)]):
  reasons.append('physical-loss-preparation-base-audit-invalid')
 objects=preservation.get('objects',{});identities={o['id']for o in initial.get('objects',[])if o['id']!='ball'}
 if (preservation.get('schema')!='hsr-object-preservation-v1'or preservation.get('physicsSteps')!=steps or set(objects)!=identities or
     any(not vector(v.get('start'))or not number(v.get('maxDisplacement'))or not 0<=v['maxDisplacement']<=.01 or not number(v.get('minHeight'))or v['minHeight']<v['start'][1]-.03 for v in objects.values())):
  reasons.append('physical-loss-non-target-preservation-invalid')
 trace=p.get('trace',[]);phases={'teacher-grasp-lift','held-settling','opening','landing'}
 if (not isinstance(trace,list)or len(trace)!=math.ceil(max(0,steps)/50)or not trace or
     any(not number(row.get('seconds'))or abs(row['seconds']-(.002+i*.1))>1e-8 or row.get('phase')not in phases or type(row.get('held'))is not bool or
         type(row.get('tableSupported'))is not bool or not vector(row.get('ball'))or row['ball'][1]<.4 or not vector(row.get('velocity'))or not vector(row.get('goal'),6)for i,row in enumerate(trace))or
     len([row for row in trace if row.get('phase')=='held-settling'and row.get('held')is True and row['ball'][1]>=.65])<19 or
     not any(row.get('phase')=='opening'and row.get('held')is False and row.get('goal',[0]*6)[5]==1 for row in trace)):
  reasons.append('physical-grasp-and-loss-trace-missing-or-invalid')
 else:
  limits=[(0,.69),(-2.62,0),(-2.09,3.84),(-1.92,1.22),(-1.92,3.67),(0,1)]
  if any(not lo<=v<=hi for row in trace for v,(lo,hi)in zip(row['goal'],limits)):
   reasons.append('physical-loss-trace-joint-goals-out-of-range')
  if all(number(v)for v in times)and number(p.get('stableSeconds')):
   frozen=None;previous=None
   for row in trace:
    time=row['seconds']-.002
    phase='teacher-grasp-lift'if time<times[0]-1e-8 else 'held-settling'if time<times[1]-1e-8 else 'opening'if time<times[3]-1e-8 else 'landing'
    if row['phase']!=phase:
     reasons.append('physical-loss-trace-phase-clock-mismatch');break
    if phase!='teacher-grasp-lift':
     if frozen is None:frozen=previous
     if frozen is None or row['goal'][:5]!=frozen or row['goal'][5]!=(1 if phase=='opening'else 0):
      reasons.append('physical-loss-trace-latch-or-opening-mismatch');break
    if phase=='held-settling'and(not row['held']or row['ball'][1]<.65):
     reasons.append('physical-loss-trace-held-settling-missing');break
    if row['seconds']>times[4]-p.get('stableSeconds',0)+1e-8 and (row['held']or not row['tableSupported']or abs(row['ball'][1]-.4335)>.003 or math.hypot(*row['velocity'])>.003):
     reasons.append('physical-loss-trace-table-settlement-missing');break
    previous=row['goal'][:5]
 return reasons

def empty_carry_preparation_rejections(scene,initial):
 def number(v):return type(v)in(int,float)and math.isfinite(v)
 def vector(v,n=3):return isinstance(v,list)and len(v)==n and all(number(x)for x in v)
 p=scene.get('emptyCarry');prefix=scene.get('physicalLoss');before=scene.get('physicalInitial');reasons=[]
 if not isinstance(prefix,dict)or not isinstance(before,dict)or not isinstance(p,dict):return ['empty-carry-physical-prefix-or-transition-missing']
 reasons.extend(physical_ball_preparation_rejections(prefix,before))
 seed=scene.get('seed');start=p.get('startJoints');goal=p.get('goalJoints');desired=p.get('desiredWorldHand');opening=p.get('opening');fraction=p.get('fraction');limits=[(0,.69),(-2.62,0),(-2.09,3.84),(-1.92,1.22),(-1.92,3.67)]
 if(scene.get('id')!='hsr-stationary-v9-empty-carry-loss'or type(seed)is not int or prefix.get('seed')!=seed or not number(opening)or opening!=seed%2 or not number(fraction)or not .2<=fraction<=1 or not vector(desired)or not(.02<=desired[0]<=.18 and .74<=desired[1]<=.86 and .44<=desired[2]<=.49)or not vector(start,5)or start!=before.get('robot',{}).get('joints')or not vector(goal,5)or any(not lo<=v<=hi for v,(lo,hi)in zip(goal or [],limits))):reasons.append('empty-carry-pose-identity-or-range-invalid')
 try:
  script="import {emptyCarryPose} from './src/learning/empty-carry-ball-recovery.ts';let t='';for await(const p of process.stdin)t+=p;const x=JSON.parse(t);console.log(JSON.stringify(emptyCarryPose(x.seed,x.robot)));"
  checked=subprocess.run(['node','--input-type=module','-e',script],cwd=ROOT,input=json.dumps({'seed':seed,'robot':before.get('robot')}).encode(),capture_output=True,timeout=30)
  if checked.returncode:raise ValueError('Pose reconstruction failed')
  expected=json.loads(checked.stdout)
  # Chrome/Node IK reconstruction differs below1e-8; seed and measured-start identity stay exact.
  if(any(p.get(k)!=expected[k]for k in ['desiredWorldHand','fraction','opening','startJoints'])or not vector(goal,5)or any(abs(a-b)>1e-8 for a,b in zip(goal,expected['goalJoints']))):reasons.append('empty-carry-seeded-pose-or-ik-goal-mismatch')
 except (ValueError,KeyError,TypeError,subprocess.SubprocessError):reasons.append('empty-carry-pose-reconstruction-failed')
 if p.get('seconds')!=3 or not number(before.get('timestamp'))or not number(initial.get('timestamp'))or abs(initial['timestamp']-before['timestamp']-3)>1e-8:reasons.append('empty-carry-physical-clock-invalid')
 quality=p.get('quality',{});base=p.get('baseAudit',{});preservation=p.get('preservation',{})
 if(quality.get('schema')!='hsr-teacher-quality-v2'or quality.get('physicsSteps')!=1500 or not number(quality.get('physicsSeconds'))or abs(quality['physicsSeconds']-3)>1e-8 or any(not number(quality.get(k))or not 0<=quality[k]<limit for k,limit in [('peakUnintendedForce',100),('above25Seconds',.1),('peakTargetFingerForce',1000),('peakTargetEnvironmentForce',1000)])):reasons.append('empty-carry-quality-or-continuous-audit-invalid')
 origin=prefix.get('preparation',{}).get('origin',{}).get('robot',{}).get('base',{});end=initial.get('robot',{}).get('base',{})
 if not vector(origin.get('position'))or not number(origin.get('yaw'))or not vector(end.get('position'))or not number(end.get('yaw')):reasons.append('empty-carry-base-origin-missing')
 else:
  reference=[origin['position'][0],origin['position'][2],origin['yaw']];distance=math.hypot(end['position'][0]-reference[0],end['position'][2]-reference[1]);yaw=abs(math.atan2(math.sin(end['yaw']-reference[2]),math.cos(end['yaw']-reference[2])))
  if(base.get('reference')!=reference or base.get('physicsSteps')!=1500 or any(not number(base.get(k))or not 0<=base[k]<=limit+1e-8 for k,limit in [('maxTranslation',.02),('maxYaw',.03)])or distance>base.get('maxTranslation',-1)+1e-8 or yaw>base.get('maxYaw',-1)+1e-8):reasons.append('empty-carry-global-base-audit-invalid')
 objects=preservation.get('objects',{});old={o['id']:o for o in before.get('objects',[])if o['id']!='ball'};final={o['id']:o for o in initial.get('objects',[])if o['id']!='ball'}
 if(preservation.get('schema')!='hsr-object-preservation-v1'or preservation.get('physicsSteps')!=1500 or set(objects)!=set(old)or set(objects)!=set(final)or any(not vector(v.get('start'))or v['start']!=old[key].get('position')or not number(v.get('maxDisplacement'))or not 0<=v['maxDisplacement']<=.01 or not number(v.get('minHeight'))or v['minHeight']<v['start'][1]-.03 or not vector(final[key].get('position'))or math.dist(v['start'],final[key]['position'])>v['maxDisplacement']+1e-8 for key,v in objects.items())):reasons.append('empty-carry-non-target-preservation-invalid')
 initial_ball=next((o for o in before.get('objects',[])if o.get('id')=='ball'),{});startBall=initial_ball.get('position');maxMotion=p.get('maxTargetDisplacement')
 if not vector(startBall)or not number(maxMotion)or not 0<=maxMotion<=.01:reasons.append('empty-carry-target-displacement-audit-invalid')
 robot=initial.get('robot',{});ball=next((o for o in initial.get('objects',[])if o.get('id')=='ball'),{})
 if(robot.get('gripperContact')is not False or not number(robot.get('gripper'))or not 0<=robot['gripper']<=1 or not number(opening)or abs(robot['gripper']-opening)>.03 or p.get('tableSupported')is not True or not vector(ball.get('position'))or abs(ball['position'][1]-.4335)>.003 or not vector(ball.get('velocity'))or math.hypot(*ball['velocity'])>.003):reasons.append('empty-carry-start-is-not-empty-and-stationary-on-table')
 trace=p.get('trace',[])
 if(not isinstance(trace,list)or len(trace)!=31 or any(not number(row.get('seconds'))or abs(row['seconds']-(3 if i==30 else .002+i*.1))>1e-8 or not vector(row.get('state'),6)or not vector(row.get('goal'),6)or not vector(row.get('ball'))or row['ball'][1]<.4 or not vector(row.get('velocity'))or row.get('held')is not False for i,row in enumerate(trace))):reasons.append('empty-carry-transition-trace-invalid')
 elif vector(start,5)and vector(goal,5)and number(opening):
  for i,row in enumerate(trace):
   u=1 if i==30 else (i*50+1)/1500;s=u*u*(3-2*u);expected=[a+(b-a)*s for a,b in zip(start,goal)]+[opening]
   if any(abs(a-b)>1e-8 for a,b in zip(row['goal'],expected)):reasons.append('empty-carry-smooth-goals-or-opening-mismatch');break
 if isinstance(trace,list)and len(trace)==31:
  last=trace[-1]
  if(last.get('state')!=robot.get('joints',[])+[robot.get('gripper')]or last.get('ball')!=ball.get('position')or last.get('velocity')!=ball.get('velocity')or last.get('held')!=robot.get('gripperContact')):reasons.append('empty-carry-terminal-trace-or-initial-mismatch')
  if any(not vector(row.get('state'),6)or any(not lo-.02<=v<=hi+.02 for v,(lo,hi)in zip(row['state'][:5],limits))or not 0<=row['state'][5]<=1 for row in trace):reasons.append('empty-carry-measured-state-out-of-range')
  if vector(startBall)and number(maxMotion)and any(not vector(row.get('ball'))or math.dist(row['ball'],startBall)>maxMotion+1e-8 for row in trace):reasons.append('empty-carry-target-transition-discontinuity')
 return reasons


def shallow_ball_preparation_rejections(scene,initial):
 def number(v):return type(v)in(int,float)and math.isfinite(v)
 def vector(v,n=3):return isinstance(v,list)and len(v)==n and all(number(x)for x in v)
 def same(a,b):return vector(a,len(b))and all(abs(x-y)<=1e-8 for x,y in zip(a,b))
 p=scene.get('preparation')or {};seed=scene.get('seed');reasons=[];before=p.get('beforePreparation')or {}
 if(scene.get('id')!='hsr-stationary-v10-shallow-reseat'or type(seed)is not int or not(1000<=seed<=1999 or 2000<=seed<=2099)or
    p.get('profile')!='synthetic-gravity-drop-shallow-contact'or p.get('purpose')!='teacher-demonstration'or p.get('initialPenetrations')!=[] or
    p.get('seconds')!=8 or p.get('settlingSeconds')!=2 or not number(initial.get('timestamp'))or abs(initial['timestamp']-8)>1e-8 or not number(before.get('timestamp'))or abs(before['timestamp']-2)>1e-8):reasons.append('shallow-identity-purpose-seed-or-clock-invalid')
 robot=initial.get('robot')or {};ball=next((o for o in initial.get('objects',[])if o.get('id')=='ball'),{})
 origin=before.get('robot')or {};ground=next((o for o in before.get('objects',[])if o.get('id')=='ball'),{}).get('position')
 pose=p.get('pose')or {};moves=p.get('moves')or [];trace=p.get('trace')or [];limits=[(0,.69),(-2.62,0),(-2.09,3.84),(-1.92,1.22),(-1.92,3.67)]
 if(not vector(ground)or not vector(origin.get('joints'),5)or not vector(origin.get('base',{}).get('position'))or not number(origin.get('base',{}).get('yaw'))):return reasons+['shallow-preparation-origin-missing']
 try:
  script="""import {createRandom} from './src/simulation/random.ts';import {shallowRecoveryPose} from './src/learning/shallow-ball-recovery.ts';import {solveHSR} from './src/robot/hsr-ik.ts';import {Vector3} from 'three';let t='';for await(const p of process.stdin)t+=p;const x=JSON.parse(t),r=createRandom(x.seed^0x42414c4c),initialBall=[.015+(r()*2-1)*.008,.54,.44+(r()*2-1)*.015],pose=shallowRecoveryPose(x.seed,x.ball,x.robot.base.yaw),ends=x.moves.map(m=>{let end=[...m.start];const local=new Vector3(...m.position).sub(new Vector3(...m.base.position)).applyAxisAngle(new Vector3(0,1,0),-m.base.yaw).toArray();for(let n=0;n<5;n++)end=solveHSR(local,pose.attitude,end);return end;});console.log(JSON.stringify({initialBall,pose,ends}));"""
  checked=subprocess.run(['node','--input-type=module','-e',script],cwd=ROOT,input=json.dumps({'seed':seed,'ball':ground,'robot':origin,'moves':moves}).encode(),capture_output=True,timeout=30)
  if checked.returncode:raise ValueError('Pose reconstruction failed')
  expected=json.loads(checked.stdout)
  if(p.get('initialBall')!=expected['initialBall']or pose!=expected['pose']or len(moves)!=2 or
     any(not same(m.get('end'),end)for m,end in zip(moves,expected['ends']))):reasons.append('shallow-seeded-pose-or-ik-goal-mismatch')
 except (ValueError,KeyError,TypeError,subprocess.SubprocessError):reasons.append('shallow-pose-reconstruction-failed')
 if(not number(p.get('groundHeight'))or abs(p['groundHeight']-ground[1])>1e-8 or abs(ground[1]-.4335)>.003 or
    not vector(ball.get('position'))or abs(ball['position'][1]-.4335)>.003 or
    not number(p.get('preparationPeakHeight'))or not number(p.get('heightGain'))or not 0<=p['heightGain']<.01 or abs(p['preparationPeakHeight']-ground[1]-p['heightGain'])>1e-8 or
    not number(p.get('heldFor'))or p['heldFor']<.2 or p['heldFor']>6 or not number(p.get('measuredOffset'))or not .025<=p['measuredOffset']<=.045 or
    not number(p.get('speed'))or not 0<=p['speed']<=.003 or p.get('tableSupported')is not True or
    robot.get('gripperContact')is not True or robot.get('contactId')!='ball'or not number(robot.get('gripper'))or not 0<=robot['gripper']<=1 or p.get('measuredOpening')!=robot['gripper']or
    not vector(robot.get('endEffectorPosition'))or not vector(ball.get('position'))or not vector(ball.get('velocity'))):reasons.append('shallow-measured-contact-height-speed-or-support-invalid')
 elif(abs(robot['endEffectorPosition'][1]-ball['position'][1]-p['measuredOffset'])>1e-8 or abs(math.hypot(*ball['velocity'])-p['speed'])>1e-8 or ball['position'][1]>p['preparationPeakHeight']+1e-8):reasons.append('shallow-measured-terminal-values-mismatch')
 metrics=[]
 for name,steps in [('quality',3000),('settlingQuality',1000)]:
  q=p.get(name)or {};metrics.append(q)
  if(q.get('schema')!='hsr-teacher-quality-v2'or q.get('physicsSteps')!=steps or not number(q.get('physicsSeconds'))or abs(q['physicsSeconds']-steps*.002)>1e-8 or
     any(not number(q.get(k))or not 0<=q[k]<limit for k,limit in [('peakUnintendedForce',100),('above25Seconds',.1),('peakTargetFingerForce',1000),('peakTargetEnvironmentForce',1000)])):reasons.append('shallow-quality-or-completed-steps-invalid:'+name)
 if all(number(q.get('above25Seconds'))for q in metrics)and sum(q['above25Seconds']for q in metrics)>=.1:reasons.append('shallow-combined-force-duration-exceeded')
 base=p.get('baseAudit')or {};end=robot.get('base')or {};reference=base.get('reference')
 if(reference!=[0,1.08,-.1]or base.get('physicsSteps')!=4000 or any(not number(base.get(k))or not 0<=base[k]<=limit+1e-8 for k,limit in [('maxTranslation',.02),('maxYaw',.03)])or not vector(end.get('position'))or not number(end.get('yaw'))):reasons.append('shallow-base-origin-audit-or-drift-invalid')
 elif(math.hypot(end['position'][0]-reference[0],end['position'][2]-reference[1])>base['maxTranslation']+1e-8 or abs(math.atan2(math.sin(end['yaw']-reference[2]),math.cos(end['yaw']-reference[2])))>base['maxYaw']+1e-8):reasons.append('shallow-terminal-base-drift-invalid')
 old={o['id']:o['position']for o in before.get('objects',[])if o['id']!='ball'};final={o['id']:o['position']for o in initial.get('objects',[])if o['id']!='ball'}
 motion=p.get('settlingMotion')or {}
 if(set(motion)!=set(old)or any(not number(v.get(k))or not 0<=v[k]<=.01 for v in motion.values()for k in ['maxHorizontal','maxRise'])):reasons.append('shallow-settling-motion-invalid')
 for name,steps in [('preservation',3000),('settlingPreservation',1000)]:
  a=p.get(name)or {};objects=a.get('objects')or {}
  if(a.get('schema')!='hsr-object-preservation-v1'or a.get('physicsSteps')!=steps or set(objects)!=set(old)or set(objects)!=set(final)):
   reasons.append('shallow-object-audit-missing:'+name);continue
  for identity,v in objects.items():
   start=v.get('start');d=v.get('maxDisplacement');h=v.get('minHeight')
   if not vector(start)or not number(d)or d<0 or not number(h):reasons.append('shallow-object-audit-invalid:'+name);continue
   minimum=start[1]-.03 if name=='preservation'else 0 if identity=='baseball'else .4 if identity in('cube','can','mug','bottle','plate')else start[1]-.03
   maximum=.01 if name=='preservation'else math.hypot(.01,max(.01,start[1]-minimum))+1e-8
   if(h<minimum or d>maximum or name=='preservation'and(start!=old[identity]or not vector(final[identity])or math.dist(start,final[identity])>d+1e-8)):reasons.append('shallow-object-displaced-or-fell:'+name)
 if(len(trace)!=60 or any(not number(row.get('time'))or abs(row['time']-(2+(i+1)/10))>1e-8 or not vector(row.get('joints'),5)or not vector(row.get('goal'),6)or
    not vector(row.get('target'))or not vector(row.get('velocity'))or not vector(row.get('tcp'))or type(row.get('held'))is not bool or not number(row.get('heldFor'))or not 0<=row['heldFor']<=6 or
    not vector(row.get('base',{}).get('position'))or not number(row.get('base',{}).get('yaw'))or
    not number(row.get('opening'))or not 0<=row['opening']<=1 or any(not lo-.02<=v<=hi+.02 for v,(lo,hi)in zip(row['joints'],limits))for i,row in enumerate(trace))):return reasons+['shallow-trace-invalid']
 if(len(moves)!=2 or not vector(p.get('closingJoints'),5)or moves[0].get('start')!=origin['joints']or moves[1].get('start')!=trace[29]['joints']or p['closingJoints']!=trace[49]['joints']or
    moves[0].get('base')!={'position':origin['base']['position'],'yaw':origin['base']['yaw']}or moves[1].get('base')!=trace[29]['base']):return reasons+['shallow-move-measured-start-invalid']
 for i,m in enumerate(moves):
  position=[pose.get('position',[0,0,0])[0],ground[1]+.18,pose.get('position',[0,0,0])[2]]if i==0 else pose.get('position')
  if(m.get('phase')!=['hover','shallow-approach'][i]or m.get('seconds')!=[3,2][i]or m.get('position')!=position or not vector(m.get('start'),5)or not vector(m.get('end'),5)):reasons.append('shallow-move-contract-invalid')
  b=m.get('base')or {}
  if(not vector(reference)or not number(base.get('maxTranslation'))or not number(base.get('maxYaw'))or not vector(b.get('position'))or not number(b.get('yaw'))or
     math.hypot(b['position'][0]-reference[0],b['position'][2]-reference[1])>base['maxTranslation']+1e-8 or
     abs(math.atan2(math.sin(b['yaw']-reference[2]),math.cos(b['yaw']-reference[2])))>base['maxYaw']+1e-8):reasons.append('shallow-move-base-exceeds-global-audit')
 for i,row in enumerate(trace):
  t=(i+1)/10;phase='hover'if i<30 else'shallow-approach'if i<50 else'shallow-close'
  m=moves[0 if i<30 else 1];u=t/3 if i<30 else (t-3)/2;s=u*u*(3-2*u)
  goal=[a+(b-a)*s for a,b in zip(m.get('start',[]),m.get('end',[]))]+[.5]if i<50 else p['closingJoints']+[0]
  if(row.get('phase')!=phase or not same(row['goal'],goal)or any(not lo<=v<=hi for v,(lo,hi)in zip(row['goal'][:5],limits))or
     row['target'][1]<.4 or row['target'][1]>p.get('preparationPeakHeight',0)+1e-8 or not row['held']and row['heldFor']!=0 or i and row['heldFor']>trace[i-1]['heldFor']+.1+1e-8):reasons.append('shallow-trace-command-or-physical-discontinuity');break
  b=row['base']
  if(not vector(reference)or not number(base.get('maxTranslation'))or not number(base.get('maxYaw'))or
     math.hypot(b['position'][0]-reference[0],b['position'][2]-reference[1])>base['maxTranslation']+1e-8 or
     abs(math.atan2(math.sin(b['yaw']-reference[2]),math.cos(b['yaw']-reference[2])))>base['maxYaw']+1e-8):reasons.append('shallow-trace-base-exceeds-global-audit');break
 last=trace[-1]
 if(last['joints']!=robot.get('joints')or last['opening']!=robot.get('gripper')or last['target']!=ball.get('position')or last['velocity']!=ball.get('velocity')or last['tcp']!=robot.get('endEffectorPosition')or last['base']!={'position':end.get('position'),'yaw':end.get('yaw')}or
    last['held']is not True or abs(last['heldFor']-p.get('heldFor',-1))>1e-8):reasons.append('shallow-terminal-trace-mismatch')
 return reasons


def quality_rejections(result, quality, duration, physics, *, require_settlement_audit=False):
    reasons = []
    if result.get('status') != 'success' or result.get('teacherFailure') or not result.get('grasped') or not result.get('lifted'):
        reasons.append('physical-transport-not-successful')
    settled = result.get('settledFor', -1)
    if not isinstance(settled, (int, float)) or not math.isfinite(settled) or settled < 2 - 1e-8:
        reasons.append('continuous-two-second-settlement-missing')
    if physics != {'robotMassKg': 33.458653999999996, 'armTorqueNm': 40, 'fixedTimestep': 0.002}:
        reasons.append('physical-configuration-changed')
    if quality.get('schema') != 'hsr-teacher-quality-v2':
        reasons.append('unsupported-quality-audit')
    required = ['physicsSteps', 'physicsSeconds', 'maxOtherMotion', 'minOtherHeight', 'peakUnintendedForce',
                'above25Seconds', 'prematureDrops', 'peakTargetFingerForce', 'peakTargetEnvironmentForce']
    if any(type(quality.get(k)) not in (int, float) or not math.isfinite(quality[k]) or quality[k] < 0 for k in required):
        return reasons + ['missing-or-invalid-physical-audit']
    steps, seconds = quality['physicsSteps'], quality['physicsSeconds']
    if steps != int(steps) or abs(seconds - steps * .002) > 1e-7 or not duration <= seconds + 1e-8 < duration + .101:
        reasons.append('audit-does-not-cover-completed-two-ms-steps')
    if quality['maxOtherMotion'] > .01 or quality['minOtherHeight'] <= .4:
        reasons.append('other-candidate-displaced-or-fell')
    if quality['peakUnintendedForce'] >= 100 or quality['above25Seconds'] >= .1:
        reasons.append('unintended-contact-limit-exceeded')
    if quality['peakTargetFingerForce'] <= 0 or quality['peakTargetEnvironmentForce'] <= 0:
        reasons.append('measured-grasp-or-environment-contact-missing')
    if max(quality['peakTargetFingerForce'], quality['peakTargetEnvironmentForce']) >= TARGET_CONTACT_OUTLIER_N:
        reasons.append('target-contact-numeric-outlier')
    drops = quality.get('drops', [])
    if quality['prematureDrops'] or len(drops) != result.get('drops') or any(d.get('phase') not in ('release', 'settle') for d in drops):
        reasons.append('unresolved-or-premature-drop')
    tail=quality.get('settlement',{})
    if require_settlement_audit:
        if(any(type(tail.get(k))not in(int,float)or not math.isfinite(tail[k])for k in ['physicsSteps','physicsSeconds','peakUnintendedForce'])or
           tail['physicsSteps']!=int(tail['physicsSteps'])or not 1000<=tail['physicsSteps']<=steps or abs(tail['physicsSeconds']-tail['physicsSteps']*.002)>1e-8 or
           abs(tail['physicsSeconds']-settled)>1e-8 or tail['peakUnintendedForce']!=0):
            reasons.append('continuous-settlement-force-audit-missing-or-pressure')
    elif quality.get('phases', {}).get('settle', {}).get('peak', -1) != 0:
        reasons.append('robot-environment-contact-continues-during-settlement')
    return reasons


def validate_demo_packet(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError('Recording exceeds size limit')
    episode = json.loads(raw)
    if episode.get('schema') != 'hsr-demonstration-v1' or episode.get('kind') != 'demonstration':
        raise ValueError('Calibration is not a newly collected demonstration')
    script = "import {validateDemonstrationEpisode} from './src/learning/recording.ts';let t='';for await(const p of process.stdin)t+=p;validateDemonstrationEpisode(JSON.parse(t));"
    checked = subprocess.run(['node', '--input-type=module', '-e', script], cwd=ROOT, input=raw, capture_output=True)
    if checked.returncode:
        raise ValueError(checked.stderr.decode().strip())
    return episode


def decode_demo_images(episode):
    decoded, unique = {}, {view: set() for view in ('head', 'hand')}
    for sample in episode['samples']:
        for view in unique:
            key = sample['images'][view]
            image = episode['images'][key]
            if abs(image['timestamp'] - sample['sourceTimestamp']) > 1e-8:
                raise ValueError('Image source time mismatch')
            data = base64.b64decode(image['dataUrl'].split(',')[1], validate=True)
            with Image.open(io.BytesIO(data)) as value:
                if value.mode != 'RGB' or value.format != 'JPEG' or value.size != (320, 240):
                    raise ValueError('Expected actual 320x240 RGB JPEG')
                pixels = np.array(value)
            if pixels.std() <= 10:
                raise ValueError('Blank camera image')
            decoded[key] = pixels
            unique[view].add(hashlib.sha256(data).hexdigest())
    return decoded, unique


TAKEOVER_SCENE = 'hsr-stationary-v12-takeover'


def takeover_seed(seed):
    """Train seeds never used for the mug before the takeover data (the parent's mug seeds end at 1149; gentle data 1200-1599)."""
    return 1150 <= seed <= 1199 or 1600 <= seed <= 1999


def takeover_integrity(report, audit, collection):
    """A takeover segment (teacher after the learned policy) must come from a takeover report that names one model,
    fetched nothing outside the model folder, and was recorded on a reserved takeover seed. Other packets fetch no model."""
    takeover = collection['sceneVersion'] == TAKEOVER_SCENE
    if (report.get('schema') == 'hsr-takeover-collection-v1') != takeover:
        raise ValueError('Takeover packets need a takeover collection report')
    if not takeover:
        if report.get('modelRequests'):
            raise ValueError('Collection execution/source integrity failed')
        return
    model = report.get('model') or {}
    if not (isinstance(model.get('id'), str) and all(re.fullmatch('[0-9a-f]{64}', str(model.get(k, ''))) for k in ('graphSha256', 'weightsSha256'))):
        raise ValueError('Takeover report must name its model')
    # Scene assets (e.g. /assets/.../models/*.gltf) are not model files; any model-like request must be under /models/.
    model_like = lambda url: re.search(r'huggingface|\.onnx|vla-probe', url) or re.match(r'^https?://[^/]+/models/', url)
    if any(model_like(url) and not re.match(r'^https?://[^/]+/models/', url) for url in report.get('modelRequests', [])):
        raise ValueError('Takeover collection fetched files outside the model folder')
    entry = audit.get('takeover') or {}
    if (entry.get('model') != model or entry.get('entryPhase') not in (0, 1, 3, 4, 5) or not isinstance(entry.get('reason'), str) or
            not takeover_seed(audit.get('scene', {}).get('seed', -1))):
        raise ValueError('Takeover audit does not match its report or reserved seeds')


def minimum_samples(collection):
    """Takeover segments start mid-task (e.g. over the basket), so 3 s of fresh images suffice; full demonstrations need 10 s."""
    return 30 if collection['sceneVersion'] == TAKEOVER_SCENE else 100


def certify_attempt(packet_path, audit_path, run, report):
    raw = Path(packet_path).read_bytes()
    episode = validate_demo_packet(raw)
    if episode['collection']['sceneVersion'] == 'hsr-stationary-v11-no-reset-sequence':
        raise ValueError('No-reset demonstrations require paired sequence certification')
    audit_raw = Path(audit_path).read_bytes()
    audit = json.loads(audit_raw)
    collection = episode['collection']
    # A new preparation file must not invalidate a previously saved source inventory.
    expected_files = json.loads(subprocess.check_output(['git', 'show', f"{collection['sourceCommit']}:tools/vla-data/collection-source-files.json"], cwd=ROOT))
    registry_hash = hashlib.sha256(Path(__file__).with_name('seed-registry.json').read_bytes()).hexdigest()
    if (set(report.get('sourceHashes', {})) != set(expected_files) or
        report.get('schema') not in ('hsr-demo-collection-v1', 'hsr-takeover-collection-v1') or report.get('kind') != 'demonstration' or
        collection['seedRegistrySha256'] != registry_hash or report.get('seedRegistrySha256') != registry_hash or
        hashlib.sha256(raw).hexdigest() != run.get('sourceSha256') or
        hashlib.sha256(audit_raw).hexdigest() != run.get('auditSha256')):
        raise ValueError('Source-file set, registry or receipt hashes mismatch')
    if (collection['episodeId'] != run['episodeId'] or episode['seed'] != run['seed'] or
        episode['instruction'] != run['instruction'] or collection['sourceCommit'] != report['sourceHead'] or
        audit['collection'] != collection or audit['scene']['id'] != collection['sceneVersion'] or
        audit['scene']['seed'] != episode['seed'] or audit['instruction'] != episode['instruction']):
        raise ValueError('Receipt, source, session or scene mismatch')
    if report.get('sourceChanged') or report.get('fatalError') or any(report.get(k) for k in ('pageErrors', 'consoleErrors')):
        raise ValueError('Collection execution/source integrity failed')
    takeover_integrity(report, audit, collection)
    for path, expected in report['sourceHashes'].items():
        data = subprocess.check_output(['git', 'show', f"{report['sourceHead']}:{path}"], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Saved collection source hash mismatch')
    decoded, unique = decode_demo_images(episode)
    reasons = quality_rejections(audit['result'], audit['quality'], episode['duration'], episode['physics'],require_settlement_audit=collection['sceneVersion']=='hsr-stationary-v10-shallow-reseat')
    if collection['sceneVersion']=='hsr-stationary-v6-recovery':
        reasons.extend(recovery_preparation_rejections(audit['scene'],audit['initial']))
    if collection['sceneVersion']=='hsr-stationary-v7-ball-regrasp':
        reasons.extend(ball_recovery_preparation_rejections(audit['scene'],audit['initial']))
    if collection['sceneVersion']=='hsr-stationary-v8-physical-ball-loss':
        reasons.extend(physical_ball_preparation_rejections(audit['scene'],audit['initial']))
    if collection['sceneVersion']=='hsr-stationary-v9-empty-carry-loss':
        reasons.extend(empty_carry_preparation_rejections(audit['scene'],audit['initial']))
    if collection['sceneVersion']=='hsr-stationary-v10-shallow-reseat':
        reasons.extend(shallow_ball_preparation_rejections(audit['scene'],audit['initial']))
    protection = audit.get('preservation', {})
    other_start = {o['id']:o['position'] for o in audit['initial']['objects'] if o['id'] != run['target']}
    tracked = protection.get('objects', {})
    if (protection.get('schema') != 'hsr-object-preservation-v1' or
        protection.get('physicsSteps') != audit['quality']['physicsSteps'] or set(tracked) != set(other_start)):
        reasons.append('all-non-target-object-audit-missing')
    else:
        for identity, values in tracked.items():
            displacement, height = values.get('maxDisplacement'), values.get('minHeight')
            if (values.get('start') != other_start[identity] or type(displacement) not in (int,float) or
                type(height) not in (int,float) or not math.isfinite(displacement) or not math.isfinite(height) or
                displacement < 0 or displacement > .01 or height < other_start[identity][1] - .03):
                reasons.append('non-target-object-displaced-or-fell:' + identity)
    if episode['error'] or run.get('runnerFailure'):
        reasons.append('capture-or-runner-failed')
    if len(episode['samples']) < minimum_samples(collection) or any(len(h) < 10 for h in unique.values()):
        reasons.append('insufficient-fresh-transport-images')
    acceptance = {'schema': 'hsr-demo-acceptance-v1', 'qualityPolicy': QUALITY_POLICY, 'status': 'rejected' if reasons else 'accepted', 'reasons': reasons,
                  'sourceSha256': hashlib.sha256(raw).hexdigest(), 'auditSha256': hashlib.sha256(audit_raw).hexdigest(),
                  'releaseDropFlagsRetained': audit['quality'].get('drops', []), 'quality': audit['quality']}
    return {'episode': episode, 'acceptance': acceptance, 'sourceSha256': acceptance['sourceSha256']}, decoded
