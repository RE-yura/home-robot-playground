import test from 'node:test';
import assert from 'node:assert/strict';
import { PlanExecution } from '../src/learning/plan-execution.ts';
import type { LanguageReceipt } from '../src/learning/language-planner.ts';

const MUG = {ja:'コップをかごに入れる',en:'Put the mug in the basket.'}, BALL = {ja:'ボールをかごに入れる',en:'Put the tennis ball in the basket.'}, OUT = {ja:'コップをかごから出す',en:'Take the mug out of the basket.'};
const skill = (en: string) => en === MUG.en ? 'mug' : en === BALL.en ? 'ball' : 'other';
const plan = (tasks: {ja:string;en:string}[]) => ({ instruction: 'テスト', goal: {status:'proposed',tasks}, rawText: JSON.stringify({status:'proposed',tasks:tasks.map(t=>({skill:skill(t.en),...t}))}) }) as unknown as LanguageReceipt;
const receipt = plan([MUG, BALL]);
function fixture(chosen = receipt) {
  let time = 1000, generation = 9;
  const instances: any[] = [];
  const sim = {resetGeneration:generation, observation:()=>({objects:[{id:'mug',position:[0,.5,0]},{id:'ball',position:[0,.5,0]}], robot:{gripperContact:false}})} as any;
  const factory = (_sim:any,instruction:string,_policy:any,options:{target:unknown}) => {
    const r = {instruction,target:options.target,stopped:false,error:null,chunks:[],inferencePending:false,result:{status:'running',elapsed:0,drops:0,settledFor:0}, afterStep(_dt:number){},stop(){this.stopped=true;}};
    instances.push(r); return r as any;
  };
  const execution = new PlanExecution(sim, {} as any, chosen, 0, ()=>time, factory);
  return {execution,sim,instances,elapse(ms:number){time+=ms;},tick(ms:number){time+=ms;execution.poll();}, now:()=>time};
}
test('a proposed LLM order is executed through VLA, preserving the world between tasks',()=>{
  const f=fixture(); f.execution.start();
  assert.equal(f.instances[0].instruction,'Put the mug in the basket.');
  f.tick(2000); const first=f.instances[0]; first.result.status='success';first.stopped=true;f.execution.poll();
  assert.equal(f.instances[1].instruction,'Put the tennis ball in the basket.');
  assert.equal(f.sim.resetGeneration,9);
  assert.equal(f.execution.tasks[0].status,'success');assert.equal(f.execution.tasks[1].status,'running');
  f.tick(500);f.instances[1].result.status='success';f.instances[1].stopped=true;f.execution.poll();
  assert.equal(f.execution.status,'success');const elapsed=f.execution.elapsedMs;f.tick(10000);assert.equal(f.execution.elapsedMs,elapsed);
});
test('failure never starts the next task and stop freezes all clocks',()=>{
  const f=fixture();f.execution.start();f.tick(900);f.instances[0].stopped=true;f.instances[0].result.status='failed';f.instances[0].result.reason='target-fell';f.execution.poll();
  assert.equal(f.instances.length,1);assert.equal(f.execution.status,'failed');assert.equal(f.execution.tasks[1].status,'queued');
  const g=fixture();g.execution.start();g.tick(500);g.execution.stop();const elapsed=g.execution.elapsedMs;g.tick(5000);assert.equal(g.execution.elapsedMs,elapsed);assert.equal(g.execution.status,'stopped');
});
test('progress requires measured hold/lift and canonical settling; a drop returns to regrasp',()=>{
  const f=fixture();f.execution.start();const r=f.instances[0];r.chunks.push({});r.afterStep(.002);
  f.tick(1000);assert.equal(f.execution.tasks[0].step,0,'elapsed time alone cannot complete a step');
  f.sim.observation=()=>({objects:[{id:'mug',position:[0,.5,0]}],robot:{gripperContact:true,contactId:'mug'}});
  for(let i=0;i<100;i++)r.afterStep(.002);assert.equal(f.execution.tasks[0].step,1);
  f.sim.observation=()=>({objects:[{id:'mug',position:[0,.66,0]}],robot:{gripperContact:true,contactId:'mug'}});r.afterStep(.002);assert.equal(f.execution.tasks[0].step,2);
  r.result.drops=1;f.sim.observation=()=>({objects:[{id:'mug',position:[0,.5,0]}],robot:{gripperContact:false}});r.afterStep(.002);
  assert.equal(f.execution.tasks[0].step,1);assert.equal(f.execution.tasks[0].recovering,true);
  r.result.settledFor=1;r.afterStep(.002);assert.equal(f.execution.tasks[0].step,3);assert.equal(f.execution.status,'running');
  r.result.status='success';r.stopped=true;f.execution.poll();assert.equal(f.execution.tasks[0].status,'success');
});
test('clarifications and mismatched receipts cannot drive a robot',()=>{
  const f=fixture();
  assert.throws(()=>new PlanExecution(f.sim,{} as any,{...receipt,goal:{status:'clarify',tasks:[]}},0));
  assert.throws(()=>new PlanExecution(f.sim,{} as any,{...receipt,goal:{status:'proposed',tasks:[BALL,MUG]}},0));
});
test('settling an already placed target does not invent observed grasp or lift; stopped clocks are latched before polling',()=>{
  const f=fixture();f.execution.start();const r=f.instances[0];
  r.result.settledFor=.002;r.afterStep(.002);
  assert.deepEqual(f.execution.tasks[0].observed,[false,false,true,false]);
  f.tick(1000);r.result.status='success';r.stop();const ended=f.execution.tasks[0].endedAt;
  assert.equal(ended,f.now());f.tick(10000);
  assert.equal(f.execution.tasks[0].endedAt,ended);assert.equal(f.execution.tasks[0].durations.reduce((a,b)=>a+b,0),1000);
});
test('stop at a successful task boundary never starts the next task',()=>{
 const f=fixture();f.execution.start();f.instances[0].result.status='success';f.execution.stop();f.tick(1000);
 assert.equal(f.instances.length,1);assert.equal(f.execution.status,'stopped');assert.equal(f.execution.tasks[0].status,'success');
});
test('overall time freezes at the final physical result even before the UI polls',()=>{
 const f=fixture();f.execution.tasks.pop();f.execution.start();f.tick(1000);const r=f.instances[0];r.result.status='success';r.stop();
 const ended=f.execution.elapsedMs;
 // Advance the injected clock without polling.
 f.elapse(10000);
 assert.equal(f.execution.elapsedMs,ended);
});

test('an unlearned step runs the VLA with its own instruction and no success target, after the learned one; it shows no grasp steps',()=>{
  const f=fixture(plan([MUG,OUT])); f.execution.start();
  assert.equal(f.instances[0].instruction,'Put the mug in the basket.'); assert.equal(f.instances[0].target,'mug');
  assert.deepEqual(f.execution.tasks.map(t=>[t.label,t.instruction,t.target]),[[MUG.ja,MUG.en,'mug'],[OUT.ja,OUT.en,null]]);
  f.tick(1000);f.instances[0].result.status='success';f.instances[0].stopped=true;f.execution.poll();
  const r=f.instances[1]; assert.equal(r.instruction,'Take the mug out of the basket.'); assert.equal(r.target,null);
  f.sim.observation=()=>({objects:[{id:'mug',position:[0,.7,0]}],robot:{gripperContact:true,contactId:'mug'}});
  for(let i=0;i<200;i++)r.afterStep(.002);
  assert.equal(f.execution.tasks[1].step,0); assert.deepEqual(f.execution.tasks[1].observed,[false,false,false,false]);
  r.result.status='failed';r.result.reason='timeout';r.stopped=true;f.execution.poll();
  assert.equal(f.execution.status,'failed'); assert.equal(f.execution.tasks[1].status,'failed');
});

test('a release over the rim counted as a drop does not erase the measured grasp once the mug settles in the basket',()=>{
  const f=fixture();f.execution.start();const r=f.instances[0];
  f.sim.observation=()=>({objects:[{id:'mug',position:[0,.5,0]}],robot:{gripperContact:true,contactId:'mug'}});
  for(let i=0;i<100;i++)r.afterStep(.002);
  f.sim.observation=()=>({objects:[{id:'mug',position:[0,.66,0]}],robot:{gripperContact:true,contactId:'mug'}});r.afterStep(.002);
  r.result.settledFor=0;f.sim.observation=()=>({objects:[{id:'mug',position:[0,.6,0]}],robot:{gripperContact:false}});
  r.result.drops=1;r.afterStep(.002);
  assert.equal(f.execution.tasks[0].recovering,true,'a real drop still shows the regrasp wait');
  r.result.settledFor=.5;r.afterStep(.002);
  assert.deepEqual(f.execution.tasks[0].observed,[true,true,true,false]);assert.equal(f.execution.tasks[0].recovering,false);
});
