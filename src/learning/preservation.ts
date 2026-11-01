import type {Simulation} from '../simulation/mujoco.ts';
import type {Vec3} from '../simulation/types.ts';

// Extra oracle audit for collection. Never read by the learned-policy input path.
export class ObjectPreservationAudit {
  readonly metrics = {schema:'hsr-object-preservation-v1',physicsSteps:0,
    objects:{} as Record<string,{start:Vec3;maxDisplacement:number;minHeight:number}>};
  constructor(privateSim:Simulation,targetId:string) {
    for(const object of privateSim.observation().objects) if(object.id!==targetId)
      this.metrics.objects[object.id]={start:[...object.position],maxDisplacement:0,minHeight:object.position[1]};
  }
  update(sim:Simulation,dt:number) {
    if(Math.abs(dt-.002)>1e-10)throw new Error('Preservation audit requires each 2ms step');
    this.metrics.physicsSteps++;
    for(const object of sim.observation().objects) {
      const audit=this.metrics.objects[object.id];if(!audit)continue;
      audit.maxDisplacement=Math.max(audit.maxDisplacement,Math.hypot(...object.position.map((v,i)=>v-audit.start[i])));
      audit.minHeight=Math.min(audit.minHeight,object.position[1]);
    }
  }
}
