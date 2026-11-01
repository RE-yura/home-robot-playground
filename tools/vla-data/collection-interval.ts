import type {LearningCapture} from '../../src/learning/capture.ts';
import type {Simulation} from '../../src/simulation/mujoco.ts';
import type {StationaryTeacher} from '../../src/learning/teacher.ts';

export type CollectionTeacher=Pick<StationaryTeacher,'result'|'evaluator'|'next'>;
export function collectionPaused(capture:Pick<LearningCapture,'stopped'>|null,teacher:Pick<CollectionTeacher,'result'>|null){
 return Boolean(capture&&(capture.stopped||(teacher&&teacher.result.status!=='running')));
}
export function advanceCollectionInterval(sim:Simulation,capture:LearningCapture,teacher:CollectionTeacher){
 for(let n=0;n<50&&!capture.stopped&&teacher.result.status==='running';n++)if(!sim.step())break;
}
