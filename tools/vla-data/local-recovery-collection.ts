import type {DemoCollection,Instruction} from '../../src/learning/recording.ts';
import type {LearningCapture} from '../../src/learning/capture.ts';
import type {Simulation} from '../../src/simulation/mujoco.ts';
import {advanceCollectionInterval,type CollectionTeacher} from './collection-interval.ts';
import {RecoveryPreparationError} from '../../src/learning/recovery-scene.ts';

type Request={instruction:Instruction;seed:number;collection:DemoCollection};
type Current={capture:LearningCapture|null;sim:Simulation|null;teacher:CollectionTeacher|null;audit:Record<string,unknown>|null};
const writer='http://127.0.0.1:5197';
async function exchange(path:string,body?:unknown){
 const response=await fetch(writer+path,body===undefined?undefined:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const result=await response.json();if(!response.ok)throw new Error(result.error);return result;
}

// Own diagnostic UI code, entered by a visible button. Cameras remain the normal
// renderer; only completed physics steps between synchronized captures accelerate.
export async function collectRecovery(start:(request:Request)=>void,read:()=>Current,status:(text:string)=>void){
 const pageErrors:string[]=[],consoleErrors:string[]=[];
 const onError=(event:ErrorEvent)=>pageErrors.push(event.message),onRejection=(event:PromiseRejectionEvent)=>pageErrors.push(String(event.reason));
 const originalError=console.error;
 console.error=(...args:unknown[])=>{consoleErrors.push(args.map(String).join(' '));originalError(...args);};
 window.addEventListener('error',onError);window.addEventListener('unhandledrejection',onRejection);
 let fatalError:string|null=null;
 try{
  const plan:Request[]=await exchange('/plan');
  for(const [index,request]of plan.entries()){
   await exchange('/start',{episodeId:request.collection.episodeId});
   status(`つかみ直し収録 ${index+1}/${plan.length} · Seed ${request.seed}`);
   const wall=performance.now();let runnerFailure:string|null=null,episodeJson:string|null=null,auditJson:string|null=null,preparationFailureJson:string|null=null;
   try{
    start(request);
    const {capture:c,teacher:t,sim:s}=read();
    if(!c||!t||!s)throw new Error('Capture unavailable');
    while(!c.stopped){
     if(t.result.status!=='running'){c.stop();break;}
     if(performance.now()-wall>1800000)throw new Error('Bounded capture wall timeout');
     if(!c.captureDue(s.data.time))advanceCollectionInterval(s,c,t);
     await new Promise<void>(resolve=>setTimeout(resolve,5));
    }
    const episode=c.episode;
    if(episode?.kind!=='demonstration'||episode.collection.episodeId!==request.collection.episodeId)throw new Error('Stale or calibration packet');
    episodeJson=JSON.stringify(episode);auditJson=JSON.stringify({...read().audit,result:t.result,final:s.observation()});
   }catch(error){
    runnerFailure=String(error);const current=read();current.capture?.stop();
    if(error instanceof RecoveryPreparationError)preparationFailureJson=JSON.stringify(error.evidence);
    const episode=current.capture?.episode;
    if(episode?.kind==='demonstration'&&episode.collection.episodeId===request.collection.episodeId){episodeJson=JSON.stringify(episode);auditJson=JSON.stringify({...current.audit,result:current.teacher?.result,final:current.sim?.observation()});}
   }
   await exchange('/attempt',{episodeId:request.collection.episodeId,runnerFailure,wallMs:performance.now()-wall,episodeJson,auditJson,preparationFailureJson});
   if(runnerFailure&&!preparationFailureJson)throw new Error(runnerFailure);
  }
  status('つかみ直し収録完了 · 品質認定待ち');
 }catch(error){fatalError=String(error);status(`収録停止: ${fatalError}`);}
 finally{
  console.error=originalError;window.removeEventListener('error',onError);window.removeEventListener('unhandledrejection',onRejection);
  const modelRequests=performance.getEntriesByType('resource').map(e=>e.name).filter(url=>/huggingface|\.onnx|vla-probe/.test(url));
  await exchange('/finish',{browserVersion:navigator.userAgent,pageErrors,consoleErrors,modelRequests,fatalError});
 }
}
