// New-session teacher data only. No model inference, fallback or publication.
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
import {mkdir,readdir,writeFile,readFile} from 'node:fs/promises';
import {randomUUID,createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {collectionRequest} from './collection-request.ts';
const repo=resolve(import.meta.dirname,'../..'),destination=process.argv[2],modulePath=process.env.PLAYWRIGHT_MODULE;
if(!destination||!modulePath||!process.env.HSR_RUNS)throw new Error('Usage: HSR_RUNS=... PLAYWRIGHT_MODULE=... node tools/cloud/collect-demos.mjs NEW_OUTPUT_DIRECTORY');
const git=(...args)=>execFileSync('git',args,{cwd:repo,encoding:'utf8'}).trim();
if(git('status','--porcelain'))throw new Error('Save all source changes before new demonstration collection');
const registryRaw=await readFile(resolve(repo,'tools/vla-data/seed-registry.json'));
const registry=JSON.parse(registryRaw),seedRegistrySha256=createHash('sha256').update(registryRaw).digest('hex');
const partition=seed=>Object.entries(registry.partitions).find(([name,[first,last]])=>name!=='learned-rollout'&&seed>=first&&seed<=last)?.[0];
const runs=JSON.parse(process.env.HSR_RUNS);
if(!Array.isArray(runs)||!runs.length||runs.some(r=>!Array.isArray(r)||r.length!==2||!['mug','ball'].includes(r[0])||!Number.isInteger(r[1])||!partition(r[1]))||new Set(runs.map(r=>JSON.stringify(r))).size!==runs.length)throw new Error('Use unique reserved training/validation seed-instruction pairs');
const sourceHead=git('rev-parse','HEAD'),sourceFiles=JSON.parse(await readFile(resolve(repo,'tools/vla-data/collection-source-files.json'),'utf8'));
const fingerprint=async()=>Object.fromEntries(await Promise.all(sourceFiles.map(async path=>[path,createHash('sha256').update(await readFile(resolve(repo,path))).digest('hex')])));
const sourceHashes=await fingerprint();
const root=resolve(destination);await mkdir(root,{recursive:true});if((await readdir(root)).length)throw new Error('Preserve prior attempts: output must be empty');
const {chromium}=await import(pathToFileURL(resolve(modulePath)).href);
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const report={schema:'hsr-demo-collection-v1',kind:'demonstration',sourceHead,sourceHashes,seedRegistrySha256,sourceChanged:false,browserVersion:browser.version(),runs:[],pageErrors:[],consoleErrors:[],modelRequests:[]};
const save=()=>writeFile(root+'/collection-report.json',JSON.stringify(report));
try{
 const page=await browser.newPage({viewport:{width:640,height:480}});
 page.on('pageerror',e=>report.pageErrors.push(String(e)));page.on('console',m=>{if(m.type()==='error')report.consoleErrors.push(m.text())});page.on('request',r=>{if(/huggingface|\.onnx|vla-probe/.test(r.url()))report.modelRequests.push(r.url())});
 await page.goto(new URL('/tools/vla-data/',process.env.HSR_DIAGNOSTIC_URL??'http://127.0.0.1:5193').href);
 await page.waitForFunction(()=>typeof window.dataPilot?.startDemonstration==='function'&&[...document.querySelectorAll('button')].some(b=>b.textContent==='教師運搬（検証）'&&!b.disabled),null,{timeout:120000});
 await page.evaluate(()=>{for(const label of document.querySelectorAll('.panel-label')){label.parentElement.style.width='0px';label.parentElement.style.height='0px'}setInterval(()=>{const{sim:s,capture:c}=window.dataPilot;if(s&&c&&!c.stopped&&!c.captureDue(s.data.time))s.advance(.1)},5)});
 for(const[target,seed]of runs){
  const {instruction,collection}=collectionRequest({target,seed,env:process.env,sourceHead,seedRegistrySha256,episodeId:randomUUID()});
  const run={episodeId:collection.episodeId,target,seed,instruction,partition:partition(seed),wallStartedAt:new Date().toISOString(),packetCreated:false,runnerFailure:null};report.runs.push(run);await save();
  let timer;const wallStart=Date.now();
  try{
   await page.evaluate(request=>window.dataPilot.startDemonstration(request),{instruction,seed,collection});
   await page.waitForFunction(id=>window.dataPilot.audit?.collection?.episodeId===id&&
     (!window.dataPilot.capture.stopped||window.dataPilot.capture.episode?.collection?.episodeId===id),collection.episodeId,{timeout:30000});
   timer=setInterval(async()=>{try{console.log(JSON.stringify(await page.evaluate(()=>({time:window.dataPilot.sim.data.time,status:document.querySelector('output')?.textContent}))))}catch{}},10000);timer.unref();
   try{await page.waitForFunction(()=>window.dataPilot.capture.stopped,null,{timeout:1800000})}catch(e){run.runnerFailure=String(e)}
   const data=await page.evaluate(()=>{const{capture:c,teacher:t,sim:s,audit:a}=window.dataPilot;c.stop();return{episode:c.episode,audit:{...a,result:t.result,final:s.observation()}}});
   if(data.episode?.kind!=='demonstration'||data.episode.collection.episodeId!==collection.episodeId)throw new Error('Stale or calibration packet returned');
   const raw=JSON.stringify(data.episode),audit=JSON.stringify(data.audit);await writeFile(root+`/${run.episodeId}.json`,raw);await writeFile(root+`/${run.episodeId}-audit.json`,audit);
   Object.assign(run,{packetCreated:true,sourceSha256:createHash('sha256').update(raw).digest('hex'),auditSha256:createHash('sha256').update(audit).digest('hex'),result:data.audit.result,error:data.episode.error,samples:data.episode.samples.length});
  }catch(e){run.runnerFailure=String(e);process.exitCode=1}finally{clearInterval(timer);run.wallMs=Date.now()-wallStart;await writeFile(root+`/${run.episodeId}-receipt.json`,JSON.stringify(run));await save();console.log(JSON.stringify(run));}
 }
 report.sourceChanged=JSON.stringify(sourceHashes)!==JSON.stringify(await fingerprint())||sourceHead!==git('rev-parse','HEAD')||Boolean(git('status','--porcelain'));
 if(report.sourceChanged||report.pageErrors.length||report.consoleErrors.length||report.modelRequests.length||report.runs.some(r=>r.runnerFailure||r.error))process.exitCode=1;
}catch(e){report.fatalError=String(e);throw e}finally{await save();await browser.close()}
