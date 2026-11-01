import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import * as download from '../src/learning/learned-model-download.ts';
import * as contract from '../src/learning/learned-onnx-contract.ts';
const expected={bytes:28640194,sha256:'8785fe59d6cf3c89a18bbef265bab4be696cdc54c3ebcb694319ff838df39c55'};
const archive={url:'/assets/learned-vla/shared-8785.onnx.gz',bytes:1882482,sha256:'4d88815bd100419288401d2d5d9d0a25f2e653871c2e37c6a2e3f93c18ec8480'};
const digest=(bytes:Uint8Array)=>createHash('sha256').update(bytes).digest('hex');
const api=download as typeof download & {loadSharedLearnedGraph:(cache:Pick<Cache,'match'|'put'|'delete'>|undefined,signal:AbortSignal)=>Promise<File>};
const metadata=contract as typeof contract & {LEARNED_SHARED_GRAPH:typeof expected};
const signal=()=>new AbortController().signal;
function memoryCache(){const entries=new Map<string,Response>();return{entries,async match(key:string){return entries.get(key)?.clone();},async put(key:string,v:Response){entries.set(key,v);},async delete(key:string){return entries.delete(key);}};}
const gzip=()=>fs.readFileSync(new URL('../public'+archive.url,import.meta.url));
test('shared graph identity permits only the verified graph with unchanged external weights',()=>{
 assert.deepEqual(contract.validateLearnedExternalIdentity({...contract.LEARNED_EXTERNAL_IDENTITY,graph:expected}).graph,expected);
 assert.deepEqual(metadata.LEARNED_SHARED_GRAPH,expected);
 assert.deepEqual(contract.validateLearnedExternalIdentity(contract.LEARNED_EXTERNAL_IDENTITY),contract.LEARNED_EXTERNAL_IDENTITY);
 assert.throws(()=>contract.validateLearnedExternalIdentity({...contract.LEARNED_EXTERNAL_IDENTITY,graph:{...expected,bytes:expected.bytes+1}}));
 assert.throws(()=>contract.validateLearnedExternalIdentity({graph:expected,weights:{...contract.LEARNED_EXTERNAL_IDENTITY.weights,sha256:'0'.repeat(64)}}));
});
test('compressed metadata and decoded graph are exact; a verified cached graph loads offline',async t=>{
 assert.equal(typeof api.loadSharedLearnedGraph,'function');const data=gzip();assert.equal(data.length,archive.bytes);assert.equal(digest(data),archive.sha256);const raw=gunzipSync(data);assert.equal(raw.length,expected.bytes);assert.equal(digest(raw),expected.sha256);
 const cache=memoryCache();let requests=0;t.mock.method(globalThis,'fetch',async(url:unknown)=>{requests++;assert.equal(url,archive.url);return new Response(data);});
 const first=await api.loadSharedLearnedGraph(cache,signal());assert.equal(first.name,'hsr-smolvla.onnx');assert.equal(first.size,expected.bytes);assert.equal(digest(new Uint8Array(await first.arrayBuffer())),expected.sha256);assert.equal(requests,1);
 t.mock.restoreAll();t.mock.method(globalThis,'fetch',async()=>{throw Error('offline');});const next=await api.loadSharedLearnedGraph(cache,signal());assert.equal(next.size,expected.bytes);
});
test('a corrupt graph cache is replaced; corrupt network bytes are never cached',async t=>{
 assert.equal(typeof api.loadSharedLearnedGraph,'function');const cache=memoryCache();cache.entries.set(archive.url,new Response('bad'));t.mock.method(globalThis,'fetch',async()=>new Response(gzip()));assert.equal((await api.loadSharedLearnedGraph(cache,signal())).size,expected.bytes);assert.equal((await cache.entries.get(archive.url)!.blob()).size,archive.bytes);
 cache.entries.clear();t.mock.restoreAll();t.mock.method(globalThis,'fetch',async()=>new Response('bad'));await assert.rejects(api.loadSharedLearnedGraph(cache,signal()),/一致|match/i);assert.equal(cache.entries.size,0);
});
test('graph storage denial still permits verified use and an aborted load does not read cache',async t=>{
 assert.equal(typeof api.loadSharedLearnedGraph,'function');t.mock.method(globalThis,'fetch',async()=>new Response(gzip()));const denied={async match(){throw Error('denied');},async put(){throw Error('quota');},async delete(){return false;}};assert.equal((await api.loadSharedLearnedGraph(denied,signal())).size,expected.bytes);
 const stop=new AbortController();stop.abort();let reads=0;await assert.rejects(api.loadSharedLearnedGraph({...denied,async match(){reads++;return undefined;}},stop.signal),{name:'AbortError'});assert.equal(reads,0);
});
test('actual hosted loader substitutes only the graph and preserves reference/weight cache and progress',async()=>{
 const vm=await import('node:vm'),ts=(await import('typescript')).default;
 const text=fs.readFileSync(new URL('../src/learning/learned-model-download.ts',import.meta.url),'utf8');
 const tree=ts.createSourceFile('download.ts',text,ts.ScriptTarget.Latest,true),fn=tree.statements.find(node=>ts.isFunctionDeclaration(node)&&node.name?.text==='loadHostedLearnedFiles')!;
 const js=ts.transpileModule(fn.getText(tree).replace(/^export /,''),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText;
 const manifest=JSON.parse(fs.readFileSync(new URL('../worker/model-manifest.json',import.meta.url),'utf8')),files=download.validateHostedModel(manifest),calls:string[]=[],progress:number[][]=[];
 const context=vm.createContext({AbortSignal,HOSTED_MODEL_PATH:download.HOSTED_MODEL_PATH,PUBLIC_LEARNED_MODEL:contract.PUBLIC_LEARNED_MODEL,hostedModelPath:download.hostedModelPath,LEARNED_EXTERNAL_IDENTITY:contract.LEARNED_EXTERNAL_IDENTITY,LEARNED_SHARED_GRAPH:expected,validateHostedModel:download.validateHostedModel,signal:signal(),progress:(...args:number[])=>progress.push(args),fetch:async(url:string)=>{assert.equal(url,download.HOSTED_MODEL_PATH+'/manifest.json');return{ok:true,json:async()=>manifest};},caches:{async open(key:string){assert.equal(key,'hsr-vla-'+contract.LEARNED_EXTERNAL_IDENTITY.weights.sha256);return{};}},async loadSharedLearnedGraph(){calls.push('shared');return{name:'hsr-smolvla.onnx',size:expected.bytes};},async downloadLearnedFile(file:download.HostedFile){assert.notEqual(file.name,'hsr-smolvla.onnx');calls.push(file.name);return{name:file.name,size:file.bytes};}});
 const loaded=await vm.runInContext(js+'\nloadHostedLearnedFiles(signal,progress)',context)as{name:string;size:number}[];
 assert.deepEqual(Array.from(loaded,v=>v.name),contract.browserFileNames);assert.deepEqual(calls,['shared',...contract.browserFileNames.slice(1)]);const total=files.reduce((sum,f)=>sum+f.bytes,0)-contract.LEARNED_EXTERNAL_IDENTITY.graph.bytes+expected.bytes;assert.deepEqual(progress.at(-1),[total,total]);
});
