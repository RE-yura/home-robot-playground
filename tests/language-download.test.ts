import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {downloadLanguageParts,validateLanguageParts} from '../src/learning/language-download.ts';
import {LANGUAGE_MODEL} from '../src/learning/language-model.ts';
const sha=(b:Uint8Array)=>createHash('sha256').update(b).digest('hex');
test('only pinned language manifest and complete bounded parts are admitted',()=>{
 const m=JSON.parse(readFileSync(new URL('../worker/language-manifest.json',import.meta.url),'utf8'));
 assert.equal(validateLanguageParts(m,'onnx/model_q4f16.onnx_data').bytes,1425786880);
 for(const mutate of [(v:any)=>v.path='/other/',(v:any)=>v.files[5].sha256='0'.repeat(64),(v:any)=>v.files[5].parts.pop()]){const bad=structuredClone(m);mutate(bad);assert.throws(()=>validateLanguageParts(bad,'onnx/model_q4f16.onnx_data'));}
});
test('language delivery assembles exact verified parts in order, limits parallel requests and resumes cached parts',async(context)=>{
 const blobs=Array.from({length:6},(_,i)=>new Uint8Array([2*i+1,2*i+2])),file={name:'onnx/model_q4f16.onnx_data',bytes:12,sha256:sha(new Uint8Array([1,2,3,4,5,6,7,8,9,10,11,12])),parts:blobs.map(b=>({bytes:b.length,sha256:sha(b)}))};
 const saved=new Map<string,Response>();let requests=0,active=0,maxActive=0;
 const cache={async match(url:string){return saved.get(url)?.clone();},async put(url:string,r:Response){saved.set(url,r);},async delete(url:string){return saved.delete(url);}} as any;
 context.mock.method(globalThis,'fetch',async(url:any)=>{requests++;active++;maxActive=Math.max(maxActive,active);const i=Number(String(url).match(/\.(\d{3})\.bin$/)![1]);await new Promise(r=>setTimeout(r,(6-i)*3));active--;return new Response(blobs[i]);});
 const progress:number[]=[];assert.deepEqual(new Uint8Array(await(await downloadLanguageParts(file,cache,undefined,n=>progress.push(n))).arrayBuffer()),new Uint8Array([1,2,3,4,5,6,7,8,9,10,11,12]));
 assert.equal(requests,6);assert.equal(maxActive,4);assert.equal(progress.at(-1),12);
 await downloadLanguageParts(file,cache);assert.equal(requests,6,'verified cache avoids repeat transfer');
 saved.set(LANGUAGE_MODEL.path+file.name+'.001.bin',new Response(new Uint8Array([0,0])));await downloadLanguageParts(file,cache);assert.equal(requests,7,'corrupt cache cannot be trusted');
 context.mock.method(globalThis,'fetch',async()=>new Response(new Uint8Array([0,0])));saved.clear();await assert.rejects(downloadLanguageParts(file,cache),/identity/);
});
