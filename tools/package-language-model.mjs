// Fixed public Apache-2.0 Qwen assets only; never accepts a VLA artifact.
import { readFile, writeFile, mkdir, open } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { LANGUAGE_MODEL } from '../src/learning/language-model.ts';
const [directory, output] = process.argv.slice(2);
if (!directory || !output) throw Error('Expected verified language directory and parts output');
const sha = b => createHash('sha256').update(b).digest('hex');
const manifest = {path:LANGUAGE_MODEL.path,upstream:LANGUAGE_MODEL.upstream,revision:LANGUAGE_MODEL.revision,files:[]};
await mkdir(output,{recursive:true});
for (const [name,[size,digest]] of Object.entries(LANGUAGE_MODEL.files)) {
  const handle = await open(`${directory}/${name}`), hash=createHash('sha256'), parts=[];
  try {
    let offset=0;
    while(offset<size) {
      const buffer=Buffer.alloc(Math.min(20*1024*1024,size-offset));
      const {bytesRead}=await handle.read(buffer,0,buffer.length,offset);
      if(bytesRead!==buffer.length) throw Error(`Incomplete language file: ${name}`);
      hash.update(buffer); const file=`${name}.${String(parts.length).padStart(3,'0')}.bin`;
      await mkdir(`${output}/${name.split('/').slice(0,-1).join('/')}`,{recursive:true});
      await writeFile(`${output}/${file}`,buffer);parts.push({bytes:buffer.length,sha256:sha(buffer)});offset+=buffer.length;
    }
    if((await handle.stat()).size!==size||hash.digest('hex')!==digest)throw Error(`Language identity mismatch: ${name}`);
    manifest.files.push({name,bytes:size,sha256:digest,parts});
  } finally {await handle.close();}
}
await writeFile(`${output}/manifest.json`,JSON.stringify(manifest,null,2)+'\n');
await writeFile('worker/language-manifest.json',JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({files:manifest.files.length,parts:manifest.files.reduce((n,f)=>n+f.parts.length,0)}));
