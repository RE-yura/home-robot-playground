import { LANGUAGE_MODEL } from './language-model.ts';
import { matchesPart, type HostedFile } from './learned-model-download.ts';
type ModelCache = Pick<Cache, 'match' | 'put' | 'delete'>;
export function validateLanguageParts(value: unknown, name: string): HostedFile {
  const m = value as {path?:string;revision?:string;files?:HostedFile[]};
  if (m?.path !== LANGUAGE_MODEL.path || m.revision !== LANGUAGE_MODEL.revision || !Array.isArray(m.files) || m.files.length !== Object.keys(LANGUAGE_MODEL.files).length) throw Error('Invalid language manifest');
  for (const [key, [bytes, sha256]] of Object.entries(LANGUAGE_MODEL.files)) {
    const matches = m.files.filter(f => f.name === key), f = matches[0];
    if (matches.length !== 1 || f.bytes !== bytes || f.sha256 !== sha256 || !Array.isArray(f.parts) || !f.parts.length || f.parts.length > 128 || f.parts.some(p => !Number.isSafeInteger(p.bytes) || p.bytes <= 0 || p.bytes > 20 * 1024 * 1024 || !/^[a-f0-9]{64}$/.test(p.sha256)) || f.parts.reduce((n,p)=>n+p.bytes,0) !== bytes) throw Error('Language manifest identity mismatch');
  }
  const file = m.files.find(f => f.name === name); if (!file) throw Error('Unknown language file'); return file;
}
// Keep individual HTTP responses small, and reuse each verified part after an
// interruption. Four bounded requests avoid one long, sequential 1.43GB stream.
export async function downloadLanguageParts(file: HostedFile, cache?: ModelCache, signal?: AbortSignal, progress?: (loaded:number,total:number)=>void, retain = true): Promise<Blob> {
  const controller = new AbortController(), combined = signal ? AbortSignal.any([signal,controller.signal]) : controller.signal;
  const blobs: Blob[] = new Array(file.parts.length); let next = 0, loaded = 0;
  const jobs = Array.from({length:Math.min(4,file.parts.length)},async()=>{
    while (next < file.parts.length) {
      combined.throwIfAborted(); const index = next++, part = file.parts[index];
      const url = `${LANGUAGE_MODEL.path}${file.name}.${String(index).padStart(3,'0')}.bin`;
      let blob: Blob | undefined;
      try { blob = await (await cache?.match(url))?.blob(); } catch { /* Storage is optional. */ }
      if (blob && !await matchesPart(blob,part)) { blob = undefined; try { await cache?.delete(url); } catch { /* Fetch verified bytes instead. */ } }
      if (!blob) {
        const r = await fetch(url,{signal:AbortSignal.any([combined,AbortSignal.timeout(120000)])});
        if (!r.ok) throw Error(`Language part download failed (${r.status})`);
        blob = await r.blob(); if (!await matchesPart(blob,part)) throw Error('Language part identity mismatch');
        combined.throwIfAborted(); try { await cache?.put(url,new Response(blob)); } catch { /* Quota does not prevent inference. */ }
      }
      combined.throwIfAborted(); if (retain) blobs[index] = blob; loaded += blob.size; progress?.(loaded,file.bytes);
    }
  });
  try { await Promise.all(jobs); } catch (error) { controller.abort(); await Promise.allSettled(jobs); throw error; }
  return retain ? new Blob(blobs) : new Blob();
}
