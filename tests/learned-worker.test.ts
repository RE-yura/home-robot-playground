import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { test } from 'node:test';
import type { LearnedPrediction } from '../src/learning/learned-rollout.ts';
import { rtcContextFeed } from '../src/learning/rtc-contract.ts';
import { prefixFeed } from '../src/learning/prefix-conditioning.ts';
import { encodeVlaInstruction } from '../src/learning/vla-instruction.ts';
import type { RtcContinuation } from '../src/learning/learned-rtc-rollout.ts';
const workerUrl = new URL('../src/learning/learned-onnx-worker.ts', import.meta.url);
const tree = ts.createSourceFile('worker.ts', fs.readFileSync(workerUrl, 'utf8'), ts.ScriptTarget.Latest, true);
const body = tree.statements.filter(node => !ts.isImportDeclaration(node)).map(node => node.getText(tree)).join('\n');
const js = ts.transpileModule(body, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;
let meanPair: ((a: Float32Array, b: Float32Array) => Float32Array) | undefined;
const meanImport = tree.statements.filter(ts.isImportDeclaration).find(node => {
  const bindings = node.importClause?.namedBindings;
  return bindings && ts.isNamedImports(bindings) && bindings.elements.some(item => item.name.text === 'meanPair');
});
if (meanImport && ts.isStringLiteral(meanImport.moduleSpecifier)) ({ meanPair } = await import(new URL(meanImport.moduleSpecifier.text, workerUrl).href));
const outputs = [16777216, 1, -16777216, 1].map(value => Float32Array.from({ length: 96 }, (_, index) => index % 6 === 5 ? -2.5 : value));
async function exercise(failAt = 0, shared = false, badMember = false, rtc = false, continuation?: RtcContinuation | null, prefixDelay = 0, members = 4, zero = false, instruction = 'Put the mug in the basket.', tokenizerLoads: string[] = []) {
  const messages: { id: number; error?: string; value?: LearnedPrediction }[] = [], calls: Float32Array[] = [], disposals: object[] = []; let draws = 0;
  class Tensor {
    type: string; data: Float32Array | BigInt64Array | Uint8Array; dims: number[];
    constructor(type: string, data: Float32Array | BigInt64Array | Uint8Array, dims: number[]) { this.type = type; this.data = data; this.dims = dims; }
    dispose() { disposals.push(this); }
  }
  const feeds: Record<string, Tensor>[] = [];
  const session = { async run(feed: Record<string, Tensor>) {
    feeds.push(feed);
    calls.push(new Float32Array(feed.initial_noise.data as Float32Array));
    if (calls.length === failAt) throw new Error('Fourth prediction failed');
    if (shared) {
      assert.ok(feed.second_noise, 'Shared graph needs an independent second noise');
      return Object.fromEntries(outputs.map((values, index) => [`__ensemble_member_${index}/hsr_action_chunk`, new Tensor('float32', badMember && index === 3 ? Float32Array.from(values, () => NaN) : values, [1, 16, 6])]));
    }
    return { hsr_action_chunk: new Tensor('float32', outputs[calls.length - 1], [1, 16, 6]) };
  }};
  // The tokenizer loader is replaced: a fake encoder that gives each character its code, verified like the real one.
  const mockLoadVlaEncode = async () => { tokenizerLoads.push('load'); return (text: string) => [...text].map(c => c.charCodeAt(0)); };
  const context = { meanPair, rtcContextFeed, prefixFeed, encodeVlaInstruction, mockLoadVlaEncode, wasmUrl: 'x', wasmModuleUrl: 'x', ort: { env: { wasm: {} }, Tensor },
    Float32Array, Uint32Array, BigInt64Array, Uint8Array, BigInt, Math,
    crypto: { getRandomValues(array: Uint32Array) { draws++; for (let i = 0; i < array.length; i++) array[i] = 10000000 + draws * 9999 + i * 99999; return array; } },
    self: { crossOriginIsolated: true, postMessage: (message: { id: number; error?: string; value?: LearnedPrediction }) => messages.push(message), onmessage: async (_event: { data: unknown }) => { throw new Error("Worker handler missing"); } },
    mockSession: session, mockManifest: { instructions: { 'Put the mug in the basket.': { tokens: Array(48).fill(1), mask: Array(48).fill(true) } } },
    fetch: async (url: string) => { assert.ok(url.startsWith('data:image/jpeg;'), 'Public inference must not contact a recording server'); return { blob: async () => ({}) }; },
    createImageBitmap: async () => ({ width: 320, height: 240, close() {} }),
    OffscreenCanvas: class { getContext() { return { drawImage() {}, getImageData() { return { data: new Uint8Array(320 * 240 * 4) }; } }; } },
  };
  vm.runInNewContext(js + ';session=mockSession;manifest=mockManifest;loadVlaEncode=mockLoadVlaEncode;if(typeof sharedGraph!=="undefined")sharedGraph='+shared+';rtcGraph='+rtc+';prefixDelayLimit='+prefixDelay+';memberCount='+members+';if(typeof zeroNoise!=="undefined")zeroNoise='+zero+';', context);
  const input = { images: ['head', 'hand'].map(view => ({ view, timestamp: 2, width: 320, height: 240, dataUrl: 'data:image/jpeg;base64,eA==' })), instruction, state: Array(6).fill(0) };
  await context.self.onmessage({ data: { id: 1, type: 'infer', input, ...(continuation !== undefined ? { continuation } : {}) } });
  return { messages, calls, disposals, draws, feeds };
}
test('public learned inference preserves four paired predictions, raw Float32 means and all-or-nothing failure', async () => {
  const good = await exercise();
  assert.equal(good.messages.at(-1)!.error, undefined);
  assert.equal(good.calls.length, 4, 'Public infer must use all four ordered predictions');
  assert.equal(good.draws, 2, 'Each observation needs two independent Gaussian draws');
  for (const [first, negative] of [[0, 1], [2, 3]]) assert.deepEqual(good.calls[negative], Float32Array.from(good.calls[first], value => -value));
  assert.notDeepEqual(good.calls[0], good.calls[2]);
  const result = good.messages.at(-1)!.value!;
  assert.ok(result.secondNoise && result.memberActions);
  assert.deepEqual(Array.from(result.actions.flat()), Array.from({ length: 96 }, (_, index) => index % 6 === 5 ? -2.5 : 0.25), 'Nested Float32 mean must be preserved before actuator clamping');
  assert.deepEqual(Array.from(result.initialNoise), Array.from(good.calls[0]));
  assert.deepEqual(Array.from(result.secondNoise), Array.from(good.calls[2]));
  assert.deepEqual(Array.from(result.memberActions, row => Array.from(row)), outputs.map(row => Array.from(row)));
  assert.equal(good.disposals.length, 28);
  const failed = await exercise(4);
  assert.equal(failed.calls.length, 4);
  assert.match(failed.messages.at(-1)!.error!, /Fourth/);
  assert.equal(failed.messages.some(message => message.value), false, 'A failed member cannot yield a partial action chunk');
  assert.equal(failed.disposals.length, 27);
  console.log(JSON.stringify({ passed: true, source: 'actual public Worker infer handler with NN/camera doubles', neuralInferencePerformed: false, fourPredictions: true, nestedFloat32Mean: true, noRecorderRequests: true, incompleteBatchCannotControl: true }));

});

test('shared public inference invokes once while preserving all four ordered members and original mean', async () => {
  const good = await exercise(0, true);
  assert.equal(good.messages.at(-1)!.error, undefined);
  assert.equal(good.calls.length, 1);
  assert.equal(good.draws, 2);
  const result = good.messages.at(-1)!.value!;
  assert.deepEqual(Array.from(result.memberActions!, row => Array.from(row)), outputs.map(row => Array.from(row)));
  assert.deepEqual(Array.from(result.actions.flat()), Array.from({length:96},(_,i)=>i%6===5?-2.5:0.25));
  assert.equal(good.disposals.length, 11);
  const bad = await exercise(0, true, true);
  assert.ok(bad.messages.at(-1)!.error);
  assert.equal(bad.messages.some(message=>message.value),false);
  assert.equal(bad.disposals.length,11);
});

test('explicit RTC inference forwards queued prefix while legacy inference rejects continuation', async () => {
 const continuation: RtcContinuation={prefix:Array.from({length:12},()=>[.3,-.2,0,-1.3,0,.5]),delaySteps:4,horizon:10};
 const actual=await exercise(0,true,false,true,continuation);
 assert.equal(actual.messages.at(-1)!.error,undefined);assert.equal(actual.calls.length,1);assert.equal(actual.draws,2);
 assert.deepEqual(Array.from(actual.feeds[0].rtc_prefix.dims),[1,16,6]);assert.deepEqual(Array.from(actual.feeds[0].rtc_weights.dims),[1,16,1]);
 assert.equal(actual.feeds[0].rtc_enabled.data[0],1);assert.equal(actual.disposals.length,14);
 const initial=await exercise(0,true,false,true,null);assert.equal(initial.feeds[0].rtc_enabled.data[0],0);
 const legacy=await exercise(0,true,false,false,continuation);assert.match(legacy.messages.at(-1)!.error!,/RTC/);assert.equal(legacy.calls.length,0);
 const invalid=await exercise(0,true,false,true,{...continuation,prefix:[[NaN,0,0,0,0,0]]});
 assert.ok(invalid.messages.at(-1)!.error);assert.equal(invalid.calls.length,0);assert.equal(invalid.disposals.length,0,"bad continuation is rejected before allocating tensors");
});

test('a prefix-conditioned model receives the committed goals as two extra inputs; other models never do', async () => {
 const continuation: RtcContinuation={prefix:Array.from({length:6},(_,i)=>[i,0,0,-1,0,.5]),delaySteps:3,horizon:6};
 const actual=await exercise(0,true,false,false,continuation,10);
 assert.equal(actual.messages.at(-1)!.error,undefined);assert.equal(actual.calls.length,1);
 assert.deepEqual(Array.from(actual.feeds[0].action_prefix.dims),[1,16,6]);assert.deepEqual(Array.from(actual.feeds[0].prefix_mask.dims),[1,16]);
 assert.deepEqual(Array.from(actual.feeds[0].prefix_mask.data as Float32Array),[1,1,1,...Array(13).fill(0)]);
 assert.deepEqual(Array.from((actual.feeds[0].action_prefix.data as Float32Array).slice(0,18)),[0,0,0,-1,0,.5,1,0,0,-1,0,.5,2,0,0,-1,0,.5]);
 const first=await exercise(0,true,false,false,null,10);assert.ok((first.feeds[0].prefix_mask.data as Float32Array).every(v=>v===0));
 const omitted=await exercise(0,true,false,false,undefined,10);assert.ok((omitted.feeds[0].prefix_mask.data as Float32Array).every(v=>v===0));
 const publicModel=await exercise(0,true,false,false,undefined,0);assert.equal('action_prefix' in publicModel.feeds[0],false);assert.equal('prefix_mask' in publicModel.feeds[0],false);
});

test('a one-member diagnostic runs the single-member graph once and returns that member without averaging', async () => {
 const actual=await exercise(0,false,false,false,undefined,0,1);
 assert.equal(actual.messages.at(-1)!.error,undefined);assert.equal(actual.calls.length,1);
 const value=actual.messages.at(-1)!.value!;assert.equal(value.memberActions!.length,1);
 assert.deepEqual(Array.from(value.actions.flat()),Array.from(outputs[0]));
});
test('a zero-noise diagnostic starts every member at zero noise and draws no randomness', async () => {
  for (const members of [1, 4]) {
    const actual = await exercise(0, false, false, false, undefined, 0, members, true);
    assert.equal(actual.messages.at(-1)!.error, undefined);
    assert.equal(actual.draws, 0, 'no Gaussian draws');
    assert.ok(actual.calls.every(noise => noise.every(value => value === 0)), 'every member starts at zero noise');
    assert.ok(actual.messages.at(-1)!.value!.initialNoise.every(value => value === 0));
  }
});

test('an instruction outside the exported table is fed through the training tokenizer, loaded once; learned ones never load it', async () => {
  const loads: string[] = [];
  const learned = await exercise(0, false, false, false, undefined, 0, 4, false, 'Put the mug in the basket.', loads);
  assert.equal(learned.messages.at(-1)!.error, undefined); assert.deepEqual(loads, []);
  const text = 'Take the mug out of the basket.', other = await exercise(0, false, false, false, undefined, 0, 4, false, text, loads);
  assert.equal(other.messages.at(-1)!.error, undefined); assert.deepEqual(loads, ['load']);
  const expected = encodeVlaInstruction(t => [...t].map(c => c.charCodeAt(0)), text);
  assert.deepEqual(Array.from(other.feeds[0].language_tokens.data as BigInt64Array, Number), expected.tokens);
  assert.deepEqual(Array.from(other.feeds[0].language_mask.data as Uint8Array, Number), expected.mask.map(Number));
  const bad = await exercise(0, false, false, false, undefined, 0, 4, false, 'コップを出して', loads);
  assert.match(bad.messages.at(-1)!.error!, /instruction/i);
});
