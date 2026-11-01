import assert from "node:assert/strict";
import { test } from "node:test";
import { hostedModelPath } from "../src/learning/learned-model-download.ts";
import { compareLearnedActions, validateLearnedManifest, validateLearnedExternalIdentity, verifyLearnedBrowserReferenceFile, learnedModel, PUBLIC_LEARNED_MODEL, LEARNED_CANDIDATES, type LearnedModelIdentity } from "../src/learning/learned-onnx-contract.ts";

test("browser parity rejects a bad tail value and evaluates normalized tolerance as well as physical tolerance", () => {
  const expected = new Float32Array(96), actual = new Float32Array(96), std = [0.1, 1, 1, 1, 1, 1];
  actual[90] = 0.0005;
  const proof = compareLearnedActions(actual, expected, [0, 0, 0, 0, 0, 0], std);
  assert.equal(proof.physical.passed, true); assert.equal(proof.normalized.passed, false);assert.equal(proof.passed, false);
  assert.equal(proof.comparedValues, 96);
  actual[95] = NaN; assert.throws(() => compareLearnedActions(actual, expected, [0, 0, 0, 0, 0, 0], std), /finite/i);
});

test("loading refuses substituted model identities, missing instruction tokens and zero normalization scale", () => {
  const row = { tokens: Array(48).fill(1), mask: Array(48).fill(true) };
  const manifest = { schema: "hsr-learned-browser-v1", graph: { bytes: 1592644520,
    sha256: "9399fb53ef8613c95b0a7e6011cb0d41236b97a77e150ef726fbef2e728739eb" },
    instructions: { "Put the mug in the basket.": row, "Put the tennis ball in the basket.": row },
    actionNormalization: { mean: [0, 0, 0, 0, 0, 0], std: [1, 1, 1, 1, 1, 1] } };
  assert.equal(validateLearnedManifest(manifest).graph.bytes, 1592644520);
  assert.throws(() => validateLearnedManifest({ ...manifest, graph: { ...manifest.graph, sha256: "0".repeat(64) } }), /model|graph/i);
  assert.throws(() => validateLearnedManifest({ ...manifest, instructions: { "Put the mug in the basket.": row } }), /instruction/i);
  assert.throws(() => validateLearnedManifest({ ...manifest, actionNormalization: { ...manifest.actionNormalization, std: [0, 1, 1, 1, 1, 1] } }), /normalization/i);
});

test("external FP32 loading requires both complete hash-bound files and the graph's exact weight location", () => {
  const identity = { graph: { bytes: 18833856, sha256: "c80224ec1719261d205ff6dd04eeb2cf516778729f956f04b3ab1251728e4b04" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "0cccb41e11e027fb92f56165748ae55364fc9432dec0f8b1afdcdef7cbf51cf1" } };
  assert.deepEqual(validateLearnedExternalIdentity(identity), identity);
  for (const changed of [
    { ...identity, graph: { ...identity.graph, sha256: "0".repeat(64) } },
    { ...identity, weights: { ...identity.weights, bytes: identity.weights.bytes - 1 } },
    { ...identity, weights: { ...identity.weights, sha256: "0".repeat(64) } },
    { ...identity, weights: { ...identity.weights, path: "other.weights" } },
    { graph: identity.graph },
  ]) assert.throws(() => validateLearnedExternalIdentity(changed), /identity|weights|manifest/i);
});

const fixtureCandidate = (references: Record<string, string> = {}): LearnedModelIdentity => ({
  ...PUBLIC_LEARNED_MODEL, id: "fixture-offset", manifestGraph: { bytes: 10, sha256: "1".repeat(64) },
  graph: { bytes: 11, sha256: "2".repeat(64) }, sharedGraph: { bytes: 12, sha256: "3".repeat(64) },
  sharedArchive: { url: "/models/fixture/hsr-smolvla-shared.onnx.gz", bytes: 13, sha256: "4".repeat(64) },
  weights: { path: "hsr-smolvla.weights", bytes: 14, sha256: "5".repeat(64) }, references: { ...PUBLIC_LEARNED_MODEL.references, ...references }, chunkOffsetSteps: 2 });

test("the page defaults to the public model and refuses unknown candidates", () => {
  for (const id of [null, undefined, ""]) assert.equal(learnedModel(id), PUBLIC_LEARNED_MODEL);
  assert.equal(PUBLIC_LEARNED_MODEL.graph.sha256, "c80224ec1719261d205ff6dd04eeb2cf516778729f956f04b3ab1251728e4b04");
  assert.equal(PUBLIC_LEARNED_MODEL.sharedGraph.sha256, "8785fe59d6cf3c89a18bbef265bab4be696cdc54c3ebcb694319ff838df39c55");
  assert.equal(PUBLIC_LEARNED_MODEL.weights.sha256, "0cccb41e11e027fb92f56165748ae55364fc9432dec0f8b1afdcdef7cbf51cf1");
  assert.equal(PUBLIC_LEARNED_MODEL.chunkOffsetSteps, 0);
  assert.throws(() => learnedModel("unknown"), /model/i);
  for (const [id, model] of Object.entries(LEARNED_CANDIDATES)) {
    assert.equal(learnedModel(id), model); assert.equal(model.id, id);
    assert.notEqual(model.weights.sha256, PUBLIC_LEARNED_MODEL.weights.sha256); assert.ok(model.chunkOffsetSteps > 0 || model.prefixConditioning, "a candidate changes how chunks are produced");
  }
});

test("every registered model has its own hosted path, and its shared archive lives there", () => {
  const models = [PUBLIC_LEARNED_MODEL, ...Object.values(LEARNED_CANDIDATES)], paths = models.map(hostedModelPath);
  assert.equal(new Set(paths).size, paths.length, paths.join());
  for (const model of Object.values(LEARNED_CANDIDATES)) assert.ok(model.sharedArchive.url.startsWith(`${hostedModelPath(model)}/`), model.id);
});

test("a candidate's files are bound to its own hashes, never to the public ones", async () => {
  const row = { tokens: Array(48).fill(1), mask: Array(48).fill(true) }, text = '{"fixture":true}';
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text))), v => v.toString(16).padStart(2, "0")).join("");
  const candidate = fixtureCandidate({ "browser-manifest.json": digest });
  const manifest = (graph: { bytes: number; sha256: string }) => ({ schema: "hsr-learned-browser-v1", graph, instructions: { "Put the mug in the basket.": row, "Put the tennis ball in the basket.": row },
    actionNormalization: { mean: [0, 0, 0, 0, 0, 0], std: [1, 1, 1, 1, 1, 1] } });
  assert.equal(validateLearnedManifest(manifest(candidate.manifestGraph), candidate).graph.sha256, "1".repeat(64));
  assert.throws(() => validateLearnedManifest(manifest(PUBLIC_LEARNED_MODEL.manifestGraph), candidate), /graph|model/i);
  assert.throws(() => validateLearnedManifest(manifest(candidate.manifestGraph)), /graph|model/i);
  for (const graph of [candidate.graph, candidate.sharedGraph]) validateLearnedExternalIdentity({ graph, weights: candidate.weights }, candidate);
  assert.throws(() => validateLearnedExternalIdentity({ graph: PUBLIC_LEARNED_MODEL.graph, weights: PUBLIC_LEARNED_MODEL.weights }, candidate), /identity/i);
  assert.throws(() => validateLearnedExternalIdentity({ graph: candidate.graph, weights: candidate.weights }), /identity/i);
  await verifyLearnedBrowserReferenceFile(new File([text], "browser-manifest.json"), candidate);
  await assert.rejects(verifyLearnedBrowserReferenceFile(new File([text], "browser-manifest.json")), /SHA-256/);
});

test("the prefix-rtc candidate is pinned to its own export and takes up to ten committed goals", () => {
  const model = learnedModel("prefix-rtc");
  assert.equal(model.chunkOffsetSteps, 0); assert.deepEqual(model.prefixConditioning, { maxDelay: 10 });
  assert.deepEqual(model.manifestGraph, { bytes: 1583548687, sha256: "d5b2996e2614d338ff03baceffc5631cc6bb09b3f2576154e2a452a81a0ddcf6" });
  assert.deepEqual(model.graph, { bytes: 9881821, sha256: "d26b52730729fcdc0aa1976043e5f6b59e52dbfaa7d32ad9d7bcaacc46dbb6db" });
  assert.deepEqual(model.sharedGraph, { bytes: 43400801, sha256: "c7d5e9145e02146fb66fd57de9fa0d7d7dcf86693951d61a730e36db9001ae51" });
  assert.deepEqual(model.sharedArchive, { url: "/models/smolvla-e7de77cd3659/hsr-smolvla-shared.onnx.gz", bytes: 4448536, sha256: "e89c1d61426b495bc49a96b1b5087466551a94699fb2491c7d0b8b8fe78518ed" });
  assert.equal(model.weights.sha256, "e7de77cd3659764abb4f004a32727cef8f0519a05c2e0eadb9b03d1c74c0e4d3");
  assert.equal(model.references["browser-case-1.json"], "83a9157a222ce25b5457e9c2500f313921048324b7d6c707a34efd530e89a39f");
  assert.equal(learnedModel().prefixConditioning, undefined, "the public model takes no committed goals");
});
test("the prefix-rtc-e5 candidate is the prefix-rtc weights with a five-step graph served from its own folder", () => {
  const model = learnedModel("prefix-rtc-e5"), parent = learnedModel("prefix-rtc");
  assert.deepEqual(model.weights, parent.weights); assert.deepEqual(model.prefixConditioning, { maxDelay: 10 }); assert.equal(model.chunkOffsetSteps, 0);
  assert.equal(model.hostedName, "smolvla-e7de77cd3659-e5");
  assert.deepEqual(model.manifestGraph, { bytes: 1579475339, sha256: "51aad9d8a669e3ac8918f715d3a1d1299455b5a1be76d380cde9ea37739e73a8" });
  assert.deepEqual(model.graph, { bytes: 5664693, sha256: "16bf7a52f4796a22ee1897d4bb801f91efa84ae8b928e7b9dd81d406e0f71043" });
  assert.deepEqual(model.sharedGraph, { bytes: 22341802, sha256: "bf3524a3a97907688ba3ec744d044293ed7b84e4bbf27fc12f0d7c769e6b4484" });
  assert.deepEqual(model.sharedArchive, { url: "/models/smolvla-e7de77cd3659-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303740, sha256: "8cd09ffea5fa97d58275d030c79c6f96383b7fba383ada24618b075d2069669a" });
  assert.equal(model.references["browser-case-1.json"], "0736dccd9e3d050ec299bd06f656cff480ec2670b3602bb42f9fc483433ac283");
});
test("the prefix-rtc-v2 candidates are pinned to the delay 0-8 retrain and take up to eight committed goals", () => {
  const model = learnedModel("prefix-rtc-v2"), fast = learnedModel("prefix-rtc-v2-e5");
  assert.deepEqual(model.prefixConditioning, { maxDelay: 8 }); assert.deepEqual(fast.prefixConditioning, { maxDelay: 8 });
  assert.equal(model.chunkOffsetSteps, 0); assert.equal(fast.chunkOffsetSteps, 0);
  assert.deepEqual(model.weights, { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "33fec139ff9e6f2928a59e2371a6319d6d702c258ef845aac77f99a09aba762e" });
  assert.deepEqual(fast.weights, model.weights); assert.equal(model.hostedName, undefined); assert.equal(fast.hostedName, "smolvla-33fec139ff9e-e5");
  assert.deepEqual(model.manifestGraph, { bytes: 1583548687, sha256: "fd48c2319796df66e3cbb18dc77dc112f5b6253146a0355ba08c90d29ed2bfae" });
  assert.deepEqual(model.graph, { bytes: 9881821, sha256: "a50cf8bcd5ad4e547528a86478f0b20c6706f4d8ecc9a368f44572efa09ca0e3" });
  assert.deepEqual(model.sharedGraph, { bytes: 43400801, sha256: "44604369380ef7b57cc86c9a95fc31159b830ed4aad9c69454f4177b4a90f248" });
  assert.deepEqual(model.sharedArchive, { url: "/models/smolvla-33fec139ff9e/hsr-smolvla-shared.onnx.gz", bytes: 4448536, sha256: "c4a1f90d888a0e59d470d13209ee7c2de37aaf6d0789eaa38fd9aed154f4e7f6" });
  assert.equal(model.references["browser-case-1.json"], "b5a7eaad35d3fd839cbbd90f41804a624007c5f9925e4b17e07bc055478afd8d");
  assert.deepEqual(fast.manifestGraph, { bytes: 1579475339, sha256: "8257d81ef1bc8b62d32694cd5797cb7d6d9e290023c2ca975cae412a72330ca5" });
  assert.deepEqual(fast.graph, { bytes: 5664693, sha256: "2abf9d91d6564f6cf68e7853d097fe66bde62ffdf05a847738359750a0ac95d5" });
  assert.deepEqual(fast.sharedGraph, { bytes: 22341802, sha256: "646516ead3b49f60d5383168ff6314753371550d3bb0adc0155ac0667d6ce182" });
  assert.deepEqual(fast.sharedArchive, { url: "/models/smolvla-33fec139ff9e-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303740, sha256: "2a8227b4fa0b32349b4592a43e6f6782d7bc17ed3f4b94b0b40a46c6b74092ff" });
  assert.equal(fast.references["browser-case-1.json"], "e307d1b2260f5aeb2fc7d2b2c6b56611fb14caa07125e01bd7766dd1cedcba17");
});
test("the delay-shift candidate is pinned to its own export and starts each chunk two goals after the observation", () => {
  const model = learnedModel("delay-shift");
  assert.equal(model.chunkOffsetSteps, 2);
  assert.deepEqual(model.manifestGraph, { bytes: 1592644520, sha256: "330813028e10f9e08c501362d6ba3e127ffc135ad397862506baa51c760ba970" });
  assert.deepEqual(model.graph, { bytes: 18833856, sha256: "9817bf35592a02be0c58820dc8c0853cdef658a61bf45d8174798d469a59582c" });
  assert.deepEqual(model.sharedGraph, { bytes: 28640194, sha256: "95d72ccddf0f3bafdb8e869e5aca7adf5b1d427a187b2ae545de9f6eb4e2f84a" });
  assert.equal(model.weights.sha256, "c63f9ccf60489ea74daf6b13f3baa3ed46f76f27df20298e7df1bf472ad6387e");
  assert.equal(model.sharedArchive.url, "/models/smolvla-c63f9ccf6048/hsr-smolvla-shared.onnx.gz");
  assert.equal(model.references["browser-manifest.json"], "379deff7e3dbc5ae28dfbc10ba17eb658fdc7a0f9f33e45735fdfcc1a9d88613");
});
test("the prefix-rtc-v3 candidates are pinned to the gentle-mug data retrain", () => {
  const model = learnedModel("prefix-rtc-v3"), fast = learnedModel("prefix-rtc-v3-e5");
  assert.deepEqual(model.prefixConditioning, { maxDelay: 8 }); assert.deepEqual(fast.prefixConditioning, { maxDelay: 8 });
  assert.deepEqual(model.weights, { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "9002612dfb4723f710608fb6dfc1b94f58901c63bae299f8187c5c8358f4cee0" });
  assert.deepEqual(fast.weights, model.weights); assert.equal(model.hostedName, undefined); assert.equal(fast.hostedName, "smolvla-9002612dfb47-e5");
  assert.deepEqual(model.sharedGraph, { bytes: 43400801, sha256: "0d2b213fcdc1be948bbd4da6e094b1c8f3b1b311c26730bbd61662004cc74b6e" });
  assert.deepEqual(fast.sharedGraph, { bytes: 22341802, sha256: "129fbcdafcb1682e81778d41916012ea7c31ba5dee87e7cf6bec1c50adc82cad" });
  assert.equal(model.sharedArchive!.url, "/models/smolvla-9002612dfb47/hsr-smolvla-shared.onnx.gz");
  assert.equal(fast.references["browser-case-0.json"], "25726bc221041e5eeee09afd6de32b7584b02bac406c4ce084086555aac64b89");
});
test("the prefix-rtc-v4 candidates are pinned to the gentle-mug plus takeover retrain", () => {
  const model = learnedModel("prefix-rtc-v4"), fast = learnedModel("prefix-rtc-v4-e5");
  assert.deepEqual(model.prefixConditioning, { maxDelay: 8 }); assert.deepEqual(fast.prefixConditioning, { maxDelay: 8 });
  assert.deepEqual(model.weights, { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "952e2d2abac8f8484db34c3533b32b1a7e3a20bf7a6d87cbc43b245b664fb524" });
  assert.deepEqual(fast.weights, model.weights); assert.equal(model.hostedName, undefined); assert.equal(fast.hostedName, "smolvla-952e2d2abac8-e5");
  assert.deepEqual(model.sharedGraph, { bytes: 43400801, sha256: "39a916e22ac5e46e9d5e401623e466f6b6e0cf68b753e241e8e58a4875e0f40b" });
  assert.deepEqual(fast.sharedGraph, { bytes: 22341802, sha256: "301a30d5f059616af90c0e1d3872da8b7e7fb056ecbdf631454e3289142ec072" });
  assert.equal(model.references["browser-case-0.json"], "bdb9f1c3aa0e9b9c9d5a96d90f0ed577d037ace09bc86a09b325265be6967761");
});
test("the prefix-rtc-v5 candidates are pinned to the gentle-mug plus late/round-2 takeover retrain", () => {
  const model = learnedModel("prefix-rtc-v5"), fast = learnedModel("prefix-rtc-v5-e5");
  assert.deepEqual(model.prefixConditioning, { maxDelay: 8 }); assert.deepEqual(fast.prefixConditioning, { maxDelay: 8 });
  assert.deepEqual(model.weights, { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "92d0c65d37aeed687b753bdb08083daf137ce2e0e74f20e8245274cf52fb25e1" });
  assert.deepEqual(fast.weights, model.weights); assert.equal(model.hostedName, undefined); assert.equal(fast.hostedName, "smolvla-92d0c65d37ae-e5");
  assert.deepEqual(model.sharedGraph, { bytes: 43400801, sha256: "7402135eefc52a3348277dcc8e5744579a6c76041ab44f2a08860c6d4d012178" });
  assert.deepEqual(fast.sharedGraph, { bytes: 22341802, sha256: "dc14670294849335d4f183e16900f4466aeba6e3d079dc7b75c1027466dc3c09" });
  assert.equal(model.references["browser-case-0.json"], "b2e0017fd70040375ba258236ed7f271ac35db572fac10204029c347ab7b93ab");
});
