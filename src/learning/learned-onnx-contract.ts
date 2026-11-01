import type { Instruction } from "./recording.ts";
type Pin = { bytes: number; sha256: string };
// Every file of a browser model is pinned by hash. The public model is the default; a candidate is selected explicitly
// (?model=<id>), is never published by this registry, and is only served where its files were packaged locally.
export type LearnedModelIdentity = {
  id: string;
  manifestGraph: Pin;            // FP32 export named in browser-manifest.json
  graph: Pin;                    // compiled external-data graph
  sharedGraph: Pin;              // four-member graph sharing the observation prefix
  sharedArchive: Pin & { url: string };
  weights: Pin & { path: "hsr-smolvla.weights" };
  references: Record<string, string>;
  chunkOffsetSteps: number;      // goals start this many 100ms steps after the observation (continuous execution only)
  prefixConditioning?: { maxDelay: number }; // graph also takes the executor's committed goals (action_prefix, prefix_mask)
  hostedName?: string;           // served folder under /models when another model already uses these weights
};
export const PUBLIC_LEARNED_MODEL: LearnedModelIdentity = {
  id: "public",
  manifestGraph: { bytes: 1592644520, sha256: "9399fb53ef8613c95b0a7e6011cb0d41236b97a77e150ef726fbef2e728739eb" },
  graph: { bytes: 18833856, sha256: "c80224ec1719261d205ff6dd04eeb2cf516778729f956f04b3ab1251728e4b04" },
  sharedGraph: { bytes: 28640194, sha256: "8785fe59d6cf3c89a18bbef265bab4be696cdc54c3ebcb694319ff838df39c55" },
  sharedArchive: { url: "/assets/learned-vla/shared-8785.onnx.gz", bytes: 1882482, sha256: "4d88815bd100419288401d2d5d9d0a25f2e653871c2e37c6a2e3f93c18ec8480" },
  weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "0cccb41e11e027fb92f56165748ae55364fc9432dec0f8b1afdcdef7cbf51cf1" },
  references: {
    "browser-manifest.json": "ed1005a15b1dece0c9db7f90d04d5a8623d51cd0ea182b6960743a3ab9f83356",
    "browser-case-0.json": "de8bab1335e63f1e3570748463c010e69321aec401424ea25a8973cc005d2801",
    "browser-case-1.json": "bbc0224eeaa9aa4af1e0c2e8e98e5fe64004f620355fa3e21900859f12825ce4",
    "browser-case-2.json": "96a26f7ea544d9bd0109c211daf8cccebb67a694a0b256c36f3ae7c21244389f",
  },
  chunkOffsetSteps: 0,
};
// Local-only candidates (files under public/models are git-excluded; nothing here is deployed by the registry).
export const LEARNED_CANDIDATES: Record<string, LearnedModelIdentity> = {
  // Public parent f289 + 2000 updates on goals 0.2-1.7 s after each observation (checkpoint 16fbc827, source a8a4397).
  "delay-shift": {
    id: "delay-shift",
    manifestGraph: { bytes: 1592644520, sha256: "330813028e10f9e08c501362d6ba3e127ffc135ad397862506baa51c760ba970" },
    graph: { bytes: 18833856, sha256: "9817bf35592a02be0c58820dc8c0853cdef658a61bf45d8174798d469a59582c" },
    sharedGraph: { bytes: 28640194, sha256: "95d72ccddf0f3bafdb8e869e5aca7adf5b1d427a187b2ae545de9f6eb4e2f84a" },
    sharedArchive: { url: "/models/smolvla-c63f9ccf6048/hsr-smolvla-shared.onnx.gz", bytes: 1890658, sha256: "595c5c1f811d015d4b9715f80a87b500d5d025c29e8cf1e0a0aec7094de0777e" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "c63f9ccf60489ea74daf6b13f3baa3ed46f76f27df20298e7df1bf472ad6387e" },
    references: {
      "browser-manifest.json": "379deff7e3dbc5ae28dfbc10ba17eb658fdc7a0f9f33e45735fdfcc1a9d88613",
      "browser-case-0.json": "650497bad32084a360dff37ffe946e9e3e9ed084e061a9b446df641fa7e40e2b",
      "browser-case-1.json": "afcb18d6979a08b8bf70f468e66b41a170faf9d4f0a4744e2e33f52a3601fc24",
      "browser-case-2.json": "d564a64d4c0195186315d4de726caa9bef8b9a476dc02a5a2cc91daba4e28f8a",
    },
    chunkOffsetSteps: 2,
  },
  // Public parent f289 + 2000 updates with training-time action-prefix conditioning, delays 0–10 (checkpoint 03758c18, source 07696e9).
  "prefix-rtc": {
    id: "prefix-rtc",
    manifestGraph: { bytes: 1583548687, sha256: "d5b2996e2614d338ff03baceffc5631cc6bb09b3f2576154e2a452a81a0ddcf6" },
    graph: { bytes: 9881821, sha256: "d26b52730729fcdc0aa1976043e5f6b59e52dbfaa7d32ad9d7bcaacc46dbb6db" },
    sharedGraph: { bytes: 43400801, sha256: "c7d5e9145e02146fb66fd57de9fa0d7d7dcf86693951d61a730e36db9001ae51" },
    sharedArchive: { url: "/models/smolvla-e7de77cd3659/hsr-smolvla-shared.onnx.gz", bytes: 4448536, sha256: "e89c1d61426b495bc49a96b1b5087466551a94699fb2491c7d0b8b8fe78518ed" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "e7de77cd3659764abb4f004a32727cef8f0519a05c2e0eadb9b03d1c74c0e4d3" },
    references: {
      "browser-manifest.json": "1f18593161160b179a9f8dd34f9fecf72b391eca3ab8d9ee2132b30cf316af0c",
      "browser-case-0.json": "51f0f9bcec21e5e92b4d61ae5d5323cdce45baac1df2fc51f5464162b673ff12",
      "browser-case-1.json": "83a9157a222ce25b5457e9c2500f313921048324b7d6c707a34efd530e89a39f",
      "browser-case-2.json": "66aee695fa04f7ef1e2ba478cf0233330e40dd09f9cbaff39c13701172a5449b",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 10 },
  },
  // The prefix-rtc checkpoint re-exported with 5 instead of 10 Euler steps (same weights, tools 65da28b).
  "prefix-rtc-e5": {
    id: "prefix-rtc-e5",
    manifestGraph: { bytes: 1579475339, sha256: "51aad9d8a669e3ac8918f715d3a1d1299455b5a1be76d380cde9ea37739e73a8" },
    graph: { bytes: 5664693, sha256: "16bf7a52f4796a22ee1897d4bb801f91efa84ae8b928e7b9dd81d406e0f71043" },
    sharedGraph: { bytes: 22341802, sha256: "bf3524a3a97907688ba3ec744d044293ed7b84e4bbf27fc12f0d7c769e6b4484" },
    sharedArchive: { url: "/models/smolvla-e7de77cd3659-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303740, sha256: "8cd09ffea5fa97d58275d030c79c6f96383b7fba383ada24618b075d2069669a" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "e7de77cd3659764abb4f004a32727cef8f0519a05c2e0eadb9b03d1c74c0e4d3" },
    references: {
      "browser-manifest.json": "eb9655fc19727c617cc91381ea632113306b3758a8c3edce4533e361fef781c4",
      "browser-case-0.json": "f473b402f5adc973cd73d282d53daae1db749de0e174a4a2e9529182e5ccb0d8",
      "browser-case-1.json": "0736dccd9e3d050ec299bd06f656cff480ec2670b3602bb42f9fc483433ac283",
      "browser-case-2.json": "e25bf2a97a019ac92ef7fd4d119f6ece028c09fb93b5c3beb141c2dcfe826f11",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 10 },
    hostedName: "smolvla-e7de77cd3659-e5",
  },
  // Public parent f289 + 8000 updates with action-prefix conditioning, delays 0–8 (checkpoint 326e2f55, source cfd071e).
  "prefix-rtc-v2": {
    id: "prefix-rtc-v2",
    manifestGraph: { bytes: 1583548687, sha256: "fd48c2319796df66e3cbb18dc77dc112f5b6253146a0355ba08c90d29ed2bfae" },
    graph: { bytes: 9881821, sha256: "a50cf8bcd5ad4e547528a86478f0b20c6706f4d8ecc9a368f44572efa09ca0e3" },
    sharedGraph: { bytes: 43400801, sha256: "44604369380ef7b57cc86c9a95fc31159b830ed4aad9c69454f4177b4a90f248" },
    sharedArchive: { url: "/models/smolvla-33fec139ff9e/hsr-smolvla-shared.onnx.gz", bytes: 4448536, sha256: "c4a1f90d888a0e59d470d13209ee7c2de37aaf6d0789eaa38fd9aed154f4e7f6" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "33fec139ff9e6f2928a59e2371a6319d6d702c258ef845aac77f99a09aba762e" },
    references: {
      "browser-manifest.json": "88cc6e7699a9e6c6d4d7008a6601bd24fed1701a6cbca70f6f977de26f0b1090",
      "browser-case-0.json": "b6af7342524aca7e4489e36af968f59383be58186d6d97a7306b1aca19249a0a",
      "browser-case-1.json": "b5a7eaad35d3fd839cbbd90f41804a624007c5f9925e4b17e07bc055478afd8d",
      "browser-case-2.json": "c664b3ab37f197c9bc2ac8300ae6d0bace31e31fd425343a493bcfeeaca5a9ad",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v2 re-exported with 5 Euler steps (same weights, tools 65da28b). Failed its synchronous gate (8/10,
  // 2026-10-26), so it has no continuous-execution results.
  "prefix-rtc-v2-e5": {
    id: "prefix-rtc-v2-e5",
    manifestGraph: { bytes: 1579475339, sha256: "8257d81ef1bc8b62d32694cd5797cb7d6d9e290023c2ca975cae412a72330ca5" },
    graph: { bytes: 5664693, sha256: "2abf9d91d6564f6cf68e7853d097fe66bde62ffdf05a847738359750a0ac95d5" },
    sharedGraph: { bytes: 22341802, sha256: "646516ead3b49f60d5383168ff6314753371550d3bb0adc0155ac0667d6ce182" },
    sharedArchive: { url: "/models/smolvla-33fec139ff9e-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303740, sha256: "2a8227b4fa0b32349b4592a43e6f6782d7bc17ed3f4b94b0b40a46c6b74092ff" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "33fec139ff9e6f2928a59e2371a6319d6d702c258ef845aac77f99a09aba762e" },
    references: {
      "browser-manifest.json": "7dfd7419ea932bb1302b24ced10376550e733b9cad510a479da1f38f87218928",
      "browser-case-0.json": "b3f663a03305533443eb724e2b8ba1e6458a6171992a274dd4e3bf89e4f8d224",
      "browser-case-1.json": "e307d1b2260f5aeb2fc7d2b2c6b56611fb14caa07125e01bd7766dd1cedcba17",
      "browser-case-2.json": "f6549196ae0e9d6809e05d37e3de5ff4df7a539bdd1a3e2399511da76b487575",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-33fec139ff9e-e5",
  },
  // f289 + 8000 updates, delays 0-8, on the gentle-mug recollection (fork 729e787b: 322 new mug + 50 ball episodes,
  // parent normalization; checkpoint b5887fa9, source cbdc513). Local candidate until it passes its gates.
  "prefix-rtc-v3": {
    id: "prefix-rtc-v3",
    manifestGraph: { bytes: 1583548687, sha256: "97f4641a60ad9545921edf539993c72f1591234d20d98ed0f6cf7578a21c359c" },
    graph: { bytes: 9881821, sha256: "ddeea8bc61a01468aa0da6432671d4cb714ab8aeae1716bd465f357653589d3e" },
    sharedGraph: { bytes: 43400801, sha256: "0d2b213fcdc1be948bbd4da6e094b1c8f3b1b311c26730bbd61662004cc74b6e" },
    sharedArchive: { url: "/models/smolvla-9002612dfb47/hsr-smolvla-shared.onnx.gz", bytes: 4448531, sha256: "a385fdeb5adc1146fee589e2c1b33a3ab43c6e82c9a03ed299812b1a7fa61ef9" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "9002612dfb4723f710608fb6dfc1b94f58901c63bae299f8187c5c8358f4cee0" },
    references: {
      "browser-manifest.json": "5ba40d62787f066c7b617aeb16481f9dc3d6ea0cddd6d9caba08180129f6c2e6",
      "browser-case-0.json": "ab7dfa934e8e2a0085f8116d2a22dbb1b8511d95b6cf40d09dc0cec965da78e1",
      "browser-case-1.json": "da542b0a8f240cba954f0ed1620d4608a8d8f8782c51dd40fdcfdc389b8b8c17",
      "browser-case-2.json": "42c46a645ed3e37099053c07012be20b77f15fbc709b8dabe774802c8d58aa85",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v3 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v3-e5": {
    id: "prefix-rtc-v3-e5",
    manifestGraph: { bytes: 1579475339, sha256: "08d76c80f00f22a803b2f4aad556e4a842cb3b5343445fa1dd4ca5685b82074c" },
    graph: { bytes: 5664693, sha256: "6ad1af1d122f8d583ed98641219c6a76f09b70a29ce52527ab6291480161b2bc" },
    sharedGraph: { bytes: 22341802, sha256: "129fbcdafcb1682e81778d41916012ea7c31ba5dee87e7cf6bec1c50adc82cad" },
    sharedArchive: { url: "/models/smolvla-9002612dfb47-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303737, sha256: "68b8904ad982065e0fa50e9baa93477a24be344e627a25d4a0b00068548a4a1d" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "9002612dfb4723f710608fb6dfc1b94f58901c63bae299f8187c5c8358f4cee0" },
    references: {
      "browser-manifest.json": "0ee4382df661d39b253b1b3af8cf15ded5c35b9da0e46c07f955c8cb55b0acb1",
      "browser-case-0.json": "25726bc221041e5eeee09afd6de32b7584b02bac406c4ce084086555aac64b89",
      "browser-case-1.json": "9088a885999e6f4313102f11d28c535c04c42ba7ba2f68ad0ee329d2321a0141",
      "browser-case-2.json": "ad13a1d7b5c6e44727b55f8857a5629887cc6c2683a3bb973c6b8465eefbb6a7",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-9002612dfb47-e5",
  },
  // Stage 4: the v3 fork plus 143 certified takeover segments (gentle teacher after prefix-rtc-v2 in continuous
  // execution; fork fa6b3aca), f289 + 8000 updates, delays 0-8 (source 100fe22). Local candidate.
  "prefix-rtc-v4": {
    id: "prefix-rtc-v4",
    manifestGraph: { bytes: 1583548687, sha256: "7faf445c7dfa031c2edc3616dc0399595cc1b3dacad33d8c46d5a201b88cf0d0" },
    graph: { bytes: 9881821, sha256: "8e35c4704fc33849e11e57c4c493fd3121ba532282fef32aea0fd30df4d80b35" },
    sharedGraph: { bytes: 43400801, sha256: "39a916e22ac5e46e9d5e401623e466f6b6e0cf68b753e241e8e58a4875e0f40b" },
    sharedArchive: { url: "/models/smolvla-952e2d2abac8/hsr-smolvla-shared.onnx.gz", bytes: 4448539, sha256: "992d21211da8d1a3fadf8223306f21338d001a401197c2f70100a25984b4849a" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "952e2d2abac8f8484db34c3533b32b1a7e3a20bf7a6d87cbc43b245b664fb524" },
    references: {
      "browser-manifest.json": "b10c36753e1b8d203699054d569381957f44122a8abd6033796a4e99472ba205",
      "browser-case-0.json": "bdb9f1c3aa0e9b9c9d5a96d90f0ed577d037ace09bc86a09b325265be6967761",
      "browser-case-1.json": "f6758d1d270ef42319f99271a28edb7f3420a27b4377df89cfb2a0738c27c5e4",
      "browser-case-2.json": "1c28a632392817e25e58ce58bb8e3c00fb6414a0d5d0cd33394fc873d393e181",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v4 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v4-e5": {
    id: "prefix-rtc-v4-e5",
    manifestGraph: { bytes: 1579475339, sha256: "1a1a71d7ea5c54dd5810af17b5fb84c2157e39800069a3a6d32b27224acabd7b" },
    graph: { bytes: 5664693, sha256: "c14f3867d96f1a519cfc3a3259e7024c269be88651dc79c7386488fd057a3e72" },
    sharedGraph: { bytes: 22341802, sha256: "301a30d5f059616af90c0e1d3872da8b7e7fb056ecbdf631454e3289142ec072" },
    sharedArchive: { url: "/models/smolvla-952e2d2abac8-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303742, sha256: "b10e292d04a501f2dbe6760a25db6c9cbd22a264e5da2eb46755bfbdd00ef4bd" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "952e2d2abac8f8484db34c3533b32b1a7e3a20bf7a6d87cbc43b245b664fb524" },
    references: {
      "browser-manifest.json": "4dc8037432c83689916af8ad3f9ce956591c326780b7c22e1f455f4555b155bf",
      "browser-case-0.json": "887af8fb66f3ea484e37554ebe9f96da2835619308c7873a11c33418fcbe70e1",
      "browser-case-1.json": "35e9eadd36c5b8ffc13b5aaac67c63635c9fbd233bd6de4aa6374085ba79ecae",
      "browser-case-2.json": "899c2433782149127789590ebfcb5ab0bdba2721279dad77612c5a5422447aa1",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-952e2d2abac8-e5",
  },
  // Stage 4 round 2: the v3 fork plus round-1 takeover segments that entered at lift/carry/lower (100) and round-2
  // segments after prefix-rtc-v3 (43; fork 6d1554dd), f289 + 8000 updates, delays 0-8 (source 98bd945). Local candidate.
  "prefix-rtc-v5": {
    id: "prefix-rtc-v5",
    manifestGraph: { bytes: 1583548687, sha256: "4246729e906c9bad3a640b3828b2500a33519e72b84da66415d811f413fd6fb2" },
    graph: { bytes: 9881821, sha256: "cf63ced1e950d5f93b99faaacf989c42df7f11dfd398e49cabee2c705b6ba59b" },
    sharedGraph: { bytes: 43400801, sha256: "7402135eefc52a3348277dcc8e5744579a6c76041ab44f2a08860c6d4d012178" },
    sharedArchive: { url: "/models/smolvla-92d0c65d37ae/hsr-smolvla-shared.onnx.gz", bytes: 4448539, sha256: "da67f72979e742b17f4f46d510ea11c46e586b37057090dbe239164e308aed96" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "92d0c65d37aeed687b753bdb08083daf137ce2e0e74f20e8245274cf52fb25e1" },
    references: {
      "browser-manifest.json": "b3f945f687e3677fdfd7156bf63f3d05ccbc075b9cdc3a23cc5974093b47c860",
      "browser-case-0.json": "b2e0017fd70040375ba258236ed7f271ac35db572fac10204029c347ab7b93ab",
      "browser-case-1.json": "c2f06a627d7463bca5721f5f77cb989279023416ecd827a8396849ec24a22079",
      "browser-case-2.json": "3d9de983d92906c82cb03e15a0289ca3d418bca79d3a4f80f782eb4f797fa2e7",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v5 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v5-e5": {
    id: "prefix-rtc-v5-e5",
    manifestGraph: { bytes: 1579475339, sha256: "dee7495d05d6bbd05e0e4ede307642a246871ab37be9168b7757db1a74dbf193" },
    graph: { bytes: 5664693, sha256: "c190a9098d39e28152aeaa85651c171a9ff6587095a5f510f539d1abf51b86ec" },
    sharedGraph: { bytes: 22341802, sha256: "dc14670294849335d4f183e16900f4466aeba6e3d079dc7b75c1027466dc3c09" },
    sharedArchive: { url: "/models/smolvla-92d0c65d37ae-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303742, sha256: "28a2c8c4fa180fc29a6bb10e8c2da7bb11ae2fa57c34059d74d00f5c89e53c27" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "92d0c65d37aeed687b753bdb08083daf137ce2e0e74f20e8245274cf52fb25e1" },
    references: {
      "browser-manifest.json": "ab3ff7ba9f814ebba3cd2c4f49ea53789b145b6ee1e739eee1c3f7e31b19aec9",
      "browser-case-0.json": "cc067f6237caf1637df8c1b2120b4e4cae1238b661599772b67fe884d7f21ec2",
      "browser-case-1.json": "8ac5722f6eea017f12d0f9b091d9afb9f2e41c08bb2ee1ad7d526de4575c7237",
      "browser-case-2.json": "7c68ecf5bcfa0240561cae58f149dad6825fe7b16f91e2ef1f905f5b8a31b0e8",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-92d0c65d37ae-e5",
  },
  // prefix-rtc-v6 (2026-10-27): v3's data with --mug-recipe-v6 (16000 batched updates of 8, first20x2), local only.
  "prefix-rtc-v6": {
    id: "prefix-rtc-v6",
    manifestGraph: { bytes: 1583548687, sha256: "04bb1f91a7869c33c7f7dcd93d689bf42af57b4b22fb80533427286c26af3f23" },
    graph: { bytes: 9881821, sha256: "4555a86009dcc864fe6eef87b48a95e66e584478aba396c232f67539efe8f9f8" },
    sharedGraph: { bytes: 43400801, sha256: "f541743e13abc4287771ea7f68479d977ba785465c1518ea3a0ea09b8a1c0fb8" },
    sharedArchive: { url: "/models/smolvla-5b4005f86425/hsr-smolvla-shared.onnx.gz", bytes: 4448539, sha256: "cdb0f0d3120afe339493c40768d2e5f3de2cee5a572c2f476ec43221f2ab21f1" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "5b4005f8642551423713e3d0da6bf643f04044e51427f3c82064c43f7e2bf59b" },
    references: {
      "browser-manifest.json": "7fcb3cea92214b8a1c34da520ea3e94db152a6686defbd77d06b836d0271b9dc",
      "browser-case-0.json": "c0b84ef68b4551c35d4e4ddb4d322faa1a39fd926cf07fc7c462204de996a2f6",
      "browser-case-1.json": "8c875b9a01b59112106e8dcb93c1bb4072ff8a70256d2cf19f6d0c4d7e1b2edd",
      "browser-case-2.json": "b4ee1c907b488b277470e49531de30624c75533bc0f10f4a926ce0033a867cd2",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v6 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v6-e5": {
    id: "prefix-rtc-v6-e5",
    manifestGraph: { bytes: 1579475339, sha256: "57fb6de125774fdba69555e2c481e20ed7d51c02e99b02a7292bc4092f42b18c" },
    graph: { bytes: 5664693, sha256: "0ae535809b4d7f87922e5a8f423dd357ae5fd6fc4a4f3e5953ab19bc95878770" },
    sharedGraph: { bytes: 22341802, sha256: "afeedc2d173babd57cdea19cdd1dfee800c85a78482a849da23113790496c819" },
    sharedArchive: { url: "/models/smolvla-5b4005f86425-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303741, sha256: "5c740f9485b429cd3bb8837165c70036a37a17f2517bba57694fc995767caaa9" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "5b4005f8642551423713e3d0da6bf643f04044e51427f3c82064c43f7e2bf59b" },
    references: {
      "browser-manifest.json": "b4c3de067324e8d2f4624ab9040e7f1f88fe7825d5de403f7eaffaae75b7a20c",
      "browser-case-0.json": "a829aa5a951066f0fd119486eadb0d0837e960156c2adb59ef13cc9b46737a18",
      "browser-case-1.json": "c7f949e8cfa6d68b0ed9cf396d720c7c7b3e21816ac6dbb4755a256ccff2774e",
      "browser-case-2.json": "0eb13022260f8186ea9af719c1b9fc096088dc2816800af7bc89985b3e045f12",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-5b4005f86425-e5",
  },
  // prefix-rtc-v8 (2026-10-27): v6 recipe plus --train-vision (SigLIP tower and connector at 1e-5), v3's data, local only.
  "prefix-rtc-v8": {
    id: "prefix-rtc-v8",
    manifestGraph: { bytes: 1583548687, sha256: "ffede0eb215e54f620a8b9424e580c94deea478f0489008706f1ecbffc271bb9" },
    graph: { bytes: 9881821, sha256: "7340da3fd50259d6d542500a124c0c53475eda83f040384f47db17ce2ece9d0b" },
    sharedGraph: { bytes: 43400801, sha256: "bdb965cc65f5aa97763fa9b19d1f1bd7fc47eed5e44b2396cefedcd623b3a2f2" },
    sharedArchive: { url: "/models/smolvla-f5ef526c6626/hsr-smolvla-shared.onnx.gz", bytes: 4448535, sha256: "032a6ab684ba2e63dacc32cf5d5a0605a7a79a63769383495715993e30159a40" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "f5ef526c66267d7e67358915f00c615fa8d5a0433da14def01f6fae79dc8e76b" },
    references: {
      "browser-manifest.json": "e9d4ff3a4970d9fc29e25ad20284a8e28b3233ae8ec614325bc1cee3b67c3fb5",
      "browser-case-0.json": "986047a145921a77fd9edeeccdef4e6cf79ebf46d634f27da2684f6cd0dc39c0",
      "browser-case-1.json": "e1596c32351a7b9deecc6159e12766815552e2d8e00d72cf01225778405e4162",
      "browser-case-2.json": "ab5564d081b265c9618c7522517abdcb8f76338f67c50bea590f59d549e9af7b",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v8 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v8-e5": {
    id: "prefix-rtc-v8-e5",
    manifestGraph: { bytes: 1579475339, sha256: "c0c73683a83cad50f205b1a19d096f4997cf32b2b65c1db62cd2d9fb4087c106" },
    graph: { bytes: 5664693, sha256: "3a4f571eb5e600c5651e12a43cef9a769e6f6440c0678fa3c4aa2ee54d6215df" },
    sharedGraph: { bytes: 22341802, sha256: "54b4b4a3bb0a401284c8613abc37a6de948ac03f8d68271ba6eb538c8e30720e" },
    sharedArchive: { url: "/models/smolvla-f5ef526c6626-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303739, sha256: "627f260bdd2fb0e0391103b11094a6077f26e1fb3e099089ff826227ffa3e13f" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "f5ef526c66267d7e67358915f00c615fa8d5a0433da14def01f6fae79dc8e76b" },
    references: {
      "browser-manifest.json": "99aed9677a0903322a62da7ef39749080181fffb49c4c2ab75deff60c162ba2c",
      "browser-case-0.json": "2522aec4c228e205b620402715c705277545d297d8d77574573a9800111a74a2",
      "browser-case-1.json": "01ccfdad666bd17afd2e5dc34e3f45da47a92751468ebe4aa8d4b6c055dfb601",
      "browser-case-2.json": "8c6cbdf547a2e1e2948017e84876fb2a0668e9eee96ef042a4f0530ff08dedf8",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-f5ef526c6626-e5",
  },
  // prefix-rtc-v7 (2026-10-27): v3 recipe on v3's data plus 293 DART mug episodes (fork 9532280e), local only.
  "prefix-rtc-v7": {
    id: "prefix-rtc-v7",
    manifestGraph: { bytes: 1583548687, sha256: "17f0380a7546b0d5d44b3d515980e7c5420e7a33fcfde64751dc9a6621f3580f" },
    graph: { bytes: 9881821, sha256: "9a3a5e2a0facbcfdc947b51c44163138a81f0e38f22b13f78a5805e4457e7422" },
    sharedGraph: { bytes: 43400801, sha256: "68009c4999b190b0d561472f28adad537c3bf34c5644f82fcdaff2fb31dcf1c3" },
    sharedArchive: { url: "/models/smolvla-52825f665b5c/hsr-smolvla-shared.onnx.gz", bytes: 4448537, sha256: "d8cfe2e944709faf14f8cfe2ddfc7f5b0f311bca6baf884f8634b9ec2f64c9a7" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "52825f665b5cf62a2a216a6733618b170df7eb9b23b7d7431517990193183e37" },
    references: {
      "browser-manifest.json": "c017557c531ca4851bd86ce2d70674c7563b3dd34ec7ac9decb6f7245520d6dc",
      "browser-case-0.json": "c41f514e77e70b8500571802a4d6ead5621a635ad79165a25fbccfa59a4dcf2c",
      "browser-case-1.json": "37964804f82b350554e9466a1ce8ac5603e89a3654e9e91943a71e354c4e3fd8",
      "browser-case-2.json": "889232a5ffb04e39bd439cf3f1cf6d297195aa038dac9cdb36c7074b6c027ee8",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v7 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v7-e5": {
    id: "prefix-rtc-v7-e5",
    manifestGraph: { bytes: 1579475339, sha256: "ce311eb2c21d29439515621067d02e03569f13f53cbdff7db83ed42efae9f5e0" },
    graph: { bytes: 5664693, sha256: "9bab2cfeb1c796c3c4bdebc18925a6b04f9c87923c8af3c62dd194a209ee1878" },
    sharedGraph: { bytes: 22341802, sha256: "c059db1397a1b42d69b51acc64dbddf55858dec1019c52c1c624cd5456f1513f" },
    sharedArchive: { url: "/models/smolvla-52825f665b5c-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303740, sha256: "be16d9437f91b14298275b93a7ad9bbb21e7d060c28c79164b8206a4caaee22b" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "52825f665b5cf62a2a216a6733618b170df7eb9b23b7d7431517990193183e37" },
    references: {
      "browser-manifest.json": "af984c11cdb62e41acea0e99e6b7e1ab0a59adcfa8e945f704dd800637ec3bb9",
      "browser-case-0.json": "118fb2f1df2f82ae48d5fd103531a6283f17c0c18dd38798a415d9c726744125",
      "browser-case-1.json": "641da3f9c980799c824d35c1de4e764c8bcfd907d760be0813cac478b401dfb1",
      "browser-case-2.json": "59e1bf8ac689b2f7f4117b99b1ce0df045285ab6b7b5b71848ace48674f7e587",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-52825f665b5c-e5",
  },
  // prefix-rtc-v9: v7's data and recipe plus --prefix-noise 0.03, local only.
  "prefix-rtc-v9": {
    id: "prefix-rtc-v9",
    manifestGraph: { bytes: 1583548687, sha256: "78a0691d017182b2c596130957e4b274996215008e0d51465d4e2996f9d83dfe" },
    graph: { bytes: 9881821, sha256: "d68f7a4afe18d3d790bdae40ec9aac5bdc983403e4c62b29fcc97c3b84a87032" },
    sharedGraph: { bytes: 43400801, sha256: "78caf4e123ea6d7fee780293e2ae65d1a7cb0e4a1eb808ace74e45e12a917dae" },
    sharedArchive: { url: "/models/smolvla-91aeb7db94a1/hsr-smolvla-shared.onnx.gz", bytes: 4448537, sha256: "1d5d87444c257eda567e81c62ed36796a6edf3e1e813afd2c89a6566cbfb2611" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "91aeb7db94a10c5db3dc0a320dbe3e083800164ee8959a24a3dd20adb55f24da" },
    references: {
      "browser-manifest.json": "afbf92681978642aa12c71630c32428cc6a4ad9da6fb9a85e65cf9984f83d0e9",
      "browser-case-0.json": "03b377d4c5c3bfc780397ea356b39b6b01a820c31fd28105f8cb25a746ec5671",
      "browser-case-1.json": "afbf0fe0e3f7c40a5b8c6839058a93cd4d95ba3419e75cd2a87f6fd70ae63507",
      "browser-case-2.json": "46af55da4c00c66dd1d78096e5b1eb886d649fdd92bea4ffd6c66f2481ccb55d",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v9 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v9-e5": {
    id: "prefix-rtc-v9-e5",
    manifestGraph: { bytes: 1579475339, sha256: "872ea34d623e6978f50e97f9b7ffb54783c804e5efd329cfd3e1015c765feabf" },
    graph: { bytes: 5664693, sha256: "21f34f9871434ab4a083592c2d48b4863a701f053a1bf7cfc32fe404ae361ac1" },
    sharedGraph: { bytes: 22341802, sha256: "1b6ea6d54dcab5f62d26c2b90262e62b4dafe59259c361d5bac1d70ea5f5f7b5" },
    sharedArchive: { url: "/models/smolvla-91aeb7db94a1-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303741, sha256: "d240edfb8568a256f5aa79e1e03249bdda35ef701ba433cfcf15537c8d86b737" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "91aeb7db94a10c5db3dc0a320dbe3e083800164ee8959a24a3dd20adb55f24da" },
    references: {
      "browser-manifest.json": "ff2c9b876031f163cd522862edbe112dc26678824b077d239bfa5a31d7fdbe08",
      "browser-case-0.json": "5d1629732283e8499a0927ea5fcbfb0aca487190f89bed2f5099268c0bcba457",
      "browser-case-1.json": "c2a9fa2bafdfe7a1e5dc260a71256bc887ce4bad24a7e1393e0e69ce776f135d",
      "browser-case-2.json": "57b3549e8eacc2ec6c3e3a8e8e9e195998ffd2429f1b575044c55f894c10df50",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-91aeb7db94a1-e5",
  },
  // prefix-rtc-v10: gentle-mug-v2 teacher + DART-v2 data (fork d970421b), v3 recipe + --prefix-noise 0.03, local only.
  "prefix-rtc-v10": {
    id: "prefix-rtc-v10",
    manifestGraph: { bytes: 1583548687, sha256: "67ba23867a5d5938ff223cbac2ff327c264bbd2292b19f1a4971bccf6bf0888d" },
    graph: { bytes: 9881821, sha256: "1d66fc266a49f358a83d863653293ae19ce94d05f2ad1377c3154e2d787e9ad4" },
    sharedGraph: { bytes: 43400801, sha256: "0a4820bebeba6d92e2976574b9c394f3cf4040f2606e9c6603428e600520c6fb" },
    sharedArchive: { url: "/models/smolvla-cee1fbdb9cb1/hsr-smolvla-shared.onnx.gz", bytes: 4448537, sha256: "ab6239f1649ec872c15fc8d374d7ac3b3db73362956481e7900ca7d50f5567a2" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "cee1fbdb9cb17420d0ab3667a329d6fd6a0960c6eddc85af56a131220a08e826" },
    references: {
      "browser-manifest.json": "e7f8b60b2a739f1ba856005999a7f492336c5a6bab7c3c174eb915d0fa977f2f",
      "browser-case-0.json": "efb0e613f2286fc418945667d16f75cf1c7614cf44ba0ae016d619c1e624579f",
      "browser-case-1.json": "be9f5efa9a3917ec48473b7511106ba8434ab42ecce81b95f551cfd04846cff5",
      "browser-case-2.json": "5ce0d9053fb245d4ac0f78a0675d80e03a4fee9b71475b8e0b2a1d5734c286dd",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v10 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v10-e5": {
    id: "prefix-rtc-v10-e5",
    manifestGraph: { bytes: 1579475339, sha256: "8b351400d05f7ce8d237c65fedb967a6361c22838ad9a94c73daf20aaf9cad7b" },
    graph: { bytes: 5664693, sha256: "e2deab75f6baca16cdac1a4aebe8bde218a4b35b4fc18a7cfddf49a19c27d8bb" },
    sharedGraph: { bytes: 22341802, sha256: "1decb351637ed13b437d9daed16c748830a44d078fd01404d49be7233e24db74" },
    sharedArchive: { url: "/models/smolvla-cee1fbdb9cb1-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303739, sha256: "32a49e8f4ba21bd2b5b965d8be9209517de27ba34e0c9f640dcf8a9f450cff96" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "cee1fbdb9cb17420d0ab3667a329d6fd6a0960c6eddc85af56a131220a08e826" },
    references: {
      "browser-manifest.json": "047c8b61091385041d9b38ec68254db110602d5c97d89d404fbebd13478f44aa",
      "browser-case-0.json": "a4f2cacf7b0090b3767f68720dc5c07aa54ea942a3cfcab9339ef292a7d20df4",
      "browser-case-1.json": "41dd44d710a21d04b89ddacb99f4f258c7eaeaeb7cb6576172159034e5d50e1d",
      "browser-case-2.json": "f4555aadb8bf108dfcfba116a79126eb4d454e03642808f51b995978e048cf0c",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-cee1fbdb9cb1-e5",
  },
  // prefix-rtc-v12: gentle-mug-v3 teacher + DART-v3 data, v3 recipe + --prefix-noise 0.03, local only.
  "prefix-rtc-v12": {
    id: "prefix-rtc-v12",
    manifestGraph: { bytes: 1583548687, sha256: "26482ef52068becc7aed8cbc4fd99328066c0de833b4af933524b2722440ba29" },
    graph: { bytes: 9881821, sha256: "7ffd15ef8576d7c4ea112727a3833fffa2c687c944a8055096fa7b1f5aa8ccc0" },
    sharedGraph: { bytes: 43400801, sha256: "3a9d8b14078e38fd92491527a7ca1b43f950c2f4693bd61b23ed69cff2cac32a" },
    sharedArchive: { url: "/models/smolvla-a1c6ae5deb14/hsr-smolvla-shared.onnx.gz", bytes: 4448537, sha256: "cb02b7792f854cc9f1174d2feb6b2da5389c7af39b42d9e0058094c95dd474c1" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "a1c6ae5deb14d8a87d047c5960671acbbb9588432175361e23f42c903e7d224b" },
    references: {
      "browser-manifest.json": "fb7d2a262761fb59aae805a4d021cb6d7dad0c30b87431971437d1341749b287",
      "browser-case-0.json": "eb044053c200ec8c4902d30a2574699a53aea964c152d74c60471a325f14c2c6",
      "browser-case-1.json": "412ccae9e576cd7058fa801dc3c56bd8b1d18803c8343b91986e52e14403f2c2",
      "browser-case-2.json": "20a463f09704e7a36be8cbf32806553af687c1f31a9a3e2f1fda396a0a6711b2",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v12 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v12-e5": {
    id: "prefix-rtc-v12-e5",
    manifestGraph: { bytes: 1579475339, sha256: "701602ad8ea3be4f446aacd69152ab20da19b7aa633f00aea98f674b1a30c95f" },
    graph: { bytes: 5664693, sha256: "ac0cbe0eba5ecc9d494016bc5992de6db7df8f4e273ba74cf2e69d561d89406c" },
    sharedGraph: { bytes: 22341802, sha256: "221b941f7fdeedb799cc8eb935fddd08fddeb5bc79593685e4fb8e12073a89af" },
    sharedArchive: { url: "/models/smolvla-a1c6ae5deb14-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303741, sha256: "0add8a2f017022f5eec28f787be0b8db82ce45a806cc5967eec9307c812e1e80" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "a1c6ae5deb14d8a87d047c5960671acbbb9588432175361e23f42c903e7d224b" },
    references: {
      "browser-manifest.json": "581a5f6e67ecc3d5f178c8545cb8e631132ef53d0b966891112cc1390357fd3b",
      "browser-case-0.json": "49349dcc2c185215b80f368899ecb8273eef404458ff81577363d43cb155504a",
      "browser-case-1.json": "a91f5be220039d17c38c44c1efe6595498da2609d41f3790064763b5f5a56cde",
      "browser-case-2.json": "f181b36ffa662c6c37f47218a13bc5b7774219f5415bd84ad780666b50c70296",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-a1c6ae5deb14-e5",
  },
  // prefix-rtc-v12s6: v12 at step 6000 (dev checkpoint screen), local only.
  "prefix-rtc-v12s6": {
    id: "prefix-rtc-v12s6",
    manifestGraph: { bytes: 1583548687, sha256: "6d00ae0f376cfb9a53cc8e00e927c5f6cee360973d175bdb6d3797d5930ea956" },
    graph: { bytes: 9881821, sha256: "bba8f11dd2218eac5d65aec882c8a49591dfdeefe60540364aa9a6a4fb5270c8" },
    sharedGraph: { bytes: 43400801, sha256: "d244ac22208558c5d1c8f260c2ba980daf27d3e5117b345ac7b49adedebb04ac" },
    sharedArchive: { url: "/models/smolvla-bd26794f5343/hsr-smolvla-shared.onnx.gz", bytes: 4448537, sha256: "de0ce02b3eb532864676ecca3f03a5440572c4c5a1f1a2a55eabd44ec838a308" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "bd26794f5343d4d98f5f6bd5ef3b16c17168cfd10795ecb7209846f85ca86078" },
    references: {
      "browser-manifest.json": "c0109c5ab8f3f300292460683fc08ddcc90e9d0e375504c93f138a0bc5dbb292",
      "browser-case-0.json": "fd90c1de12f2e79d692a7d4b75c09c2d01ef2ca9224a35c86fcd49020a1e6cf0",
      "browser-case-1.json": "c9c5233ac7810432a6a07a7713b9cf9bca8773e373598fac8b9fedbb527f6644",
      "browser-case-2.json": "38dfeec0966839d5e1cf389254046bc10591ab689f985c7bf26bbe193300fb16",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v12s6 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v12s6-e5": {
    id: "prefix-rtc-v12s6-e5",
    manifestGraph: { bytes: 1579475339, sha256: "e442c5209e74cfdcba370693a8fa9164392ff1f7c3f6e007423f21bbe94c1f9d" },
    graph: { bytes: 5664693, sha256: "101a5c9ae35770b7808d15c6dde01c2a4399586181a9a9c7643709bd5c375652" },
    sharedGraph: { bytes: 22341802, sha256: "b38adb42dcd432047b33a84580f0fc81b636de96c90e2056272785fa663480e1" },
    sharedArchive: { url: "/models/smolvla-bd26794f5343-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303740, sha256: "8e1396872709568c110892e9b70bcc91a724d32ea99ad41683840bf3e43ec115" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "bd26794f5343d4d98f5f6bd5ef3b16c17168cfd10795ecb7209846f85ca86078" },
    references: {
      "browser-manifest.json": "9cfe4ad255dbb426901c94f1fe60053543e2dbd826d6facb5362ea90a08cbc93",
      "browser-case-0.json": "b73f7ea0a37d9af4aec997b1a982cc9f0bbd493c305cf300e933cc887b7bedde",
      "browser-case-1.json": "675200682faa78d8a03ea7022decf396083f4ce54d10630a63b11e4938caadac",
      "browser-case-2.json": "eb6daa4e17af4ab3efc1fd636b1e5f517286a7cfdc651e5f63f91a8b2d7f453d",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-bd26794f5343-e5",
  },
  // prefix-rtc-v12s2: v12 recipe with --seed 2 (same v3-teacher data), local only.
  "prefix-rtc-v12s2": {
    id: "prefix-rtc-v12s2",
    manifestGraph: { bytes: 1583548687, sha256: "eeb0a21d6a37ff4ae978423e6da69621df7282d3ca1e046f8ad36a26fe633416" },
    graph: { bytes: 9881821, sha256: "665d2218e54d6de6cafffe04bd945837e09b1f7411fb428e184594f82f636ca9" },
    sharedGraph: { bytes: 43400801, sha256: "e441b09ae1e672155376d900ed4f35fa67e28eafe606e309bdefc907d34c963e" },
    sharedArchive: { url: "/models/smolvla-ca7315687552/hsr-smolvla-shared.onnx.gz", bytes: 4448538, sha256: "2175f1611767a65bcbfa4309ff8949280c9a90bc7df7410025ab892414e2b995" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "ca7315687552041a47e63241383830b7d4884e7bb1c6b180d4c9648e8b00ca64" },
    references: {
      "browser-manifest.json": "f4842e070e2c21aa0fd0993f50a76a38c2502a89f4715c27eb9575091656f991",
      "browser-case-0.json": "16aec000a94705ee56b3f27856f94ab32e33986c68a2d72364af397f348df388",
      "browser-case-1.json": "cfb9d1cc2b400b11efd83a1eb0e3524349dea36a5af7443f92942bcc4167aded",
      "browser-case-2.json": "51478c6cf5069f27ef035a9939899a95d62cac746ab029674555a88d2e8187fe",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v12s2 re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v12s2-e5": {
    id: "prefix-rtc-v12s2-e5",
    manifestGraph: { bytes: 1579475339, sha256: "fd1c975835eb3bc8a20243967c06cd163a528801be3e756da31756142df8eaec" },
    graph: { bytes: 5664693, sha256: "db91b4b80bc59dc3546fc8e50204b81f2a22bc107584237790395e914dc42c35" },
    sharedGraph: { bytes: 22341802, sha256: "4835c46ce804f05c3320c6ac8a04c446ebcdb7bcdf9da6165070debc05aaad15" },
    sharedArchive: { url: "/models/smolvla-ca7315687552-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303742, sha256: "57173b25cb2ed7d111b358d4e8ca13375bd04080e92c67833e74850b2ecb77c6" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "ca7315687552041a47e63241383830b7d4884e7bb1c6b180d4c9648e8b00ca64" },
    references: {
      "browser-manifest.json": "5350db667fa4bc773b9d0d6e790670323ba7413af1c60e1d2b6cc0fbab2db995",
      "browser-case-0.json": "9e8374e07352329b83f158d63ef16a67e78bdf35535ec53ff7c88904a04130ae",
      "browser-case-1.json": "efcad863343a4157fbd01bf2a0ffd532178d549f155b309b627fe04393e71320",
      "browser-case-2.json": "e2f3eff59834d1340a55764a450c96bc4302148e25b80daf0615dffc19bcba93",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-ca7315687552-e5",
  },
  // prefix-rtc-v9a: uniform weight mean of v9 steps 6000 and 8000 (checkpoint_average.py), local only.
  "prefix-rtc-v9a": {
    id: "prefix-rtc-v9a",
    manifestGraph: { bytes: 1583548687, sha256: "c8ddcec01dbdb538fe015cbd0cc95770982a188c7c718c70224d4083f6798e8a" },
    graph: { bytes: 9881821, sha256: "6d56032738bc7f19008760b4cedb66a3d9906868bc877f0a2ab97de3b9bb0454" },
    sharedGraph: { bytes: 43400801, sha256: "c450cf5aa30b218ee685de5df3acfa808f485325ad671e6151f457e5c3efe7bc" },
    sharedArchive: { url: "/models/smolvla-8fb599721f26/hsr-smolvla-shared.onnx.gz", bytes: 4448538, sha256: "eb68c7567db5c2e538fcbf877efd6038e0f985b8579e6bdc3acaa3a1eefdaec2" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "8fb599721f262dd81302f21cc9b348fd28208c07392f5d8fba7ff07bb6e5a888" },
    references: {
      "browser-manifest.json": "45f520448ae2f722d40aad7fa3677267b0b4949fc811d081943337363a49e4ac",
      "browser-case-0.json": "16a72f41b457590a8f80dd06be9b785b4f893ff3b018e1fefd2882dfbdce25de",
      "browser-case-1.json": "53f4d4fcf2429e5e45f0492ffec23ae7d690dc2e11682a72d209bbee2b04d334",
      "browser-case-2.json": "90f95234e17b6b0879af319b1654e6ca428a8051c028d31f289651c9b4ffce10",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
  },
  // prefix-rtc-v9a re-exported with 5 Euler steps (same weights).
  "prefix-rtc-v9a-e5": {
    id: "prefix-rtc-v9a-e5",
    manifestGraph: { bytes: 1579475339, sha256: "8e88392952598bb9572e37f330794ff664e9016708d99f7d9f318c5659284551" },
    graph: { bytes: 5664693, sha256: "eee820cc5742f9ab5f81a919f55dae4d994569edc9f1fbb680db95649e9c3da2" },
    sharedGraph: { bytes: 22341802, sha256: "e72164502d428b39e376b512530b1299c4df1760dc6653f20e14c9de8a271b4f" },
    sharedArchive: { url: "/models/smolvla-8fb599721f26-e5/hsr-smolvla-shared.onnx.gz", bytes: 2303742, sha256: "86275f8e7826df301643daa3dfbb701a97f595582cf2179bc62fc5b9a918cac4" },
    weights: { path: "hsr-smolvla.weights", bytes: 1573985536, sha256: "8fb599721f262dd81302f21cc9b348fd28208c07392f5d8fba7ff07bb6e5a888" },
    references: {
      "browser-manifest.json": "2ec596d47a9e521804458862fbde82ff20da0ebe0329baba6ef7fa354b3ab607",
      "browser-case-0.json": "0fdf855a05a4ec27d763169c22f64f96149fc652d8902ddb0b80829466bedeec",
      "browser-case-1.json": "d2305e7284d89bf9a9ce64021dda47c657a23f70c112783c81fa9f83d29bd2c1",
      "browser-case-2.json": "fc30b71bd39e435753796d239c8b871b451248ca25e3d110da5d106a703e760e",
    },
    chunkOffsetSteps: 0,
    prefixConditioning: { maxDelay: 8 },
    hostedName: "smolvla-8fb599721f26-e5",
  },
};
export function learnedModel(id?: string | null): LearnedModelIdentity {
  if (!id || id === PUBLIC_LEARNED_MODEL.id) return PUBLIC_LEARNED_MODEL;
  const model = Object.hasOwn(LEARNED_CANDIDATES, id) ? LEARNED_CANDIDATES[id] : undefined;
  if (!model) throw new Error(`Unknown learned model: ${id}`);
  return model;
}
export const LEARNED_GRAPH_SHA256 = PUBLIC_LEARNED_MODEL.manifestGraph.sha256;
export const LEARNED_GRAPH_BYTES = PUBLIC_LEARNED_MODEL.manifestGraph.bytes;
export const browserFileNames = ["hsr-smolvla.onnx", "hsr-smolvla.weights", "browser-manifest.json", "browser-case-0.json", "browser-case-1.json", "browser-case-2.json"];
export const browserReferenceHashes: Record<string, string> = PUBLIC_LEARNED_MODEL.references;
export function selectLearnedBrowserFiles(files: File[]): File[] {
  return browserFileNames.map(name => {
    const matches = files.filter(file => file.name === name);
    if (matches.length > 1) throw new Error(`ファイル名が重複しています: ${name}`);
    if (!matches.length) throw new Error(`必要なファイルがありません: ${name}`);
    return matches[0];
  });
}
export async function verifyLearnedBrowserReferenceFile(file: File, model = PUBLIC_LEARNED_MODEL): Promise<void> {
  const expected = model.references[file.name];
  if (!expected) throw new Error("Unknown learned reference file");
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", await file.arrayBuffer()));
  const actual = Array.from(digest, byte => byte.toString(16).padStart(2, "0")).join("");
  if (actual !== expected) throw new Error(`検証ファイルのSHA-256が一致しません: ${file.name}`);
}
export type LearnedExternalIdentity = { graph: { bytes: number; sha256: string }; weights: { path: string; bytes: number; sha256: string } };
export const LEARNED_EXTERNAL_IDENTITY: LearnedExternalIdentity = { graph: PUBLIC_LEARNED_MODEL.graph, weights: PUBLIC_LEARNED_MODEL.weights };
export const LEARNED_SHARED_GRAPH = PUBLIC_LEARNED_MODEL.sharedGraph;
export function validateLearnedExternalIdentity(value: unknown, model = PUBLIC_LEARNED_MODEL): LearnedExternalIdentity {
  const v = object(value), graph = object(v.graph), weights = object(v.weights), expected = model;
  if (![expected.graph, expected.sharedGraph].some(pin => graph.bytes === pin.bytes && graph.sha256 === pin.sha256) || weights.path !== expected.weights.path ||
      weights.bytes !== expected.weights.bytes || weights.sha256 !== expected.weights.sha256)
    throw new Error("External FP32 model/weights identity differs from the verified export");
  return structuredClone(value) as LearnedExternalIdentity;
}
export type LearnedManifest = { schema: "hsr-learned-browser-v1"; graph: { bytes: number; sha256: string };
  instructions: Record<Instruction, { tokens: number[]; mask: boolean[] }>;
  actionNormalization: { mean: number[]; std: number[] } };
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid learned manifest");
  return value as Record<string, unknown>;
}
function normalization(mean: number[], std: number[]) {
  if (!Array.isArray(mean) || !Array.isArray(std) || mean.length !== 6 || std.length !== 6 ||
    !mean.every(Number.isFinite) || !std.every(v => Number.isFinite(v) && v > 0)) throw new Error("Invalid action normalization");
}
export function validateLearnedManifest(value: unknown, model = PUBLIC_LEARNED_MODEL): LearnedManifest {
  const v = object(value), graph = object(v.graph), instructions = object(v.instructions), n = object(v.actionNormalization);
  if (v.schema !== "hsr-learned-browser-v1" || graph.bytes !== model.manifestGraph.bytes || graph.sha256 !== model.manifestGraph.sha256)
    throw new Error("Learned graph identity differs from the verified learned model");
  const names: Instruction[] = ["Put the mug in the basket.", "Put the tennis ball in the basket."];
  if (Object.keys(instructions).length !== 2 || !names.every(name => name in instructions)) throw new Error("Missing learned instruction tokens");
  for (const name of names) {
    const row = object(instructions[name]);
    if (!Array.isArray(row.tokens) || row.tokens.length !== 48 || !row.tokens.every(v => Number.isSafeInteger(v) && v >= 0) ||
      !Array.isArray(row.mask) || row.mask.length !== 48 || !row.mask.every(v => typeof v === "boolean") || !row.mask.some(Boolean))
      throw new Error("Invalid learned instruction tokens/mask");
  }
  normalization(n.mean as number[], n.std as number[]);
  return structuredClone(value) as LearnedManifest;
}
export function compareLearnedActions(actual: Float32Array, expected: Float32Array, mean: number[], std: number[]) {
  normalization(mean, std);
  if (actual.length !== 96 || expected.length !== 96 || !actual.every(Number.isFinite) || !expected.every(Number.isFinite))
    throw new Error("Expected 96 finite learned actions");
  const physical = { passed: true, maxAbs: 0 }, normalized = { passed: true, maxAbs: 0 };
  for (let i = 0; i < 96; i++) {
    const d = Math.abs(actual[i] - expected[i]);
    physical.maxAbs = Math.max(physical.maxAbs, d); physical.passed &&= d <= 0.001 + 0.001 * Math.abs(expected[i]);
    const k = i % 6, a = Math.fround((actual[i] - mean[k]) / std[k]), e = Math.fround((expected[i] - mean[k]) / std[k]);
    const nd = Math.abs(a - e); normalized.maxAbs = Math.max(normalized.maxAbs, nd);
    normalized.passed &&= nd <= 0.001 + 0.001 * Math.abs(e);
  }
  return { passed: physical.passed && normalized.passed, comparedValues: 96, physical, normalized };
}
export type LearnedReferenceCase = { head_rgb: number[]; hand_rgb: number[]; language_tokens: number[];
  language_mask: boolean[]; hsr_state: number[]; initial_noise: number[]; expected: number[]; action_prefix?: number[]; prefix_mask?: number[] };
