import { useEffect, useRef, useState } from "react";
import wasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import { useProgress } from "@react-three/drei";
import SimulationCanvas from "../playground/SimulationCanvas.tsx";
import { createSimulation, type Simulation } from "../simulation/mujoco.ts";
import { prepareLearningScene } from "./scene.ts";
import { LearnedRollout } from "./learned-rollout.ts";
import { LearnedOnnxPolicy } from "./learned-onnx.ts";
import { learnedTarget } from "./language-goals.ts";
import { learnedModel, validateLearnedManifest, selectLearnedBrowserFiles, verifyLearnedBrowserReferenceFile, PUBLIC_LEARNED_MODEL, type LearnedModelIdentity, type LearnedReferenceCase } from "./learned-onnx-contract.ts";
import type { Instruction } from "./recording.ts";
import { loadHostedLearnedFiles } from "./learned-model-download.ts";
import "./learned-pilot.css";
import { LearnedRtcRollout } from "./learned-rtc-rollout.ts";
import { chooseRtcTiming, continuousExecutionSteps, loadRtcDiagnosticGraph } from "./rtc-runtime.ts";
import type { RtcGraphVariant } from "./rtc-contract.ts";
import { PlanExecution } from "./plan-execution.ts";
import type { LanguageReceipt } from "./language-planner.ts";
import { LanguagePlanPanel } from "./LanguagePlanPanel.tsx";
import { RoomMap } from "../playground/RoomMap.tsx";
import { AppHeader } from "../app/AppHeader.tsx";

function elapsedLabel(seconds: number) {
  const value = Math.floor(seconds); return `${Math.floor(value / 60).toString().padStart(2, "0")}:${(value % 60).toString().padStart(2, "0")}`;
}

// Page defaults (user decision 2026-10-29): prefix-rtc-v9, the model that passed the phase-2 final (60/60 on held-out
// seeds), runs continuously with the settings of that evaluation. URL overrides: ?model=public|<candidate>,
// ?executor=sync, ?rtcCoverage=<ms>, ?gripPause=1, ?rtcHold=0..5, ?success=rim, and the private RTC diagnostics.
function pageQuery() { const search = globalThis.location?.search; return search ? new URLSearchParams(search) : null; }
// Continuous execution of the unchanged public policy: each chunk is sampled from images taken 0.2 s of physics earlier
// while the queued goals keep running, and 12 goals of every chunk run before the next capture. No prefix guidance:
// in browser trials it degraded grasping. Chunks that open or close the hand run to their end and the next plan waits
// for a fresh image with physics held (see docs/rtc-continuous-execution.md). A candidate trained on delay-shifted
// goals places each chunk offset goals after its capture; ?gripPause=1 turns the gripper pause on (off by default).
// ?rtcCoverage=<ms of physics per inference> trades playback speed for inference delay; each chunk then runs at most
// 16 − delaySteps goals so that the goals covering the next inference stay queued.
function continuousOptions(proofs: { inferenceMs: number }[], model: LearnedModelIdentity) {
  const coverage = Number(pageQuery()?.get("rtcCoverage") ?? 400), timing = chooseRtcTiming(proofs, coverage);
  // One request in flight needs more executed goals than the delay, and the delay's goals must stay queued: delay <= 7.
  // ?rtcExec=<goals per chunk> replans sooner (diagnostic); the default is min(12, 16 - delay).
  const executionSteps = continuousExecutionSteps(timing.delaySteps, pageQuery()?.get("rtcExec") ?? null);
  return { ...timing, executionSteps, guidance: false, pauseForGripperChanges: pageQuery()?.get("gripPause") === "1",
    ...(model.chunkOffsetSteps ? { chunkOffsetSteps: model.chunkOffsetSteps } : {}), ...(model.prefixConditioning ? { prefixConditioning: model.prefixConditioning } : {}),
    // ?assist=1: opt-in rule assist (restart from the start pose on a regrasp loop or a 35 s stall), reported separately.
    ...(pageQuery()?.get("assist") === "1" ? { assist: true } : {}),
    // ?rtcHold=<0..5>: late-reply grace, the last goal is kept for up to that many 100 ms boundaries (default 3, 0 = off).
    ...(Number(pageQuery()?.get("rtcHold") ?? 3) ? { lateHoldSteps: Number(pageQuery()?.get("rtcHold") ?? 3) } : {}) };
}
// Mug runs succeed once the mug rests inside the basket's side bars (user decision 2026-10-28); ?success=rim keeps the
// original rim-opening box. The ball keeps its own criterion.
function successOption(instruction: string) { return pageQuery()?.get("success") !== "rim" && learnedTarget(instruction) === "mug" ? { success: "walls" as const } : {}; }
// active=false while the other menu is shown: the canvas (rendering, physics and therefore any run) is frozen, the models stay loaded.
export default function LearnedPilot({ rtcDiagnostic = false, planningDiagnostic = false, active: shown = true }: { rtcDiagnostic?: boolean; planningDiagnostic?: boolean; active?: boolean } = {}) {
  const [sim, setSim] = useState<Simulation | null>(null), [ready, setReady] = useState<string[]>([]);
  const [instruction, setInstruction] = useState<Instruction>("Put the mug in the basket."), [seed, setSeed] = useState("3000");
  const [status, setStatus] = useState("部屋とモデルを準備しています。");
  const [stage, setStage] = useState<"download" | "initialize" | "verify" | "ready" | "error" | "cancelled">("download");
  const [downloadProgress, setDownloadProgress] = useState<{ loaded: number; total: number } | null>(null);
  const [loaded, setLoaded] = useState(false), [busy, setBusy] = useState(false), [running, setRunning] = useState(false), [starting, setStarting] = useState(false);
  const [outcome, setOutcome] = useState("ready"), [metrics, setMetrics] = useState({ wall: 0, simulation: 0, inferences: 0 });
  const head = useRef<HTMLDivElement>(null), hand = useRef<HTMLDivElement>(null);
  const [overviewPane, setOverviewPane] = useState<HTMLDivElement | null>(null), [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  // ?direct=1 keeps the single-task VLA start (no language model) for scripted trials; the RTC diagnostic always uses it.
  const [direct] = useState(() => rtcDiagnostic || pageQuery()?.get("direct") === "1");
  const planningWallMs = useRef<number | null>(null);
  const worker = useRef<LearnedOnnxPolicy | null>(null), rollout = useRef<LearnedRollout | LearnedRtcRollout | null>(null);
  const plan = useRef<PlanExecution | null>(null);
  const [planning, setPlanning] = useState(false);
  const rtcTiming = useRef<{ playbackRate: number; delaySteps: number; executionSteps?: number; guidance?: boolean; pauseForGripperChanges?: boolean; graph?: string }>({ playbackRate: .5, delaySteps: 4 });
  const [continuous, setContinuous] = useState(pageQuery()?.get("executor") !== "sync");
  const continuousTiming = useRef<ReturnType<typeof continuousOptions> | null>(null), continuousTimingError = useRef("");
  const download = useRef<AbortController | null>(null), clock = useRef<{ start: number; end: number | null } | null>(null);
  const parity = useRef<unknown[]>([]), operation = useRef(0), mounted = useRef(true), sceneSeed = useRef(3000);
  const runInfo = useRef<{ instruction: string; seed: number; initialSimulationTime: number; playbackRate: number; model: string; members: number; noise: string; protocol: string; successCriterion: string; rtcSchedule?: object } | null>(null);
  const modelIdentity = useRef<Awaited<LearnedOnnxPolicy["ready"]> | null>(null), selectedModel = useRef<LearnedModelIdentity>(PUBLIC_LEARNED_MODEL);
  const active = useProgress(s => s.active), errors = useProgress(s => s.errors);
  useEffect(() => {
    let owned: Simulation | null = null, cancelled = false; mounted.current = true;
    createSimulation({ locateFile: () => wasmUrl }).then(value => {
      if (cancelled) value.dispose(); else { owned = value; prepareLearningScene(value, 3000); value.advance(2); setSim(value); }
    }).catch(error => { if (!cancelled) { setOutcome("failed"); setStatus(String(error)); } });
    void load();
    const timer = setInterval(() => {
      plan.current?.poll();
      if (plan.current?.rollout && rollout.current !== plan.current.rollout) {
        rollout.current = plan.current.rollout;
        setInstruction(plan.current.tasks[plan.current.current].target === 'mug' ? 'Put the mug in the basket.' : 'Put the tennis ball in the basket.');
      }
      const r = rollout.current; if (!r) return;
      const result = r.result, timing = clock.current;
      if (r.stopped) {
        if (timing && timing.end === null) timing.end = performance.now();
        setRunning(false); if (worker.current?.isDisposed) { setLoaded(false); setStage("error"); }
        setOutcome(r.error || result.status === "failed" ? "failed" : result.status === "success" ? "success" : "stopped");
      }
      setMetrics({ wall: plan.current ? plan.current.elapsedMs / 1000 : timing ? ((timing.end ?? performance.now()) - timing.start) / 1000 : 0, simulation: plan.current ? plan.current.rollouts.reduce((sum, item) => sum + item.result.elapsed, 0) : result.elapsed, inferences: plan.current ? plan.current.rollouts.reduce((sum, item) => sum + item.chunks.length, 0) : r.chunks.length });
      setStatus(r.error ? `実行できませんでした: ${r.error}` : r.stopped ? result.status === "success" ? "かごへの運搬が完了しました。次の指示を実行できます。" : result.status === "failed" ? `運搬を終了しました (${result.reason})。配置はそのまま残っています。` : "停止しました。姿勢と配置を保持しています。" : r.inferencePending ? r instanceof LearnedRtcRollout ? r.pausedForGripper ? "指の開閉の直後のため、物理を止めて次の動作を計算しています。" : "実行しながら、次の動作を先読みしています。" : "次の動作を計算中です。この間は物理時間を一時停止します。" : "カメラの映像から、ロボットを操作しています。");
    }, 100);
    return () => { cancelled = true; mounted.current = false; operation.current++; download.current?.abort(); clearInterval(timer); plan.current?.stop(); rollout.current?.stop(); worker.current?.dispose(); owned?.dispose(); };
  }, []);
  function stop() {
    operation.current++; plan.current?.stop(); rollout.current?.stop();
    if (clock.current && clock.current.end === null) clock.current.end = performance.now();
    if (download.current) {
      download.current.abort(); download.current = null; worker.current?.dispose(); worker.current = null;
      setLoaded(false); setStage("cancelled");
    }
    setDownloadProgress(null); setBusy(false); setRunning(false); setStarting(false); setOutcome("stopped");
    setStatus("停止しました。姿勢と配置を保持しています。");
  }
  async function load() {
    stop(); worker.current?.dispose(); worker.current = null; rollout.current = null; runInfo.current = null; clock.current = null;
    plan.current = null; parity.current = []; modelIdentity.current = null; setMetrics({ wall: 0, simulation: 0, inferences: 0 });
    const generation = ++operation.current; setBusy(true); setLoaded(false); setStage("download"); setOutcome("ready");
    const controller = new AbortController(); download.current = controller;
    try {
      const model = learnedModel(pageQuery()?.get("model") ?? "prefix-rtc-v9"); selectedModel.current = model;
      // ?members=1: speed diagnostic with one sample from the single-member graph instead of the four-member mean.
      const oneMember = pageQuery()?.get("members") === "1";
      // ?noise=zero: diagnostic, every member starts its flow at zero noise (deterministic output).
      const zeroNoise = pageQuery()?.get("noise") === "zero";
      setStatus("モデルを準備しています。初回は約1.6GB、次回から端末の保存データを使います。");
      const files = await loadHostedLearnedFiles(controller.signal, (loaded, total) => {
        if (operation.current !== generation || !mounted.current) return;
        setDownloadProgress({ loaded, total });
      }, model, { singleGraph: oneMember });
      if (operation.current !== generation || !mounted.current) return;
      setDownloadProgress(null); setStage("initialize"); setStatus("モデルを確認し、この端末で起動しています。少し時間がかかります。");
      const [graph, weights, metadata, ...references] = selectLearnedBrowserFiles(files);
      for (const file of [metadata, ...references]) await verifyLearnedBrowserReferenceFile(file, model);
      const manifest = validateLearnedManifest(JSON.parse(await metadata.text()), model);
      if (operation.current !== generation || !mounted.current) return;
      const selectedGraph = rtcDiagnostic ? await loadRtcDiagnosticGraph(controller.signal, (pageQuery()?.get("rtcGraph") ?? "max10") as RtcGraphVariant) : graph;
      if (operation.current !== generation || !mounted.current) return;
      const policy = new LearnedOnnxPolicy(selectedGraph, manifest, weights, { rtc: rtcDiagnostic, modelId: model.id, ...(oneMember ? { members: 1 as const } : {}), ...(zeroNoise ? { noise: "zero" as const } : {}) }); worker.current = policy;
      const identity = await policy.ready;
      if (operation.current !== generation || !mounted.current) { policy.dispose(); return; }
      modelIdentity.current = identity; setStage("verify");
      const proofs: unknown[] = [];
      for (let i = 0; i < 3; i++) {
        if (operation.current !== generation) return;
        setStatus(`モデルの動作を確認しています (${i + 1}/3)。`);
        const reference = JSON.parse(await references[i].text()) as LearnedReferenceCase;
        if (operation.current !== generation || !mounted.current) { policy.dispose(); return; }
        const result = await policy.parity(reference) as { passed: boolean };
        proofs.push(result); if (!result.passed) throw new Error(`ブラウザ数値比較 ${i + 1} が許容差を超えました`);
      }
      if (operation.current !== generation) return;
      if (rtcDiagnostic) {
        // Diagnostic overrides: ?rtcCoverage=<ms of physics covered per inference>&rtcExec=<actions executed per chunk before the next capture>.
        const query = pageQuery(), exec = query?.get("rtcExec");
        rtcTiming.current = { ...chooseRtcTiming(proofs as { inferenceMs: number }[], Number(query?.get("rtcCoverage") ?? 600)), ...(exec ? { executionSteps: Number(exec) } : {}), ...(query?.get("rtcGuide") === "0" ? { guidance: false } : {}), ...(query?.get("rtcGripPause") === "1" ? { pauseForGripperChanges: true } : {}), graph: query?.get("rtcGraph") ?? "max10" };
      } else { try { continuousTiming.current = continuousOptions(proofs as { inferenceMs: number }[], model); continuousTimingError.current = ""; } catch (error) { continuousTiming.current = null; continuousTimingError.current = error instanceof Error ? error.message : String(error); } }
      parity.current = proofs; setLoaded(true); setStage("ready"); setStatus(rtcDiagnostic ? "RTC実行部分の検証モードです。上位計画・画像による段階確認はまだ未接続です。" : "配置を保ったまま、次の指示を実行できます。");
    } catch (error) { if (operation.current === generation) { worker.current?.dispose(); worker.current = null; setStage("error"); setStatus(`読み込めませんでした: ${String(error)}`); } }
    finally { if (operation.current === generation) { download.current = null; setDownloadProgress(null); setBusy(false); } }
  }
  async function start(receipt?: LanguageReceipt, requestedAt = performance.now(), plannedAt = requestedAt) {
    const policy = worker.current;
    if (planningDiagnostic || !sim || !policy || !loaded || policy.isDisposed || running || starting || planning) return;
    const generation = ++operation.current; setStarting(true); setStatus("前の推論が終了するのを待っています。");
    try {
      await policy.whenIdle;
      if (!mounted.current || operation.current !== generation || policy !== worker.current || policy.isDisposed) return;
      const rtcOptions = rtcDiagnostic ? rtcTiming.current : continuous ? continuousTiming.current : null;
      if (!rtcDiagnostic && continuous && !rtcOptions) { setStatus(`連続実行の設定を使えません: ${continuousTimingError.current}`); return; }
      if (!rtcOptions && selectedModel.current.chunkOffsetSteps) { setStatus("この候補モデルは連続実行専用です。設定で「連続実行」を有効にしてください。"); return; }
      plan.current?.stop(); plan.current = null; rollout.current?.stop(); sim.coordinateJointTargets = true;
      const firstInstruction = receipt ? receipt.goal.tasks[0].en : instruction;
      if (learnedTarget(firstInstruction) !== null) setInstruction(firstInstruction as Instruction);
      runInfo.current = { instruction: firstInstruction, seed: sceneSeed.current, initialSimulationTime: sim.data.time, playbackRate: rtcOptions?.playbackRate ?? 1, model: selectedModel.current.id, members: pageQuery()?.get("members") === "1" ? 1 : 4, noise: pageQuery()?.get("noise") === "zero" ? "zero" : "gaussian", protocol: rtcDiagnostic ? "rtc-identity-guided-diagnostic" : rtcOptions ? "continuous-unguided-chunking" : "synchronous-sixteen-actions", successCriterion: !receipt && successOption(instruction).success ? "walls" : "rim", ...(rtcOptions ? { rtcSchedule: { ...rtcOptions } } : {}) };
      if (receipt) {
        planningWallMs.current = plannedAt - requestedAt;
        plan.current = new PlanExecution(sim, policy, receipt, requestedAt, undefined, rtcOptions ? (s, i, p, o) => new LearnedRtcRollout(s, i, p, { ...rtcOptions, ...o, ...successOption(i) }) : (s, i, p, o) => new LearnedRollout(s, i, p, { ...o, ...successOption(i) })); plan.current.start();
        rollout.current = plan.current.rollout;
      } else rollout.current = rtcOptions ? new LearnedRtcRollout(sim, instruction, policy, { ...rtcOptions, disposePolicy: false, ...successOption(instruction) }) : new LearnedRollout(sim, instruction, policy, { disposePolicy: false, ...successOption(instruction) });
      clock.current = { start: receipt ? requestedAt : performance.now(), end: null }; setMetrics({ wall: 0, simulation: 0, inferences: 0 });
      setOutcome("running"); setRunning(true);
    } catch (error) { if (operation.current === generation) { stop(); setOutcome("failed"); setStatus(String(error)); } }
    finally { if (operation.current === generation) setStarting(false); }
  }
  function reset() {
    const value = Number(seed); if (!sim || !Number.isInteger(value) || value < 3000 || value > 3199) return;
    stop(); plan.current = null; rollout.current = null; runInfo.current = null; clock.current = null;
    prepareLearningScene(sim, value); sim.advance(2); sceneSeed.current = value;
    setMetrics({ wall: 0, simulation: 0, inferences: 0 }); setOutcome("ready"); setStatus("配置を初期状態に戻しました。");
  }
  function save() {
    const r = rollout.current, info = runInfo.current; if (!r || !info) return;
    const common = { modelIdentity: modelIdentity.current, browserParity: parity.current, source: "learned-policy-no-fallback",
      seed: info.seed, startedFrom: "current-scene", physics: { fixedTimestep: 0.002, robotMassKg: sim?.robotMassKg, armTorqueNm: sim?.armTorqueNm } };
    const p = plan.current;
    const evidence = p ? { ...common, schema: "hsr-learned-browser-plan-v1", languagePlan: p.receipt, planningWallMs: planningWallMs.current,
      progressSource: "simulation-sensors-not-image-classification", status: p.status, elapsedMs: p.elapsedMs,
      initialSimulationTime: info.initialSimulationTime, tasks: p.tasks,
      taskRollouts: p.rollouts.map((item, index) => ({ target: p.tasks[index].target, label: p.tasks[index].label,
        instruction: p.tasks[index].instruction,
        result: item.result, error: item.error, chunks: item.chunks }))
    } : { ...common, schema: "hsr-learned-browser-rollout-v1", ...info, error: r.error, result: r.result, chunks: r.chunks,
      ...(r instanceof LearnedRtcRollout ? { rtcRequests: r.requests, quality: r.quality, gripperPauses: r.gripperPauses, assistEvents: r.assistEvents, lateHolds: r.lateHolds } : {}) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(evidence)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = `learned-${p ? "plan" : info.instruction.includes("mug") ? "mug" : "ball"}-${info.seed}.json`; a.click(); URL.revokeObjectURL(url);
  }
  // Panes only show the views; the policy's head/hand images come from the separate 320×240 capture.
  const available = Boolean(sim && !active && !errors.length && ready.includes("head") && ready.includes("hand"));
  // Evaluation layouts 3000-3199: outside every data partition of tools/vla-data/seed-registry.json (train 1000-1999,
  // validation 2000-2099); 3100-3199 serves later final evaluations without changing the registry's hash.
  const invalidSeed = !Number.isInteger(Number(seed)) || Number(seed) < 3000 || Number(seed) > 3199;
  const label = busy ? stage === "download" ? "モデルを準備中" : stage === "verify" ? "動作を確認中" : "モデルを起動中" : starting ? "前の推論の終了を待っています" : planning ? "計画を作成中" : stage === "error" || outcome === "failed" ? "実行を終了しました" : running ? rollout.current?.inferencePending && !(rollout.current instanceof LearnedRtcRollout && !rollout.current.pausedForGripper) ? "次の動作を計算中" : "ロボットを操作中" : outcome === "success" ? "運搬完了" : outcome === "stopped" ? "停止中" : planningDiagnostic && loaded ? "計画を確認できます" : available && loaded ? "実行できます" : "部屋を準備中";
  const feed = (view: string) => !ready.includes(view) && <div className="feed-skeleton" aria-hidden="true"/>;
  return <main className="auto-page">
    <AppHeader current="auto"/>
    <div className="workbench"><aside id="sidebar" hidden={sidebarCollapsed}>
      {direct && <section className="direct-run" aria-label="VLA を直接実行">
        <label htmlFor="learned-instruction">{rtcDiagnostic ? "RTC検証 · " : ""}かごへ運ぶもの</label>
        <div className="plan-buttons">
          <select id="learned-instruction" aria-label="指示" disabled={running || starting} value={instruction} onChange={e => setInstruction(e.target.value as Instruction)}>
            <option value="Put the mug in the basket.">コップ</option><option value="Put the tennis ball in the basket.">テニスボール</option>
          </select>
          <button onClick={() => void start()} disabled={planningDiagnostic || planning || !available || !loaded || running || busy || starting}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5 13 8l-9 5.5z" fill="currentColor"/></svg>実行</button>
          {rtcDiagnostic && <button className="secondary" onClick={stop} disabled={!running && !busy && !starting}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3h10v10H3z" fill="currentColor"/></svg>停止</button>}
        </div>
      </section>}
      {!rtcDiagnostic && <LanguagePlanPanel disabled={running || starting || busy} onBusyChange={setPlanning} execution={plan.current}
        canExecute={!planningDiagnostic && available && loaded && !busy && !starting && !running && !planning}
        warm={stage === "ready" || stage === "error" || stage === "cancelled"}
        stopEnabled={running || busy || starting} onStop={stop} onExecute={(receipt, requestedAt, plannedAt) => void start(receipt, requestedAt, plannedAt)}/>}
      <section className="learned-status-rail" aria-label="実行状況" data-outcome={outcome} data-loading={busy || starting}>
        <div className="learned-status-main"><span className="learned-state-dot" aria-hidden="true"/><div>
          <output aria-live="polite" data-provider={modelIdentity.current?.provider} data-threads={modelIdentity.current?.numThreads} data-sampling={modelIdentity.current?.sampling}>{label}</output>
          <p>{planningDiagnostic && loaded ? "画像による達成判定を検証中です。動作はまだ開始できません。" : status}</p>
          {busy && <div className="learned-load-progress"><progress aria-label="モデルの読み込み" max={downloadProgress?.total ?? 1} value={downloadProgress?.loaded}/>
            {downloadProgress && <span>{Math.round(downloadProgress.loaded / downloadProgress.total * 100)}% · {Math.round(downloadProgress.loaded / 1048576)} / {Math.round(downloadProgress.total / 1048576)} MB</span>}</div>}
          {stage === "error" || stage === "cancelled" ? <button className="learned-link" onClick={() => void load()} disabled={busy}>読み込みを再開</button> : null}
        </div></div>
        <dl className="learned-run-metrics">{(rtcDiagnostic || continuous) && <div><dt>再生速度</dt><dd>{(rtcDiagnostic ? rtcTiming.current.playbackRate : continuousTiming.current?.playbackRate ?? 1).toFixed(2)}×</dd></div>}<div><dt>経過</dt><dd>{elapsedLabel(metrics.wall)}</dd></div><div><dt>シミュレーション</dt><dd>{metrics.simulation.toFixed(1)}秒</dd></div><div><dt>推論</dt><dd>{metrics.inferences}回</dd></div></dl>
      </section>
      <div className="auto-utilities">
        <button className="secondary" onClick={reset} disabled={!available || busy || invalidSeed || planning || starting}>配置を戻す</button>
        <details className="learned-settings"><summary>設定<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5"/></svg></summary>
          <div className="learned-settings-body">
            <label>配置のSeed<input aria-label="Seed" type="number" min="3000" max="3199" value={seed} onChange={e => setSeed(e.target.value)}/></label>
            <p>{invalidSeed ? "3000〜3199を指定してください。" : "「配置を戻す」を押すと反映します。"}</p>
            {!rtcDiagnostic && <><hr/><label className="learned-check"><input type="checkbox" checked={continuous} disabled={running || starting || busy || planning} onChange={e => setContinuous(e.target.checked)}/>連続実行（推論中も動き続ける）</label>
            <p>{continuous ? `次の動作を計算する間も、前の計画の続きを実行します。${pageQuery()?.get("gripPause") === "1" ? "指を開閉する動作の後だけ止まって次を計算します。" : "指の開閉でも止まらずに動き続けます。"}物理時間は約${(continuousTiming.current?.playbackRate ?? 0).toFixed(2)}倍速で進みます。` : "次の動作を計算する間は、物理時間を一時停止します。"}</p></>}
            <hr/><div className="settings-buttons"><button className="secondary" onClick={() => void load()} disabled={busy || running || starting || planning}>モデルを再読み込み</button>
            <button className="secondary" onClick={save} disabled={running || starting || busy || !runInfo.current}>検証結果を保存</button></div>
            <p>画像と関節状態は、このブラウザ内で処理します。対応タスクは机上のコップ・ボール運搬です。</p>
            <small>数値比較 {parity.current.length}/3 · {modelIdentity.current?.provider ?? "準備中"}</small>
          </div>
        </details>
      </div>
      <div className="asset-credits"><a href="/assets/hsr-original-v1/ATTRIBUTION.html" target="_blank" rel="noreferrer">HSR: Toyota / CC BY-NC-ND 4.0（非商用）</a><br/>
        <a href="https://polyhaven.com" target="_blank" rel="noreferrer">3D assets / materials: Poly Haven</a></div>
    </aside><section className="viewport" aria-label="MuJoCo robot simulation">
      <button type="button" className="panel-label sidebar-toggle" aria-controls="sidebar" aria-expanded={!sidebarCollapsed}
        aria-label={sidebarCollapsed ? "サイドメニューを開く" : "サイドメニューを折りたたむ"} title={sidebarCollapsed ? "サイドメニューを開く" : "サイドメニューを折りたたむ"}
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/><path d={sidebarCollapsed ? "m14 9 3 3-3 3" : "m17 9-3 3 3 3"}/>
        </svg>
      </button>
      <SimulationCanvas sim={sim} active={shown} paused={() => !running || !rollout.current || rollout.current.stopped} mobile={false} controlView="head" headPane={head} handPane={hand} overviewPane={overviewPane}
        onReady={view => setReady(previous => previous.includes(view) ? previous : [...previous, view])}/>
      <div className="view-grid">
        <section className="camera-panel" data-view="head" aria-label="頭カメラの映像" aria-busy={!ready.includes("head")}><span className="panel-label">頭カメラ</span><div className="camera-feed" ref={head}>{feed("head")}</div></section>
        <section className="camera-panel" data-view="hand" aria-label="手先カメラの映像" aria-busy={!ready.includes("hand")}><span className="panel-label">手先カメラ</span><div className="camera-feed" ref={hand}>{feed("hand")}</div></section>
        <section className="camera-panel map-panel" data-view="map" aria-label="部屋の Map"><span className="panel-label">Map</span>
          <div className="camera-feed"><RoomMap observation={sim?.observation() ?? null} headPan={sim?.q("pan") ?? 0} doorAngle={sim?.q("fridge_hinge") ?? 0}/></div></section>
        <section className="camera-panel" data-view="overview" aria-label="俯瞰の映像" aria-busy={!ready.includes("overview")}><span className="panel-label" title="ドラッグで回転 · スクロールで拡大">俯瞰</span>
          <div className="camera-feed overview-feed" ref={setOverviewPane}>{feed("overview")}</div></section>
      </div>
    </section></div>
  </main>;
}
