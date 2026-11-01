import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { useProgress } from "@react-three/drei";
import SimulationCanvas from "../../src/playground/SimulationCanvas.tsx";
import { createSimulation, type Simulation } from "../../src/simulation/mujoco.ts";
import { LearningCapture } from "../../src/learning/capture.ts";
import type { Instruction, DemoCollection } from "../../src/learning/recording.ts";
import { validateDemoCollection } from "../../src/learning/recording.ts";
import type { ArmJointTarget } from "../../src/simulation/types.ts";
import { prepareLearningScene, LEARNING_SCENE_VERSION } from "../../src/learning/scene.ts";
import { prepareRecoveryScene, RECOVERY_SCENE_VERSION, prepareBallRecoveryScene, prepareBallTransportScene, BALL_RECOVERY_SCENE_VERSION } from "../../src/learning/recovery-scene.ts";
import {prepareEmptyCarryBallRecovery,EMPTY_CARRY_BALL_RECOVERY_SCENE_VERSION} from "../../src/learning/empty-carry-ball-recovery.ts";
import { preparePhysicalBallRecovery, PHYSICAL_BALL_RECOVERY_SCENE_VERSION } from "../../src/learning/physical-ball-recovery.ts";
import { StationaryTeacher } from "../../src/learning/teacher.ts";
import { DartNoise, DART_VERSION, DART_VERSIONS, dartActive, perturbGoal } from "../../src/learning/dart.ts";
import {ReseatBallTeacher} from "../../src/learning/reseat-ball-teacher.ts";
import {prepareShallowBallRecovery,SHALLOW_BALL_RECOVERY_SCENE_VERSION} from "../../src/learning/shallow-ball-recovery.ts";
import { TeacherQualityAudit } from "../../src/learning/quality.ts";
import { ObjectPreservationAudit } from "../../src/learning/preservation.ts";
import { collectRecovery } from "./local-recovery-collection.ts";
import {collectionPaused,type CollectionTeacher} from "./collection-interval.ts";
import "../../src/index.css";

type DemoRequest = {instruction:Instruction;seed:number;collection:DemoCollection};
declare global { interface Window { dataPilot: { capture: LearningCapture | null; sim: Simulation | null;
  startDemonstration: ((request:DemoRequest)=>void) | null;
  teacher: CollectionTeacher | null; audit: Record<string, unknown> | null } } }
window.dataPilot = { capture: null, sim: null, teacher: null, audit: null, startDemonstration:null };
function Pilot() {
  const [sim, setSim] = useState<Simulation | null>(null), [instruction, setInstruction] = useState<Instruction>("Put the mug in the basket.");
  const [ready, setReady] = useState<string[]>([]), [status, setStatus] = useState("準備中");
  const [seed, setSeed] = useState("42");
  const [recovery, setRecovery] = useState(false);
  const [collecting,setCollecting]=useState(false),[collectionStatus,setCollectionStatus]=useState('');
  const head = useRef<HTMLDivElement>(null), hand = useRef<HTMLDivElement>(null);
  const quality = useRef<TeacherQualityAudit | null>(null);
  const active = useProgress(s => s.active), errors = useProgress(s => s.errors);
  useEffect(() => {
    let cancelled = false, owned: Simulation | null = null;
    createSimulation().then(value => { if (cancelled) value.dispose(); else { owned = value; window.dataPilot.sim = value; setSim(value); } });
    const timer = setInterval(() => {
      const { capture: c, teacher, sim: current } = window.dataPilot;
      if (teacher && c) {
        const result = teacher.result;
        if (!c.stopped && result.status !== "running") {
          c.stop();
          window.dataPilot.audit = { ...window.dataPilot.audit, result, final: current!.observation() };
        }
        setStatus(c.error ? `失敗: ${c.error}` : result.status === "success" ? `成功 · ${result.elapsed.toFixed(1)}秒` :
          result.status === "failed" ? `失敗: ${result.teacherFailure ?? result.reason}` : c.stopped ? "停止" : `${result.phase} · ${result.elapsed.toFixed(1)}秒`);
        return;
      }
      setStatus(c ? c.error ? `失敗: ${c.error}` : c.stopped ? `完了 · ${c.episode!.samples.length}組 · ${c.episode!.duration.toFixed(1)}秒` : "収録中" : "準備完了");
    }, 100);
    return () => { cancelled = true; clearInterval(timer); quality.current?.dispose(); quality.current=null; owned?.dispose(); };
  }, []);
  function start() {
    if (!sim) return;
    window.dataPilot.teacher = null; window.dataPilot.audit = null;
    const initial = [...sim.observation().robot!.joints] as ArmJointTarget;
    window.dataPilot.capture = new LearningCapture(sim, { instruction, maxSamples: 30,
      actionForSample: index => ({ armJointTarget: [initial[0] + 0.025 * Math.sin(index * Math.PI / 15), ...initial.slice(1)] as ArmJointTarget,
        gripper: index < 15 ? 0.65 : 1 }) });
    setStatus("収録中");
  }
  function transport(request?:DemoRequest) {
    if (!sim) return;
    try {
      if (window.dataPilot.capture && !window.dataPilot.capture.stopped) throw new Error('A capture is already running');
      const selected=request?.instruction??instruction;
      if(request)validateDemoCollection(request.collection,request.seed);
      if(request && ![LEARNING_SCENE_VERSION,RECOVERY_SCENE_VERSION,BALL_RECOVERY_SCENE_VERSION,PHYSICAL_BALL_RECOVERY_SCENE_VERSION,EMPTY_CARRY_BALL_RECOVERY_SCENE_VERSION,SHALLOW_BALL_RECOVERY_SCENE_VERSION].some(version=>request.collection.sceneVersion===version))throw new Error('Collection scene version does not match this saved source');
      if(selected!=="Put the mug in the basket." && selected!=="Put the tennis ball in the basket.")throw new Error('Unsupported instruction');
      const shallowRecovery=request?.collection.sceneVersion===SHALLOW_BALL_RECOVERY_SCENE_VERSION;
      const ballTeacher=request?['hsr-ball-teacher-v1','hsr-ball-reseat-v1'].includes(request.collection.teacherVersion):selected==='Put the tennis ball in the basket.';
      if(ballTeacher&&selected!=='Put the tennis ball in the basket.')throw new Error('Ball teacher requires the ball instruction');
      if(request && (sim.armTorqueNm!==40 || Math.abs(sim.robotMassKg-33.458654)>1e-9))throw new Error('Demonstration physics configuration changed');
      const recoveryStart=request?request.collection.sceneVersion===RECOVERY_SCENE_VERSION:recovery;
      const emptyCarryRecovery=request?.collection.sceneVersion===EMPTY_CARRY_BALL_RECOVERY_SCENE_VERSION;
      const physicalRecovery=request?.collection.sceneVersion===PHYSICAL_BALL_RECOVERY_SCENE_VERSION;
      const ballRecovery=emptyCarryRecovery||physicalRecovery||(request?request.collection.sceneVersion===BALL_RECOVERY_SCENE_VERSION:recovery&&ballTeacher);
      const shallowScene=shallowRecovery?prepareShallowBallRecovery(sim,request!.seed):null;
      const scene = shallowScene??(emptyCarryRecovery?prepareEmptyCarryBallRecovery(sim,request!.seed):physicalRecovery?preparePhysicalBallRecovery(sim,request!.seed):ballRecovery?prepareBallRecoveryScene(sim,request?.seed??Number(seed)):recoveryStart?prepareRecoveryScene(sim,request?.seed??Number(seed)):ballTeacher?prepareBallTransportScene(sim,request?.seed??Number(seed)):prepareLearningScene(sim,request?.seed??Number(seed)));
      if(!recoveryStart&&!ballRecovery&&!ballTeacher)sim.advance(2);
      sim.mj.mj_kinematics(sim.model, sim.data);
      const teacher = shallowScene?new ReseatBallTeacher(sim,shallowScene.preparation.pose.attitude):new StationaryTeacher(sim, selected,ballTeacher?ballRecovery?'ball-recovery':'stable-ball':request&&['hsr-mug-gentle-v1',DART_VERSION].includes(request.collection.teacherVersion)?'gentle-mug':request&&['hsr-mug-gentle-v2','hsr-mug-gentle-dart-v2'].includes(request.collection.teacherVersion)?'gentle-mug-v2':request&&['hsr-mug-gentle-v3','hsr-mug-gentle-dart-v3'].includes(request.collection.teacherVersion)?'gentle-mug-v3':request&&['hsr-mug-gentle-v4','hsr-mug-gentle-dart-v4'].includes(request.collection.teacherVersion)?'gentle-mug-v4':'legacy');
      // DART: noise-injected execution with clean labels (src/learning/dart.ts).
      const dart=request&&DART_VERSIONS.includes(request.collection.teacherVersion)?new DartNoise(request.seed):null;
      quality.current?.dispose(); quality.current = new TeacherQualityAudit(sim, selected.includes('mug')?'mug':'ball');
      const preservation=request?new ObjectPreservationAudit(sim,selected.includes('mug')?'mug':'ball'):null;
      window.dataPilot.teacher = teacher;
      window.dataPilot.audit = { source: request?"oracle-teacher-demonstration":"oracle-teacher-verification", scene, instruction:selected, initial: sim.observation(), quality:quality.current.metrics,
        ...(request?{collection:request.collection,preservation:preservation!.metrics}:{}) };
      window.dataPilot.capture = new LearningCapture(sim, { instruction:selected, maxSamples: 1210, collection:request?.collection,
        taskEvaluator: teacher.evaluator,
        ...(dart?{perturb:(goal:{armJointTarget:ArmJointTarget;gripper:number},index:number)=>perturbGoal(goal,dart.next(dartActive(index)))}:{}),
        actionForSample: () => {
          const observed = sim.observation(), goal = teacher.next(observed);
          window.dataPilot.audit = { ...window.dataPilot.audit, evaluated: observed, result: teacher.result };
          return goal;
        } });
      const capture=window.dataPilot.capture,after=capture.afterStep.bind(capture),audit=quality.current;
      capture.afterStep=dt=>{after(dt);if(teacher instanceof ReseatBallTeacher)teacher.afterStep(sim.observation(),dt);audit.update(dt,teacher.result);preservation?.update(sim,dt);};
      setStatus("教師運搬中");
    } catch (error) { setStatus(`失敗: ${String(error)}`); if(request)throw error; }
  }
  window.dataPilot.startDemonstration=request=>{if(!sim)throw new Error('Simulator unavailable');transport(request);};
  function download() {
    const e = window.dataPilot.capture?.episode;
    if (!e || e.error) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(e)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = `hsr-${e.kind}-${e.instruction.includes("mug") ? "mug" : "ball"}.json`; a.click(); URL.revokeObjectURL(url);
  }
  const running = collecting||Boolean(window.dataPilot.capture && !window.dataPilot.capture.stopped);
  const validSeed = seed.trim() !== "" && Number.isInteger(Number(seed)) && Number(seed) >= 0 && Number(seed) <= 0xffffffff;
  return <main style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
    <div style={{ padding: 12, display: "flex", alignItems: "center", gap: 12 }}>
      <strong>HSR 10 Hz 収録検証</strong>
      <select aria-label="指示" value={instruction} onChange={e => setInstruction(e.target.value as Instruction)} disabled={running}>
        <option>Put the mug in the basket.</option><option>Put the tennis ball in the basket.</option>
      </select>
      <label>Seed <input aria-label="Seed" type="number" min={0} max={0xffffffff} step={1} value={seed} onChange={e => setSeed(e.target.value)} disabled={running} style={{ width: 72 }}/></label>
      <label><input type="checkbox" checked={recovery} onChange={e=>setRecovery(e.target.checked)} disabled={running}/>つかみ直し</label>
      <button onClick={start} disabled={!sim || active || errors.length > 0 || ready.length !== 2 || running}>収録開始</button>
      <button onClick={()=>transport()} disabled={!validSeed || !sim || active || errors.length > 0 || ready.length !== 2 || running}>教師運搬（検証）</button>
      <button onClick={() => window.dataPilot.capture?.stop()} disabled={!running}>停止</button>
      <button onClick={download} disabled={!window.dataPilot.capture?.episode || Boolean(window.dataPilot.capture?.error)}>保存</button>
      {new URLSearchParams(location.search).has('recoveryCollection')&&<button disabled={!sim||active||errors.length>0||ready.length!==2||running} onClick={async()=>{setCollecting(true);try{await collectRecovery(request=>transport(request),()=>window.dataPilot,setCollectionStatus);}catch(error){setCollectionStatus(`収録保存失敗: ${String(error)}`);}finally{setCollecting(false);}}}>つかみ直しデータ収録</button>}
      <output>{status}</output>
      {collectionStatus&&<output>{collectionStatus}</output>}
    </div>
    <p style={{ margin: "0 12px 12px" }}>同期検証は3秒。教師運搬は物体の位置を使う状態ベース制御です。保存データは検証用で、学習済みVLAの実行ではありません。</p>
    <div className="viewport" style={{ minHeight: 300 }}>
      <SimulationCanvas sim={sim} paused={()=>collectionPaused(window.dataPilot.capture,window.dataPilot.teacher)} mobile={false} controlView="head" headPane={head} handPane={hand} overviewPane={null}
        onReady={view => setReady(previous => previous.includes(view) ? previous : [...previous, view])}/>
      <div style={{ position: "absolute", inset: 0, display: "grid", gridTemplateColumns: "1fr 1fr", pointerEvents: "none" }}>
        <div ref={head} style={{ position: "relative" }}><span className="panel-label">頭カメラ</span></div><div ref={hand} style={{ position: "relative" }}><span className="panel-label">手先カメラ</span></div>
      </div>
    </div>
  </main>;
}
createRoot(document.getElementById("root")!).render(<Pilot/>);
