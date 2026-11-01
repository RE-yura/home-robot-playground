import { useCallback, useEffect, useRef, useState } from "react";
import { useProgress } from "@react-three/drei";
import wasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import SimulationCanvas from "../playground/SimulationCanvas.tsx";
import { createSimulation, type Simulation } from "../simulation/mujoco.ts";
import type { CameraImage } from "../simulation/types.ts";
import type { LearningCaptureGate } from "./capture.ts";
import type { RecognitionResult } from "./recognition-detector.ts";
import { prepareLearningScene } from "./scene.ts";
import "./recognition-preview.css";

const labels = { mug: "コップ", ball: "ボール", basket: "かご" };
const colors = { mug: "#ca5038", ball: "#267bc5", basket: "#8a49b5" };
export default function RecognitionPreview() {
  const [sim, setSim] = useState<Simulation | null>(null), [views, setViews] = useState<string[]>([]);
  const [ready, setReady] = useState(false), [busy, setBusy] = useState(false), [reload, setReload] = useState(0);
  const [status, setStatus] = useState("認識モデルを読み込んでいます。"), [photo, setPhoto] = useState<string | null>(null);
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const head = useRef<HTMLDivElement>(null), hand = useRef<HTMLDivElement>(null), worker = useRef<Worker | null>(null);
  const captureCancel = useRef<(() => void) | null>(null), generation = useRef(0), timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stopCapture = useCallback(() => captureCancel.current?.(), []);
  const closeWorker = useCallback((current: Worker) => {
    generation.current++; captureCancel.current?.(); current.onmessage = null; current.onerror = null; current.terminate();
    if (worker.current === current) worker.current = null;
  }, []);
  const assetsLoading = useProgress(s => s.active), assetErrors = useProgress(s => s.errors);
  useEffect(() => {
    let owned: Simulation | null = null, cancelled = false;
    createSimulation({ locateFile: () => wasmUrl }).then(value => {
      if (cancelled) value.dispose();
      else { owned = value; prepareLearningScene(value, 3000); value.advance(2); setSim(value); }
    }).catch(error => { if (!cancelled) setStatus(`部屋を準備できませんでした: ${String(error)}`); });
    return () => { cancelled = true; stopCapture(); owned?.dispose(); };
  }, [stopCapture]);
  useEffect(() => {
    const current = new Worker(new URL("./recognition-worker.ts", import.meta.url), { type: "module" });
    worker.current = current;
    const clear = () => { if (timer.current) clearTimeout(timer.current); timer.current = null; };
    const fail = (message: string) => {
      if (worker.current !== current) return;
      clear(); setReady(false); setBusy(false); setStatus(message); closeWorker(current);
    };
    current.onmessage = event => {
      if (worker.current !== current) return;
      clear();
      if (event.data.type === "ready") { setReady(true); setStatus("準備できました。頭カメラの1枚を認識できます。"); }
      else if (event.data.type === "result") { setResult(event.data.value as RecognitionResult); setBusy(false); setStatus("撮影した頭カメラ画像の認識が完了しました。"); }
      else if (event.data.type === "error") fail(String(event.data.error));
    };
    current.onerror = event => fail(`認識を開始できませんでした: ${event.message}`);
    timer.current = setTimeout(() => fail("モデルの読み込みが時間内に完了しませんでした。"), 120000);
    current.postMessage({ type: "load" });
    return () => { clear(); closeWorker(current); };
  }, [reload, closeWorker]);
  function cancel() {
    const current = worker.current;
    if (current) closeWorker(current);
    else { generation.current++; stopCapture(); }
    if (timer.current) clearTimeout(timer.current); timer.current = null;
    setBusy(false); setReady(false); setStatus("認識を停止しました。再読み込みすると試せます。");
  }
  function restart() {
    cancel(); setResult(null); setStatus("認識モデルを読み込んでいます。初回は少し時間がかかります。"); setReload(n => n + 1);
  }
  async function snapshot() {
    if (!sim || !worker.current || !ready || busy || sim.learningCapture) return;
    const operation = generation.current; setBusy(true); setResult(null); setStatus("頭カメラの新しい画像を認識しています。");
    try {
      const image = await new Promise<CameraImage>((resolve, reject) => {
        let ended = false;
        const stop = () => {
          if (ended) return; ended = true; clearTimeout(timeout);
          if (sim.learningCapture === gate) sim.learningCapture = null;
          captureCancel.current = null; reject(new Error("撮影を停止しました。"));
        };
        const gate: LearningCaptureGate = {
          needsImages: true, captureDue: () => !ended,
          capture(images) {
            const frame = images.find(i => i.view === "head");
            if (!frame) { stop(); return; }
            ended = true; clearTimeout(timeout); if (sim.learningCapture === gate) sim.learningCapture = null;
            captureCancel.current = null; resolve(frame);
          }, stop,
        };
        const timeout = setTimeout(stop, 5000); captureCancel.current = stop; sim.learningCapture = gate;
      });
      if (generation.current !== operation || !worker.current) return;
      setPhoto(image.dataUrl);
      timer.current = setTimeout(() => { cancel(); setStatus("認識が時間内に完了しませんでした。再読み込みして試してください。"); }, 60000);
      worker.current.postMessage({ type: "snapshot", dataUrl: image.dataUrl });
    } catch (error) { if (generation.current === operation) { setBusy(false); setStatus(String(error)); } }
  }
  const available = Boolean(sim && views.length === 2 && !assetsLoading && !assetErrors.length);
  return <main className="recognition-console">
    <header className="recognition-toolbar">
      <div><strong>Home Robot</strong><span className="recognition-badge">物体認識・実験</span><p>T7 · 28,800更新 · WASM / CPU 1スレッド</p></div>
      <nav><button onClick={() => void snapshot()} disabled={!available || !ready || busy}>{busy ? "認識中…" : "いまの画像を認識"}</button>
        {busy && <button className="secondary" onClick={cancel}>キャンセル</button>}
        {!ready && !busy && <button className="secondary" onClick={restart}>モデルを再読み込み</button>}
        <a href="?mode=learned">自律操作へ</a><a href="/">手動操作へ</a></nav>
    </header>
    <div className="recognition-layout">
      <section className="viewport recognition-live" aria-label="現在のカメラ映像">
        <SimulationCanvas sim={sim} paused={true} mobile={false} controlView="head" headPane={head} handPane={hand} overviewPane={null}
          onReady={view => setViews(previous => previous.includes(view) ? previous : [...previous, view])}/>
        <div className="recognition-cameras"><div ref={head}><span className="panel-label">頭カメラ · 認識に使用</span></div>
          <div ref={hand}><span className="panel-label">手先カメラ · 表示のみ</span></div></div>
      </section>
      <section className="recognition-snapshot" aria-label="撮影画像の認識結果">
        <h1>画像から見つけたもの</h1>
        {photo ? <svg viewBox="0 0 320 240" role="img" aria-label="頭カメラの撮影画像と推定位置">
          <image href={photo} width="320" height="240"/>
          {result?.detections.map(d => <g key={d.sourceIndex} fill={colors[d.label]}>
            <rect x={d.box[0]} y={d.box[1]} width={d.box[2] - d.box[0]} height={d.box[3] - d.box[1]} fill="none" stroke={colors[d.label]} strokeWidth="1.5"/>
            <text x={Math.min(245, d.box[0])} y={Math.max(11, d.box[1] - 3)} stroke="#fff" strokeWidth="2.5" paintOrder="stroke" fontSize="10" fontWeight="600">{labels[d.label]} {Math.round(d.score * 100)}%</text>
          </g>)}
        </svg> : <div className="recognition-placeholder">「いまの画像を認識」で、コップ・ボール・かごの推定位置を表示します。</div>}
        {result && <><p>{result.detections.length ? `${result.detections.length}件の候補` : "この画像では候補が見つかりませんでした。"}</p>
          <dl className="recognition-timing"><div><dt>画像全体</dt><dd>{result.elapsedMs.toFixed(0)} ms</dd></div>
            {result.inferenceMs.map((ms, i) => <div key={i}><dt>{i === 0 ? "全体推論" : `部分推論 ${i}`}</dt><dd>{ms.toFixed(1)} ms</dd></div>)}</dl></>}
      </section>
    </div>
    <footer className="recognition-status"><output aria-live="polite">{status}</output>
      <p>実験用の物体検出です。見落としや誤検出があります。ロボットは動かず、把持・運搬の成功を判定するものではありません。</p></footer>
  </main>;
}
