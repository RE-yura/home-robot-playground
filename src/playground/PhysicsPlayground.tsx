/* eslint-disable react/immutability -- MuJoCo handles and Three.js cameras are mutable external simulation objects. */
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import wasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import type { Simulation } from "../simulation/mujoco.ts";
import { CONTROL_KEYS } from "../simulation/input.ts";
import { TABLE_HEIGHT } from "../simulation/table.ts";
import { RoomMap } from "./RoomMap.tsx";
import { TouchControls } from "./TouchControls.tsx";
import { AppHeader } from "../app/AppHeader.tsx";

const SimulationCanvas = lazy(() => import("./SimulationCanvas.tsx"));
const CAMERA_LABELS = { head: "頭カメラ", hand: "手先カメラ" };
type VideoView = "head" | "hand" | "overview";

function FeedSkeleton() {
  return <div className="feed-skeleton" aria-hidden="true"/>;
}

// active=false while 自動実行 is shown: keys are released and the canvas (rendering and physics) is frozen.
export function PhysicsPlayground({ active = true }: { active?: boolean } = {}) {
  const [started, setStarted] = useState(false);
  const [sim, setSim] = useState<Simulation | null>(null), [error, setError] = useState("");
  const [paused, setPaused] = useState(false), [controlView, setControlView] = useState<"head" | "hand">("head");
  const [mobile, setMobile] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [mobileView, setMobileView] = useState<"head" | "hand" | "map" | "overview">("head");
  const [inputGeneration, setInputGeneration] = useState(0);
  const [seedInput, setSeedInput] = useState("42"), [seedError, setSeedError] = useState("");
  const [overviewPane, setOverviewPane] = useState<HTMLDivElement | null>(null);
  const [readyViews, setReadyViews] = useState<Record<VideoView, boolean>>({ head: false, hand: false, overview: false });
  const viewReady = useCallback((view: VideoView) => setReadyViews(ready => ready[view] ? ready : { ...ready, [view]: true }), []);
  const [, refresh] = useState(0), headPane = useRef<HTMLDivElement>(null), handPane = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const query = window.matchMedia("(max-width: 960px)"), change = () => setMobile(query.matches);
    change(); query.addEventListener("change", change); return () => query.removeEventListener("change", change);
  }, []);
  useEffect(() => {
    let alive = true, loaded: Simulation | undefined;
    setStarted(true);
    import("../simulation/mujoco.ts").then(({ createSimulation }) => createSimulation({ locateFile: () => wasmUrl })).then(s => {
      loaded = s; if (alive) setSim(s); else s.dispose();
    }).catch(e => { if (alive) setError(String(e)); });
    return () => { alive = false; loaded?.dispose(); };
  }, []);
  useEffect(() => {
    if (!sim || !active) return;
    const accepted = new Set(CONTROL_KEYS);
    const down = (e: KeyboardEvent) => {
      if (!accepted.has(e.code) || e.metaKey || e.ctrlKey || e.altKey || e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement ||
        (e.target instanceof HTMLElement && e.target.isContentEditable)) return;
      e.preventDefault(); if (paused) return;
      sim.setKey(e.code, true);
    };
    const up = (e: KeyboardEvent) => { if (accepted.has(e.code)) sim.setKey(e.code, false); };
    const clear = () => { for (const k of Array.from(sim.keys)) sim.setKey(k, false); };
    window.addEventListener("keydown", down); window.addEventListener("keyup", up); window.addEventListener("blur", clear);
    const timer = window.setInterval(() => refresh(n => n + 1), 200);
    return () => { clearInterval(timer); clear(); window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); window.removeEventListener("blur", clear); };
  }, [sim, paused, active]);
  const reset = (seed = sim?.seed ?? 42) => {
    sim?.reset(seed); setSeedInput(String(seed)); setSeedError(""); refresh(n => n + 1);
    setInputGeneration(n => n + 1);
  };
  const applySeed = () => {
    const seed = Number(seedInput);
    if (!seedInput.trim() || !Number.isInteger(seed) || seed < 0 || seed > 4294967295) {
      setSeedError("0〜4294967295 の整数を入力してください。"); return;
    }
    reset(seed);
  };
  const mug = sim?.position(sim.mugId), base = sim?.position(sim.baseId);
  const status = !sim ? "準備中" : paused ? "一時停止" : sim.gripperContactId ? "物体を把持中" : "操作できます";
  return <main>
    <AppHeader current="manual"/>
    <div className="mobile-toolbar">
      <div className="mobile-actions">
        <span role="status">{status}</span>
        <button type="button" className="secondary" disabled={!sim} onClick={() => setPaused(!paused)}>{paused ? "再開" : "一時停止"}</button>
        <button type="button" className="secondary" disabled={!sim} onClick={() => reset()}>リセット</button>
        <button type="button" className="secondary" disabled={!sim} aria-expanded={mobileMenu} aria-controls="sidebar" onClick={() => setMobileMenu(!mobileMenu)}>{mobileMenu ? "閉じる" : "設定"}</button>
      </div>
      <nav className="mobile-views" aria-label="表示する映像">
        {(["head", "hand", "map", "overview"] as const).map(view => <button key={view} type="button" aria-pressed={mobileView === view}
          onClick={() => { setMobileView(view); if (view === "head" || view === "hand") setControlView(view); }}>
          {{ head: "頭カメラ", hand: "手先カメラ", map: "Map", overview: "俯瞰" }[view]}
        </button>)}
      </nav>
    </div>
    <div className="workbench"><aside id="sidebar" hidden={mobile ? !mobileMenu : sidebarCollapsed} data-mobile-menu={mobile && mobileMenu}>
      <section className="world-controls" aria-label="World controls">
        <form onSubmit={e => { e.preventDefault(); applySeed(); }}>
          <label htmlFor="seed">Seed</label>
          <input id="seed" type="number" min="0" max="4294967295" step="1" value={seedInput}
            disabled={!sim} aria-invalid={Boolean(seedError)} aria-describedby={seedError ? "seed-help" : undefined}
            title="整数を入力して Enter で適用"
            onChange={e => setSeedInput(e.target.value)}/>
          <button type="button" disabled={!sim} onClick={() => reset(((sim?.seed ?? 42) + 1) >>> 0)}>New world</button>
          {seedError && <p id="seed-help" role="alert">{seedError}</p>}
        </form>
        <div className="run-controls">
          <span role="status">{status}</span>
          <button type="button" className="secondary" disabled={!sim} onClick={() => setPaused(!paused)}>{paused ? "再開" : "一時停止"}</button>
          <button type="button" className="secondary" disabled={!sim} onClick={() => reset()}>リセット</button>
        </div>
      </section>
      <details className="sidebar-section" open><summary>操作キー</summary><dl className="keys">
        <dt>台車</dt><dd><kbd>W S</kbd> 前後 / <kbd>A D</kbd> 横<br/><kbd>Q E</kbd> 旋回</dd>
        <dt>手先移動</dt><dd><kbd>I K</kbd> 画面の上下 / <kbd>J L</kbd> 左右<br/><kbd>U O</kbd> 奥へ / 手前へ<br/>向きを保つ · 台車も補助</dd>
        <dt>手先姿勢</dt><dd><kbd>Shift + I K</kbd> 上下に傾ける<br/><kbd>Shift + J L</kbd> 左右に向ける<br/><kbd>Shift + U O</kbd> 軸まわりに回す<br/>位置を保つ · 台車も補助</dd>
        <dt>グリッパー</dt><dd><kbd>Space</kbd> 開閉</dd>
        <dt>頭カメラ</dt><dd><kbd>↑ ↓ ← →</kbd> 上下左右</dd>
      </dl></details>
      <section className="parameters sidebar-section" aria-label="グリッパーの姿勢">
        <div className="parameter-heading"><h2>手首の向き</h2>
          <button type="button" className="secondary" disabled={!sim || paused || sim.armReturning}
            onClick={() => { sim?.returnArmHome(); refresh(n => n + 1); }}>アームを標準姿勢へ</button>
        </div>
        {["Yaw · ひねり", "Pitch · 傾き", "Roll · 横回転"].map((label, axis) => <div key={label}>
          <label htmlFor={`wrist-${axis}`}>{label} <output>{sim ? (sim.wristTarget[axis] * 180 / Math.PI).toFixed(0) : "0"}°</output></label>
          <input id={`wrist-${axis}`} type="range" min={axis === 1 ? -90 : -180} max={axis === 1 ? 90 : 180} step="1" disabled={!sim || paused}
            value={sim ? sim.wristTarget[axis] * 180 / Math.PI : 0} onChange={e => { sim?.setWristAngle(axis, Number(e.target.value) * Math.PI / 180); refresh(n => n + 1); }}/>
        </div>)}
        <p className="pose-error" data-testid="pose-error">{sim ? `目標との差: 位置 ${(sim.targetError() * 100).toFixed(1)} cm / 向き ${(sim.attitudeError() * 180 / Math.PI).toFixed(1)}°` : ""}</p>
      </section>
      <details className="sidebar-section"><summary>物理設定とテスト</summary>
        <section className="parameters" aria-label="ロボットの物理設定">
          <label htmlFor="robot-mass">ロボット総質量 <output>{sim ? sim.robotMassKg.toFixed(1) : "—"} kg</output></label>
          <input id="robot-mass" type="range" min="10" max="80" step="0.5" disabled={!sim}
            value={sim?.robotMassKg ?? 34.5} onChange={e => { if (sim) { sim.setParameters(Number(e.target.value), sim.armTorqueNm); refresh(n => n + 1); } }}/>
          <label htmlFor="arm-torque">アームの最大トルク <output>{sim?.armTorqueNm ?? 40} N·m / 回転軸</output></label>
          <input id="arm-torque" type="range" min="5" max="100" step="5" disabled={!sim}
            value={sim?.armTorqueNm ?? 40} onChange={e => { if (sim) { sim.setParameters(sim.robotMassKg, Number(e.target.value)); refresh(n => n + 1); } }}/>
          <label htmlFor="press-height">押し付けテストの目標高さ <output>{(sim?.contactTargetY ?? TABLE_HEIGHT - 0.1).toFixed(2)} m</output></label>
          <input id="press-height" type="range" min={TABLE_HEIGHT - 0.25} max={TABLE_HEIGHT + 0.02} step="0.01" disabled={!sim}
            value={sim?.contactTargetY ?? TABLE_HEIGHT - 0.1} onChange={e => { if (sim) { sim.contactTargetY = Number(e.target.value); sim.reset(); refresh(n => n + 1); } }}/>
        </section>
        <div className="buttons">
          <button disabled={!sim} onClick={() => { sim?.startProbe("grasp"); setPaused(false); }}>把持テスト</button>
          <button aria-label="机への接触テスト" disabled={!sim} onClick={() => { sim?.startProbe("contact"); setPaused(false); }}>接触テスト</button>
        </div>
      </details>
      <details className="sidebar-section"><summary>計測</summary><dl className="metrics">
        <dt>状態</dt><dd data-testid="phase">{paused ? "一時停止" : sim?.phase ?? "準備中"}</dd>
        <dt>コップの高さ</dt><dd data-testid="mug-height">{mug ? mug[1].toFixed(3) + " m" : "—"}</dd>
        <dt>両指の接触</dt><dd data-testid="contact">{sim?.gripperContactId ?? "なし"}</dd>
        <dt>冷蔵庫の扉</dt><dd><span data-testid="door-angle">{sim ? (sim.q("fridge_hinge") * 180 / Math.PI).toFixed(0) + "°" : "—"}</span>
          <button className="secondary door-probe" aria-label="冷蔵庫の扉テスト" disabled={!sim}
            onClick={() => { sim?.startProbe("door"); setPaused(false); }}>扉テスト</button></dd>
        <dt>キッチンの収納扉</dt><dd><span data-testid="kitchen-angle">{sim ? (sim.q("kitchen_hinge") * 180 / Math.PI).toFixed(0) + "°" : "—"}</span>
          <button className="secondary door-probe" aria-label="キッチンの収納扉テスト" disabled={!sim}
            onClick={() => { sim?.startProbe("kitchen"); setPaused(false); }}>扉テスト</button></dd>
        <dt>オーブンの扉</dt><dd><span data-testid="oven-angle">{sim ? (sim.q("oven_hinge") * 180 / Math.PI).toFixed(0) + "°" : "—"}</span>
          <button className="secondary door-probe" aria-label="オーブンの扉テスト" disabled={!sim}
            onClick={() => { sim?.startProbe("oven"); setPaused(false); }}>扉テスト</button></dd>
        <dt>保持中の最低高さ</dt><dd data-testid="hold-height">{sim && Number.isFinite(sim.minHoldHeight) ? sim.minHoldHeight.toFixed(3) + " m" : "—"}</dd>
        <dt>机への最大食い込み</dt><dd data-testid="penetration">{sim ? (sim.maxTablePenetration * 1000).toFixed(2) + " mm" : "—"}</dd>
        <dt>机への接触力</dt><dd data-testid="table-force">{sim ? sim.tableForce.toFixed(1) + " N" : "—"}</dd>
        <dt>最大接触力</dt><dd data-testid="peak-table-force">{sim ? sim.peakTableForce.toFixed(1) + " N" : "—"}</dd>
        <dt>保持中の平均接触力</dt><dd data-testid="mean-table-force">{sim ? sim.meanTableForce().toFixed(1) + " N" : "—"}</dd>
        <dt>目標 TCP 高さ</dt><dd data-testid="target-height">{sim ? sim.target[1].toFixed(3) + " m" : "—"}</dd>
        <dt>肩 / 手首のトルク</dt><dd data-testid="arm-torque">{sim ? ["arm_flex_joint", "wrist_flex_joint"].map(n => sim.data.actuator_force[sim.joints[n].actuator].toFixed(1)).join(" / ") + " N·m" : "—"}</dd>
        <dt>昇降軸の力</dt><dd>{sim ? sim.data.actuator_force[sim.joints.arm_lift_joint.actuator].toFixed(1) + " N" : "—"}</dd>
        <dt>物理計算 / frame</dt><dd>{sim ? sim.physicsMs.toFixed(2) + " ms" : "—"}</dd>
        <dt>台車 x / z / yaw</dt><dd data-testid="base-pose">{base && sim ? [base[0], base[2], sim.q("base_yaw")].map(v => v.toFixed(3)).join(" / ") : "—"}</dd>
        <dt>台車の傾き</dt><dd data-testid="base-tilt">{sim ? `${(sim.baseTilt * 180 / Math.PI).toFixed(1)}°${sim.tipped ? " · 転倒" : ""}` : "—"}</dd>
        <dt>自己接触</dt><dd>{sim?.selfContactCount ?? "—"}</dd>
        <dt>TCP x / y / z</dt><dd data-testid="tcp-pose">{sim ? sim.tcp().map(v => v.toFixed(3)).join(" / ") : "—"}</dd>
        <dt>頭 tilt / pan</dt><dd data-testid="head-pose">{sim ? ["tilt", "pan"].map(n => (sim.q(n) * 180 / Math.PI).toFixed(1)).join(" / ") + "°" : "—"}</dd>
        <dt>手首 yaw / pitch / roll</dt><dd data-testid="wrist-pose">{sim ? sim.actualWristAngles().map(v => (v * 180 / Math.PI).toFixed(1)).join(" / ") + "°" : "—"}</dd>
        <dt>指の開き</dt><dd data-testid="opening">{sim ? (sim.gripperOpening() * 100).toFixed(0) + "%" : "—"}</dd>
        <dt>物理時間</dt><dd data-testid="sim-time">{sim ? sim.data.time.toFixed(1) + " s" : "—"}</dd>
      </dl></details>
      <div className="asset-credits"><a href="/assets/hsr-original-v1/ATTRIBUTION.html" target="_blank" rel="noreferrer">HSR: Toyota / CC BY-NC-ND 4.0（非商用）</a><br/>
        <a href="https://polyhaven.com" target="_blank" rel="noreferrer">3D assets / materials: Poly Haven</a></div>
    </aside><section className="viewport" aria-label="MuJoCo robot simulation">
      <button type="button" className="panel-label sidebar-toggle" aria-controls="sidebar" aria-expanded={!sidebarCollapsed}
        aria-label={sidebarCollapsed ? "サイドメニューを開く" : "サイドメニューを折りたたむ"}
        title={sidebarCollapsed ? "サイドメニューを開く" : "サイドメニューを折りたたむ"}
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/>
          <path d={sidebarCollapsed ? "m14 9 3 3-3 3" : "m17 9-3 3 3 3"}/>
        </svg>
      </button>
      <Suspense fallback={null}>{started && <SimulationCanvas sim={sim} active={active} paused={paused} mobile={mobile} controlView={controlView}
        headPane={headPane} handPane={handPane} overviewPane={overviewPane} onReady={viewReady}/>}</Suspense>
      <div className="view-grid" data-mobile-view={mobileView}>
        {(["head", "hand"] as const).map(view => <section key={view} className="camera-panel" data-view={view} data-active={controlView === view}
          aria-label={`${CAMERA_LABELS[view]}の映像`} aria-busy={!readyViews[view]}>
          <button type="button" className="panel-label" aria-label={`${CAMERA_LABELS[view]}を操作基準にする`} aria-pressed={controlView === view}
            title="手先操作の基準を切り替え"
            disabled={!sim} onClick={() => setControlView(view)}>
            {CAMERA_LABELS[view]}
          </button>
          <div className="camera-feed" ref={view === "head" ? headPane : handPane}>{!readyViews[view] && <FeedSkeleton/>}</div>
        </section>)}
        <section className="camera-panel map-panel" data-view="map" aria-label="部屋の Map">
          <span className="panel-label">Map</span>
          <button type="button" className="panel-label map-align" disabled={!sim || paused || sim.headReturning}
            onClick={() => { sim?.centerHead(); refresh(n => n + 1); }}>頭カメラを前方へ</button>
          <div className="camera-feed"><RoomMap observation={sim?.observation() ?? null} headPan={sim?.q("pan") ?? 0} doorAngle={sim?.q("fridge_hinge") ?? 0}/></div>
        </section>
        <section className="camera-panel" data-view="overview" aria-label="俯瞰の映像" aria-busy={!readyViews.overview}>
          <span className="panel-label" title="ドラッグで回転 · スクロールで拡大">俯瞰</span>
          <div className="camera-feed overview-feed" ref={setOverviewPane}>{!readyViews.overview && <FeedSkeleton/>}</div>
        </section>
      </div>
      {error && <div className="viewport-error" role="alert">{error}</div>}
    </section></div>
    <TouchControls key={`${paused}-${mobileView}-${inputGeneration}`} sim={sim} paused={paused}/>
  </main>;
}
