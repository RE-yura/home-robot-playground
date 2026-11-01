import { useEffect, useRef, useState, type PointerEvent } from "react";
import type { Simulation } from "../simulation/mujoco.ts";
import { TOUCH_STOP } from "../simulation/input.ts";

type Axes = [number, number];

function Joystick({ label, horizontal, vertical, disabled, onChange }: {
  label: string; horizontal: string; vertical: string; disabled: boolean; onChange: (axes: Axes) => void;
}) {
  const pointer = useRef<number | null>(null), change = useRef(onChange);
  const [point, setPoint] = useState<Axes>([0, 0]);
  useEffect(() => { change.current = onChange; }, [onChange]);
  useEffect(() => () => { change.current([0, 0]); }, []);
  const update = (e: PointerEvent<HTMLButtonElement>) => {
    const rect = e.currentTarget.getBoundingClientRect(), radius = rect.width * 0.3;
    const x = (e.clientX - rect.left - rect.width / 2) / radius;
    const y = (e.clientY - rect.top - rect.height / 2) / radius;
    const distance = Math.hypot(x, y), scale = Math.max(1, distance);
    setPoint([x / scale, y / scale]);
    // A small dead zone prevents a resting thumb from creeping the robot.
    const speed = Math.min(1, Math.max(0, (distance - 0.12) / 0.88));
    onChange(distance ? [x / distance * speed, y / distance * speed] : [0, 0]);
  };
  const stop = () => { pointer.current = null; setPoint([0, 0]); onChange([0, 0]); };
  return <button type="button" className="joystick" aria-label={label} disabled={disabled}
    onPointerDown={e => {
      if (pointer.current !== null) return;
      e.preventDefault(); pointer.current = e.pointerId; e.currentTarget.setPointerCapture(e.pointerId); update(e);
    }} onPointerMove={e => { if (pointer.current === e.pointerId) update(e); }}
    onPointerUp={e => { if (pointer.current === e.pointerId) stop(); }}
    onPointerCancel={stop} onLostPointerCapture={stop} onBlur={stop}
    onKeyDown={e => {
      const axes: Record<string, Axes> = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };
      if (!axes[e.key]) return;
      e.preventDefault(); e.stopPropagation(); setPoint(axes[e.key]); onChange(axes[e.key]);
    }} onKeyUp={e => { if (e.key.startsWith("Arrow")) { e.preventDefault(); e.stopPropagation(); stop(); } }}>
    <span className="stick-vertical">{vertical}</span><span className="stick-horizontal">{horizontal}</span>
    <span className="stick-knob" style={{ transform: `translate(${point[0] * 60}%, ${point[1] * 60}%)` }}/>
  </button>;
}

function TurnButton({ left, disabled, sim }: { left: boolean; disabled: boolean; sim: Simulation | null }) {
  const pointer = useRef<number | null>(null);
  useEffect(() => () => { sim?.setTouchInput({ turn: 0 }); }, [sim]);
  const stop = () => { pointer.current = null; sim?.setTouchInput({ turn: 0 }); };
  return <button type="button" className="secondary turn-button" aria-label={left ? "左へ旋回" : "右へ旋回"}
    disabled={disabled} onPointerDown={e => {
      e.preventDefault(); pointer.current = e.pointerId; e.currentTarget.setPointerCapture(e.pointerId);
      sim?.setTouchInput({ turn: left ? 1 : -1 });
    }} onPointerUp={stop} onPointerCancel={stop} onLostPointerCapture={stop} onBlur={stop}
    onKeyDown={e => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); e.stopPropagation(); sim?.setTouchInput({ turn: left ? 1 : -1 }); } }}
    onKeyUp={e => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); e.stopPropagation(); stop(); } }}>
    {left ? "↶ 左旋回" : "右旋回 ↷"}
  </button>;
}

export function TouchControls({ sim, paused }: { sim: Simulation | null; paused: boolean }) {
  const [generation, setGeneration] = useState(0), disabled = !sim || paused;
  useEffect(() => {
    const stop = () => { sim?.setTouchInput(TOUCH_STOP); setGeneration(n => n + 1); };
    const hidden = () => { if (document.hidden) stop(); };
    window.addEventListener("blur", stop); document.addEventListener("visibilitychange", hidden);
    return () => {
      sim?.setTouchInput(TOUCH_STOP);
      window.removeEventListener("blur", stop); document.removeEventListener("visibilitychange", hidden);
    };
  }, [sim]);
  return <section className="touch-controls" aria-label="タッチ操作" key={generation}>
    <div className="stick-column base-stick">
      <strong>台車</strong>
      <Joystick label="台車の移動ジョイスティック" horizontal="左右" vertical="前後" disabled={disabled}
        onChange={([x, y]) => sim?.setTouchInput({ strafe: x, forward: -y })}/>
      <div className="turn-buttons"><TurnButton left sim={sim} disabled={disabled}/><TurnButton left={false} sim={sim} disabled={disabled}/></div>
    </div>
    <div className="stick-column arm-stick">
      <strong>手先 <small>{sim ? `${sim.tcp()[1].toFixed(2)} m` : ""}</small></strong>
      <Joystick label="手先の上下・伸縮ジョイスティック" horizontal="縮む・伸びる" vertical="上下" disabled={disabled}
        onChange={([x, y]) => sim?.setTouchInput({ reach: x, lift: -y })}/>
      <button type="button" className="gripper-button" disabled={disabled} onClick={() => {
        sim?.setKey("Space", true); sim?.setKey("Space", false);
      }}>{sim && sim.opening < 0.5 ? "グリッパーを開く" : "グリッパーを閉じる"}</button>
      <button type="button" className="secondary" disabled={disabled || sim?.armReturning}
        onClick={() => sim?.returnArmHome()}>アームを標準姿勢へ</button>
    </div>
    <p className="touch-hint">指を離すと停止 · 右スティックは上げ下げ／左右で伸縮</p>
  </section>;
}
