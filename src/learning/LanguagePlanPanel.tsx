import { useEffect, useRef, useState } from "react";
import { LanguagePlanner, type LanguageReceipt } from "./language-planner.ts";
import { PLAN_STEPS, type PlanExecution } from "./plan-execution.ts";
import { learnedTarget } from "./language-goals.ts";

function duration(ms: number) { const seconds = Math.floor(ms / 1000); return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`; }
// warm: the robot model has finished verifying, so the planning model may start on the GPU (it downloads from the start).
export function LanguagePlanPanel({ disabled = false, canExecute = false, stopEnabled = false, warm = true, execution, onExecute, onStop, onBusyChange }: {
  disabled?: boolean; canExecute?: boolean; stopEnabled?: boolean; warm?: boolean; execution?: PlanExecution | null; onBusyChange?: (busy: boolean) => void;
  onExecute?: (receipt: LanguageReceipt, requestedAt: number, plannedAt: number) => void; onStop?: () => void;
}) {
  const [instruction, setInstruction] = useState("コップをかごに入れて。");
  const [receipt, setReceipt] = useState<LanguageReceipt | null>(null), [busy, setBusy] = useState(false), [message, setMessage] = useState("指示から作業順を作ります。");
  const [now, setNow] = useState(performance.now()), [requestedAt, setRequestedAt] = useState<number | null>(null), [plannedAt, setPlannedAt] = useState<number | null>(null);
  const [download, setDownload] = useState<{loaded:number;total:number} | null>(null);
  const [model, setModel] = useState<"downloading" | "downloaded" | "ready" | "failed">("downloading");
  // 「計画して実行」: the proposed plan starts as soon as the robot can take it, exactly once.
  const [queued, setQueued] = useState(false);
  const planner = useRef<LanguagePlanner | null>(null), revision = useRef(0);
  function createPlanner(allowed: boolean) {
    const created: LanguagePlanner = new LanguagePlanner(value => {
      const p = value as {status?:string;loaded?:number;total?:number};
      if (planner.current === created && p.status === "language-download" && Number.isFinite(p.loaded) && Number.isFinite(p.total)) setDownload({loaded:p.loaded!,total:p.total!});
    }, { deferLoad: true });
    created.downloaded.then(() => { if (planner.current === created) { setDownload(null); setModel("downloaded"); } }, () => {});
    created.ready.then(() => { if (planner.current === created) setModel("ready"); }, () => { if (planner.current === created) setModel("failed"); });
    if (allowed) created.startLoading();
    return created;
  }
  useEffect(() => {
    const timer = setInterval(() => setNow(performance.now()), 100);
    planner.current = createPlanner(false);
    return () => { clearInterval(timer); revision.current++; planner.current?.dispose(); planner.current = null; };
  }, []);
  useEffect(() => { if (warm) planner.current?.startLoading(); }, [warm]);
  useEffect(() => {
    if (!queued || busy || !canExecute || receipt?.goal.status !== "proposed" || requestedAt === null) return;
    // One-shot hand-off: clearing the queue here keeps a refused start from retrying on every render.
    // eslint-disable-next-line react/set-state-in-effect
    setQueued(false); onExecute?.(receipt, requestedAt, plannedAt ?? requestedAt);
  }, [queued, busy, canExecute, receipt, requestedAt, plannedAt, onExecute]);
  function cancel() {
    // Stopping a generation needs a fresh worker; it reloads from the verified cache in the background.
    revision.current++; planner.current?.dispose(); setModel("downloading"); planner.current = createPlanner(warm);
    setBusy(false); setDownload(null); onBusyChange?.(false); setReceipt(null); setPlannedAt(performance.now()); setMessage("解釈を停止しました。");
  }
  function stop() {
    setQueued(false);
    if (busy) cancel();
    if (stopEnabled) onStop?.();
  }
  async function interpret(run: boolean) {
    const current = ++revision.current; setReceipt(null); setQueued(false); setRequestedAt(performance.now()); setPlannedAt(null); setBusy(true); onBusyChange?.(true);
    if (!planner.current) { setModel("downloading"); planner.current = createPlanner(true); }
    const owned = planner.current;
    setMessage(model === "ready" ? "LLMが作業の対象と順番を考えています。" : "計画用モデルの準備を待っています。");
    try {
      owned.startLoading(); await owned.ready;
      if (revision.current !== current) return;
      setMessage("LLMが作業の対象と順番を考えています。");
      const result = await owned.interpret(instruction);
      if (revision.current !== current) return;
      setReceipt(result); setPlannedAt(performance.now());
      const proposed = result.goal.status === "proposed";
      if (run && proposed) setQueued(true);
      const unlearned = result.goal.tasks.some(step => learnedTarget(step.en) === null);
      setMessage(result.goal.status === "clarify" ? "対象と作業を具体的にしてください。" : (run ? "計画どおりに実行します。" : "対象と順番を確認して、「この計画で実行」を押してください。") +
        (unlearned ? "未学習の作業は VLA に英語の指示で試させ、成功の判定基準がないため120秒で失敗として終わります。" : ""));
    } catch (error) {
      if (revision.current === current) { owned.dispose(); if (planner.current === owned) planner.current = null; setPlannedAt(performance.now()); setMessage(`解釈できませんでした: ${String(error)}`); }
    } finally { if (revision.current === current) { setBusy(false); onBusyChange?.(false); } }
  }
  const linked = Boolean(receipt && execution?.receipt === receipt), tasks = linked ? execution!.tasks : null;
  const total = linked ? execution!.elapsedMs : requestedAt === null ? 0 : (receipt?.goal.status === "proposed" ? now : plannedAt ?? now) - requestedAt;
  const planning = requestedAt === null ? 0 : (plannedAt ?? now) - requestedAt;
  const state = linked ? execution!.status : busy ? 'planning' : queued ? 'queued' : receipt?.goal.status === 'proposed' ? 'ready' : 'idle';
  const stateLabels = {planning:'計画中', queued:'実行準備中', ready:'実行待ち', running:'実行中', success:'全作業完了', failed:'作業失敗', stopped:'停止', idle:'未開始'};
  return <section className="language-plan" aria-label="LLMの動作計画" aria-busy={busy} data-plan-status={state}>
    <form onSubmit={e => { e.preventDefault(); void interpret(true); }}>
      <label htmlFor="language-instruction">ロボットへの指示</label>
      <textarea id="language-instruction" rows={3} maxLength={500} value={instruction} disabled={disabled || busy} onChange={e => { revision.current++; setInstruction(e.target.value); setReceipt(null); setQueued(false); setRequestedAt(null); setPlannedAt(null); setMessage("指示から作業順を作ります。"); }}/>
      <div className="plan-buttons">
        <button disabled={disabled || busy || !instruction.trim()} type="submit"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5 13 8l-9 5.5z" fill="currentColor"/></svg>計画して実行</button>
        <button className="secondary" type="button" disabled={disabled || busy || !instruction.trim()} onClick={() => void interpret(false)}>計画のみ</button>
        <button className="secondary" type="button" disabled={!busy && !queued && !stopEnabled} onClick={stop}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3h10v10H3z" fill="currentColor"/></svg>停止</button>
      </div>
    </form>
    <p role="status">{linked ? execution!.status === 'running' ? `${execution!.current + 1}/${execution!.tasks.length} · ${execution!.tasks[execution!.current].target === null ? '未学習の作業を VLA が実行中（120秒で終了）' : execution!.tasks[execution!.current].recovering ? '取り落としを検出。VLAによる再把持を待っています。' : PLAN_STEPS[execution!.tasks[execution!.current].step]}` : execution!.status === 'success' ? 'すべて、かご内での静止まで確認しました。' : execution!.status === 'failed' ? `作業を終了しました (${execution!.tasks[execution!.current].error ?? execution!.tasks[execution!.current].result?.reason})。未完了の作業を残して停止しました。` : '停止しました。配置と結果を保持しています。' : queued && !canExecute ? 'ロボットの準備ができしだい実行します。' : message}</p>
    {model !== "ready" && <div className="language-download"><small>{model === "failed" ? "計画用モデルを準備できませんでした。計画を押すと再試行します。" : model === "downloaded" ? warm ? "計画用モデルを起動しています。" : "計画用モデルはダウンロード済みです。ロボットのモデルの確認後に起動します。" : "計画用モデルをダウンロードしています（初回約1.43GB、次回から保存データを使います）。"}</small>
      {model === "downloading" && <><progress aria-label="計画用モデルの準備" max={download?.total ?? 1} value={download?.loaded}/>{download && <small>{Math.floor(download.loaded / download.total * 100)}% · {Math.round(download.loaded / 1048576)} / {Math.round(download.total / 1048576)} MB</small>}</>}</div>}
    {requestedAt !== null && <dl className="plan-time" data-state={state}><div><dt>{stateLabels[state]}</dt></div><div><dt>計画</dt><dd>{duration(planning)}</dd></div><div><dt>合計</dt><dd>{duration(total)}</dd></div></dl>}
    {receipt?.goal.status === "proposed" && <>
      <ol className="plan-tasks">{receipt.goal.tasks.map((planned, index) => {
        const task = tasks?.[index], active = task?.status === 'running', complete = task?.status === 'success', learned = learnedTarget(planned.en) !== null;
        return <li key={index} data-task-status={task?.status ?? 'queued'} data-learned={learned}>
          <div className="plan-task-title"><strong>{planned.ja}</strong><small className="plan-skill">{learned ? '学習済み' : '未学習'}</small><span>{complete ? '完了' : active ? '実行中' : task?.status === 'failed' ? '失敗' : task?.status === 'stopped' ? '停止' : '待機'}</span></div>
          {learned ? <ol className="plan-steps">{PLAN_STEPS.map((label, step) => <li key={label} aria-current={active && task.step === step ? 'step' : undefined} data-complete={Boolean(task?.observed[step])}><span>{task?.observed[step] ? '✓' : active && task.step === step ? '●' : complete ? '—' : '○'} {label}</span><time>{task ? duration(execution!.stepElapsed(task, step)) : '0:00'}</time></li>)}</ol>
            : <>{receipt.stages?.[index] && <ol className="plan-steps plan-steps-unverified">{receipt.stages[index]!.map((stage, i) => <li key={i}><span>・ {stage}</span><small>未確認</small></li>)}</ol>}
              <p className="plan-unlearned">VLA への指示「{planned.en}」{active ? 'を実行中' : ''}（成功判定なし・120秒で終了）</p></>}
          {task?.recovering && <small>再把持待ち · 取り落とし {execution!.rollout?.result.drops}回</small>}
        </li>;
      })}</ol>
      {!linked && !queued && <button className="plan-execute" disabled={!canExecute || busy || disabled} onClick={() => onExecute?.(receipt, requestedAt!, plannedAt ?? requestedAt!)}>この計画で実行</button>}
      <p className="plan-note">LLMの解釈は誤ることがあります。達成表示は接触・位置・静止の実測に基づきます。</p>
    </>}
  </section>;
}
