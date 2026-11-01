import "./learned-pilot.css";
import { AppHeader } from "../app/AppHeader.tsx";

export function LearnedPilotShell() {
  return <main className="auto-page" aria-busy="true">
    <AppHeader current="auto"/>
    <div className="workbench"><aside>
      <section className="language-plan"><form>
        <label htmlFor="pending-instruction">ロボットへの指示</label>
        <textarea id="pending-instruction" rows={3} disabled defaultValue="コップをかごに入れて。"/>
        <div className="plan-buttons"><button type="button" disabled>計画して実行</button><button type="button" className="secondary" disabled>計画のみ</button><button type="button" className="secondary" disabled>停止</button></div>
      </form></section>
      <section className="learned-status-rail" data-loading="true"><div className="learned-status-main"><span className="learned-state-dot" aria-hidden="true"/><div><output role="status">部屋とモデルを準備しています</output><p>準備できたカメラから表示します。</p></div></div></section>
    </aside><section className="viewport"><div className="view-grid">
      {["頭カメラ", "手先カメラ", "Map", "俯瞰"].map(label => <section key={label} className="camera-panel"><span className="panel-label">{label}</span><div className="camera-feed"><div className="feed-skeleton"/></div></section>)}
    </div></section></div>
  </main>;
}
