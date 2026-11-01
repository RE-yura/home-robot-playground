import { lazy, Suspense, useEffect, useState } from "react";
import { PhysicsPlayground } from "../playground/PhysicsPlayground";
import { LearnedPilotShell } from "../learning/LearnedPilotShell.tsx";
import { NavigationContext, menuFromSearch, type Menu } from "./navigation.ts";
const LearnedPilot = lazy(() => import("../learning/LearnedPilot.tsx"));
const RecognitionPreview = lazy(() => import("../learning/RecognitionPreview.tsx"));
// The two menus switch without a page load. Once opened, a menu stays mounted with its own world and loaded models;
// only the shown one runs its canvas.
export function App({ learned = false, rtcDiagnostic = false, recognition = false }: { learned?: boolean; rtcDiagnostic?: boolean; recognition?: boolean }) {
  const [menu, setMenu] = useState<Menu>(learned ? "auto" : "manual");
  const [opened, setOpened] = useState({ manual: !learned, auto: learned });
  // 自動実行 keeps the query it was opened with (?model=, ?executor=, ?direct=1, ?mode=rtc ...).
  const [autoSearch] = useState(() => learned ? globalThis.location?.search || "?mode=learned" : "?mode=learned");
  useEffect(() => {
    const back = () => {
      const next = menuFromSearch(location.search);
      if (!next) { location.reload(); return; }
      setMenu(next); setOpened(previous => previous[next] ? previous : { ...previous, [next]: true });
    };
    addEventListener("popstate", back); return () => removeEventListener("popstate", back);
  }, []);
  function navigate(next: Menu) {
    if (next === menu) return;
    history.pushState(null, "", next === "auto" ? `/${autoSearch}` : "/");
    setMenu(next); setOpened(previous => previous[next] ? previous : { ...previous, [next]: true });
  }
  if (recognition) return <Suspense fallback={<main><p role="status">物体認識ページを準備しています。</p></main>}><RecognitionPreview/></Suspense>;
  return <NavigationContext value={{ navigate, autoHref: `/${autoSearch}` }}>
    {opened.manual && <div className="menu-page" hidden={menu !== "manual"}><PhysicsPlayground active={menu === "manual"}/></div>}
    {opened.auto && <div className="menu-page" hidden={menu !== "auto"}><Suspense fallback={<LearnedPilotShell/>}><LearnedPilot rtcDiagnostic={rtcDiagnostic} active={menu === "auto"}/></Suspense></div>}
  </NavigationContext>;
}
