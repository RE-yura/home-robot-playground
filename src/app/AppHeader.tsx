import { useContext, type MouseEvent } from "react";
import { NavigationContext, type Menu } from "./navigation.ts";

// The two top-level menus share one header. Inside the app a click switches menus without a page load;
// the links still work as ordinary pages (new tab, before hydration).
export function AppHeader({ current }: { current: Menu }) {
  const navigation = useContext(NavigationContext);
  const follow = (menu: Menu) => (event: MouseEvent) => {
    if (!navigation || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault(); navigation.navigate(menu);
  };
  return <header className="app-header">
    <strong>Home Robot</strong>
    <nav aria-label="メニュー">
      <a href="/" aria-current={current === "manual" ? "page" : undefined} onClick={follow("manual")}>手動操作</a>
      <a href={navigation?.autoHref ?? "/?mode=learned"} aria-current={current === "auto" ? "page" : undefined} onClick={follow("auto")}>自動実行</a>
    </nav>
  </header>;
}
