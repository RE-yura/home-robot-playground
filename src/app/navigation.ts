import { createContext } from "react";

export type Menu = "manual" | "auto";
// The recognition preview is not a menu; it stays a separate document (null).
export function menuFromSearch(search: string): Menu | null {
  const mode = new URLSearchParams(search).get("mode");
  return mode === "recognition" ? null : mode === "learned" || mode === "rtc" ? "auto" : "manual";
}
export const NavigationContext = createContext<{ navigate: (menu: Menu) => void; autoHref: string } | null>(null);
