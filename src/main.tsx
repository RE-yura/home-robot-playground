import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import "./index.css";
import { App } from "./app/App.tsx";

const root = document.getElementById("root")!;
const mode = new URLSearchParams(window.location.search).get("mode");
const learned = mode === "learned" || mode === "rtc";
const recognition = mode === "recognition";
const app = <StrictMode><App learned={learned} rtcDiagnostic={mode === "rtc"} recognition={recognition} /></StrictMode>;
if (root.hasChildNodes() && !learned && !recognition) hydrateRoot(root, app);
else createRoot(root).render(app);
