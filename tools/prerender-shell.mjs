import { readFile, writeFile } from "node:fs/promises";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

// Render the same initial UI used by React so the browser can paint before JS loads.
const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
try {
  const { App } = await server.ssrLoadModule("/src/app/App.tsx");
  let html = await readFile("dist/index.html", "utf8");
  html = html.replace('<div id="root"></div>', () => `<div id="root">${renderToString(createElement(App))}</div>`);
  for (const match of html.matchAll(/<link\b[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g)) {
    const css = await readFile(`dist${match[1]}`, "utf8");
    html = html.replace(match[0], `<style>${css}</style>`);
  }
  await writeFile("dist/index.html", html);
} finally {
  await server.close();
}
