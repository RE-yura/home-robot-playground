// Drive one learned run in real Chrome (WebGPU) and save the page's own evidence JSON.
// Usage: node run-task.mjs <learned|rtc> <mug|ball> <outdir> [seed]
import { chromium } from 'playwright-core';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const [mode, target, outArg, seedArg] = process.argv.slice(2);
if (!['learned', 'rtc'].includes(mode) || !['mug', 'ball'].includes(target) || !outArg) throw new Error('Usage: node run-task.mjs <learned|rtc> <mug|ball> <outdir> [seed]');
const out = resolve(outArg); await mkdir(out, { recursive: true });
const seed = seedArg ?? '3000';
const base = process.env.HSR_URL ?? 'http://127.0.0.1:5183';
const profile = resolve(new URL('.', import.meta.url).pathname, process.env.HSR_PROFILE ?? 'chrome-profile');

const context = await chromium.launchPersistentContext(profile, { channel: 'chrome', headless: true, viewport: { width: 1280, height: 800 }, args: ['--enable-unsafe-webgpu'], acceptDownloads: true });
const page = context.pages()[0] ?? await context.newPage();
const log = [];
const note = (...a) => { const line = `[${new Date().toISOString()}] ${a.join(' ')}`; log.push(line); console.log(line); };
page.on('console', m => { if (['error', 'warning'].includes(m.type())) note('console', m.type(), m.text().slice(0, 300)); });
page.on('pageerror', e => note('pageerror', String(e).slice(0, 300)));
// Vite dev serves *.gz with Content-Encoding: gzip, so the browser would inflate it before the
// hash check. Production serves it as plain octet-stream; mirror that here.
if (!process.env.HSR_NO_ROUTE) await page.route('**/assets/learned-vla/shared-8785.onnx.gz', route => route.fulfill({ path: new URL('../../public/assets/learned-vla/shared-8785.onnx.gz', import.meta.url).pathname, contentType: 'application/octet-stream' }));
const label = () => page.locator('output[aria-live]').innerText().catch(() => '');
try {
  await page.goto(`${base}/?mode=${mode}${process.env.HSR_QUERY ?? ""}`);
  note('opened', mode);
  const t0 = Date.now();
  for (;;) {
    const l = await label();
    if (l.startsWith('実行できま')) break;
    if (l.includes('終了') || Date.now() - t0 > 900000) throw new Error(`load failed: ${l} / ${await page.locator('.learned-status-main p').innerText()}`);
    await page.waitForTimeout(1000);
  }
  note('ready after', ((Date.now() - t0) / 1000).toFixed(1), 's', 'provider', await page.locator('output[aria-live]').getAttribute('data-provider'));
  if (seed !== '3000') {
    await page.locator('summary', { hasText: '設定' }).click();
    await page.getByLabel('Seed').fill(seed);
    await page.getByRole('button', { name: '配置を戻す' }).click();
    await page.waitForTimeout(1500);
  }
  await page.locator('select#learned-instruction').selectOption(target === 'mug' ? 'Put the mug in the basket.' : 'Put the tennis ball in the basket.');
  await page.getByRole('button', { name: '実行', exact: true }).click();
  note('started', target, 'seed', seed);
  const t1 = Date.now(); let shot = 0;
  for (;;) {
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${out}/frame-${String(shot++).padStart(3, '0')}.jpg`, quality: 60 });
    const l = await label();
    const sim = await page.locator('.learned-run-metrics').innerText().catch(() => '');
    note('status', l, '|', sim.replace(/\s+/g, ' '));
    if (['運搬完了', '実行を終了しました', '停止中'].includes(l)) break;
    if (Date.now() - t1 > 1200000) { note('timeout: stopping'); await page.getByRole('button', { name: '停止' }).click(); await page.waitForTimeout(3000); break; }
  }
  await page.screenshot({ path: `${out}/final.png` });
  // The save button sits inside the collapsed 設定 panel on the current UI; open it first.
  if (!(await page.locator('details.learned-settings').evaluate(d => d.open))) await page.locator('summary', { hasText: '設定' }).click();
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '検証結果を保存' }).click()]);
  await download.saveAs(`${out}/record.json`);
  note('saved', `${out}/record.json`, 'status text:', await page.locator('.learned-status-main p').innerText());
} finally {
  await writeFile(`${out}/driver.log`, log.join('\n') + '\n');
  await context.close();
}
