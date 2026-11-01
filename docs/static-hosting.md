# 静的ホスティング（Cloudflare Workers）— 2026-10-26

同じアプリとモデルを Cloudflare Workers の静的アセットとして無料で公開した。2026-10-26 にユーザーが Workers を選び、URL のサブドメインもそのまま使うと決めた。

- URL: https://home-robot-playground.yurao.workers.dev （メニューは手動操作 `/` と自動実行 `?mode=learned`。試験用に `?mode=learned&direct=1`、`?mode=recognition` もリンクなしで残している）
- 現在の中身（2026-10-29〜）: コミット ccd13f7 のビルド（Workers version 3e395f23）。4e54ed4 に、計画の 2 つ目以降の作業が前の作業の推論とぶつかる不具合の修正（64999fa）と、縁の上で放したときに「把持・持ち上げ」の ✓ が消える表示の修正を加えた版。公開 URL で「コップをかごに入れてから、かごから出して。」を最後まで実行して確認（コップ成功 27.2 秒、未学習の作業は 120 秒で時間切れ）。前の版の公開フォルダは `work/deploy-workers/site-64999fa`。それ以前は 4e54ed4（Workers version c19e1a8e、未学習の作業の段階表示と作業の種類の個別判定を追加。前の版 389edea／36e0c40b の公開フォルダは `work/deploy-workers/site-389edea`）。それ以前は 389edea（Workers version 36e0c40b）。自動実行の既定は v9 の連続実行（下の「6a73962 版」）。389edea では、どんな作業でも計画し、未学習の作業は VLA に試させて時間切れで失敗とする（`docs/open-language-plans.md`）。
- 以前の中身: コミット f78d288 のビルド（2026-10-26 更新、Workers version 8fc5b65f。初回公開は c4e2622）、公開 VLA（graph 8785 / weights 0cccb41e、81パーツ）、計画用言語モデル Qwen3-1.7B（68パーツ＋小さいファイル）、物体認識 T7（11283b24）。合計352ファイル、3.0GB（モデルは c4e2622 版とバイト単位で同一）。候補モデル（`smolvla-c63f9ccf6048`）と RTC 検証用グラフは載せていない。
- 無料枠の条件: 1回の公開あたり2万ファイルまで、1ファイル25MiBまで。静的ファイルへのリクエストは無料・無制限。Worker のスクリプトは置かない（置くと1日10万リクエストの制限に数えられる）。

## 公開の手順

モデルは Git に含めない。手元のモデルを、リポジトリに固定したハッシュと照合しながら集めてから公開する。

1. モデルを含まない状態でアプリをビルドする（`public/models/smolvla-*`、`public/models/language/`、`public/diagnostics/` が無いクリーンな作業コピーで `tsc -b && vite build && node tools/prerender-shell.mjs`）。
2. 公開用フォルダを作る。各パーツをコピー後にハッシュで照合する。不一致、25MiB 超、アプリ側にモデルや検証用ファイルが残っている場合は止まる。

   ```
   node tools/assemble-static-site.mjs --dist <ビルドの dist> \
     --vla public/models/smolvla-0cccb41e11e0 --language work/language-local-qwen3-1.7b \
     --recognition <model.onnx（sha256 11283b24…）> --out work/deploy-workers/site
   ```

3. `cd deploy/cloudflare-workers && wrangler deploy`（`wrangler login` 済みのアカウント）。

## 確認したこと

- 配信: `.gz`（`shared-8785.onnx.gz`）は `Content-Encoding` なしの `application/gzip` で返り、SHA-256 4d88815b… が一致。モデルパーツは `application/octet-stream`。候補モデルの URL は 404。
- 実ブラウザ（Chrome、ヘッドレス、WebGPU）で公開 URL を開き、モデルの取得と検証に110秒、数値確認 3/3。コップ seed 3000 を同期実行で運搬成功（シミュレーション 24.9秒、推論16回）。
- 公開物に氏名・所属・メールアドレスは含まれない（トークナイザーの語彙にある「gmail」などの一般語のみ）。

## f78d288 版の公開 URL での確認（2026-10-26）

キャッシュ空の Chrome（ヘッドレス、WebGPU）で公開 URL を開き、手動操作で台車を動かしてから自動実行へ切り替えた。

- 自動実行を開いた直後から VLA（6.1〜60.3秒）と Qwen（6.4秒〜、39.1秒でダウンロード完了）を並列に取得。VLA は88.0秒で実行可能、Qwen は VLA の確認後に GPU で起動し178.7秒で準備完了。
- 初期値「コップをかごに入れて。」で「計画して実行」: 計画0:03、コップ seed 3000 を運搬成功（シミュレーション24.5秒、推論16回）。
- 実行中に手動操作へ切り替えると物理時間が3.5秒で止まり、戻ると続きから成功。手動操作で動かした台車の位置は保持。ページの読み直しは一度も起きず、ブラウザの「戻る」でもメニューが切り替わる。

## 6a73962 版：自動実行の既定を v9・連続実行に（2026-10-29）

ユーザーの依頼（「良いタイミングで公開サイトのモデル更新して、触ってみたい」）と選択（「連続実行を既定に」）に沿って更新した。

- 中身: コミット 6a73962 のビルド（モデルを置かないクリーンな作業コピーで `tsc -b && vite build && node tools/prerender-shell.mjs`）。自動実行の既定は `prefix-rtc-v9`（第2段の最終評価で連続実行 60/60、`docs/evidence/mug-continuous-phase2-final-2026-10-28.json`）の連続実行で、評価と同じ設定（推論中に物理を0.4秒進める、指の開閉で止めない、返事の遅れの猶予3コマ、コップは壁の内側の判定）。従来の公開モデル（8785/0ccb）も同じサイトに残し、`?mode=learned&model=public&executor=sync&success=rim` で以前と同じ動作になる。
- 組み立て: `node tools/assemble-static-site.mjs --dist dist --vla public/models/smolvla-0cccb41e11e0 --language work/language-local-qwen3-1.7b --recognition <T7 model.onnx> --candidate prefix-rtc-v9 --candidate-dir work/prefix-rtc-v9-2026-10-27/hosted/models/smolvla-91aeb7db94a1 --out work/deploy-workers/site`。435ファイル、4.5GB。候補のパーツと共有グラフの `.gz` は登録表（`learned-onnx-contract.ts`）のハッシュで照合した。
- 公開: `wrangler deploy`（新規 37ファイル、398ファイルは既にアップロード済み）。前の版の公開フォルダは `work/deploy-workers/site-f78d288` に残した（戻すときは `site` と入れ替えて `wrangler deploy`）。
- 確認: 公開 URL から v9 の重みのパーツを取得し、手元とハッシュが一致。手元で組み立てたフォルダ（ポート5300）と公開 URL の両方で、URL に `?mode=learned&direct=1` だけを付けて開き、既定で v9・連続実行（再生0.47〜0.48倍、猶予3、壁の判定）になることを確認。WebGPU の数値確認 3/3、コップ seed 3000 を運搬成功（手元 24.7秒、公開 URL 27.7秒）。公開 URL ではキャッシュ空の Chrome で実行可能になるまで 227.5秒（f78d288 版の確認時は 88秒。ネットワークの条件で変わる）。
- 未確認: 公開 URL での「計画して実行」（言語モデル経由）とボールの運搬。v9 はボールを評価していない。

## 未確認・注意

- 物体認識は公開 URL ではまだ動かしていない（手元の同じファイルでは確認済み）。連続実行は 6a73962 版で確認した。
- 切り替え前の版の公開フォルダは `work/deploy-workers/site-c4e2622` に残している。戻すときは `site` と入れ替えて `wrangler deploy`。
- 静的ファイルの帯域上限は公式文書に明記がない（静的リクエストは無料・無制限と明記）。
- `Cache-Control: max-age=0, must-revalidate` のため、再訪時はブラウザ内のキャッシュ（Cache API）と ETag の再検証で済む。
