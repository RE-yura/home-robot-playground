# ブラウザだけで動く家庭用ロボットの VLA

家の中の HSR（トヨタの生活支援ロボット）を MuJoCo で物理シミュレーションし、頭と手先のカメラ画像から腕を動かす学習モデル（SmolVLA を追加学習した VLA）を、**サーバーなしでブラウザの中だけ**で動かすプロジェクトです。日本語で「コップをかごに入れて。」と頼むと、ブラウザ内の SLM（小規模言語モデル、Qwen3-1.7B）が作業を計画し、VLA がコップを掴んでかごに入れます。推論中もロボットは止まりません（連続実行）。

- 試す：https://home-robot-playground.yurao.workers.dev （「自動実行」メニュー。PC の Chrome など WebGPU に対応したブラウザ。初回はモデル約 3 GB の取得に数分）
- 動画：https://youtu.be/OWMNRIN2UOc
- 説明資料：`docs/submission/report.html`（PDF は提出用）

Physical AI 応用1（松尾研究室）の最終課題として作りました。

## できること

| メニュー | 内容 |
|---|---|
| 手動操作 | キーボード・ボタンで台車・腕・手・頭を操作。頭カメラ・手先カメラ・地図・俯瞰の4画面 |
| 自動実行 | 日本語の指示 → SLM が作業の並びを計画（どんな作業でも計画し、学習済み／未学習を表示）→ VLA がカメラ画像と関節の実測値だけから腕を動かす。学習済みの作業はコップ（またはテニスボール）をかごに入れること。未学習の作業も VLA に試させ、120 秒で失敗として終わる |

結果（コップ、学習モデルのみ、最終評価用に取っておいた 60 配置。学習にも、モデル・設定の比較にも使っていない）：連続実行 **60/60**（95% 信頼下限 95.1%）、同期実行 60/60。詳しくは [docs/rtc-continuous-execution.md](docs/rtc-continuous-execution.md)。

## しくみ

- **物理**：MuJoCo WASM（2 ms 刻み）。カメラは 320×240 の RGB。
- **VLA**：LeRobot の SmolVLA を HSR 用に追加学習し、ONNX に書き出して ONNX Runtime WebGPU で実行。入力は頭・手先の RGB、指示文、6 チャンネルの関節の実測、連続実行で確定済みの目標列。出力は 100 ms ごとの関節目標 16 個。
- **計画**：Qwen3-1.7B（q4f16）を transformers.js / WebGPU で実行し、JSON で作業を出力。
- **学習**：シミュレーターの状態を使う台本の教師でデモを集め、GPU で追加学習（`tools/vla-train`、`tools/vla-data`）。教師が使う状態は VLA の入力には使わない。
- **配信**：Cloudflare Workers の静的ファイル。モデルはハッシュで照合してからブラウザにキャッシュ（[docs/static-hosting.md](docs/static-hosting.md)）。

## 手元で動かす

Node.js 24 以上と pnpm。

```sh
pnpm install
pnpm dev          # http://127.0.0.1:5183/
pnpm test         # 実際の MuJoCo WASM で物理・採点・実行器を検証（資料に関わる部分のテストだけを収録）
pnpm build
```

学習済みモデルは Git に含めていません。手元で自動実行を動かすときのモデルの置き方は [docs/rtc-continuous-execution.md](docs/rtc-continuous-execution.md) の「手元（pnpm dev）で動かすときのモデル配置」を参照してください。

評価用スクリプト（`tools/eval`）は `playwright-core` を使います（`npm i --no-save playwright-core`）。Python の学習コードは GPU 環境で動かす前提で、依存は `tools/vla-train/requirements-colab.txt` にあります。

## 主なフォルダ

| 場所 | 中身 |
|---|---|
| `src/simulation` | MuJoCo の読み込み、HSR の制御、採点（`task.ts`） |
| `src/learning` | VLA の実行器（同期・連続）、ブラウザ推論、言語計画、教師、データ記録 |
| `tools/vla-train` | 学習（`hsr_training.py`、確定済みの目標の入力は `prefix_conditioning.py`）・ONNX 書き出し・重みの平均 |
| `tools/vla-data`、`tools/cloud` | デモの収集・検査・データセットの組み立て |
| `tools/vla-learned` | 評価の集計（`final_summary.py`）、ONNX の書き出しとブラウザ用の変換 |
| `tools/eval` | 実ブラウザ（Chrome・WebGPU）で 1 回ずつ実行して記録を保存する評価用スクリプト、記録を CPU で再生する `replay-rtc.ts` |
| `docs/evidence` | 資料の数値の元になる記録（要約 JSON）。最終モデルの学習は `final-model-training.json` |

## 限界

- 配置はほぼ固定（コップ・ボール・かごの位置が数 mm ずれる程度）。大きく動かした配置は未検証。
- 学習した作業は「コップ／テニスボールをかごに入れる」だけ。最終モデルのボールは未評価。
- 物理は実時間の約 0.47 倍で進む（推論 約 0.9 秒/回）。
- シミュレーションのみ。実機では検証していない。

## 使用したモデル・素材のライセンス

[public/models/ATTRIBUTION.md](public/models/ATTRIBUTION.md) に記載。
