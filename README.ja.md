# PawMatch Guard

[🇬🇧 English](README.md) | 🇯🇵 日本語

犬猫の里親マッチングアプリに、投稿の **AI審査(コンテンツモデレーション)** を組み込んだポートフォリオです。マッチングは軽めにし、見せ場は画像・動画の審査パイプラインに置いています。分類・検出・セグメンテーションの3モデルで「公開可 / 要確認 / 却下」を判定し、迷うものは人の確認に回します。

> 状況: **フェーズ1(分類と評価の土台)を実装済み。** フェーズ2〜4は計画中です(下記)。

## 機能(フェーズ1)

- Oxford-IIIT Petを品種で層別した70/15/15分割。seed固定、内容ハッシュでの重複チェック、分割結果はCSVで保存
- ResNet50の転移学習。種別(犬/猫)と37品種の2ヘッド
- クラス不均衡への対応: class weight、`WeightedRandomSampler`、Focal Loss
- 評価指標を自前実装(混同行列、正解率、クラス別P/R/F1、macro-F1)し、scikit-learnとの一致をテスト
- 設定ファイル駆動。結果は `runs/<日時>/` に設定・指標・重みを保存
- CLI出力は英語(既定)/日本語を `--lang` または `PAWMATCH_LANG` で切り替え

## クイックスタート

[uv](https://docs.astral.sh/uv/) が必要です(Python 3.12は自動で用意されます)。

```bash
uv sync
uv run pawmatch prepare-data          # データ取得(約800MB)+ data/splits/*.csv 作成
# 公式サーバは英国外だと非常に遅いため、Hugging Faceミラーも使えます:
uv run pawmatch prepare-data --source hf
uv run pawmatch train                 # configs/classification.yaml を使用
uv run pawmatch report runs/<run_dir> # テスト指標を表示
```

日本語で出力: `PAWMATCH_LANG=ja uv run pawmatch train` または `uv run pawmatch --lang ja train`

Docker(CPUのみ、GPU不要):

```bash
docker compose run --rm prepare
docker compose run --rm train
```

品質チェック: `uv run ruff check . && uv run pytest`

## 構成

```
src/data        取得・分割・前処理
src/models      分類モデル・学習
src/metrics     自前の評価指標
src/i18n        英語/日本語メッセージ
src/quality     (フェーズ2) ノイズラベル検出
src/moderation  (フェーズ3) 審査ルール・動画
src/api         (フェーズ3) FastAPI
configs/        実験設定YAML
docs/           デバッグ記録、アノテーションガイドライン
infra/          (フェーズ4) AWS
```

## 審査ルール(フェーズ3で実装予定)

1. 動物が検出されない → **却下**
2. 顔・ナンバー・文字を検出 → 自動でぼかして **公開可**。信頼度が低ければ **要確認**
3. 分類結果が投稿内容と食い違う(例:「猫」と投稿されたが犬と判定)→ **要確認**
4. それ以外 → **公開可**

## 結果

### フェーズ1: 分類(テスト分割 1,104枚、ResNet50、8エポック、seed 42、CE損失、リサンプリングなし)

| タスク | 正解率 | macro-F1 |
|---|---|---|
| 種別(猫/犬) | 0.9955 | 0.9948 |
| 品種(37クラス) | 0.9076 | 0.9070 |

種別の混同行列(行=正解、列=予測):

| | 予測:猫 | 予測:犬 |
|---|---|---|
| **正解:猫** | 353 | 2 |
| **正解:犬** | 3 | 746 |

F1が低い品種: staffordshire_bull_terrier 0.71、beagle 0.79、american_pit_bull_terrier 0.80、ragdoll 0.82、american_bulldog 0.82。
多い取り違え: ragdoll→birman(8)、newfoundland→leonberger(5)、british_shorthair→russian_blue(5)、basset_hound→beagle(4)、staffordshire_bull_terrier→american_bulldog(4)。いずれも見た目が似た組み合わせです。

クラス別の全指標と37×37の混同行列: [`docs/results/phase1_metrics.json`](docs/results/phase1_metrics.json)、設定: [`docs/results/phase1_config.yaml`](docs/results/phase1_config.yaml)
注意: 1回・1seedの結果です。検証精度はエポック4以降0.90前後で頭打ちかつ変動(0.896〜0.923)しており、1ポイント程度の差は有意ではありません。不均衡対策(weighted CE / sampler / Focal)は実装済みですが、まだ比較していません。

## ロードマップ

1. 分類と評価の土台 ✅(コード完了、結果は下記)
2. セグメンテーション・検出、アノテーション品質管理(Label Studio)、ノイズラベル検出、`docs/debugging.md`
3. 審査パイプライン、動画(ffmpeg)、FastAPIアプリ
4. AWS(S3 → Lambda → ECS、Batchで一括再評価)、ONNX化と量子化

## データとライセンス

Oxford-IIIT PetはCC BY-SA 4.0です。コミットするのは分割CSVとハッシュのみで、画像は含めません。
