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
uv run pawmatch train-seg             # セグメンテーション(configs/segmentation.yaml)
uv run pawmatch train-det             # 動物の頭部検出(configs/detection.yaml)
uv run pawmatch detect-noise runs/<run_dir>  # ラベル5%入れ替え実験
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
src/quality     アノテーション整合チェック、一致度、ノイズラベル検出
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

### フェーズ2(進行中): セグメンテーション、アノテーション品質管理

**セグメンテーション**(DeepLabV3-MobileNetV3、256px、4エポック、train全量、テスト1,104枚、自前指標):

| IoU | Dice | 境界IoU |
|---|---|---|
| 0.897 | 0.943 | 0.290 |

境界IoUがIoUより大幅に低いのは、Oxfordのトライマップの「境界」クラスが毛の縁の不確かな帯であり、指標が256px上の幅3pxの輪郭帯を見るためです。エッジ品質を厳しく見る指標で、欠陥ではありません。設定・指標: [`docs/results/phase2_seg_full_metrics.json`](docs/results/phase2_seg_full_metrics.json)

**ノイズラベル検出**(val+testの2,206枚で品種ラベルを5%入れ替え、ノイズ110件。フェーズ1の分類器を使用、この画像は学習に未使用):

| 手法 | 検出数 | 的中 | 検出率(Recall) | 適合率(Precision) |
|---|---|---|---|---|
| 与えたラベルの損失が上位5% | 110 | 101 | 0.918 | 0.918 |
| 予測 ≠ ラベル すべて | 285 | 110 | 1.000 | 0.386 |

正しいラベルでもモデル自身が8.5%を間違えるため、「食い違い」は多く拾います。詳細: [`docs/results/noise_detection.json`](docs/results/noise_detection.json)。1 seedのみ。

**前処理バグの実験:** [`docs/debugging.md`](docs/debugging.md) 参照。仕込んだ3つのバグでIoUは0.1〜1.7ポイントしか動かず(ノイズの範囲内)、マスクのリサイズバグだけが*境界*IoUの34%低下として現れました。

**物体検出**(Faster R-CNN MobileNetV3-FPN、COCO事前学習、3エポック、クラスは`animal_head`の1つ、テストは矩形のある571枚、自前指標):

| AP@0.5 | mAP@0.5:0.95 | Recall@IoU0.5 | 検出あり画像の割合(スコア0.5以上) |
|---|---|---|---|
| 0.988 | 0.723 | 0.993 | 0.991 |

注意: (1) Oxfordの矩形は動物の**頭部**のみで、矩形がある画像も約3.7千枚(train 2,551 / val 549 / test 571)です。そのため「動物が写っている」の代理としての頭部検出器です。(2) テスト画像はすべてペットが写っているので、動物のいない写真での誤検知は分かりません。審査ルール1(動物なし→却下)には、動物のいない画像(ネガティブ)での評価が別途必要です。(3) ペット中心の写真なのでAP@0.5は上限に近く、mAP@0.5:0.95(0.72)のほうが情報があります。(4) 顔・ナンバー・文字クラスはLabel Studioでの自作アノテーションが必要で、未学習です。1回のみの実行。ファイル: [`docs/results/phase2_det_metrics.json`](docs/results/phase2_det_metrics.json)

**フェーズ2で未実施:** Label Studioでの顔・ナンバー・文字の自作アノテーションと、それに依存する検出クラスです。指標(IoU、AP@0.5、mAP@0.5:0.95、クラス別Recall)はtorchmetrics/pycocotoolsと一致しています。

## ロードマップ

1. 分類と評価の土台 ✅(コード完了、結果は下記)
2. セグメンテーション ✅、アノテーション品質管理 ✅(ガイドライン、チェック、ノイズ検出)、デバッグ記録 ✅。動物の頭部検出 ✅。顔・ナンバー・文字のLabel Studioアノテーションは未実施
3. 審査パイプライン、動画(ffmpeg)、FastAPIアプリ
4. AWS(S3 → Lambda → ECS、Batchで一括再評価)、ONNX化と量子化

## データとライセンス

Oxford-IIIT PetはCC BY-SA 4.0です。コミットするのは分割CSVとハッシュのみで、画像は含めません。
