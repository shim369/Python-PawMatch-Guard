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
uv run pawmatch evaluate-moderation   # 審査全体の混同行列
uv run pawmatch serve                 # デモ用Webアプリ(認証なし)
```

日本語で出力: `PAWMATCH_LANG=ja uv run pawmatch train` または `uv run pawmatch --lang ja train`

AWSのコード経路をローカルで(S3 → SeaweedFS、DynamoDB → DynamoDB Local): `docker compose --profile aws-local up`

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
src/moderation  審査ルール、ぼかし、動画(ffmpeg)、パイプライン、評価
src/api         FastAPIアプリ(投稿・検索・確認キュー)
configs/        実験設定YAML
docs/           デバッグ記録、アノテーションガイドライン
infra/          Terraform(S3、Lambda、DynamoDB、ECR、Budgets)。未適用
```

## 審査ルール(`src/moderation/rules.py` に実装済み)

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

### フェーズ3: 審査パイプラインとアプリ

フェーズ1の**テスト分割**(ペット写真1,104枚。各画像を「正しい種別」と「誤った種別」で1回ずつ投稿、計2,208ケース。分類器+頭部検出器+ルール。`uv run pawmatch evaluate-moderation`)での全体評価:

| ケース(期待される判定) | 公開可 | 要確認 | 却下 | n |
|---|---|---|---|---|
| 種別が正しい(公開可) | 1,089 | 12 | 3 | 1,104 |
| 種別が誤り(要確認) | **4** | 1,097 | 3 | 1,104 |
| **動物なし(却下)** | **31** | 81 | 188 | 300 |

- 見逃し(種別が誤った投稿が公開された): 1,104件中4件。不要な確認: 12件(1.1%)。誤って却下された正しい投稿: 3件(検出器が頭部を見つけられなかった)。
- JSONの「人の確認に回る割合 50%」は**テスト設計による見かけの値**です(半分は意図的な誤り)。実際の投稿ではずっと低くなります。
- **動物のいない投稿(ルール1)は、実際の弱点でした。** 動物ラベルのないOpen ImagesのCC BY 2.0写真300枚(`scripts/build_negatives.py`。ランダムな48枚と、公開可になった31枚全部を目視で確認し、動物は写っていませんでした)では、**却下188枚(62.7%)、要確認81枚(27.0%)、公開可31枚(10.3%)** でした。頭部検出器はペットの写真だけで学習し、ネガティブを見ていないため、人の顔・頭(スコア0.6〜0.96)や、ときに食べ物・物(0.5〜0.8)に反応します。動物のしきい値を上げるだけでは、ペットの再現率を犠牲にしないと解決しません。0.9にしてもネガティブの4.3%が通り、ペット写真の3.0%が動物と判定されなくなります([`docs/results/negatives_score_analysis.json`](docs/results/negatives_score_analysis.json))。対策はネガティブ(人物を含む)を使った検出器の再学習ですが、未実施です。注意: Open Imagesでラベルがないことは動物がいない証明ではなく(目視確認の結果を参照)、300枚は小さなサンプルです。
- テスト写真は整ったペットの写真で、種別分類器の正解率も99.5%のため、雑多な実際の投稿に対しては楽観的な数字です。`configs/moderation.yaml` のしきい値は検証セットで調整していません。

**ネガティブを使った検出器の再学習。** ネガティブ300枚を150/150に分割しました(`data/splits/negatives_split.csv`、seed 42)。ペット写真と、*学習用*のネガティブ150枚(すべて背景の画像として3回ずつ。`configs/detection_neg.yaml`)で検出器を再学習し、どちらの検出器も見ていない残り150枚のネガティブで、同じパイプラインとしきい値(`--negatives-split`)により評価しました。

| 検出器 | 動物なし→却下 | 要確認 | **公開可(見逃し)** | 誤って却下されたペット写真 | 公開された種別誤りの投稿 |
|---|---|---|---|---|---|
| ペットのみ(前) | 91(60.7%) | 44 | **15(10.0%)** | 3 / 1,104 | 4 / 1,104 |
| +ネガティブ(後) | 116(77.3%) | 25 | **9(6.0%)** | 2 / 1,104 | 4 / 1,104 |

ペットのテストセットでも新しい検出器は同等以上です(AP@0.5 0.989、mAP@0.5:0.95 0.742。前は0.988 / 0.723。[`docs/results/phase3_det_neg_metrics.json`](docs/results/phase3_det_neg_metrics.json))。ネガティブでの学習は効きましたが、**動物のいない写真の6%はまだ公開されます**。見逃してはいけない審査としては多すぎます。テスト画像は150枚のみで、見逃し率(10.0%対6.0%)の差は約4ポイント、それぞれの標本誤差は約±4〜5ポイントなので、改善の方向は確かですが、大きさは正確には測れていません。次の手は、ネガティブ(人物・食べ物・物)を大幅に増やすこと、ハードネガティブの採掘、大きな評価用データの確保です。改善後の検出器は `configs/pipeline_neg.yaml` で使えます。`configs/pipeline.yaml`(ペットのみの検出器)は上の表を再現します。生データ: [`moderation_eval_heldout_old.json`](docs/results/moderation_eval_heldout_old.json)、[`moderation_eval_heldout_new.json`](docs/results/moderation_eval_heldout_new.json)

**既知の制約(重要):**
- 顔・ナンバー・文字の検出は**未学習**です。そのため、ぼかしの処理は実装済みでダミーによる単体テストはありますが、実モデルでは動作しません。
- 動画はフレームごと(1fps、最も危険なフレームを採用)に審査しますが、動画ファイル自体はぼかせません。個人情報を検出した動画は公開せず、要確認に回します。
- Webアプリには**認証がありません**(デモ用)。確認キューは誰でも操作できます。公開サーバーに置かないでください。
- 動画のフレーム抽出には動作するffmpegが必要です。開発機のHomebrew版は `libvpx` 欠落で壊れているため、単体テストはそこではスキップされます。ffmpegを入れたDockerイメージ内では、合成した3秒のクリップが3フレームに分割されることを確認しました。実際のユーザー動画をパイプライン全体・Webアプリで通した確認は未実施です。
- マッチングは意図的に軽くしてあり、種別(分類器の判定)による絞り込みとタイトル・説明の検索のみです。

アプリの起動: `uv run pawmatch serve` のあと http://127.0.0.1:8000 を開きます(日本語は `?lang=ja`)。先に `configs/pipeline.yaml` を自分の学習結果に向けてください(重みはGit管理外)。

### フェーズ4(一部): ONNX化、量子化、CPU速度

種別+品種の分類器(ResNet50)、CPUのみ。1枚あたりの推論時間 p50 / p95(ms、ウォームアップ5回後に40回)、精度はテスト先頭400枚。測定環境は **Apple M1(8コア)**、torch 2.15.0、onnxruntime 1.32.0で、AWSのCPUではありません。

| 方式 | 入力 | バッチ1(ms) | バッチ8(ms/枚) | 品種正解率 | 種別正解率 | PyTorchと一致 |
|---|---|---|---|---|---|---|
| pytorch_fp32 | 224 | 24.3 / 34.1 | 24.7 / 26.1 | 0.895 | 0.995 | 1.000 |
| onnx_fp32 | 224 | 52.7 / 81.9 | 52.6 / 65.5 | 0.895 | 0.995 | 1.000 |
| onnx_int8_dynamic | 224 | 26.5 / 34.1 | 27.0 / 29.3 | 0.885 | 0.998 | 0.965 |
| pytorch_fp32 | 160 | 15.9 / 21.0 | 13.2 / 17.1 | 0.900 | 0.995 | 1.000 |
| onnx_fp32 | 160 | 32.5 / 47.9 | 32.6 / 49.8 | 0.900 | 0.995 | 1.000 |
| onnx_int8_dynamic | 160 | 17.2 / 20.2 | 16.3 / 18.4 | 0.880 | 0.990 | 0.950 |

モデルサイズ: ONNX fp32 94MB、動的INT8 24MB。

表から言えること(1台・1回の測定、n=400のため精度には約±1.5ポイントの標本誤差があります):
- **このMacでは、ONNX Runtime fp32がPyTorchの約2倍*遅い*結果でした**(約52ms対約24ms)。スレッド数やグラフ最適化レベルを変えても変わらなかった(46〜57ms)ため、高速化とは主張せず、測定値のまま記載します。x86(AWSなど)で同じかは未測定です。
- 動的INT8はPyTorchと同等の速度(約26ms)で、サイズは4分の1。代償は品種正解率が約1ポイント低下(0.885対0.895)、予測一致率は96.5%です。
- 入力160pxは224pxより約35%速く、このサンプル数では精度低下は測定できませんでした。ただし学習は224pxなので、採用前にもっと大きなセットで確認が必要です。
- バッチサイズはCPUでは1枚あたりの時間にほとんど影響しませんでした。

未実施: 検出器(Faster R-CNN)とセグメンテーションのONNX化、AWSインフラ。生データ: [`docs/results/onnx_benchmark.json`](docs/results/onnx_benchmark.json)。再現: `uv run pawmatch benchmark runs/<分類器のrun>`

### フェーズ4(一部): AWS設計とコスト

設計(詳細と図: [`docs/architecture.md`](docs/architecture.md)、コード: [`infra/terraform`](infra/terraform)、[`src/aws_lambda.py`](src/aws_lambda.py)): S3へのアップロード → **コンテナLambda1つ**が審査 → 結果を**DynamoDB**へ。モデル更新後は、運用者が同じLambdaを `{"action": "reevaluate"}` で呼び、公開中の投稿を20件ずつ再確認します(自分自身を呼び直して続行)。再評価は判定を厳しくする方向にしか変えず、人が判断した投稿には触れません。

**コスト。** *AWSには何もデプロイしておらず、AWSの費用は発生していません。自分で `terraform apply` しない限り、このリポジトリがAWSリソースを作ることはありません。* 無料枠に収まるよう設計を変更しました。ECS/FargateとBatchは廃止(Fargateの無料枠は見つからず、Batch自体は下のFargate/EC2の料金が加わるだけ)、DynamoDBはプロビジョンド(オンデマンドは無料の容量単位がない)、AWS Budgetsの警告も追加しています。以下の条件はAWSのページと第三者の解説によるもので、デプロイ前に必ず再確認してください。

| サービス | 用途 | 無料枠 | 想定コスト |
|---|---|---|---|
| Lambda | 審査・再評価 | 常時無料: 月100万リクエスト、40万GB秒 | ポートフォリオ規模なら0円(下の測定値参照) |
| DynamoDB | 投稿 | 常時無料: 25GB、プロビジョンド25 WCU/RCU | 0円(25のうち10+10を使用) |
| S3 | アップロード、出力、重み | 無料枠はアカウント種別による。**未確認** | 数円以下 |
| CloudWatch Logs | 保持7日 | 無料枠は**未確認** | 数円以下 |
| ECR | Lambdaイメージ1つ、圧縮後約0.6GB | 新規は500MBを最初の12か月のみ無料 | **無料ではない**: 月およそ$0.01〜0.07 |
| AWS Budgets | 月$1の警告 | 最初の予算は無料 | 0円 |

したがって正直な答えは「ほぼ0円だが、ぴったり0円ではない」です。無料枠の対象外になるのはECRのコンテナイメージだけです(Lambdaのコンテナイメージには ECR が必須で、PyTorch は zip パッケージに収まりません)。

ローカルでの測定(Lambdaランタイムエミュレーター、Apple M1上のDocker、arm64イメージ。AWS上では**未測定**): S3から重みを取得するコールドスタートは2件で約11.4秒、ウォームは1件あたり約0.26秒、S3イベントの重複は約30ms。メモリ3GBでは、40万GB秒は「毎回コールドスタート」なら月約1万件程度、ウォームならはるかに多い規模です。実際の数字はデプロイして測る必要があります。

確認済み: `terraform validate` と `fmt` が通る(構文とスキーマのみの確認で、`apply` の成功やIAMの十分性は確認していません)。Lambdaイメージがビルドでき(1.78GB、CPU専用PyTorch)、Lambdaランタイムエミュレーター経由で、S3互換ストアから重みを取得し、投稿を審査してDynamoDB Localに書き込む。Webアプリが、SeaweedFS+DynamoDB Localで一連の流れを実行する。ストアはmotoで契約テスト済み。**未確認:** 実際の `terraform apply`、S3 → Lambdaのイベント連携、Lambdaの自己呼び出し(偽のクライアントによる単体テストのみ)、AWS上の実際の性能。

既知の不足: Lambdaイメージに `ffmpeg` が無いため、**AWSでは動画の投稿が `error` になります**(ローカルとDockerイメージでは動作)。FastAPIアプリはTerraformの対象外です。ONNX化は分類器のみで(MacではONNX RuntimeがPyTorchより遅かった。上記参照)、LambdaはPyTorchを使います。

## ロードマップ

1. 分類と評価の土台 ✅(コード完了、結果は下記)
2. セグメンテーション ✅、アノテーション品質管理 ✅(ガイドライン、チェック、ノイズ検出)、デバッグ記録 ✅。動物の頭部検出 ✅。顔・ナンバー・文字のLabel Studioアノテーションは未実施
3. 審査パイプライン ✅、FastAPIアプリ ✅。動画のフレーム抽出はDocker内で合成クリップのみ確認、ネガティブ画像での評価は実施済み(動物なし写真の10.3%が通過。検出器のネガティブ学習が必要)
4. ONNX化と量子化 ✅(分類器)。AWS: 無料枠に収まる設計を作成し、検証とローカルでの動作確認まで実施。**デプロイはしていません**(S3 → Lambda → DynamoDB、再評価は自己連鎖するLambda)

## データとライセンス

Oxford-IIIT PetはCC BY-SA 4.0です。コミットするのは分割CSVとハッシュのみで、画像は含めません。

ネガティブ画像300枚は[Open Images](https://github.com/openimages/dataset)由来です(画像は画像ごとのメタデータでCC BY 2.0、アノテーションはCC BY 4.0)。画像はコミットせず、[`data/negatives_manifest.csv`](data/negatives_manifest.csv) に帰属表示のための作者・タイトル・出典を記載し、`scripts/build_negatives.py` で再構築できます。
