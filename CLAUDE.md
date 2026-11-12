# Python-PawMatch-Guard

## 目的
犬猫の里親マッチングアプリの投稿をAIで審査するポートフォリオ。
PyTorchでの分類・検出・セグメンテーション、評価指標の自前実装、
データセット設計、アノテーション品質管理、Docker、AWSのスキルを示す。

## ルール
- Python 3.12、uvで依存管理。型ヒント必須、ruffとpytestを通す
- 実験は configs/*.yaml で設定し、seedを固定。結果は runs/<日時>/ に設定と指標を保存
- 評価指標は src/metrics/ に自前で実装し、ライブラリの値と一致するかをテストで確認
- データは data/ に置き、Gitには入れない。分割CSVとハッシュの一覧だけコミット
- 学習・推論とも docker compose で再現できるようにする(GPUがない環境ではCPUで動くこと)
- フェーズの完了時に、READMEの結果の節を更新する

## 構成
src/data       データ取得・分割・前処理(OpenCV)・整合チェック
src/models     分類 / 検出 / セグメンテーション
src/metrics    IoU、Dice、AP、P/R/F1
src/quality    ノイズラベルの検出、アノテーション一致度
src/moderation 審査ルール、動画処理(ffmpeg)
src/api        FastAPI(投稿・一覧・検索・審査結果)
infra/         AWS構成(TerraformまたはCDK)
docs/          アノテーションガイドライン、debugging.md

## フェーズ1
Oxford-IIIT Petの取得と品種で層別した分割(70/15/15)。
ResNet50の転移学習で犬猫+37品種を分類。
P/R/F1と混同行列を自前で実装。Dockerfileとdocker composeを用意。

## フェーズ2〜4
企画書の「フェーズ計画」に従う。着手前に設計を提示し、承認を得てから実装する。
