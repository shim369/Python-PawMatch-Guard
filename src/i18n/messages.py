"""Message catalog. Keep keys identical across languages (checked by tests)."""

from __future__ import annotations

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "split.done": "Split saved to {path}: train={train}, val={val}, test={test}",
        "split.dup": "Removed {n} duplicate images (same content hash)",
        "train.epoch": "epoch {epoch}/{epochs} loss={loss:.4f} val_breed_acc={acc:.4f}",
        "train.done": "Training finished. Results saved to {path}",
        "eval.header": "Test results",
        "eval.accuracy": "Accuracy",
        "eval.macro_f1": "macro-F1",
        "eval.species": "Species (cat/dog)",
        "eval.breed": "Breed (37 classes)",
        "cli.prepare": "Download Oxford-IIIT Pet and create stratified splits",
        "cli.train": "Train the classifier from a YAML config",
        "cli.evaluate": "Evaluate a saved run on the test split",
        "cli.noise": "Inject 5% label noise and measure how many are detected",
        "cat": "cat",
        "dog": "dog",
    },
    "ja": {
        "split.done": "分割を {path} に保存しました: train={train}, val={val}, test={test}",
        "split.dup": "重複画像(同一ハッシュ)を {n} 枚除外しました",
        "train.epoch": "エポック {epoch}/{epochs} 損失={loss:.4f} 検証品種精度={acc:.4f}",
        "train.done": "学習が完了しました。結果は {path} に保存しました",
        "eval.header": "テスト結果",
        "eval.accuracy": "正解率",
        "eval.macro_f1": "macro-F1",
        "eval.species": "種別(犬/猫)",
        "eval.breed": "品種(37クラス)",
        "cli.prepare": "Oxford-IIIT Petを取得し、層別分割を作成する",
        "cli.train": "YAML設定から分類モデルを学習する",
        "cli.evaluate": "保存済みの実行結果をテスト分割で評価する",
        "cli.noise": "ラベルを5%入れ替え、ノイズの検出率を測定する",
        "cat": "猫",
        "dog": "犬",
    },
}
