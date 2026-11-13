# Debugging log / デバッグ記録

Three preprocessing bugs were injected on purpose (`bug:` in `configs/segmentation.yaml`),
one at a time, to record how the metrics break and how each bug is found.
前処理バグを意図的に1つずつ仕込み(`configs/segmentation.yaml` の `bug:`)、指標がどう崩れ、どう特定するかを記録します。

**Setup / 条件**: DeepLabV3-MobileNetV3, 256 px, 800 train images, 2 epochs, seed 42,
test split (1,104 images). **One run per row** – differences below ~1 pt IoU are within the
noise I would expect and were not checked with multiple seeds.
Raw numbers: [`results/phase2_bug_runs.json`](results/phase2_bug_runs.json).

| Run | IoU | Dice | Boundary IoU |
|---|---|---|---|
| baseline (no bug) | 0.831 | 0.903 | 0.167 |
| RGB/BGR swap | 0.813 (−1.7) | 0.891 | 0.163 |
| Normalization applied twice | 0.824 (−0.7) | 0.898 | 0.162 |
| Mask resized linearly – training labels only | 0.829 (−0.1) | 0.902 | 0.165 |
| Mask resized linearly – also used as evaluation GT | 0.817 (−1.4) | 0.894 | **0.111 (−34 %)** |

## What I found / 分かったこと

1. **The metrics barely moved for three of the bugs.** An ImageNet-pretrained backbone is
   tolerant of a channel swap or a double normalization, and the first two epochs mostly learn
   a coarse animal/background split. A drop of 1–2 pt IoU is easy to explain away as noise –
   which is exactly why these bugs survive in real projects. 3つのバグは指標がほとんど動かず、
   ノイズとして見過ごしやすい。
2. **The mask bug is visible only in the boundary metric.** Region IoU changed by 1.4 pt, boundary
   IoU by 34 %. When IoU looks fine but boundary IoU collapses, suspect the *label/resize
   pipeline*, not the model. 領域IoUは正常でも境界IoUだけ崩れるときは、モデルではなくラベルの
   リサイズ処理を疑う。
3. **Where a bug is applied matters.** Linear-resized masks used only for training changed
   almost nothing (−0.1 pt); the same masks used as evaluation ground truth made a good model
   look bad. 同じバグでも、学習ラベルに入ると影響は小さく、評価の正解に入ると良いモデルが悪く見える。

## How each bug is found and fixed / 特定と修正の手順

| Bug | Symptom | How to detect | Fix |
|---|---|---|---|
| RGB/BGR swap | small, uniform drop | Compare per-channel mean of a batch against ImageNet stats; view the image after the transform – skin/fur tones look blue | Convert once, right after loading; PIL gives RGB, OpenCV gives BGR |
| Mask resized linearly | boundary IoU collapses, region IoU ≈ normal | `np.unique` on the resized mask: values other than {0, 1} (or ids other than {1, 2, 3}) mean interpolation | Resize masks with nearest-neighbour only (`Image.NEAREST`) |
| Double normalization | small drop; slower convergence | Print `x.mean(), x.std()` of a batch: should be ≈0 / ≈1, not ≈0 / ≈4 | Normalize in exactly one place |

`tests/test_segmentation_data.py` pins each behaviour (the clean mask stays binary; BGR and
double-normalization change the tensor; the linear mask differs from the clean one only at
edges). 各挙動はテストで固定しています。

## Safeguards in place / 既存の安全策

- `tests/test_overfit.py`: the classifier loop, the segmentation model and the loss functions
  must overfit a tiny batch – a broken label/loss/optimizer path fails here first.
  少量データで過学習できるかのテスト。
- `src/quality/annotation_checks.py`: out-of-bounds, zero-area, wrong class ID, mask shape/values
  and mask/box mismatch. 整合チェック。

## Not yet covered / 未実施

- Multiple seeds to quantify run-to-run variance.
- The same three bugs on the classifier and the detector (detector is not trained yet).
