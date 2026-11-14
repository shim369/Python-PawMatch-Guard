# PawMatch Guard

🇬🇧 English | [🇯🇵 日本語](README.ja.md)

A pet-adoption matching app with an **AI moderation pipeline** for user posts. Matching is kept light; the showcase is the image/video review pipeline: classification, detection and segmentation models decide *publish / needs review / reject*, and anything uncertain goes to a human.

> Status: **Phase 1 (classification + evaluation foundation) implemented.** Phases 2–4 are planned (see below).

## Features (Phase 1)

- Breed-stratified 70/15/15 split of Oxford-IIIT Pet, fixed seed, duplicate check by content hash, split saved as CSV
- ResNet50 transfer learning with two heads: species (cat/dog) and 37 breeds
- Class-imbalance options: weighted CE, `WeightedRandomSampler`, Focal Loss
- Metrics implemented from scratch (confusion matrix, accuracy, per-class P/R/F1, macro-F1) and tested against scikit-learn
- Config-driven runs saved to `runs/<timestamp>/` (config, metrics, weights)
- English (default) / Japanese UI for CLI output via `--lang` or `PAWMATCH_LANG`

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically).

```bash
uv sync
uv run pawmatch prepare-data          # download dataset (~800 MB) + create data/splits/*.csv
# official server can be very slow outside the UK; use the Hugging Face mirror instead:
uv run pawmatch prepare-data --source hf
uv run pawmatch train                 # uses configs/classification.yaml
uv run pawmatch report runs/<run_dir> # print test metrics
uv run pawmatch train-seg             # segmentation (configs/segmentation.yaml)
uv run pawmatch train-det             # animal-head detector (configs/detection.yaml)
uv run pawmatch detect-noise runs/<run_dir>  # 5% label-noise experiment
uv run pawmatch evaluate-moderation   # whole-pipeline confusion matrix
uv run pawmatch serve                 # demo web app (no authentication)
```

Japanese output: `PAWMATCH_LANG=ja uv run pawmatch train` or `uv run pawmatch --lang ja train`.

With Docker (CPU, no GPU needed):

```bash
docker compose run --rm prepare
docker compose run --rm train
```

Quality checks: `uv run ruff check . && uv run pytest`

## Layout

```
src/data        download, split, preprocessing
src/models      classifier, training
src/metrics     from-scratch metrics
src/i18n        en / ja messages
src/quality     annotation checks, agreement, noisy-label detection
src/moderation  rules, blur, video (ffmpeg), pipeline, evaluation
src/api         FastAPI app (posts, search, review queue)
configs/        experiment YAMLs
docs/           debugging log, annotation guidelines
infra/          (Phase 4) AWS
```

## Moderation rules (implemented in `src/moderation/rules.py`)

1. No animal detected → **reject**
2. Face / license plate / text detected → auto-blur, then **publish**; low confidence → **needs review**
3. Classification disagrees with the post (e.g. "cat" but model says dog) → **needs review**
4. Otherwise → **publish**

## Results

### Phase 1: classification (test split, 1,104 images, ResNet50, 8 epochs, seed 42, CE loss, no resampling)

| Task | Accuracy | macro-F1 |
|---|---|---|
| Species (cat / dog) | 0.9955 | 0.9948 |
| Breed (37 classes) | 0.9076 | 0.9070 |

Species confusion matrix (rows = true, columns = predicted):

| | pred cat | pred dog |
|---|---|---|
| **true cat** | 353 | 2 |
| **true dog** | 3 | 746 |

Weakest breeds (F1): staffordshire_bull_terrier 0.71, beagle 0.79, american_pit_bull_terrier 0.80, ragdoll 0.82, american_bulldog 0.82.
Most frequent confusions: ragdoll→birman (8), newfoundland→leonberger (5), british_shorthair→russian_blue (5), basset_hound→beagle (4), staffordshire_bull_terrier→american_bulldog (4). These are visually similar pairs.

Full per-class metrics and the 37×37 confusion matrix: [`docs/results/phase1_metrics.json`](docs/results/phase1_metrics.json); config: [`docs/results/phase1_config.yaml`](docs/results/phase1_config.yaml).
Caveats: single run, single seed; validation accuracy plateaued around 0.90 from epoch 4 and was noisy (0.896–0.923), so differences of ~1 point are not meaningful. Imbalance options (weighted CE / sampler / Focal) are implemented but not yet compared.

### Phase 2 (in progress): segmentation, annotation QA

**Segmentation** (DeepLabV3-MobileNetV3, 256 px, 4 epochs, full train split, test = 1,104 images, own metrics):

| IoU | Dice | Boundary IoU |
|---|---|---|
| 0.897 | 0.943 | 0.290 |

Boundary IoU is much lower than IoU because the Oxford trimap's "boundary" class is an uncertain band around the fur edge; the metric measures a 3 px contour band at 256 px, so it is a strict edge-quality indicator, not a defect. Config/metrics: [`docs/results/phase2_seg_full_metrics.json`](docs/results/phase2_seg_full_metrics.json).

**Noisy-label detection** (5 % of breed labels swapped on val+test, 2,206 images, 110 noisy; classifier from Phase 1, which never trained on these images):

| Method | Flagged | Caught | Recall | Precision |
|---|---|---|---|---|
| Top 5 % by loss of the given label | 110 | 101 | 0.918 | 0.918 |
| Any prediction ≠ label | 285 | 110 | 1.000 | 0.386 |

The model itself is wrong on 8.5 % of clean labels, which is why "disagree" flags so many. Details: [`docs/results/noise_detection.json`](docs/results/noise_detection.json). Single seed.

**Preprocessing-bug experiments:** see [`docs/debugging.md`](docs/debugging.md). Main finding: three injected bugs moved IoU by only 0.1–1.7 pt (within plausible noise); the mask-resize bug showed up as a 34 % drop in *boundary* IoU.

**Object detection** (Faster R-CNN MobileNetV3-FPN, COCO-pretrained, 3 epochs, one class `animal_head`, test = 571 images with boxes, own metrics):

| AP@0.5 | mAP@0.5:0.95 | Recall@IoU0.5 | Images with a detection (score ≥ 0.5) |
|---|---|---|---|
| 0.988 | 0.723 | 0.993 | 0.991 |

Caveats: (1) Oxford's boxes cover the animal's **head** only and only ~3.7k images have them (train 2,551 / val 549 / test 571), so this is a head detector used as a proxy for "an animal is visible". (2) Every test image contains a pet, so the numbers say nothing about false positives on photos *without* animals, which matters for moderation rule 1 ("no animal → reject"); that needs negative images. (3) The photos are pet-centred, so AP@0.5 is near its ceiling; mAP@0.5:0.95 (0.72) is the more informative number. (4) Face / plate / text classes need the custom Label Studio annotations and are not trained yet. Single run. Files: [`docs/results/phase2_det_metrics.json`](docs/results/phase2_det_metrics.json).

**Not done yet in Phase 2:** the custom face/plate/text annotations in Label Studio and the detector classes that depend on them. The metric code (IoU, AP@0.5, mAP@0.5:0.95, per-class recall) matches torchmetrics/pycocotools.

### Phase 3: moderation pipeline and app

End-to-end evaluation on the Phase 1 **test split** (1,104 pet photos, each submitted twice: once with the correct species and once with the wrong one; 2,208 cases; classifier + animal-head detector + rules; `uv run pawmatch evaluate-moderation`):

| Case (expected verdict) | Publish | Needs review | Reject | n |
|---|---|---|---|---|
| Correct species (publish) | 1,089 | 12 | 3 | 1,104 |
| Wrong species (needs review) | **4** | 1,097 | 3 | 1,104 |

- Misses (a wrong-species post published): 4 of 1,104. Unnecessary reviews: 12 of 1,104 (1.1 %). Valid posts wrongly rejected: 3 of 1,104 (the detector found no head).
- The 50 % "human review rate" in the JSON is **an artefact of the test design** (half the cases are deliberately wrong); on real traffic it would be much lower.
- **Not evaluated: posts without any animal** (rule 1). That needs negative photos; the harness accepts a folder (`--negatives DIR`) but no such set is included yet, so the reject rate on real non-animal photos is unknown. Only a flat grey image was checked by hand (rejected).
- Test photos are clean pet portraits and the species classifier is 99.5 % accurate on them, so these numbers are optimistic for messy user uploads. Thresholds in `configs/moderation.yaml` were not tuned on a validation set.

**Known limitations (important):**
- Face / licence-plate / text detection is **not trained**, so the PII blur path is implemented and unit-tested with fakes but never triggers with the real models.
- Videos are screened frame by frame (1 fps, worst frame wins) but the video file itself is not blurred, so any video with detected personal information is sent to review rather than published.
- The web app has **no authentication** (demo only); the review queue is open. Do not expose it publicly.
- Video extraction needs a working `ffmpeg`. It could not be run on the development machine (the Homebrew build was missing `libvpx`) and the Docker image was not rebuilt/run after adding ffmpeg, so that path is covered by a test that is skipped here and is unverified end to end.
- Matching is intentionally light: filter by species (taken from the classifier) and search title/description.

Run the app: `uv run pawmatch serve` then open http://127.0.0.1:8000 (`?lang=ja` for Japanese). Point `configs/pipeline.yaml` at your trained runs first (weights are not in Git).

## Roadmap

1. Classification & evaluation foundation ✅ (code), results below
2. Segmentation ✅, annotation QA ✅ (guidelines, checks, noise detection), debugging log ✅; animal-head detector ✅; Label Studio annotation of face/plate/text pending
3. Moderation pipeline ✅, FastAPI app ✅, video (ffmpeg) implemented but not verified end to end; negative-photo evaluation pending
4. AWS (S3 → Lambda → ECS, Batch re-evaluation), ONNX + quantization

## Data & license

Oxford-IIIT Pet is licensed CC BY-SA 4.0. Only split CSVs and hashes are committed; images are not.
