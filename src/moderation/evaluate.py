"""End-to-end evaluation of the moderation pipeline.

Cases (expected verdict):
  * normal   - a pet photo, declared species correct          -> PUBLISH
  * mismatch - the same photo, declared species wrong          -> REVIEW
  * negative - a photo without any animal (optional folder)    -> REJECT

Costly errors are *misses*: a mismatch published, or a negative published.
Unnecessary reviews (normal -> REVIEW) only cost human time and are reported separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.oxford_pet import DATASET_DIR
from src.data.split import read_split_csv
from src.metrics import confusion_matrix

from .pipeline import ModerationPipeline
from .rules import Verdict, decide

NAMES = [v.name for v in (Verdict.PUBLISH, Verdict.REVIEW, Verdict.REJECT)]
EXPECTED = {"normal": Verdict.PUBLISH, "mismatch": Verdict.REVIEW, "negative": Verdict.REJECT}


def evaluate(pipe: ModerationPipeline, data_root: Path, split_csv: Path,
             negatives_dir: Path | None = None, limit: int | None = None,
             negative_ids: set[str] | None = None) -> dict:
    root = data_root / DATASET_DIR / "images"
    entries = read_split_csv(split_csv)["test"][:limit]
    cases: list[tuple[str, str, str, Verdict]] = []  # (kind, id, expected-name, verdict)
    for e in entries:
        img = np.asarray(Image.open(root / f"{e.name}.jpg").convert("RGB"))
        pred = pipe.predict(img)
        truth = "dog" if e.species_id else "cat"
        wrong = "cat" if truth == "dog" else "dog"
        cases.append(("normal", e.name, "", decide(pred, truth, pipe.th).verdict))
        cases.append(("mismatch", e.name, "", decide(pred, wrong, pipe.th).verdict))
    n_neg = 0
    if negatives_dir and negatives_dir.exists():
        for p in sorted(negatives_dir.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and (
                    negative_ids is None or p.stem in negative_ids):
                img = np.asarray(Image.open(p).convert("RGB"))
                cases.append(("negative", p.name, "", pipe.analyze(img, "cat").verdict))
                n_neg += 1

    y_true = np.array([int(EXPECTED[k]) for k, *_ in cases])
    y_pred = np.array([int(v) for *_, v in cases])
    cm = confusion_matrix(y_true, y_pred, 3)
    by_kind = {}
    for kind in EXPECTED:
        vs = [int(v) for k, *_, v in cases if k == kind]
        if vs:
            by_kind[kind] = {NAMES[i].lower(): vs.count(i) for i in range(3)} | {"n": len(vs)}
    n = len(cases)
    misses = [(k, i) for k, i, _, v in cases
              if (k == "mismatch" and v is Verdict.PUBLISH)
              or (k == "negative" and v is Verdict.PUBLISH)]
    return {
        "n_cases": n, "n_images": len(entries), "n_negatives": n_neg,
        "labels": NAMES, "confusion_matrix_rows_expected_cols_actual": cm.tolist(),
        "by_kind": by_kind,
        "human_review_rate": float((y_pred == int(Verdict.REVIEW)).mean()) if n else 0.0,
        "miss_count": len(misses), "miss_examples": misses[:20],
        "note": "negatives not evaluated" if n_neg == 0 else "includes negatives",
    }


def main(config: str, out: str, negatives: str | None, limit: int | None,
         negatives_split: str | None = None) -> None:
    from src.data.negatives import read_negatives_split

    from .torch_models import load_pipeline

    held_out = (set(read_negatives_split(Path(negatives_split))["test"])
                if negatives_split else None)

    res = evaluate(load_pipeline(Path(config)), Path("data"),
                   Path("data/splits/oxford_pet_seed42.csv"),
                   Path(negatives) if negatives else None, limit, held_out)
    Path(out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
