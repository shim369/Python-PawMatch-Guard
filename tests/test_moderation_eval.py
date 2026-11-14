import csv

import numpy as np
from PIL import Image

from src.moderation.evaluate import evaluate
from src.moderation.pipeline import ModerationPipeline
from src.moderation.rules import Detection


class Det:
    def detect(self, image):
        return [Detection("animal", 0.95, (0, 0, 9, 9))] if image.mean() > 50 else []


class Clf:
    def predict(self, image):
        return "cat", 0.99


def test_confusion_matrix_on_synthetic_data(tmp_path) -> None:
    root = tmp_path / "oxford-iiit-pet" / "images"
    root.mkdir(parents=True)
    rows = []
    for i, species in enumerate([0, 1, 0, 1]):  # 0 = cat, 1 = dog
        Image.fromarray(np.full((20, 20, 3), 200, np.uint8)).save(root / f"p{i}.jpg")
        rows.append([f"p{i}", 0, species, f"h{i}", "test"])
    with (tmp_path / "s.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "breed_id", "species_id", "sha256", "split"])
        w.writerows(rows)
    neg = tmp_path / "neg"
    neg.mkdir()
    Image.fromarray(np.zeros((20, 20, 3), np.uint8)).save(neg / "n.png")

    res = evaluate(ModerationPipeline(Det(), Clf()), tmp_path, tmp_path / "s.csv", neg)
    # classifier always says "cat": dogs declared dog -> mismatch review (unnecessary);
    # dogs declared cat -> published (a MISS).
    assert res["n_cases"] == 4 * 2 + 1 and res["n_negatives"] == 1
    assert res["by_kind"]["negative"]["reject"] == 1
    assert res["by_kind"]["normal"] == {"publish": 2, "review": 2, "reject": 0, "n": 4}
    assert res["by_kind"]["mismatch"]["publish"] == 2 and res["miss_count"] == 2
    cm = np.array(res["confusion_matrix_rows_expected_cols_actual"])
    assert cm.sum() == 9 and cm[2, 2] == 1
