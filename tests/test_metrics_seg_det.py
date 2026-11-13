import numpy as np
import pytest
import torch
from torchmetrics.detection import MeanAveragePrecision

from src.metrics.detection import (
    ImageDetections,
    average_precision,
    box_iou,
    class_recall,
    mean_ap,
)
from src.metrics.segmentation import boundary_band, boundary_iou, dice, iou


def _rand_masks(seed: int):
    rng = np.random.default_rng(seed)
    t = np.zeros((64, 64), bool)
    t[10:50, 15:55] = True
    p = np.roll(t, shift=int(rng.integers(-6, 7)), axis=int(rng.integers(0, 2)))
    return p, t


@pytest.mark.parametrize("seed", range(4))
def test_iou_dice_match_torchmetrics(seed: int) -> None:
    from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index

    p, t = _rand_masks(seed)
    pt, tt = torch.from_numpy(p).int(), torch.from_numpy(t).int()
    assert iou(p, t) == pytest.approx(binary_jaccard_index(pt, tt).item(), abs=1e-6)
    # binary Dice is identical to F1
    assert dice(p, t) == pytest.approx(binary_f1_score(pt, tt).item(), abs=1e-6)


def test_empty_and_boundary() -> None:
    z = np.zeros((8, 8), bool)
    assert iou(z, z) == 1.0 and dice(z, z) == 1.0
    p, t = _rand_masks(0)
    assert boundary_iou(t, t) == 1.0
    assert boundary_iou(p, t) <= iou(p, t) + 1e-9  # edges are harder than area
    band = boundary_band(t, 2)
    assert band.sum() < t.sum() and (band & ~t).sum() == 0


def test_box_iou() -> None:
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(box_iou(a, b), [[1 / 3, 0.0]])


def _scene(seed: int):
    rng = np.random.default_rng(seed)
    ims, tm_p, tm_t = [], [], []
    for _ in range(25):
        n = int(rng.integers(1, 4))
        xy = rng.uniform(0, 150, (n, 2))
        gt = np.hstack([xy, xy + rng.uniform(30, 80, (n, 2))])
        gl = rng.integers(0, 3, n)
        keep = rng.random(n) < 0.8
        pb = gt[keep] + rng.normal(0, 4, (int(keep.sum()), 4))
        pl = np.where(rng.random(len(pb)) < 0.9, gl[keep], rng.integers(0, 3, len(pb)))
        extra = rng.uniform(0, 200, (2, 4)).clip(0)
        extra[:, 2:] = extra[:, :2] + 40
        pb = np.vstack([pb, extra])
        pl = np.concatenate([pl, rng.integers(0, 3, 2)])
        ps = rng.uniform(0.05, 1.0, len(pb))
        ims.append(ImageDetections(pb, ps, pl, gt, gl))
        tm_p.append({"boxes": torch.tensor(pb, dtype=torch.float32),
                     "scores": torch.tensor(ps, dtype=torch.float32),
                     "labels": torch.tensor(pl)})
        tm_t.append({"boxes": torch.tensor(gt, dtype=torch.float32), "labels": torch.tensor(gl)})
    return ims, tm_p, tm_t


@pytest.mark.parametrize("seed", [0, 1])
def test_map_matches_torchmetrics(seed: int) -> None:
    ims, tp, tt = _scene(seed)
    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
    m.update(tp, tt)
    ref = m.compute()
    classes = [0, 1, 2]
    ap50 = np.mean([average_precision(ims, c, 0.5) for c in classes])
    assert ap50 == pytest.approx(ref["map_50"].item(), abs=1e-3)
    assert mean_ap(ims, classes) == pytest.approx(ref["map"].item(), abs=1e-3)


def test_recall_bounds() -> None:
    ims, _, _ = _scene(0)
    for c in range(3):
        assert 0.0 <= class_recall(ims, c) <= 1.0
    assert average_precision(ims, 99) == -1.0
