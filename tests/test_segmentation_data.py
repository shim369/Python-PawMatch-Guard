import numpy as np
import pytest
import torch
from PIL import Image

from src.data.segmentation import PetSegmentationDataset, trimap_to_mask
from src.data.split import Entry


def test_trimap_to_mask() -> None:
    tri = np.array([[1, 2], [3, 2]])
    assert trimap_to_mask(tri).tolist() == [[1, 0], [1, 0]]


@pytest.fixture
def root(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations" / "trimaps").mkdir(parents=True)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)
    Image.fromarray(img).save(tmp_path / "images" / "a.jpg")
    tri = np.full((90, 120), 2, np.uint8)
    tri[20:70, 30:90] = 1
    tri[18:20, 30:90] = 3  # thin boundary line
    tri[40:60, 90:100] = 3  # boundary band on the right side
    Image.fromarray(tri).save(tmp_path / "annotations" / "trimaps" / "a.png")
    return tmp_path


def _get(root, bug):
    return PetSegmentationDataset([Entry("a", 0, 0)], root, size=48, bug=bug)[0]


def test_clean_pipeline_keeps_binary_mask(root) -> None:
    x, y = _get(root, "none")
    assert x.shape == (3, 48, 48) and set(y.unique().tolist()) <= {0, 1}


def test_bgr_swaps_channels(root) -> None:
    x, _ = _get(root, "none")
    xb, _ = _get(root, "bgr")
    assert not torch.allclose(x, xb)
    assert not torch.allclose(x[0], xb[0])


def test_double_norm_shifts_statistics(root) -> None:
    x, _ = _get(root, "none")
    xd, _ = _get(root, "double_norm")
    assert abs(xd.mean().item() - x.mean().item()) > 0.1


def test_linear_mask_differs_at_edges_only(root) -> None:
    _, y = _get(root, "none")
    _, yl = _get(root, "linear_mask")
    assert (y != yl).any()
    assert (y != yl).float().mean() < 0.05  # damage is concentrated on the boundary


def test_unknown_bug_rejected(root) -> None:
    with pytest.raises(ValueError):
        PetSegmentationDataset([Entry("a", 0, 0)], root, bug="nope")
