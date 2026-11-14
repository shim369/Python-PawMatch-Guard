import numpy as np
import pytest
import torch
from PIL import Image

from src.data.detection import PetDetectionDataset, parse_xml, with_boxes
from src.data.split import Entry

XML = (
    "<annotation><size><width>100</width><height>60</height></size><object><name>cat</name>"
    "<bndbox><xmin>10</xmin><ymin>5</ymin><xmax>40</xmax><ymax>30</ymax></bndbox></object>"
    "</annotation>"
)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations" / "xmls").mkdir(parents=True)
    Image.fromarray(np.zeros((60, 100, 3), np.uint8)).save(tmp_path / "images" / "a.jpg")
    (tmp_path / "annotations" / "xmls" / "a.xml").write_text(XML)
    return tmp_path


def test_parse_and_filter(root) -> None:
    boxes, w, h = parse_xml(root / "annotations" / "xmls" / "a.xml")
    assert boxes.tolist() == [[10, 5, 40, 30]] and (w, h) == (100, 60)
    entries = [Entry("a", 0, 0), Entry("missing", 0, 0)]
    assert [e.name for e in with_boxes(entries, root)] == ["a"]


def test_flip_mirrors_box(root, monkeypatch) -> None:
    monkeypatch.setattr("src.data.detection.random.random", lambda: 0.0)  # always flip
    x, t = PetDetectionDataset([Entry("a", 0, 0)], root, train=True)[0]
    assert x.shape == (3, 60, 100)
    assert t["boxes"].tolist() == [[60.0, 5.0, 90.0, 30.0]]
    assert t["labels"].dtype == torch.int64 and t["labels"].tolist() == [1]


def test_negative_dataset_gives_an_empty_target(tmp_path) -> None:
    from src.data.detection import NegativeDetectionDataset

    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(tmp_path / "n.jpg")
    x, t = NegativeDetectionDataset([tmp_path / "n.jpg"])[0]
    assert x.shape == (3, 20, 30)
    assert t["boxes"].shape == (0, 4) and t["labels"].shape == (0,)


def test_negatives_split_is_disjoint_deterministic(tmp_path) -> None:
    from src.data.negatives import read_negatives_split, split_negatives

    for i in range(10):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / f"{i}.jpg")
    a = split_negatives(tmp_path, tmp_path / "s.csv", train_n=4)
    b = split_negatives(tmp_path, tmp_path / "s2.csv", train_n=4)
    assert a == b and len(a["train"]) == 4 and len(a["test"]) == 6
    assert not set(a["train"]) & set(a["test"])
    assert read_negatives_split(tmp_path / "s.csv") == a
